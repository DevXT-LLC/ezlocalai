#include "ezlocalai/hub_snapshot.hpp"
#include "ezlocalai/whisper_model.hpp"
#include "ezlocalai/stt_runtime.hpp"
#include "whisper_transcription_test.hpp"
#include <arpa/inet.h>
#include <atomic>
#include <mutex>
#include <poll.h>
#include <sys/socket.h>
#include <thread>
using namespace transcription_test;
namespace {
std::string hex(const std::string &bytes) { std::ostringstream out; for (unsigned char c : bytes) out << std::hex << std::setfill('0') << std::setw(2) << unsigned(c); return out.str(); }
void write(const fs::path &path,const std::string &bytes) { fs::create_directories(path.parent_path()); std::ofstream out(path,std::ios::binary); out.write(bytes.data(),bytes.size()); expect(bool(out),"fixture write"); }
std::string replace(std::string value,const std::string &from,const std::string &to) { for (size_t pos = 0; (pos = value.find(from,pos)) != std::string::npos; pos += to.size()) value.replace(pos,from.size(),to); return value; }
struct Reply { int status = 200; std::map<std::string,std::string> headers; std::string body; bool length = true; };
class Server {
public:
    using Handler = std::function<Reply(const std::string &,const std::string &,const std::map<std::string,std::string> &)>;
    explicit Server(Handler handler) : handler_(std::move(handler)) {
        fd_ = socket(AF_INET,SOCK_STREAM,0); expect(fd_ >= 0,"server socket"); sockaddr_in address{}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        expect(!bind(fd_,reinterpret_cast<sockaddr *>(&address),sizeof(address)),"server bind"); socklen_t size = sizeof(address); getsockname(fd_,reinterpret_cast<sockaddr *>(&address),&size);
        url = "http://127.0.0.1:" + std::to_string(ntohs(address.sin_port)); expect(!listen(fd_,32),"server listen"); thread_ = std::thread([this] { serve(); });
    }
    ~Server() { stopped_ = true; shutdown(fd_,SHUT_RDWR); thread_.join(); close(fd_); }
    std::string url;
private:
    int fd_; std::atomic<bool> stopped_{false}; std::thread thread_; Handler handler_;
    void serve() {
        while (!stopped_) {
            pollfd ready{fd_,POLLIN,0}; if (poll(&ready,1,100) <= 0) continue; const int client = accept(fd_,nullptr,nullptr); if (client < 0) continue;
            timeval timeout{2,0}; setsockopt(client,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout)); std::string wire; char bytes[4096];
            while (wire.find("\r\n\r\n") == std::string::npos) { const auto length = recv(client,bytes,sizeof(bytes),0); if (length <= 0) break; wire.append(bytes,length); }
            std::istringstream input(wire); std::string first,method,path; std::getline(input,first); std::istringstream(first) >> method >> path; std::map<std::string,std::string> headers;
            for (std::string line; std::getline(input,line);) { const auto at = line.find(':'); if (at != std::string::npos) { auto key = line.substr(0,at); for (auto &c : key) c = std::tolower(c); auto value = line.substr(at + 1); while (!value.empty() && std::isspace(value.front())) value.erase(0,1); while (!value.empty() && std::isspace(value.back())) value.pop_back(); headers[key] = value; } }
            const auto reply = handler_(method,path,headers); const std::map<int,std::string> reasons{{200,"OK"},{206,"Partial Content"},{302,"Found"},{307,"Temporary Redirect"},{400,"Bad Request"},{401,"Unauthorized"},{403,"Forbidden"},{404,"Not Found"},{429,"Too Many Requests"},{503,"Service Unavailable"}};
            std::ostringstream response; response << "HTTP/1.0 " << reply.status << ' ' << reasons.at(reply.status) << "\r\n";
            for (const auto &header : reply.headers) response << header.first << ": " << header.second << "\r\n";
            if (reply.length) response << "Content-Length: " << reply.body.size() << "\r\n";
            response << "Content-Type: " << (path.rfind("/api/",0) == 0 || reply.status >= 400 ? "application/json" : "application/octet-stream") << "\r\n\r\n";
            if (method != "HEAD") response << reply.body; const auto text = response.str(); size_t sent = 0;
            while (sent < text.size()) { const auto length = send(client,text.data() + sent,text.size() - sent,MSG_NOSIGNAL); if (length <= 0) break; sent += length; } close(client);
        }
    }
};
void test(const Json &row,bool single_file = false) {
    const auto mode = row.at("mode").get<std::string>(); Temporary temporary; const auto cache = temporary.path/"cache", repo = cache/"models--Owner--Model";
    unsetenv("HF_HUB_OFFLINE"); unsetenv("HF_HUB_DISABLE_IMPLICIT_TOKEN"); setenv("HF_TOKEN","test-fixture",1);
    const std::string commit(40,'a'), newer(40,'b'), original("model fixture\0\n",15); auto current = commit,payload = original; const auto blob = hash(original);
    Json trace = Json::array(); std::map<std::string,int> counts; std::mutex mutex; std::string url,cdn_url;
    const auto handler = [&](const std::string &host,const std::string &method,const std::string &path,const std::map<std::string,std::string> &request) {
        std::lock_guard<std::mutex> guard(mutex); const auto range = request.count("range") ? request.at("range") : "";
        trace.push_back({{"host",host},{"method",method},{"path",path},{"auth",request.count("authorization") != 0},{"range",range}});
        const auto count = ++counts[host + method + path]; Reply out; out.body = payload;
        if (host == "cdn") out.headers["Content-Disposition"] = "attachment; filename=\"model.bin\"";
        else if (path.rfind("/api/",0) == 0) {
            out.body = Json({{"id","Owner/Model"},{"sha",current},{"siblings",Json::array({Json{{"rfilename","model.bin"}},Json{{"rfilename","config.json"}},Json{{"rfilename",mode == "parallel_shared" ? "vocabulary.txt" : "nested/unused.txt"}}})}}).dump();
            if (mode == "tree" || mode == "tree_pagination" || mode == "tree_failure") {
                if (path.find("/tree/") == std::string::npos) out.body = Json({{"id","Owner/Model"},{"sha",current},{"siblings",Json::array()}}).dump();
                else {
                    out.body = Json::array({Json{{"type","file"},{"path","model.bin"},{"oid",std::string(40,'c')},{"size",payload.size()}}}).dump();
                    if (mode == "tree_failure") { out.status = 403; out.body = R"({"error":"tree denied"})"; }
                    if (mode == "tree_pagination" && path.find("page=2") == std::string::npos) { out.body = "[]"; out.headers["Link"] = "<" + url + "/api/models/Owner/Model/tree/main?page=2>; rel=\"next\""; }
                }
            }
            if (mode == "api_forbidden" || mode == "api_forbidden_cached") { out.status = 403; out.body = R"({"error":"denied"})"; }
            if (mode == "api_unauthorized") { out.status = 401; out.body = R"({"error":"credentials invalid"})"; out.headers["X-Error-Message"] = "Invalid credentials in Authorization header"; }
            if (mode == "api_missing") { out.status = 404; out.body = R"({"error":"absent"})"; out.headers["X-Error-Code"] = "RepoNotFound"; }
            if (mode == "api_revision_cached") { out.status = 404; out.body = R"({"error":"gone"})"; out.headers["X-Error-Code"] = "RevisionNotFound"; }
        } else if (method == "HEAD") {
            out.headers = {{"X-Repo-Commit",current},{"ETag","\"" + hash(payload) + "\""},{"X-Linked-Size",std::to_string(payload.size() + (mode == "wrong_size"))}};
            if (mode == "missing_etag") out.headers.erase("ETag");
            if (mode == "missing_commit") out.headers.erase("X-Repo-Commit");
            if (mode == "missing_size") { out.headers.erase("X-Linked-Size"); out.length = false; }
            if (mode == "head_missing") { out.status = 404; out.headers["X-Error-Code"] = "EntryNotFound"; }
            if (mode == "head_revision") { out.status = 404; out.headers["X-Error-Code"] = "RevisionNotFound"; }
            if (mode == "head_forbidden") out.status = 403;
            if (mode == "head_retry" && count == 1) out.status = 503;
            if (mode == "relative_redirect" && path != "/canonical") { out.status = 307; out.headers = {{"Location","/canonical"}}; }
            if (mode == "cdn_redirect") { out.status = 302; out.headers["Location"] = cdn_url + "/blob"; }
        } else {
            out.headers["Content-Disposition"] = "attachment; filename=\"model.bin\"";
            if (mode == "get_failure") { out.status = 400; out.body = R"({"error":"download denied"})"; }
            if (mode == "get_rate_limit" && count == 1) { out.status = 429; out.body.clear(); }
            if (!range.empty() && mode != "resume_ignored") { const auto start = std::stoul(range.substr(6)); out.status = 206; out.body = payload.substr(start); out.headers["Content-Range"] = "bytes " + std::to_string(start) + "-" + std::to_string(payload.size() - 1) + "/" + std::to_string(payload.size()); }
        } return out;
    };
    Server cdn([&](const auto &method,const auto &path,const auto &headers) { return handler("cdn",method,path,headers); }); cdn_url = cdn.url;
    Server server([&](const auto &method,const auto &path,const auto &headers) { return handler("hub",method,path,headers); }); url = server.url;
    if (mode == "api_forbidden_cached" || mode == "api_revision_cached" || mode == "offline_ref" || mode == "offline_commit" || mode == "offline_partial") {
        write(repo/"refs/main",commit); fs::create_directories(repo/"snapshots"/commit); if (mode != "offline_partial") write(repo/"snapshots"/commit/"model.bin",payload);
    }
    if (mode == "blob_only") write(repo/"blobs"/blob,payload);
    if (mode == "resume" || mode == "resume_ignored" || mode == "resume_complete") write(repo/"blobs"/(blob + ".incomplete"),mode == "resume_complete" ? payload : payload.substr(0,5));
    if (mode == "refs_directory") fs::create_directories(repo/"refs/main");
    if (mode.rfind("offline_",0) == 0) setenv("HF_HUB_OFFLINE","1",1);
    HubSnapshotOptions options; options.cache_directory = cache; options.endpoint = url; options.max_workers = 1; options.allow_patterns = {mode == "filter_none" ? "absent" : "model.bin"}; options.revision = mode == "offline_commit" ? commit : "main"; options.local_files_only = mode == "local_only_missing";
    if (mode == "parallel_shared") { options.max_workers = 8; options.allow_patterns = {"model.bin","config.json","vocabulary.*"}; }
    const auto normalize = [&](std::string value) {
        // The single-file consistency error truncates its URL before formatting,
        // leaving only the ephemeral port and path rather than the whole origin.
        value = replace(value,"(…)" + url.substr(url.rfind(':') + 1),"(…){hub-port}");
        return replace(replace(replace(value,temporary.path,"/work"),url,"{hub}"),cdn_url,"{cdn}");
    };
    Json outcomes = Json::array();
    for (int step = 0; step < (mode == "reuse" || mode == "update" || mode == "same_blob" || mode == "commit_reuse" ? 2 : 1); ++step) {
        if (step) {
            std::lock_guard<std::mutex> guard(mutex);
            if (mode == "update" || mode == "same_blob") current = newer;
            if (mode == "update") payload = "updated weights\n"; if (mode == "commit_reuse") options.revision = commit;
        }
        try { outcomes.push_back({{"path",normalize(single_file ? hub_snapshot_file("Owner/Model","model.bin",options) : hub_snapshot_download("Owner/Model",options))}}); }
        catch (const HubError &error) { outcomes.push_back({{"error",{{"category",error.category},{"message",normalize(error.what())}}}}); }
    }
    Json files = Json::object(), directories = Json::array(); std::vector<fs::path> paths;
    if (fs::exists(cache)) for (const auto &entry : fs::recursive_directory_iterator(cache)) paths.push_back(entry.path());
    std::sort(paths.begin(),paths.end());
    for (const auto &path : paths) {
        const auto name = path.lexically_relative(cache).string();
        if (fs::is_symlink(path)) files[name] = {{"link",fs::read_symlink(path).string()}};
        else if (fs::is_regular_file(path)) files[name] = {{"hex",hex(read(path))}};
        else if (fs::is_directory(path)) directories.push_back(name);
    }
    compare_json(outcomes,row.at("outcomes"),mode + ".outcomes");
    {
        std::lock_guard<std::mutex> guard(mutex);
        if (mode == "parallel_shared") {
            for (auto &event : trace) if (event["method"] == "GET" && event["path"].get<std::string>().find("/resolve/") != std::string::npos) {
                auto path = event["path"].get<std::string>(); event["path"] = path.substr(0,path.rfind('/')) + "/{shared-blob}";
            }
            auto expected = row.at("trace");
            const auto order = [](const Json &a,const Json &b) { return a.dump() < b.dump(); };
            std::sort(trace.begin(),trace.end(),order); std::sort(expected.begin(),expected.end(),order); compare_json(trace,expected,mode + ".trace");
        } else compare_json(trace,row.at("trace"),mode + ".trace");
    }
    compare_json(files,row.at("files"),mode + ".files"); compare_json(directories,row.at("directories"),mode + ".directories");
}
void model_policy(const fs::path &fixtures) {
    const auto cases = records(fixtures/"whisper_model_cases.jsonl"); expect(cases.size() == 505,"complete model policy fixtures");
    expect(cases[0]["source_sha256"]["legacy"] == "0da077f09077bb428a380a5ca1ea909a04d8d9379de3eb632252d7d40ec22647" &&
           cases[0]["source_sha256"]["modern"] == "ad0401e8d5aeb9e5f226b52e04a0544a2be6afaaa1a00b1593928c0e61446438","pinned model helper sources");
    for (size_t i = 1; i < cases.size(); ++i) {
        const auto &row = cases[i]; Json calls = Json::array(),outcome;
        const auto download = [&](const std::string &repo,const HubSnapshotOptions &options) {
            calls.push_back({{"repo",repo},{"cache",options.cache_directory},{"local_only",options.local_files_only},{"patterns",options.allow_patterns},{"revision",options.revision}});
            if (calls.size() == 1 && !row["failure"].is_null()) {
                const auto type = row["failure"].get<std::string>(); throw HubError(type,"fixture failure",0,type == "HfHubHTTPError" || type == "RepositoryNotFoundError");
            } return std::string("/snapshot/model");
        };
        try { outcome["path"] = download_whisper_model(row.at("name"),"models",row["profile"] == "legacy" ? WhisperTranscriptionProfile::faster_whisper_1_0_3 : WhisperTranscriptionProfile::faster_whisper_1_2_0,download); }
        catch (const HubError &error) { outcome["error"] = {{"category",error.category},{"message",error.what()}}; }
        catch (const SttError &error) { outcome["error"] = {{"category",error.category},{"message",error.what()}}; }
        const auto label = "model policy " + std::to_string(i); auto expected = Json::object();
        if (row.contains("path")) expected["path"] = row["path"]; else expected["error"] = row["error"];
        compare_json(outcome,expected,label); compare_json(calls,row.at("calls"),label + ".calls");
    }
    Temporary temporary; HubSnapshotOptions options; options.cache_directory = temporary.path; options.local_files_only = true;
    const auto repositories = records(fixtures/"hub_repository_cases.jsonl"); expect(repositories.size() == 30,"complete repository cases");
    for (const auto &row : repositories) {
        Json actual;
        try { hub_snapshot_download(row.at("name"),options); throw std::runtime_error("unexpected cached repository"); }
        catch (const HubError &error) { if (error.category == "LocalEntryNotFoundError") actual["valid"] = true; else actual["error"] = {{"category",error.category},{"message",error.what()}}; }
        auto expected = row; expected.erase("name"); compare_json(actual,expected,"repository " + row.at("name").get<std::string>());
    }
    const auto local = temporary.path/"local"; fs::create_directory(local);
    expect(resolve_whisper_model(local,"unused",WhisperTranscriptionProfile::faster_whisper_1_0_3) == local,"local directory bypasses alias/download validation");
}
}
int main(int argc,char **argv) { try {
    if (argc == 5 && std::string(argv[1]) == "--download-tokenizer") {
        HubSnapshotOptions options; options.cache_directory = argv[4]; options.revision = argv[3];
        std::cout << hub_snapshot_file(argv[2],"tokenizer.json",options) << '\n'; return 0;
    }
    if (argc == 5 && std::string(argv[1]) == "--download") {
        HubSnapshotOptions options; options.cache_directory = argv[4]; options.revision = argv[3]; options.allow_patterns = {"config.json","preprocessor_config.json","model.bin","tokenizer.json","vocabulary.*"};
        std::cout << hub_snapshot_download(argv[2],options) << '\n'; return 0;
    }
    expect(argc == 2,"Hub snapshot fixture required"); const auto cases = records(argv[1]); expect(cases.size() == 38 && cases[0]["hub_version"] == "1.5.0","pinned complete snapshot captures");
    compare_json(cases[0].at("source_sha256"),Json({
        {"_snapshot_download.py","6109e5fdfa77522a8c25fd59f8760d1e08b09b840e78285076e4205e89a2580f"},
        {"file_download.py","2688890695096b61aa1e0619f0aaa227de281b3d14574ca2d890e891096ad14b"},
        {"utils/_http.py","34eff3ea1b15afc83e3caa3684bfb8499f4ce7486c1445a6bc3fa95afc1dad73"},
        {"utils/_validators.py","0370645dba57e0a9d40f6585b183a09179d87692e25d71bc29bc8d76ad2acfbf"}}),"Hub source identities");
    expect(cases[0].at("dependencies").at("requests") == "2.33.0","reference requests pin");
    for (size_t i = 1; i < cases.size(); ++i) test(cases[i]);
    const auto files = records(fs::path(argv[1]).parent_path()/"hub_file_cases.jsonl");
    expect(files.size() == 28,"complete single-file Hub captures");
    compare_json(files[0].at("source_sha256"),cases[0].at("source_sha256"),"single-file source identities");
    for (size_t i = 1; i < files.size(); ++i) test(files[i],true);
    model_policy(fs::path(argv[1]).parent_path());
    std::cout << cases.size() - 1 << " snapshot and " << files.size() - 1 << " single-file Hub 1.5.0 cases match results, errors, requests and complete cache state; 504 model policy and 30 repository cases match\n"; return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; } }
