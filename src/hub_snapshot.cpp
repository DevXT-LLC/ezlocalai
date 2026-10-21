#include "ezlocalai/hub_snapshot.hpp"
#include "ezlocalai/hub_assets.hpp"
#include "ezlocalai/configuration.hpp"
#include "ezlocalai/http_url.hpp"
#include "ezlocalai/process.hpp"
#include "ezlocalai/llm_startup.hpp"
#include <nlohmann/json.hpp>
#include <unicode/uchar.h>
#include <unicode/utf8.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <fnmatch.h>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <sys/file.h>
#include <sys/stat.h>
#include <thread>

namespace ezlocalai {
namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;
std::string env(const char *name,const std::string &fallback = {}) { const auto *value = std::getenv(name); return value ? value : fallback; }
std::string lower(std::string value) { for (auto &c : value) if (c >= 'A' && c <= 'Z') c += 'a' - 'A'; return value; }
bool enabled(const char *name) { const auto value = lower(env(name)); return value == "1" || value == "yes" || value == "true" || value == "on"; }
std::string read(const fs::path &path) { std::ifstream in(path,std::ios::binary); return {std::istreambuf_iterator<char>(in),{}}; }
void write(const fs::path &path,const std::string &bytes) {
    fs::create_directories(path.parent_path()); std::ofstream out(path,std::ios::binary); out.write(bytes.data(),bytes.size());
    if (!out) throw HubError("OSError","Cannot write Hub cache file: " + path.string());
}
bool path_exists(const fs::path &path) { std::error_code ignored; return fs::exists(path,ignored); }
bool commit(const std::string &value) { return value.size() == 40 && value.find_first_not_of("0123456789abcdef") == std::string::npos; }
std::string escape(const std::string &value,bool slash = true) {
    const char *hex = "0123456789ABCDEF"; std::string result;
    for (unsigned char c : value) {
        if (std::isalnum(c) && c < 128 || c == '-' || c == '_' || c == '.' || c == '~' || slash && c == '/') result += c;
        else { result += '%'; result += hex[c >> 4]; result += hex[c & 15]; }
    } return result;
}
void validate(const std::string &repository) {
    const auto fail = [&](const std::string &message) { throw HubError("HFValidationError",message); };
    if (std::count(repository.begin(),repository.end(),'/') > 1)
        fail("Repo id must be in the form 'repo_name' or 'namespace/repo_name': '" + repository + "'. Use \x60repo_type\x60 argument if needed.");
    // Python's regex end anchor permits one final newline.
    auto match = repository; if (!match.empty() && match.back() == '\n') match.pop_back();
    bool valid = true; size_t begin = 0;
    for (;;) {
        const auto end = match.find('/',begin); const auto part = match.substr(begin,end == std::string::npos ? end : end - begin);
        int32_t at = 0; size_t count = 0; bool first = false,last = false;
        while (at < static_cast<int32_t>(part.size())) {
            UChar32 c; U8_NEXT(part.data(),at,static_cast<int32_t>(part.size()),c);
            const auto type = u_charType(c);
            const bool word = c == '_' || u_isalpha(c) || type == U_DECIMAL_DIGIT_NUMBER || type == U_LETTER_NUMBER || type == U_OTHER_NUMBER;
            if (!count++) first = word; last = word; if (!word && c != '-' && c != '.') valid = false;
        }
        if (!count || !first || !last || end == std::string::npos && count > 96) valid = false;
        if (end == std::string::npos) break; begin = end + 1;
    }
    if (!valid) fail("Repo id must use alphanumeric chars, '-', '_' or '.'. The name cannot start or end with '-' or '.' and the maximum length is 96: '" + repository + "'.");
    if (repository.find("--") != std::string::npos || repository.find("..") != std::string::npos) fail("Cannot have -- or .. in repo_id: '" + repository + "'.");
    if (repository.size() >= 4 && repository.substr(repository.size() - 4) == ".git") fail("Repo_id cannot end by '.git': '" + repository + "'.");
}
std::string folder(const std::string &repository) { std::string out = "models--"; for (char c : repository) out += c == '/' ? "--" : std::string(1,c); return out; }
std::string token() {
    if (enabled("HF_HUB_DISABLE_IMPLICIT_TOKEN")) return {};
    auto value = env("HF_TOKEN"); if (value.empty()) value = env("HUGGING_FACE_HUB_TOKEN");
    if (value.empty()) {
        auto home = env("HF_HOME",env("XDG_CACHE_HOME",env("HOME") + "/.cache") + "/huggingface");
        if (home == "~" || home.rfind("~/",0) == 0) home.replace(0,1,env("HOME"));
        value = read(env("HF_TOKEN_PATH",home + "/token"));
    }
    value.erase(std::remove_if(value.begin(),value.end(),[](char c) { return c == '\r' || c == '\n'; }),value.end()); return strip_config_value(value);
}
struct Temporary {
    fs::path path;
    Temporary() {
        auto value = (fs::temp_directory_path()/"ezlocalai-hub-XXXXXX").string();
        if (!mkdtemp(value.data())) throw std::system_error(errno,std::generic_category(),"Hub temporary directory"); path = value;
    }
    ~Temporary() { std::error_code ignored; fs::remove_all(path,ignored); }
};
struct Response {
    int status = 0, transport = 0; std::string url,reason;
    std::map<std::string,std::string> headers;
    std::shared_ptr<Temporary> temporary;
    fs::path body() const { return temporary->path/"body"; }
};
Response request(const std::string &url,bool head,const std::string &auth,const std::string &range = {},int timeout = 10) {
    Response out; out.temporary = std::make_shared<Temporary>();
    if (enabled("HF_HUB_OFFLINE")) throw HubError("OfflineModeIsEnabled","Cannot reach " + url + ": offline mode is enabled.");
    auto args = std::vector<std::string>{"curl","--silent","--show-error","--proto","=http,https","--proto-redir","=http,https","--dump-header",(out.temporary->path/"headers").string(),"--output",out.body().string(),"--write-out","%{url_effective}"};
    if (head) args.insert(args.end(),{"--head","--header","Accept-Encoding: identity"}); else args.push_back("--location");
    args.insert(args.end(),{"--connect-timeout",std::to_string(timeout),"--speed-limit","1","--speed-time",std::to_string(timeout)});
    std::string headers;
    if (!auth.empty()) headers += "Authorization: Bearer " + auth + "\n";
    if (!range.empty()) headers += "Range: " + range + "\n";
    if (!headers.empty()) { write(out.temporary->path/"request-headers",headers); args.insert(args.end(),{"--header","@" + (out.temporary->path/"request-headers").string()}); }
    args.insert(args.end(),{"--",url});
    out.transport = run_process_to_files(args,(out.temporary->path/"url").string(),(out.temporary->path/"error").string()).status;
    if (head) write(out.body(),"");
    out.url = read(out.temporary->path/"url"); if (out.url.empty()) out.url = url;
    std::istringstream input(read(out.temporary->path/"headers"));
    for (std::string line; std::getline(input,line);) {
        if (line.rfind("HTTP/",0) == 0) { out.headers.clear(); std::istringstream first(line); std::string protocol; first >> protocol >> out.status; std::getline(first,out.reason); out.reason = strip_config_value(out.reason); }
        else { const auto colon = line.find(':'); if (colon != std::string::npos) out.headers[lower(line.substr(0,colon))] = strip_config_value(line.substr(colon + 1)); }
    }
    return out;
}
bool retry_status(int status) { return status == 429 || status == 500 || status == 502 || status == 503 || status == 504; }
void status(const Response &response) {
    const auto &h = response.headers; const auto get = [&](const std::string &key,const std::string &fallback = "") { const auto at = h.find(key); return at == h.end() ? fallback : at->second; };
    if (response.transport) {
        if (response.transport == 28) throw HubError("ReadTimeout","timed out");
        if (response.transport == 18) throw HubError("RemoteProtocolError","peer closed connection without sending complete message body");
        throw HubError("ConnectError","All connection attempts failed");
    }
    if (response.status < 400) return;
    const auto code = get("x-error-code"), server = get("x-error-message"), url = response.url, number = std::to_string(response.status);
    std::string category = "HfHubHTTPError", message;
    if (code == "RevisionNotFound") { category = "RevisionNotFoundError"; message = number + " Client Error.\n\nRevision Not Found for url: " + url + "."; }
    else if (code == "EntryNotFound") { category = "RemoteEntryNotFoundError"; message = number + " Client Error.\n\nEntry Not Found for url: " + url + "."; }
    else if (code == "GatedRepo") { category = "GatedRepoError"; message = number + " Client Error.\n\nCannot access gated repo for url " + url + "."; }
    else if (server == "Access to this resource is disabled.") { category = "DisabledRepoError"; message = number + " Client Error.\n\nCannot access repository for url " + url + ".\n" + server; }
    else if (code == "RepoNotFound" || response.status == 401 && server != "Invalid credentials in Authorization header" && url.rfind("https://",0) == 0 && (url.find("/api/models/") != std::string::npos || url.find("/resolve/") != std::string::npos)) {
        category = "RepositoryNotFoundError"; message = number + " Client Error.\n\nRepository Not Found for url: " + url + ".\nPlease make sure you specified the correct \x60repo_id\x60 and \x60repo_type\x60.\nIf you are trying to access a private or gated repo, make sure you are authenticated. For more details, see https://huggingface.co/docs/huggingface_hub/authentication";
    } else if (response.status == 400) { category = "BadRequestError"; message = "\n\nBad request:"; }
    else if (response.status == 403) message = "\n\n403 Forbidden: " + get("x-error-message","None") + ".\nCannot access content at: " + url + ".\nMake sure your token has the correct permissions.";
    else if (response.status == 429) message = "\n\n429 Too Many Requests for url: " + url + ".";
    else message = std::string(response.status >= 500 ? "Server error '" : "Client error '") + number + " " + response.reason + "' for url '" + url + "'\nFor more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/" + number;
    std::vector<std::string> errors;
    const auto add = [&](const std::string &text) { const auto clean = strip_config_value(text); if (!clean.empty() && std::find(errors.begin(),errors.end(),clean) == errors.end()) errors.push_back(clean); };
    add(server); const auto body = read(response.body()); const auto data = Json::parse(body,nullptr,false);
    if (data.is_object()) {
        if (data.contains("error")) { const auto &error = data["error"]; if (error.is_array()) for (const auto &part : error) add(part.is_string() ? part.get<std::string>() : part.dump()); else add(error.is_string() ? error.get<std::string>() : error.dump()); }
        if (data.contains("errors") && data["errors"].is_array()) for (const auto &part : data["errors"]) if (part.contains("message")) add(part["message"].get<std::string>());
    } else if (data.is_discarded() && lower(get("content-type")).find("html") == std::string::npos) add(body);
    std::string detail; for (const auto &error : errors) { if (!detail.empty()) detail += '\n'; detail += error; }
    if (!detail.empty() && lower(message).find(lower(detail)) == std::string::npos) message += (message.find("\n\n") == std::string::npos ? "\n\n" : "\n") + detail;
    for (const auto &pair : {std::pair{"x-request-id","Request ID"},std::pair{"x-amzn-trace-id","Amzn Trace ID"},std::pair{"x-amz-cf-id","Amz CF ID"}})
        if (!get(pair.first).empty()) { const auto id = get(pair.first); if (lower(message).find(lower(id)) == std::string::npos) { auto at = message.find('\n'); message.insert(at == std::string::npos ? message.size() : at," (" + std::string(pair.second) + ": " + id + ")"); } break; }
    HubError error(category,strip_config_value(message),response.status,true); error.commit_hash = get("x-repo-commit"); throw error;
}
struct Configuration {
    HubSnapshotOptions options; std::string endpoint,auth; fs::path storage;
    Configuration(const std::string &repository,const HubSnapshotOptions &input) : options(input) {
        validate(repository); if (options.cache_directory.empty()) options.cache_directory = default_hub_cache_directory();
        storage = fs::path(options.cache_directory)/folder(repository); endpoint = options.endpoint.empty() ? env("HF_ENDPOINT","https://huggingface.co") : options.endpoint;
        while (!endpoint.empty() && endpoint.back() == '/') endpoint.pop_back(); auth = token();
    }
};
fs::path pointer(const fs::path &storage,const std::string &revision,const std::string &filename) {
    const auto snapshots = storage/"snapshots", path = snapshots/revision/filename;
    const auto base = fs::absolute(snapshots).lexically_normal().string() + "/", full = fs::absolute(path).lexically_normal().string();
    if (full.rfind(base,0) != 0) throw HubError("ValueError","Invalid pointer path: cannot create pointer path in snapshot folder if \x60storage_folder='" + storage.string() + "'\x60, \x60revision='" + revision + "'\x60 and \x60relative_filename='" + filename + "'\x60.");
    return path;
}
void reference(const Configuration &config,const std::string &revision,const std::string &sha) {
    if (revision != sha) try { write(config.storage/"refs"/revision,sha); } catch (const std::exception &) {}
}
[[noreturn]] void reference_io_error(const fs::path &path,int error) {
    const auto category = error == EISDIR ? "IsADirectoryError" : error == ENOTDIR ? "NotADirectoryError" :
        error == ENOENT ? "FileNotFoundError" : error == EACCES || error == EPERM ? "PermissionError" : error == EEXIST ? "FileExistsError" : "OSError";
    throw HubError(category,"[Errno " + std::to_string(error) + "] " + std::strerror(error) + ": " + python_string_repr(path.string()));
}
void file_reference(const Configuration &config,const std::string &revision,const std::string &sha) {
    if (revision == sha) return;
    const auto path = config.storage/"refs"/revision; std::error_code error;
    fs::create_directories(path.parent_path(),error); if (error) reference_io_error(path.parent_path(),error.value());
    // Unlike snapshot_download, hf_hub_download propagates reference I/O
    // failures and avoids rewriting an already-current reference.
    if (path_exists(path)) {
        const int fd = open(path.c_str(),O_RDONLY | O_CLOEXEC); if (fd < 0) reference_io_error(path,errno);
        std::string value; char buffer[4096]; int failure = 0;
        for (;;) { const auto count = ::read(fd,buffer,sizeof(buffer)); if (count < 0 && errno == EINTR) continue; if (count < 0) failure = errno; if (count <= 0) break; value.append(buffer,count); }
        close(fd); if (failure) reference_io_error(path,failure); if (value == sha) return;
    }
    const int fd = open(path.c_str(),O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC,0666); if (fd < 0) reference_io_error(path,errno);
    size_t offset = 0; int failure = 0;
    while (offset < sha.size()) { const auto count = ::write(fd,sha.data() + offset,sha.size() - offset); if (count < 0 && errno == EINTR) continue; if (count <= 0) { failure = count < 0 ? errno : EIO; break; } offset += count; }
    if (close(fd) && !failure) failure = errno; if (failure) reference_io_error(path,failure);
}
std::string cached_commit(const Configuration &config) {
    if (commit(config.options.revision)) return config.options.revision;
    const auto path = config.storage/"refs"/config.options.revision; return path_exists(path) ? read(path) : "";
}
struct Lock {
    int fd = -1; fs::path path; bool soft = false;
    explicit Lock(fs::path path) : path(std::move(path)) {
        fs::create_directories(this->path.parent_path());
        for (;;) {
            fd = open(this->path.c_str(),O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW | (soft ? O_EXCL : O_TRUNC),0664);
            if (fd < 0) {
                if (soft && errno == EEXIST || errno == ENOENT) { std::this_thread::sleep_for(std::chrono::milliseconds(50)); continue; }
                throw std::system_error(errno,std::generic_category(),"Hub lock");
            }
            fchmod(fd,0664); if (soft) break;
            while (flock(fd,LOCK_EX)) {
                if (errno == EINTR) continue;
                const auto error = errno; close(fd); fd = -1;
                if (error == ENOSYS) { ::unlink(this->path.c_str()); soft = true; break; }
                throw std::system_error(error,std::generic_category(),"Hub lock");
            }
            if (fd < 0) continue;
            struct stat info{}; if (!fstat(fd,&info) && info.st_nlink) break; close(fd); fd = -1;
        }
    }
    ~Lock() { ::unlink(path.c_str()); if (!soft) flock(fd,LOCK_UN); close(fd); }
};
void link(const fs::path &blob,const fs::path &target) {
    if (path_exists(target)) return;
    std::error_code ignored; fs::remove(target,ignored); fs::create_symlink(fs::relative(blob,target.parent_path()),target);
}
bool fatal(const HubError &error) { return error.category == "RepositoryNotFoundError" || error.category == "GatedRepoError" || error.http && error.status == 401; }
Response metadata(std::string url,const std::string &auth,bool retry) {
    for (int redirects = 0; redirects < 50; ++redirects) {
        Response response;
        for (int attempt = 0;; ++attempt) {
            response = request(url,true,auth,{},retry ? 60 : std::stoi(env("HF_HUB_ETAG_TIMEOUT","10")));
            if (!retry || attempt >= 5 || !retry_status(response.status) && response.transport != 7 && response.transport != 28) break;
            std::this_thread::sleep_for(std::chrono::seconds(std::min(1 << attempt,8)));
        }
        status(response);
        const auto location = response.headers.find("location");
        if (response.status >= 300 && response.status < 400 && location != response.headers.end() && !HttpUrl::parse(location->second).authority) { url = HttpUrl::parse(url).resolve(location->second).str(); continue; }
        return response;
    } throw HubError("TooManyRedirects","Exceeded maximum allowed redirects.");
}
std::string file(const std::string &repository,const std::string &filename,Configuration config) {
    const auto revision = config.options.revision;
    if (commit(revision)) { const auto path = pointer(config.storage,revision,filename); if (path_exists(path)) return path.string(); }
    const auto url = config.endpoint + "/" + escape(repository) + "/resolve/" + escape(revision,false) + "/" + escape(filename);
    Response response; std::unique_ptr<HubError> error;
    for (int attempt = 0; attempt < 2; ++attempt) {
        try {
            if (config.options.local_files_only) throw HubError("OfflineModeIsEnabled","local files only");
            response = metadata(url,config.auth,attempt != 0);
            const auto &h = response.headers;
            if (!h.count("x-repo-commit")) throw HubError("FileMetadataError","Missing commit");
            if (!h.count("x-linked-etag") && !h.count("etag")) throw HubError("FileMetadataError","Missing ETag");
            if (!h.count("x-linked-size") && !h.count("content-length")) throw HubError("FileMetadataError","Missing size");
            error.reset(); break;
        } catch (const HubError &caught) {
            if (caught.category == "RemoteEntryNotFoundError") {
                if (!caught.commit_hash.empty()) {
                    try { write(config.storage/".no_exist"/caught.commit_hash/filename,""); } catch (const std::exception &) {}
                    file_reference(config,revision,caught.commit_hash);
                }
                throw;
            }
            if (caught.category == "RevisionNotFoundError" || caught.category == "ProxyError") throw;
            error = std::make_unique<HubError>(caught);
            const auto sha = cached_commit(config); if (!sha.empty()) { const auto path = pointer(config.storage,sha,filename); if (path_exists(path)) return path.string(); }
            if (attempt || !retry_status(caught.status) && caught.category != "ConnectError" && caught.category != "ReadTimeout") break;
        }
    }
    if (error) {
        if (config.options.local_files_only) throw HubError("LocalEntryNotFoundError","Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable hf.co look-ups and downloads online, set 'local_files_only' to False.");
        if (fatal(*error)) throw *error;
        throw HubError("LocalEntryNotFoundError","An error happened while trying to locate the file on the Hub and we cannot find the requested files in the local cache. Please check your connection and try again or make sure your Internet connection is on.");
    }
    auto &h = response.headers; auto etag = h.count("x-linked-etag") ? h["x-linked-etag"] : h["etag"];
    if (etag.rfind("W/",0) == 0) etag.erase(0,2); if (etag.size() >= 2 && etag.front() == '"' && etag.back() == '"') etag = etag.substr(1,etag.size() - 2);
    const auto sha = h["x-repo-commit"]; const auto target = pointer(config.storage,sha,filename), blob = config.storage/"blobs"/etag;
    const uint64_t expected = std::stoull(h.count("x-linked-size") ? h["x-linked-size"] : h["content-length"]);
    fs::create_directories(blob.parent_path()); fs::create_directories(target.parent_path()); file_reference(config,revision,sha);
    const auto lockpath = fs::path(config.options.cache_directory)/".locks"/folder(repository)/(etag + ".lock"); fs::create_directories(lockpath.parent_path());
    if (path_exists(target)) return target.string();
    Lock lock(lockpath);
    if (!path_exists(blob)) {
        const auto incomplete = fs::path(blob.string() + ".incomplete"); { std::ofstream create(incomplete,std::ios::binary | std::ios::app); }
        const auto resume = fs::file_size(incomplete);
        if (resume != expected) {
            auto download_url = h.count("location") ? h["location"] : response.url;
            auto auth = config.auth; if (HttpUrl::parse(url).host_header() != HttpUrl::parse(download_url).host_header()) auth.clear();
            Response downloaded;
            for (int attempt = 0;; ++attempt) {
                downloaded = request(download_url,false,auth,resume ? "bytes=" + std::to_string(resume) + "-" : "",std::stoi(env("HF_HUB_DOWNLOAD_TIMEOUT","10")));
                if (downloaded.status != 429 || attempt >= 5) break; std::this_thread::sleep_for(std::chrono::seconds(std::min(1 << attempt,8)));
            }
            if (!downloaded.transport) status(downloaded);
            if (downloaded.status < 400) {
                std::ifstream in(downloaded.body(),std::ios::binary); std::ofstream out(incomplete,std::ios::binary | (resume && downloaded.status != 200 ? std::ios::app : std::ios::trunc));
                char buffer[65536]; while (in) { in.read(buffer,sizeof(buffer)); if (in.gcount()) out.write(buffer,in.gcount()); }
                if (in.bad() || !out) throw HubError("OSError","Cannot write incomplete Hub file");
            }
            status(downloaded);
            if (fs::file_size(incomplete) != expected) {
                auto name = download_url; const auto disp = downloaded.headers.find("content-disposition");
                if (disp != downloaded.headers.end()) { const auto begin = disp->second.find("filename=\""); if (begin != std::string::npos) { const auto end = disp->second.find("\";",begin + 10); if (end != std::string::npos) name = disp->second.substr(begin + 10,end - begin - 10); } }
                std::vector<size_t> characters; for (int32_t at = 0; at < static_cast<int32_t>(name.size());) { characters.push_back(at); UChar32 c; U8_NEXT(name.data(),at,static_cast<int32_t>(name.size()),c); }
                if (characters.size() > 40) name = "(…)" + name.substr(characters[characters.size() - 40]);
                throw HubError("OSError","Consistency check failed: file should be of size " + std::to_string(expected) + " but has size " + std::to_string(fs::file_size(incomplete)) + " (" + name + ").\nThis is usually due to network issues while downloading the file. Please retry with \x60" "force_download=True\x60.");
            }
        }
        fs::rename(incomplete,blob);
    }
    link(blob,target); return target.string();
}
}
std::string hub_snapshot_file(const std::string &repository,const std::string &filename,const HubSnapshotOptions &options) { return file(repository,filename,Configuration(repository,options)); }
std::string hub_snapshot_download(const std::string &repository,const HubSnapshotOptions &options) {
    Configuration config(repository,options); Json info; std::unique_ptr<HubError> error;
    if (!options.local_files_only) try {
        const auto response = request(config.endpoint + "/api/models/" + escape(repository) + "/revision/" + escape(options.revision,false),false,config.auth);
        status(response); info = Json::parse(read(response.body()));
    } catch (const HubError &caught) {
        if (caught.category == "RevisionNotFoundError" || caught.category == "ProxyError") throw;
        error = std::make_unique<HubError>(caught);
    }
    if (info.is_null()) {
        const auto sha = cached_commit(config); if (!sha.empty()) { const auto path = config.storage/"snapshots"/sha; if (path_exists(path)) return path.string(); }
        if (options.local_files_only) throw HubError("LocalEntryNotFoundError","Cannot find an appropriate cached snapshot folder for the specified revision on the local disk and outgoing traffic has been disabled. To enable repo look-ups and downloads online, pass 'local_files_only=False' as input.");
        if (error && error->category == "OfflineModeIsEnabled") throw HubError("LocalEntryNotFoundError","Cannot find an appropriate cached snapshot folder for the specified revision on the local disk and outgoing traffic has been disabled. To enable repo look-ups and downloads online, set 'HF_HUB_OFFLINE=0' as environment variable.");
        if (error && fatal(*error)) throw *error;
        throw HubError("LocalEntryNotFoundError","An error happened while trying to locate the files on the Hub and we cannot find the appropriate snapshot folder for the specified revision on the local disk. Please check your internet connection and try again.");
    }
    if (!info.contains("sha") || info["sha"].is_null()) throw HubError("AssertionError","Repo info returned from server must have a revision sha.");
    const auto sha = info["sha"].get<std::string>(); std::vector<std::string> files; bool reference_saved = false;
    if (info.contains("siblings") && info["siblings"].is_array() && !info["siblings"].empty() && info["siblings"].size() <= 1000)
        for (const auto &entry : info["siblings"]) files.push_back(entry.at("rfilename"));
    else {
        // The source tree iterator is consumed after recording the branch ref.
        // Preserve that cache side effect even when the tree request fails.
        reference(config,options.revision,sha); reference_saved = true;
        auto url = config.endpoint + "/api/models/" + escape(repository) + "/tree/" + escape(options.revision,false) + "?recursive=true&expand=false";
        while (!url.empty()) {
            const auto response = request(url,false,config.auth); status(response);
            for (const auto &entry : Json::parse(read(response.body()))) if (entry.value("type","") == "file") files.push_back(entry.at("path"));
            url.clear(); const auto links = response.headers.find("link");
            if (links != response.headers.end()) { const auto next = links->second.find("rel=\"next\""); if (next != std::string::npos) { const auto begin = links->second.rfind('<',next), end = links->second.find('>',begin); if (begin != std::string::npos && end != std::string::npos) url = links->second.substr(begin + 1,end - begin - 1); } }
        }
    }
    files.erase(std::remove_if(files.begin(),files.end(),[&](const auto &name) {
        if (options.allow_patterns.empty()) return false; for (const auto &pattern : options.allow_patterns) if (!fnmatch(pattern.c_str(),name.c_str(),0)) return false; return true;
    }),files.end());
    if (!reference_saved) reference(config,options.revision,sha); config.options.revision = sha;
    if (options.max_workers <= 0) throw HubError("ValueError","max_workers must be greater than 0");
    std::atomic<size_t> next{0}; std::vector<std::exception_ptr> errors(files.size()); std::vector<std::thread> workers;
    try {
        for (size_t i = 0; i < std::min<size_t>(options.max_workers,files.size()); ++i) workers.emplace_back([&] {
            for (;;) { const auto index = next.fetch_add(1); if (index >= files.size()) break; try { file(repository,files[index],config); } catch (...) { errors[index] = std::current_exception(); } }
        });
    } catch (...) { for (auto &worker : workers) worker.join(); throw; }
    for (auto &worker : workers) worker.join(); for (const auto &failure : errors) if (failure) std::rethrow_exception(failure);
    return (config.storage/"snapshots"/sha).string();
}
}
