#include "ezlocalai/hub_assets.hpp"
#include "ezlocalai/configuration.hpp"
#include "ezlocalai/json_validation.hpp"
#include "ezlocalai/process.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <openssl/evp.h>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <sys/file.h>
#include <sys/stat.h>

namespace ezlocalai {
namespace {
namespace fs = std::filesystem;
std::string env(const char *key, const std::string &fallback = "") { const auto value = std::getenv(key); return value ? value : fallback; }
std::string lower(std::string value) { for (auto &c : value) if (c >= 'A' && c <= 'Z') c += 'a' - 'A'; return value; }
bool enabled(const char *key) { const auto value = lower(env(key)); return value == "1" || value == "on" || value == "yes" || value == "true"; }
std::string read(const fs::path &path) { std::ifstream in(path, std::ios::binary); return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()}; }
std::string escape(const std::string &value, bool slash = true) {
    std::string out; const char *hex = "0123456789ABCDEF";
    for (const unsigned char c : value) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' || c == '~' || (slash && c == '/')) out += c;
        else { out += '%'; out += hex[c >> 4]; out += hex[c & 15]; }
    }
    return out;
}
std::string expand_home(std::string path) {
    if (path == "~" || path.rfind("~/", 0) == 0) path.replace(0, 1, env("HOME"));
    return path;
}
std::string hub_home() { return expand_home(env("HF_HOME", env("XDG_CACHE_HOME", env("HOME") + "/.cache") + "/huggingface")); }
std::string cache_directory() { return expand_home(env("HF_HUB_CACHE", env("HUGGINGFACE_HUB_CACHE", hub_home() + "/hub"))); }
std::string endpoint() { auto value = env("HF_ENDPOINT", "https://huggingface.co"); while (!value.empty() && value.back() == '/') value.pop_back(); return value; }
void valid_repository(const std::string &repository) {
    if (repository.empty() || repository.size() > 193 || repository.find("..") != std::string::npos || repository.find("--") != std::string::npos || repository.front() == '/' || repository.back() == '/' || std::count(repository.begin(), repository.end(), '/') > 1) throw std::invalid_argument("Invalid Hub repository");
    for (const auto &part : fs::path(repository)) { const auto name = part.string(); if (name.empty() || name.size() > 96 || name.front() == '.' || name.front() == '-' || name.back() == '.' || name.back() == '-') throw std::invalid_argument("Invalid Hub repository"); }
    for (unsigned char c : repository) if (c < 128 && !std::isalnum(c) && c != '/' && c != '-' && c != '_' && c != '.') throw std::invalid_argument("Invalid Hub repository");
}
void valid_filename(const std::string &filename) {
    if (filename.empty() || fs::path(filename).is_absolute() || filename.find('\0') != std::string::npos || filename.find('\\') != std::string::npos) throw std::invalid_argument("Invalid Hub filename");
    for (const auto &part : fs::path(filename)) if (part == "..") throw std::invalid_argument("Invalid Hub filename");
}
bool hash_text(const std::string &value, size_t size) { return value.size() == size && value.find_first_not_of("0123456789abcdef") == std::string::npos; }
struct Temporary {
    fs::path path;
    explicit Temporary(const fs::path &parent) {
        auto pattern = (parent / ".hub-XXXXXX").string();
        if (!mkdtemp(pattern.data())) throw std::system_error(errno, std::generic_category(), "Hub temporary directory");
        path = pattern;
    }
    ~Temporary() { std::error_code ignored; fs::remove_all(path, ignored); }
};
struct Lock {
    int fd;
    explicit Lock(const fs::path &path) : fd(open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600)) {
        if (fd < 0) throw std::system_error(errno, std::generic_category(), "Hub file lock");
        while (flock(fd, LOCK_EX)) if (errno != EINTR) { const auto error = errno; close(fd); throw std::system_error(error, std::generic_category(), "Hub file lock"); }
    }
    ~Lock() { flock(fd, LOCK_UN); close(fd); }
};
struct MissingHubEntry : std::runtime_error { using std::runtime_error::runtime_error; };
std::string implicit_token() {
    if (enabled("HF_HUB_DISABLE_IMPLICIT_TOKEN")) return {};
    auto token = env("HF_TOKEN"); if (token.empty()) token = env("HUGGING_FACE_HUB_TOKEN");
    if (token.empty()) token = read(expand_home(env("HF_TOKEN_PATH", hub_home() + "/token")));
    token.erase(std::remove_if(token.begin(), token.end(), [](char c) { return c == '\r' || c == '\n'; }), token.end());
    return strip_config_value(token);
}
std::vector<std::string> curl_command(const Temporary &temp) {
    std::vector<std::string> args{"curl", "--silent", "--show-error", "--fail", "--location", "--proto", "=http,https", "--proto-redir", "=http,https"};
    const auto token = implicit_token();
    if (!token.empty()) {
        const auto path = temp.path / "authorization"; std::ofstream out(path); out << "Authorization: Bearer " << token << '\n'; out.close();
        args.insert(args.end(), {"--header", "@" + path.string()});
    }
    return args;
}
std::map<std::string, std::string> headers(const fs::path &path) {
    std::map<std::string, std::string> result; std::istringstream input(read(path));
    for (std::string line; std::getline(input, line);) {
        const auto colon = line.find(':');
        if (colon != std::string::npos) result[lower(line.substr(0, colon))] = strip_config_value(line.substr(colon + 1));
    }
    return result;
}
void request(std::vector<std::string> args, const Temporary &temp, const std::string &url, bool head = false, const fs::path &output = {}) {
    if (enabled("HF_HUB_OFFLINE")) throw std::runtime_error("Offline mode is enabled");
    if (head) {
        args.erase(std::remove(args.begin(), args.end(), "--location"), args.end());
        args.insert(args.end(), {"--head", "--header", "Accept-Encoding: identity"});
    }
    args.insert(args.end(), {"--dump-header", (temp.path / "headers").string(), "--output", output.empty() ? (temp.path / "response").string() : output.string()});
    if (head || output.empty()) args.insert(args.end(), {"--max-time", env("HF_HUB_ETAG_TIMEOUT", "10")});
    else args.insert(args.end(), {"--speed-limit", "1", "--speed-time", env("HF_HUB_DOWNLOAD_TIMEOUT", "10")});
    args.insert(args.end(), {"--write-out", "%{url_effective}", "--", url});
    const auto status = run_process_to_files(args, (temp.path / "url").string(), (temp.path / "error").string());
    if (status.status != 0) {
        const auto response = headers(temp.path / "headers"); const auto code = response.find("x-error-code");
        const auto message = "Hub request failed: " + strip_config_value(read(temp.path / "error"));
        if (code != response.end() && (code->second == "EntryNotFound" || code->second == "RevisionNotFound")) throw MissingHubEntry(message);
        throw std::runtime_error(message);
    }
}
// Only the tree API's type/path strings are retained. Full syntax validation
// runs first, then nested metadata values are skipped without interpreting them.
class TreeJson {
    const std::string &input_; size_t at_ = 0;
    void space() { while (at_ < input_.size() && std::string(" \r\n\t").find(input_[at_]) != std::string::npos) ++at_; }
    char peek() { space(); return at_ < input_.size() ? input_[at_] : '\0'; }
    void take(char expected) { if (peek() != expected) throw std::runtime_error("Invalid Hub tree response"); ++at_; }
    unsigned hex4() { const auto value = std::stoul(input_.substr(at_, 4), nullptr, 16); at_ += 4; return value; }
    static void utf8(std::string &out, unsigned value) {
        if (value < 128) out += static_cast<char>(value);
        else if (value < 2048) { out += static_cast<char>(0xc0 | (value >> 6)); out += static_cast<char>(0x80 | (value & 63)); }
        else if (value < 65536) { out += static_cast<char>(0xe0 | (value >> 12)); out += static_cast<char>(0x80 | ((value >> 6) & 63)); out += static_cast<char>(0x80 | (value & 63)); }
        else { out += static_cast<char>(0xf0 | (value >> 18)); out += static_cast<char>(0x80 | ((value >> 12) & 63)); out += static_cast<char>(0x80 | ((value >> 6) & 63)); out += static_cast<char>(0x80 | (value & 63)); }
    }
    std::string string() {
        take('"'); std::string out;
        while (at_ < input_.size()) {
            char c = input_[at_++]; if (c == '"') return out;
            if (c != '\\') { out += c; continue; }
            c = input_[at_++];
            if (c == 'u') {
                unsigned value = hex4();
                if (value >= 0xd800 && value <= 0xdbff && input_.compare(at_, 2, "\\u") == 0) {
                    const auto low = std::stoul(input_.substr(at_ + 2, 4), nullptr, 16);
                    if (low >= 0xdc00 && low <= 0xdfff) { at_ += 6; value = 0x10000 + ((value - 0xd800) << 10) + low - 0xdc00; }
                }
                utf8(out, value);
            } else {
                const auto where = std::string("bfnrt").find(c);
                out += where == std::string::npos ? c : "\b\f\n\r\t"[where];
            }
        }
        throw std::runtime_error("Invalid Hub tree string");
    }
    void skip() {
        if (peek() == '"') { string(); return; }
        if (peek() == '[' || peek() == '{') {
            const char end = input_[at_++] == '[' ? ']' : '}';
            while (peek() != end) { if (peek() == ',' || peek() == ':') ++at_; else skip(); }
            ++at_; return;
        }
        while (at_ < input_.size() && std::string(",]} \t\r\n").find(input_[at_]) == std::string::npos) ++at_;
    }
public:
    explicit TreeJson(const std::string &input) : input_(input) {}
    std::vector<std::string> files() {
        if (json_decode_error(input_)) throw std::runtime_error("Invalid Hub tree JSON");
        std::vector<std::string> files; take('[');
        while (peek() != ']') {
            take('{'); std::string type, path; bool has_path = false;
            while (peek() != '}') {
                const auto key = string(); take(':');
                if ((key == "type" || key == "path") && peek() == '"') {
                    const auto value = string(); if (key == "type") type = value; else { path = value; has_path = true; }
                } else skip();
                if (peek() != '}') take(',');
            }
            take('}'); if (type == "file") { if (!has_path) throw std::runtime_error("Missing Hub file path"); files.push_back(path); }
            if (peek() != ']') take(',');
        }
        take(']'); return files;
    }
};
std::string file_sha256(const fs::path &path) {
    std::ifstream input(path, std::ios::binary); if (!input) throw std::runtime_error("Cannot hash local Hub file");
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!context || !EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr)) throw std::runtime_error("Cannot initialize Hub checksum");
    char buffer[65536];
    while (input) { input.read(buffer, sizeof(buffer)); if (!EVP_DigestUpdate(context.get(), buffer, static_cast<size_t>(input.gcount()))) throw std::runtime_error("Cannot hash Hub file"); }
    if (!input.eof()) throw std::runtime_error("Cannot hash local Hub file");
    unsigned char digest[EVP_MAX_MD_SIZE]; unsigned length = 0;
    if (!EVP_DigestFinal_ex(context.get(), digest, &length)) throw std::runtime_error("Cannot finish Hub checksum");
    std::ostringstream out; for (unsigned i = 0; i < length; ++i) out << std::hex << std::setfill('0') << std::setw(2) << unsigned(digest[i]); return out.str();
}
struct LocalMetadata { std::string commit, etag; double timestamp; };
std::optional<LocalMetadata> local_metadata(const fs::path &path, const fs::path &file) {
    if (!fs::exists(path)) return std::nullopt;
    try {
        std::ifstream input(path); LocalMetadata m; std::string timestamp;
        if (!std::getline(input, m.commit) || !std::getline(input, m.etag) || !std::getline(input, timestamp)) throw std::runtime_error("Invalid Hub metadata");
        m.commit = strip_config_value(m.commit); m.etag = strip_config_value(m.etag); timestamp = strip_config_value(timestamp);
        size_t end = 0; m.timestamp = std::stod(timestamp, &end); if (end != timestamp.size()) throw std::runtime_error("Invalid Hub timestamp");
        struct stat status{};
        if (!stat(file.c_str(), &status) && status.st_mtim.tv_sec + status.st_mtim.tv_nsec / 1e9 - 1 <= m.timestamp) return m;
    } catch (const std::exception &) { std::error_code ignored; fs::remove(path, ignored); }
    return std::nullopt;
}
void save_metadata(const fs::path &path, const std::string &commit, const std::string &etag) {
    std::ofstream output(path);
    const auto now = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    output << commit << '\n' << etag << '\n' << std::setprecision(17) << now << '\n'; output.close();
    if (!output) throw std::runtime_error("Cannot write Hub local metadata");
}
}
std::vector<std::string> list_hub_files(const std::string &repository, const std::string &revision) {
    valid_repository(repository);
    Temporary temp(fs::temp_directory_path()); const auto command = curl_command(temp);
    const auto base = endpoint();
    auto url = base + "/api/models/" + escape(repository) + "/tree/" + escape(revision, false) + "?recursive=true&expand=false";
    const auto scheme = base.find("://"), end = scheme == std::string::npos ? std::string::npos : base.find('/', scheme + 3);
    const auto origin = base.substr(0, end);
    std::set<std::string> visited; std::vector<std::string> result;
    while (!url.empty()) {
        if (!visited.insert(url).second) throw std::runtime_error("Repeated Hub pagination link");
        request(command, temp, url);
        const auto body = read(temp.path / "response"); const auto page = TreeJson(body).files(); result.insert(result.end(), page.begin(), page.end());
        const auto response = headers(temp.path / "headers"); const auto link = response.find("link");
        url.clear();
        if (link != response.end()) {
            static const std::regex next(R"REGEX(<([^>]+)>\s*;\s*rel\s*=\s*"?next"?)REGEX"); std::smatch match;
            if (std::regex_search(link->second, match, next)) {
                url = match[1]; if (url.front() == '/') url = origin + url;
                if (url.rfind(origin + '/', 0) != 0) throw std::runtime_error("Cross-origin Hub pagination link");
            }
        }
    }
    return result;
}
std::string default_hub_cache_directory() { return cache_directory(); }
std::string download_hub_file(const std::string &repository, const std::string &filename, const std::string &local_dir, const std::string &revision) {
    valid_repository(repository); valid_filename(filename);
    const auto file = fs::path(local_dir) / filename, cache = fs::path(local_dir) / ".cache/huggingface";
    const auto metadata = cache / "download" / (filename + ".metadata"), lock_path = cache / "download" / (filename + ".lock");
    fs::create_directories(file.parent_path()); fs::create_directories(metadata.parent_path());
    { std::ofstream ignore(cache / ".gitignore"); ignore << "*"; }
    Lock lock(lock_path); auto previous = local_metadata(metadata, file);
    if (hash_text(revision, 40) && fs::is_regular_file(file) && previous && previous->commit == revision) return file.string();
    Temporary temp(metadata.parent_path()); const auto command = curl_command(temp);
    const auto url = endpoint() + '/' + escape(repository) + "/resolve/" + escape(revision, false) + '/' + escape(filename);
    std::map<std::string, std::string> response;
    std::string etag, commit, location; uint64_t size = 0;
    try {
        auto head_url = url; std::set<std::string> visited;
        while (true) {
            if (!visited.insert(head_url).second) throw std::runtime_error("Repeated Hub metadata redirect");
            request(command, temp, head_url, true);
            response = headers(temp.path / "headers");
            location = response.count("location") ? response.at("location") : head_url;
            if (location.find("://") != std::string::npos) break;
            const auto host_end = head_url.find('/', head_url.find("://") + 3);
            head_url = !location.empty() && location.front() == '/' ? head_url.substr(0, host_end) + location : head_url.substr(0, head_url.rfind('/') + 1) + location;
        }
        etag = response.count("x-linked-etag") ? response.at("x-linked-etag") : response["etag"];
        if (etag.rfind("W/", 0) == 0) etag.erase(0, 2);
        if (etag.size() >= 2 && etag.front() == '"' && etag.back() == '"') etag = etag.substr(1, etag.size() - 2);
        commit = response["x-repo-commit"];
        const auto length = response.count("x-linked-size") ? response.at("x-linked-size") : response["content-length"];
        if (etag.empty() || etag.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_") != std::string::npos || !hash_text(commit, 40) || length.empty() || length.find_first_not_of("0123456789") != std::string::npos) throw std::runtime_error("Missing Hub file metadata");
        size = std::stoull(length);
    } catch (const MissingHubEntry &) { throw; }
    catch (...) { if (fs::is_regular_file(file)) return file.string(); throw; }
    if (fs::is_regular_file(file) && ((previous && previous->etag == etag) || (!previous && hash_text(etag, 64) && file_sha256(file) == etag))) {
        save_metadata(metadata, commit, etag); return file.string();
    }
    const auto cached = cached_hub_asset({"", repository, filename, commit}, cache_directory());
    const auto incoming = temp.path / "asset";
    if (!cached.empty()) fs::copy_file(cached, incoming);
    else {
        // Use the resolved immutable URL from HEAD. A signed CDN URL receives
        // no Hub authorization header, matching the source downloader.
        auto get = command;
        auto origin = [](const std::string &value) { const auto scheme = value.find("://"); return value.substr(0, value.find('/', scheme == std::string::npos ? 0 : scheme + 3)); };
        if (origin(location) != origin(url)) {
            for (size_t i = 0; i < get.size(); ++i) if (get[i] == "--header") { get.erase(get.begin() + i, get.begin() + i + 2); break; }
        }
        fs::remove(file);
        request(get, temp, location, false, incoming);
        if (fs::file_size(incoming) != size) throw std::runtime_error("Hub download size mismatch for " + filename);
    }
    fs::rename(incoming, file); save_metadata(metadata, commit, etag);
    return file.string();
}
}
