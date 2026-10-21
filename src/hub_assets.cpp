#include "ezlocalai/wan_video.hpp"
#include "ezlocalai/configuration.hpp"
#include "ezlocalai/process.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <sys/file.h>

namespace ezlocalai {
namespace {
std::string env(const char *key, const char *fallback = "") { const auto value = std::getenv(key); return value ? value : fallback; }
std::string repository_folder(const std::string &repository) {
    std::string name = "models--";
    for (char c : repository) name += c == '/' ? "--" : std::string(1, c);
    return name;
}
std::filesystem::path snapshot_path(const HubAsset &asset, const std::string &cache_dir) {
    for (const auto &part : std::filesystem::path(asset.filename)) if (part == ".." || part == "/") throw std::invalid_argument("Invalid Hub filename");
    if (asset.repository.empty() || asset.repository.find("..") != std::string::npos || asset.revision.empty()
        || asset.revision.find('/') != std::string::npos) throw std::invalid_argument("Invalid Hub repository or revision");
    return std::filesystem::path(cache_dir) / repository_folder(asset.repository) / "snapshots" / asset.revision / asset.filename;
}
std::string read(const std::filesystem::path &path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
std::string url_escape(const std::string &value) {
    const char *hex = "0123456789ABCDEF";
    std::string result;
    for (unsigned char c : value) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' || c == '~' || c == '/') result += c;
        else { result += '%'; result += hex[c >> 4]; result += hex[c & 15]; }
    }
    return result;
}
struct TemporaryDirectory {
    std::filesystem::path path;
    explicit TemporaryDirectory(const std::filesystem::path &parent) {
        auto pattern = (parent / ".download-XXXXXX").string();
        if (!mkdtemp(pattern.data())) throw std::system_error(errno, std::generic_category(), "download temporary directory");
        path = pattern;
    }
    ~TemporaryDirectory() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
};
struct FileLock {
    int fd = -1;
    explicit FileLock(const std::filesystem::path &path) {
        fd = open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
        if (fd < 0) throw std::system_error(errno, std::generic_category(), "asset lock");
        while (flock(fd, LOCK_EX) != 0) if (errno != EINTR) { const int error = errno; close(fd); throw std::system_error(error, std::generic_category(), "asset lock"); }
    }
    ~FileLock() { if (fd >= 0) { flock(fd, LOCK_UN); close(fd); } }
};
} // namespace

std::string cached_hub_asset(const HubAsset &asset, const std::string &cache_dir) {
    const auto path = snapshot_path(asset, cache_dir);
    return std::filesystem::is_regular_file(path) ? path.string() : std::string();
}
std::string download_hub_asset(const HubAsset &asset, const std::string &cache_dir) {
    const auto snapshot = snapshot_path(asset, cache_dir);
    if (std::filesystem::is_regular_file(snapshot)) return snapshot.string();
    auto offline = env("HF_HUB_OFFLINE");
    std::transform(offline.begin(), offline.end(), offline.begin(), [](unsigned char c) { return std::toupper(c); });
    if (offline == "1" || offline == "ON" || offline == "YES" || offline == "TRUE") throw std::runtime_error("Cannot find the requested files in the disk cache and outgoing traffic has been disabled.");
    const auto repo = std::filesystem::path(cache_dir) / repository_folder(asset.repository);
    std::filesystem::create_directories(repo / "blobs");
    // Lock the immutable snapshot target before probing, then deduplicate by ETag.
    const auto locks = std::filesystem::path(cache_dir) / ".locks" / repository_folder(asset.repository);
    std::filesystem::create_directories(locks);
    auto filename_key = asset.filename;
    std::replace(filename_key.begin(), filename_key.end(), '/', '_');
    FileLock lock(locks / (asset.revision + "-" + filename_key + ".lock"));
    if (std::filesystem::is_regular_file(snapshot)) return snapshot.string();
    TemporaryDirectory temp(repo);
    auto endpoint = env("HF_ENDPOINT", "https://huggingface.co");
    while (!endpoint.empty() && endpoint.back() == '/') endpoint.pop_back();
    const auto url = endpoint + "/" + url_escape(asset.repository) + "/resolve/" + url_escape(asset.revision) + "/" + url_escape(asset.filename);
    std::vector<std::string> command = {"curl", "--silent", "--show-error", "--fail", "--location", "--proto", "=http,https", "--proto-redir", "=http,https"};
    auto token = env("HF_TOKEN", env("HUGGING_FACE_HUB_TOKEN").c_str());
    if (token.empty() && env("HF_HUB_DISABLE_IMPLICIT_TOKEN") != "1") {
        const auto home = env("HF_HOME", (std::filesystem::path(env("HOME")) / ".cache/huggingface").c_str());
        token = strip_config_value(read(env("HF_TOKEN_PATH", (std::filesystem::path(home) / "token").c_str())));
    }
    if (!token.empty()) {
        // Keep credentials out of argv and logs; the temporary directory is 0700.
        std::ofstream headers(temp.path / "authorization"); headers << "Authorization: Bearer " << token << '\n'; headers.close();
        command.insert(command.end(), {"--header", "@" + (temp.path / "authorization").string()});
    }
    auto probe = command;
    probe.insert(probe.end(), {"--head", "--max-time", env("HF_HUB_ETAG_TIMEOUT", "10"), "--dump-header", (temp.path / "headers").string(), "--output", "/dev/null", "--", url});
    auto status = run_process_to_files(probe, "/dev/null", (temp.path / "error").string());
    if (status.status != 0) throw std::runtime_error("Hub metadata request failed for " + asset.filename + ": " + strip_config_value(read(temp.path / "error")));
    std::map<std::string, std::string> headers;
    std::istringstream lines(read(temp.path / "headers"));
    for (std::string line; std::getline(lines, line);) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        auto key = line.substr(0, colon);
        std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) { return std::tolower(c); });
        headers[key] = strip_config_value(line.substr(colon + 1));
    }
    auto etag = headers.count("x-linked-etag") ? headers["x-linked-etag"] : headers["etag"];
    if (etag.rfind("W/", 0) == 0) etag.erase(0, 2);
    if (etag.size() >= 2 && etag.front() == '"' && etag.back() == '"') etag = etag.substr(1, etag.size() - 2);
    if (etag.empty() || etag.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_") != std::string::npos) throw std::runtime_error("Hub metadata did not contain a usable ETag for " + asset.filename);
    const auto blob = repo / "blobs" / etag;
    FileLock blob_lock(locks / (etag + ".lock"));
    if (!std::filesystem::is_regular_file(blob)) {
        const auto incoming = temp.path / "asset";
        auto get = command;
        get.insert(get.end(), {"--speed-limit", "1", "--speed-time", env("HF_HUB_DOWNLOAD_TIMEOUT", "10"), "--output", incoming.string(), "--", url});
        status = run_process_to_files(get, "/dev/null", (temp.path / "error").string());
        if (status.status != 0) throw std::runtime_error("Hub download failed for " + asset.filename + ": " + strip_config_value(read(temp.path / "error")));
        const auto size = headers.count("x-linked-size") ? headers["x-linked-size"] : headers["content-length"];
        if (!size.empty() && std::filesystem::file_size(incoming) != std::stoull(size)) throw std::runtime_error("Hub download size mismatch for " + asset.filename);
        std::filesystem::rename(incoming, blob);
    }
    std::filesystem::create_directories(snapshot.parent_path());
    const auto link = temp.path / "snapshot";
    std::filesystem::create_symlink(std::filesystem::relative(blob, snapshot.parent_path()), link);
    std::filesystem::rename(link, snapshot);
    return snapshot.string();
}
} // namespace ezlocalai
