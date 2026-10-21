#pragma once
#include <stdexcept>
#include <string>
#include <vector>

namespace ezlocalai {
struct HubError : std::runtime_error {
    std::string category;
    int status = 0;
    bool http = false;
    std::string commit_hash;
    HubError(std::string category,std::string message,int status = 0,bool http = false)
        : std::runtime_error(std::move(message)),category(std::move(category)),status(status),http(http) {}
};
// Model-cache operations used by the source STT constructor. Behavior follows
// huggingface_hub 1.5.0; separate from the older local-directory downloader.
struct HubSnapshotOptions {
    std::string cache_directory, revision = "main", endpoint;
    std::vector<std::string> allow_patterns;
    bool local_files_only = false;
    int max_workers = 8;
};
std::string hub_snapshot_download(const std::string &repository,const HubSnapshotOptions &options);
std::string hub_snapshot_file(const std::string &repository,const std::string &filename,const HubSnapshotOptions &options);
}
