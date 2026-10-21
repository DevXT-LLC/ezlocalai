#pragma once
#include <string>
#include <vector>

namespace ezlocalai {
struct HubAsset { std::string role, repository, filename, revision; };
std::string default_hub_cache_directory();
std::string cached_hub_asset(const HubAsset &asset, const std::string &cache_dir);
std::string download_hub_asset(const HubAsset &asset, const std::string &cache_dir);
std::vector<std::string> list_hub_files(const std::string &repository, const std::string &revision = "main");
std::string download_hub_file(const std::string &repository, const std::string &filename,
    const std::string &local_dir, const std::string &revision = "main");
}
