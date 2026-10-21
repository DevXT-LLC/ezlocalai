#pragma once
#include <optional>
#include <string>
#include <utility>

namespace ezlocalai {
struct HttpUrl {
    std::string scheme, host, path, query, fragment;
    std::optional<int> explicit_port;
    std::optional<std::string> raw_user, raw_password;
    bool authority = false;

    static HttpUrl parse(const std::string &value);
    HttpUrl resolve(const std::string &location) const;
    void require_http() const;
    int port() const;
    std::string str() const;
    std::string target() const;
    std::string request_target() const;
    std::string decoded_path() const;
    std::string host_header() const;
    bool same_origin(const HttpUrl &other) const;
    std::optional<std::pair<std::string,std::string>> credentials() const;
    void remove_credentials();
};
}
