#pragma once

#include "ezlocalai/http_url.hpp"
#include <functional>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace ezlocalai {

// One instance belongs to one ClientSession/request chain. Use one parser
// profile consistently for its lifetime.
class HttpCookieJar {
public:
    explicit HttpCookieJar(std::function<double()> now = {});
    void update(const std::vector<std::string> &set_cookie_headers, const HttpUrl &response_url);
    // Installed aiohttp 3.14 response fields use a distinct, bounded parser.
    void update_from_headers(const std::vector<std::string> &set_cookie_headers, const HttpUrl &response_url);
    std::string header(const HttpUrl &request_url);

private:
    struct Cookie {
        std::string name, value, coded_value;
        std::map<std::string, std::string> attributes;
    };
    using Key = std::tuple<std::string, std::string, std::string>;
    std::function<double()> now_;
    std::map<Key, Cookie> cookies_;
    std::map<Key, double> expirations_;
    std::set<std::pair<std::string, std::string>> host_only_;
    std::set<Key> response_host_only_;
    bool response_headers_ = false;

    static std::vector<Cookie> parse(const std::vector<std::string> &headers);
    static std::vector<Cookie> parse_response_headers(const std::vector<std::string> &headers);
    void store(const std::vector<Cookie> &cookies, const HttpUrl &response_url);
    void expire();
};

} // namespace ezlocalai
