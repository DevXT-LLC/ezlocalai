#include <fstream>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<std::string> split_tsv(const std::string &line) {
    std::vector<std::string> fields;
    std::string field;
    std::istringstream stream(line);
    while (std::getline(stream, field, '\t')) {
        fields.push_back(field);
    }
    return fields;
}

std::string route_key(const std::string &mode, const std::string &method, const std::string &path) {
    return mode + "\t" + method + "\t" + path;
}

std::string method_label(const std::string &decorator_method) {
    if (decorator_method == "websocket") {
        return "WS";
    }
    std::string out = decorator_method;
    for (char &ch : out) {
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    return out;
}

std::string fastapi_url_arg(
    const std::string &content,
    const std::string &name,
    const std::string &fallback,
    bool *enabled
) {
    const std::regex pattern(name + R"(\s*=\s*(None|["']([^"']+)["']))");
    std::smatch match;
    if (!std::regex_search(content, match, pattern)) {
        *enabled = true;
        return fallback;
    }
    if (match[1].str() == "None") {
        *enabled = false;
        return "";
    }
    *enabled = true;
    return match[2].str();
}

std::set<std::string> scan_python_routes(const std::filesystem::path &file, const std::string &mode) {
    std::ifstream in(file);
    std::set<std::string> routes;
    if (!in) {
        std::cerr << "warning: failed to open route source " << file << "\n";
        return routes;
    }

    std::ostringstream buffer;
    buffer << in.rdbuf();
    const std::string content = buffer.str();
    if (content.find("FastAPI(") != std::string::npos) {
        bool openapi_enabled = true;
        const std::string openapi_url = fastapi_url_arg(content, "openapi_url", "/openapi.json", &openapi_enabled);
        if (openapi_enabled) {
            routes.insert(route_key(mode, "GET", openapi_url));
            routes.insert(route_key(mode, "HEAD", openapi_url));
        }

        bool docs_enabled = true;
        const std::string docs_url = fastapi_url_arg(content, "docs_url", "/docs", &docs_enabled);
        if (docs_enabled) {
            routes.insert(route_key(mode, "GET", docs_url));
            routes.insert(route_key(mode, "HEAD", docs_url));
            bool oauth_enabled = true;
            const std::string oauth_url = fastapi_url_arg(
                content,
                "swagger_ui_oauth2_redirect_url",
                "/docs/oauth2-redirect",
                &oauth_enabled
            );
            if (oauth_enabled) {
                routes.insert(route_key(mode, "GET", oauth_url));
                routes.insert(route_key(mode, "HEAD", oauth_url));
            }
        }

        bool redoc_enabled = true;
        const std::string redoc_url = fastapi_url_arg(content, "redoc_url", "/redoc", &redoc_enabled);
        if (redoc_enabled) {
            routes.insert(route_key(mode, "GET", redoc_url));
            routes.insert(route_key(mode, "HEAD", redoc_url));
        }
    }

    const std::regex decorator(R"(^\s*@app\.(get|post|put|patch|delete|head|options|websocket)\s*\((.*)$)");
    const std::regex static_mount(R"(^\s*app\.mount\s*\(\s*["']([^"']+)["']\s*,\s*StaticFiles\s*\()");
    const std::regex quoted_path(R"(["']([^"']+)["'])");
    std::string line;
    std::istringstream lines(content);
    while (std::getline(lines, line)) {
        std::smatch mount_match;
        if (std::regex_search(line, mount_match, static_mount)) {
            std::string mount_path = mount_match[1].str();
            if (!mount_path.empty() && mount_path.back() == '/') {
                mount_path.pop_back();
            }
            routes.insert(route_key(mode, "GET", mount_path + "/{path:path}"));
            routes.insert(route_key(mode, "HEAD", mount_path + "/{path:path}"));
            continue;
        }

        std::smatch match;
        if (!std::regex_search(line, match, decorator)) {
            continue;
        }
        const std::string method = method_label(match[1].str());
        std::string rest = match[2].str();
        std::smatch path_match;
        while (!std::regex_search(rest, path_match, quoted_path)) {
            if (!std::getline(lines, rest)) {
                break;
            }
        }
        if (std::regex_search(rest, path_match, quoted_path)) {
            routes.insert(route_key(mode, method, path_match[1].str()));
        }
    }
    return routes;
}

} // namespace

int main(int argc, char **argv) {
    std::string path;
    std::filesystem::path source_root;
    bool enforce = false;
    bool worker_only = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--enforce") {
            enforce = true;
        } else if (arg == "--worker-only") {
            worker_only = true;
        } else if (arg == "--source-root") {
            if (++i == argc) {
                std::cerr << "--source-root requires an ezlocalai checkout path\n";
                return 2;
            }
            source_root = argv[i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "usage: parity_report [--enforce] [--worker-only] [--source-root <ezlocalai>] <parity_manifest.tsv>\n";
            return 0;
        } else if (path.empty()) {
            path = arg;
        } else {
            std::cerr << "unexpected argument: " << arg << "\n";
            std::cerr << "usage: parity_report [--enforce] [--worker-only] [--source-root <ezlocalai>] <parity_manifest.tsv>\n";
            return 2;
        }
    }

    if (path.empty()) {
        std::cerr << "usage: parity_report [--enforce] [--worker-only] [--source-root <ezlocalai>] <parity_manifest.tsv>\n";
        return 2;
    }

    std::ifstream in(path);
    if (!in) {
        std::cerr << "failed to open " << path << "\n";
        return 2;
    }

    std::map<std::string, int> counts;
    std::set<std::string> manifest_routes;
    int rows = 0;
    int malformed = 0;
    bool first = true;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        if (first) {
            first = false;
            continue;
        }
        const auto fields = split_tsv(line);
        if (fields.size() < 6 || (fields[4] != "implemented" && fields[4] != "partial" && fields[4] != "missing")) {
            malformed++;
            continue;
        }
        if (worker_only && fields[0] == "router") continue;
        counts[fields[4]]++;
        if (fields[0] == "api" || fields[0] == "router") {
            if (!manifest_routes.insert(route_key(fields[0], fields[1], fields[2])).second) ++malformed;
        }
        rows++;
    }

    const auto manifest_path = std::filesystem::path(path);
    const auto repo_root = std::filesystem::absolute(manifest_path).parent_path().parent_path();
    if (source_root.empty()) {
        source_root = std::filesystem::exists(repo_root / "app.py") ? repo_root : repo_root.parent_path() / "ezlocalai";
    }
    const auto app_source = source_root / "app.py";
    const auto router_source = source_root / "router_app.py";
    const bool route_sources_available =
        std::filesystem::exists(app_source) && (worker_only || std::filesystem::exists(router_source));
    std::set<std::string> python_routes;
    if (route_sources_available) {
        python_routes = scan_python_routes(app_source, "api");
        if (!worker_only) {
            auto router_routes = scan_python_routes(router_source, "router");
            python_routes.insert(router_routes.begin(), router_routes.end());
        }
    }

    std::vector<std::string> uncovered_routes;
    if (route_sources_available) {
        for (const auto &route : python_routes) {
            if (manifest_routes.find(route) == manifest_routes.end()) {
                uncovered_routes.push_back(route);
            }
        }
    }
    std::vector<std::string> extra_manifest_routes;
    if (route_sources_available) {
        for (const auto &route : manifest_routes) {
            if (python_routes.find(route) == python_routes.end()) {
                extra_manifest_routes.push_back(route);
            }
        }
    }
    const int implemented = counts["implemented"];
    const int partial = counts["partial"];
    const int missing = counts["missing"];
    std::cout << "scope: " << (worker_only ? "worker (router service owned by WorkerRouter)" : "worker and legacy router") << "\n";
    std::cout << "parity rows: " << rows << "\n";
    std::cout << "implemented: " << implemented << "\n";
    std::cout << "partial: " << partial << "\n";
    std::cout << "missing: " << missing << "\n";
    if (malformed) {
        std::cout << "malformed: " << malformed << "\n";
    }
    std::cout << "source routes: " << python_routes.size() << "\n";
    std::cout << "source root: " << source_root.string() << "\n";
    if (!route_sources_available) {
        std::cout << "source route scan: skipped\n";
    }
    std::cout << "uncovered source routes: " << uncovered_routes.size() << "\n";
    std::cout << "extra manifest routes: " << extra_manifest_routes.size() << "\n";
    for (const auto &route : uncovered_routes) {
        std::cout << "uncovered: " << route << "\n";
    }
    for (const auto &route : extra_manifest_routes) {
        std::cout << "extra: " << route << "\n";
    }

    if (malformed || !uncovered_routes.empty() || !extra_manifest_routes.empty()) {
        return 2;
    }
    if (enforce && (!route_sources_available || rows == 0)) {
        std::cerr << "parity enforcement failed: source route scan and a nonempty manifest are required\n";
        return 1;
    }
    if (enforce && (partial > 0 || missing > 0)) {
        std::cerr << "parity enforcement failed: partial or missing rows remain\n";
        return 1;
    }
    return 0;
}
