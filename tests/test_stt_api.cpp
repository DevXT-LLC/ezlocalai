#include "stt_api_test.hpp"
int main(int argc,char **argv) { try {
    expect(argc == 3,"source and API fixture paths"); const auto cases = records(argv[2]); expect(hash(read(fs::path(argv[1])/"app.py")) == cases[0].at("source_sha256"),"unchanged source API"); expect(hash(read(fs::path(argv[1])/"Pipes.py")) == cases[0].at("pool_sha256"),"unchanged source pool/guard"); size_t count = 0;
    Temporary temporary; struct Restore { fs::path path = fs::current_path(); ~Restore() { fs::current_path(path); } } restore; fs::current_path(temporary.path);
    for (size_t i = 1; i < cases.size(); ++i) { Environment env; env.config = cases[i].at("config"); count += run_api_case(env,cases[i],"API scenario " + std::to_string(i)); }
    std::cout << cases.size() - 1 << " source API scenarios and " << count << " native HTTP requests match\n"; return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; } }
