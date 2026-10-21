#include "stt_pipeline_test.hpp"
int main(int argc,char **argv) { try {
    expect(argc == 3,"source and pipeline fixture paths"); const auto cases = records(argv[2]); expect(hash(read(fs::path(argv[1])/"Pipes.py")) == cases[0].at("source_sha256"),"unchanged source pipeline"); expect(cases.size() == cases[0].at("cases").get<size_t>() + 1,"complete pipeline cases"); size_t count = 0;
    for (size_t i = 1; i < cases.size(); ++i) { Environment env; env.config = cases[i].at("config"); count += run_pipeline_case(env,cases[i],"pipeline " + std::to_string(i)); }
    std::cout << cases.size()-1 << " unchanged-source pipeline scenarios and " << count << " calls match\n"; return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; } }
