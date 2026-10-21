#include "stt_api_test.hpp"
#include "ezlocalai/subtitles.hpp"
int main(int argc,char **argv) { try {
    expect(argc == 3,"source and fixture directories"); const auto fixtures = fs::absolute(argv[2]); const auto source_hash = hash(read(fs::path(argv[1])/"app.py"));
    const auto cases = records(fixtures/"subtitle_translation_cases.jsonl"); expect(source_hash == cases[0].at("source_sha256"),"unchanged source translation helper");
    for (size_t i = 1; i < cases.size(); ++i) {
        const auto &row = cases[i]; Json calls = Json::array(),result; bool error = false;
        try { result = Json::parse(translate_subtitle_segments(row.at("segments").dump(),"en","fr","fixture-text",[&](const auto &body) { calls.push_back(Json::parse(body)); return row.at("response").dump(); },row.at("max_words"))); } catch (...) { error = true; }
        const auto label = "translation helper " + std::to_string(i); compare_json(error,row.at("error"),label + " error"); compare_json(result,row.at("result"),label + " result"); compare_json(calls,row.at("calls"),label + " calls");
    }
    // Both translations must enter before either is allowed to complete.
    std::mutex mutex; std::condition_variable cv; int active = 0;
    const auto concurrent = generate_native_subtitles(R"({"language":"en","segments":[{"id":0,"start":0,"end":1,"text":"source"}]})","","fr,de","json","fixture-text",[&](const auto &) {
        std::unique_lock<std::mutex> lock(mutex); ++active; cv.notify_all(); expect(cv.wait_for(lock,std::chrono::seconds(5),[&] { return active == 2; }),"parallel target-language execution"); return R"({"choices":[{"message":{"content":"[1] translated"}}]})";
    }); expect(Json::parse(concurrent).at("subtitles").size() == 2,"parallel target results");
    const auto api = records(fixtures/"subtitles_api_cases.jsonl"); expect(source_hash == api[0].at("source_sha256"),"unchanged subtitle API"); expect(hash(read(fs::path(argv[1])/"Pipes.py")) == api[0].at("pool_sha256"),"unchanged pool/guard");
    Temporary temporary; struct Restore { fs::path path = fs::current_path(); ~Restore() { fs::current_path(path); } } restore; fs::current_path(temporary.path); size_t count = 0;
    for (size_t i = 1; i < api.size(); ++i) { Environment env; env.config = api[i].at("config"); count += run_api_case(env,api[i],"subtitle API " + std::to_string(i)); }
    std::cout << cases.size()-1 << " translation helper cases and " << count << " subtitle HTTP requests match; concurrent target languages verified\n"; return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; } }
