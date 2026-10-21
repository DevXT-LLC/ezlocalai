#include "ezlocalai/whisper_model.hpp"
#include "ezlocalai/stt_runtime.hpp"
#include "whisper_transcription_test.hpp"
using namespace transcription_test;
int main(int argc,char **argv) { try {
    expect(argc == 2,"tokenizer loader fixtures"); const fs::path fixtures(argv[1]); size_t total = 0;
    for (const bool legacy : {false,true}) {
        const std::string profile = legacy ? "legacy" : "modern";
        const auto cases = records(fixtures/("whisper_tokenizer_loader_" + profile + ".jsonl"));
        expect(cases.size() == 109 && cases[0].at("hub") == "1.5.0" && cases[0].at("tokenizers") == "0.22.1","complete pinned loader captures");
        expect(cases[0].at("source_sha256") == (legacy ? "3356e6a48c801d97719cf0d16d785b8880e7cf2a5c642efb4434951c3cc485a5" : "ecec022994e56d085351d5d4ff309db9017311680fc9171783b737139c739a6e"),"pinned source constructor");
        for (size_t i = 1; i < cases.size(); ++i) {
            const auto &row = cases[i]; Temporary temporary; const auto folder = temporary.path/"model", path = folder/"tokenizer.json", target = temporary.path/"target", fallback = temporary.path/"download";
            fs::create_directory(folder); fs::create_directory(fallback); std::ofstream(target) << "fixture";
            const auto kind = row.at("kind").get<std::string>();
            if (kind == "file") std::ofstream(path) << "fixture";
            else if (kind == "directory") fs::create_directory(path);
            else if (kind == "symlink_file") fs::create_symlink(target,path);
            else if (kind == "symlink_directory") fs::create_directory_symlink(fallback,path);
            else if (kind == "dangling") fs::create_symlink(temporary.path/"absent",path);
            else if (kind == "loop") fs::create_symlink("tokenizer.json",path);
            else if (kind == "parentfile") { fs::remove(folder); std::ofstream(folder) << "fixture"; }
            auto directory = folder.string(); if (kind == "nul_directory") directory += std::string("\0x",2);
            Json calls = Json::array(), actual;
            const auto download = [&](const std::string &repo,const std::string &filename,const HubSnapshotOptions &options) {
                expect(options.cache_directory.empty() && options.endpoint.empty() && !options.local_files_only && options.allow_patterns.empty(),"source tokenizer default download settings");
                calls.push_back({{"repo_id",repo},{"filename",filename},{"revision",options.revision}});
                if (!row.at("failure").is_null()) { const auto type = row.at("failure").get<std::string>(); throw HubError(type,"fixture failure",type == "HfHubHTTPError" ? 403 : 0,type == "HfHubHTTPError"); }
                return (fallback/"tokenizer.json").string();
            };
            try { auto selected = resolve_whisper_tokenizer(directory,row.at("multilingual"),download); selected.replace(0,temporary.path.string().size(),"/work"); actual["path"] = selected; }
            catch (const SttError &error) { actual["error"] = {{"category",error.category},{"message",error.what()}}; }
            const auto label = profile + " tokenizer resolution " + std::to_string(i); Json expected;
            if (row.contains("path")) expected["path"] = row.at("path"); else expected["error"] = row.at("error");
            compare_json(actual,expected,label); compare_json(calls,row.at("calls"),label + ".calls"); ++total;
        }
    }
    std::cout << total << " unchanged-source tokenizer fallback path and download/error cases match\n"; return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; } }
