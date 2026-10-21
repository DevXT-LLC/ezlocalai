#include "ezlocalai/ctts_paths.hpp"
#include "ezlocalai/llama_tts.hpp"
#include "whisper_transcription_test.hpp"
using namespace transcription_test;
int main(int argc,char **argv) { try { expect(argc==3,"source and fixture"); const auto rows=records(argv[2]); expect(hash(read(fs::path(argv[1])/"ezlocalai/CTTS.py"))==rows[0].at("source_sha256"),"pinned source");
    for(size_t i=1;i<rows.size();++i) { const auto &row=rows[i]; Temporary temporary; const auto root=temporary.path; fs::create_directories(root/"voices"); fs::create_directories(root/"outside");
        for(const auto &entry:row.at("setup")) { const auto path=root/entry.at("path").get<std::string>(); fs::create_directories(path.parent_path()); if(entry.at("kind")=="dir") fs::create_directories(path); else if(entry.at("kind")=="symlink") fs::create_symlink(entry.at("target").get<std::string>(),path); else { std::ofstream out(path,std::ios::binary); out<<unhex(entry.at("data")); out.close(); if(entry.contains("mode")) fs::permissions(path,fs::perms(entry.at("mode").get<unsigned>())); } }
        const auto replace=[&](std::string value) { size_t p=0; while((p=value.find(root.string(),p))!=std::string::npos) { value.replace(p,root.string().size(),"<ROOT>"); p+=6; } return value; };
        const auto value=row.at("value").is_null()?std::nullopt:std::optional<std::string>(row.at("value").get<std::string>()); Json actual;
        try { std::optional<std::string> result; if(row.at("method")=="contained") result=ctts_contained_file_path((root/row.at("base").get<std::string>()).string(),value,row.at("extension")); else if(row.at("method")=="voice") result=ctts_voice_audio_path((root/"voices").string(),value); else result=ctts_voice_ref_text((root/"voices").string(),value); actual["value_result"]=result?Json(replace(*result)):Json(); }
        catch(const TtsError &e) { actual["error"]={e.category,replace(e.what())}; }
        compare_json(actual,row.contains("error")?Json{{"error",row.at("error")}}:Json{{"value_result",row.at("value_result")}},"voice path case "+std::to_string(i));
    } expect(rows.size()-1==rows[0].at("cases"),"all path cases"); std::cout<<rows.size()-1<<" CTTS real-filesystem path/voice/transcript cases match source\n"; return 0;
} catch(const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; } }
