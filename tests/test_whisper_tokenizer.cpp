#include "ezlocalai/whisper_tokenizer.hpp"
#include "whisper_test_data.hpp"
#include <nlohmann/json.hpp>
#include <iostream>
#include <map>
#include <unistd.h>
using namespace whisper_test;
using namespace ezlocalai;
using Json = nlohmann::json;
std::string unpack(const fs::path &path,size_t size) {
    const auto compressed = read(path); std::string output(size + 1,'\0'); uLongf length = output.size();
    expect(uncompress(reinterpret_cast<Bytef *>(output.data()),&length,reinterpret_cast<const Bytef *>(compressed.data()),compressed.size()) == Z_OK && length == size,"compressed tokenizer fixture");
    output.resize(size); return output;
}
int main(int argc,char **argv) { try {
    expect(argc == 2,"tokenizer fixture argument"); const fs::path fixture(argv[1]); const auto assets = fixture.parent_path()/"whisper_tokenizer_assets";
    struct Temporary { fs::path path; ~Temporary() { std::error_code ignored; fs::remove_all(path,ignored); } };
    auto pattern = (fs::temp_directory_path()/"ezlocalai-tokenizer-XXXXXX").string();
    expect(mkdtemp(pattern.data()),"tokenizer temporary directory"); Temporary directory{pattern};
    const auto cases = rows(fixture); std::map<std::string,WhisperTokenizer> models;
    for (const auto &row : cases) if (row[0] == "H") {
        expect(row.size() == 5,"tokenizer identity columns");
        const auto raw = unpack(assets/row[3],std::stoul(row[4])); expect(hash(raw) == row[2],"pinned tokenizer bytes");
        const auto path = directory.path/(row[1] + ".json"); std::ofstream file(path,std::ios::binary); file.write(raw.data(),raw.size()); file.close();
        models.emplace(row[1],WhisperTokenizer(path.string(),row[1] == "multi"));
    }
    expect(models.size() == 2,"English and multilingual tokenizers");
    std::map<std::string,size_t> counts;
    size_t singles = 0, number = 0;
    for (const auto &row : cases) {
        ++number; const auto &kind = row[0]; if (kind == "H") continue;
        ++counts[kind]; const auto label = "tokenizer row " + std::to_string(number) + " (" + kind + ")";
        if (kind == "R") { expect(Json(whisper_pre_tokenize(row[1])) == Json::parse(row[2]),label + " pre-tokenization"); continue; }
        const auto &model = models.at(row[1]);
        if (kind == "A") {
            const auto expected = Json::parse(unpack(assets/row[2],std::stoul(row[3]))); expect(expected.size() == std::stoul(row[4]),label + " vocabulary count");
            for (size_t i = 0; i < expected.size(); ++i) {
                expect(model.decode({static_cast<uint32_t>(i)}) == expected[i][0] && model.decode_with_timestamps({static_cast<uint32_t>(i)}) == expected[i][1],label + " vocabulary token " + std::to_string(i)); ++singles;
            }
        } else if (kind == "E") {
            expect(Json(model.encode(row[2])) == Json::parse(row[3]),label + " BPE encoding for " + row[2]);
        } else if (kind == "D") {
            const auto ids = Json::parse(row[2]).get<std::vector<uint32_t>>();
            expect(model.decode(ids) == row[3] && model.decode_with_timestamps(ids) == row[4],label + " UTF-8/timestamp decoding");
        } else if (kind == "W") {
            const auto configured = model.configured(row[1] == "multi","transcribe",row[2]); const auto ids = Json::parse(row[4]).get<std::vector<uint32_t>>();
            const auto words = row[3] == "unicode" ? configured.split_tokens_on_unicode(ids) : row[3] == "spaces" ? configured.split_tokens_on_spaces(ids) : configured.split_to_word_tokens(ids);
            Json texts = Json::array(), groups = Json::array(); for (const auto &word : words) { texts.push_back(word.word); groups.push_back(word.tokens); }
            expect(texts == Json::parse(row[5]) && groups == Json::parse(row[6]),label + " word splitting");
        } else if (kind == "C") {
            try {
                const auto configured = model.configured(row[2] == "1",row[3],row[4]);
                expect(row[5].rfind("ERROR:",0) != 0 && Json(configured.sot_sequence()) == Json::parse(row[5]) && configured.language_code() == row[6],label + " configuration");
            } catch (const std::invalid_argument &error) { expect(row[5] == "ERROR:" + std::string(error.what()),label + " configuration error: " + error.what()); }
        } else if (kind == "N") {
            expect(Json(model.non_speech_tokens()) == Json::parse(row[2]),label + " non-speech tokens");
        } else if (kind == "S") {
            const auto input = Json::parse(row[3]);
            try {
                const auto result = whisper_suppressed_tokens(model,input.is_null() ? std::optional<std::vector<int>>{} : input.get<std::vector<int>>(),
                    row[2] == "legacy" ? WhisperTokenizerProfile::faster_whisper_1_0_3 : WhisperTokenizerProfile::faster_whisper_1_2_0);
                expect(row[4].rfind("ERROR:",0) != 0 && (result ? Json(*result) : Json()) == Json::parse(row[4]),label + " suppression");
            } catch (const std::invalid_argument &error) { expect(row[4] == "ERROR:" + std::string(error.what()),label + " suppression error"); }
        } else if (kind == "P") {
            const auto actual = whisper_prompt(model,Json::parse(row[3]).get<std::vector<uint32_t>>(),row[4] == "1",row[5],row[6],std::stoll(row[7]));
            expect(Json(actual) == Json::parse(row[8]),label + " prompt");
        } else throw std::runtime_error(label + " unknown fixture type");
    }
    expect(counts["R"] > 1700 && counts["E"] == counts["R"] * 2 && counts["W"] == 7100 && counts["D"] == 800 && counts["C"] == 1260 && counts["N"] == 2 && counts["S"] == 28 && counts["P"] == 640,"complete tokenizer fixtures");
    std::cout << number - 2 << " tokenizer cases and " << singles << " exhaustive single-token decodes pass\n"; return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; } }
