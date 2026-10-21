#include "ezlocalai/whisper_vad.hpp"
#include "whisper_test_data.hpp"
#include <cmath>
#include <iostream>
using namespace whisper_test;
using namespace ezlocalai;
namespace {
WhisperVadProfile profile(const std::string &value) { return value == "legacy" ? WhisperVadProfile::faster_whisper_1_0_3 : WhisperVadProfile::faster_whisper_1_2_0; }
std::vector<WhisperSpeechChunk> chunks(const std::string &value) {
    const auto numbers = integers<int64_t>(value); expect(numbers.size() % 2 == 0,"chunk pairs");
    std::vector<WhisperSpeechChunk> result;
    for (size_t i = 0; i < numbers.size(); i += 2) result.push_back({numbers[i],numbers[i + 1]});
    return result;
}
void equal_chunks(const std::vector<WhisperSpeechChunk> &actual,const std::string &value,const std::string &label) {
    const auto expected = chunks(value);
    expect(actual.size() == expected.size(),label + " chunk count: " + std::to_string(actual.size()) + " expected " + value);
    for (size_t i = 0; i < actual.size(); ++i) expect(actual[i].start == expected[i].start && actual[i].end == expected[i].end,
        label + " chunk " + std::to_string(i) + " actual " + std::to_string(actual[i].start) + "," + std::to_string(actual[i].end) + " expected " + value);
}
std::vector<std::string> split(const std::string &value,char delimiter) {
    std::vector<std::string> out; std::istringstream in(value);
    for (std::string item;std::getline(in,item,delimiter);) out.push_back(item);
    return out;
}
void policy(const fs::path &fixture) {
    size_t segments = 0, collected = 0, maps = 0;
    for (const auto &row : rows(fixture)) {
        if (row[0] == "P") {
            expect(row.size() == 12,"VAD policy columns"); ++segments;
            auto options = whisper_vad_options(profile(row[1])); options.threshold = std::stod(row[4]);
            if (!row[5].empty()) options.neg_threshold = std::stod(row[5]);
            options.min_speech_duration_ms = std::stod(row[6]); options.max_speech_duration_s = std::stod(row[7]);
            options.min_silence_duration_ms = std::stod(row[8]); options.speech_pad_ms = std::stod(row[9]);
            equal_chunks(whisper_speech_timestamps(integers<float>(row[10]),std::stoll(row[3]),options,profile(row[1]),std::stoi(row[2])),row[11],"policy " + row[1] + " " + std::to_string(segments));
        } else if (row[0] == "C") {
            expect(row.size() == 7,"VAD collection columns"); ++collected;
            const auto audio = integers<float>(row[3]);
            const auto actual = whisper_collect_chunks(audio,chunks(row[4]),std::stoi(row[1]),std::stod(row[2]));
            const auto expected = split(row[5],';'); expect(actual.size() == expected.size(),"collected audio group count");
            for (size_t i = 0; i < actual.size(); ++i) {
                const auto fields = split(expected[i],'|'); expect(fields.size() == 4,"collection result columns");
                expect(actual[i].offset == std::stod(fields[0]) && actual[i].duration == std::stod(fields[1]),"audio group offset/duration");
                expect(actual[i].audio == integers<float>(fields[2]),"audio group samples");
                equal_chunks(actual[i].segments,fields[3],"audio group metadata");
            }
            expect(whisper_collect_chunks_legacy(audio,chunks(row[4])) == integers<float>(row[6]),"legacy concatenated samples");
        } else if (row[0] == "M") {
            expect(row.size() == 10,"VAD map columns"); ++maps;
            const WhisperSpeechTimestampsMap map(chunks(row[4]),std::stoi(row[2]),std::stoi(row[3]),profile(row[1]));
            const auto time = std::stod(row[5]); const bool end = row[6] == "1";
            expect(map.chunk_index(time,end) == std::stoll(row[8]),"restored timestamp chunk index");
            const auto index = row[7].empty() ? std::optional<int64_t>{} : std::optional<int64_t>{std::stoll(row[7])};
            try {
                const auto result = map.original_time(time,index,end);
                expect(row[9] != "ERROR" && result == std::stod(row[9]),"restored timestamp case " + std::to_string(maps));
            } catch (const std::out_of_range &) { expect(row[9] == "ERROR","unexpected timestamp index error"); }
        } else throw std::runtime_error("Unknown VAD case");
    }
    expect(segments == 1120 && collected == 90 && maps == 8256,"complete VAD policy fixtures");
    std::cout << segments << " segmentation, " << collected << " collection and " << maps << " timestamp-map reference cases\n";
}
#ifdef EZLOCALAI_TEST_VAD_MODEL
void models(const fs::path &fixture,const fs::path &modern_directory,const fs::path &legacy_directory) {
    const auto cases = rows(fixture); const auto assets = fixture.parent_path()/"whisper_vad_assets";
    for (const auto &row : cases) if (row[0] == "H") {
        expect(row.size() == 4,"VAD model identity columns");
        expect(hash(read((row[1] == "modern" ? modern_directory : legacy_directory)/row[2])) == row[3],"pinned Silero model " + row[2]);
    }
    WhisperVadModel modern(WhisperVadProfile::faster_whisper_1_2_0,modern_directory.string());
    WhisperVadModel legacy(WhisperVadProfile::faster_whisper_1_0_3,legacy_directory.string());
    size_t count = 0; float maximum = 0;
    for (const auto &row : cases) {
        if (row[0] == "H") continue;
        expect(row[0] == "R" && row.size() == 8,"real VAD case columns"); ++count;
        const auto audio = packed(assets/row[2],std::stoul(row[3])), original = audio;
        const auto expected = packed(assets/row[4],std::stoul(row[5]));
        auto &model = row[1] == "modern" ? modern : legacy;
        const auto actual = model.probabilities(audio);
        expect(actual.size() == expected.size(),"real VAD probability count " + row[2]);
        float difference = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            expect(std::isfinite(actual[i]),"finite VAD probability");
            difference = std::max(difference,std::abs(actual[i] - expected[i]));
        }
        maximum = std::max(maximum,difference);
        expect(difference <= 0.000001f,"real VAD probability error " + row[1] + " " + row[2] + " " + std::to_string(difference));
        const auto options = whisper_vad_options(profile(row[1]));
        equal_chunks(whisper_speech_timestamps(actual,audio.size(),options,profile(row[1])),row[6],"real VAD default timestamps " + row[2]);
        auto stt = options; stt.min_silence_duration_ms = 500; stt.speech_pad_ms = 200;
        equal_chunks(model.speech_timestamps(audio,stt),row[7],"real VAD STT timestamps " + row[2]);
        expect(audio == original,"VAD leaves caller audio intact");
        expect(model.probabilities(audio) == actual,"VAD state resets between calls");
    }
    expect(count == 29,"complete native Silero model fixtures");
    std::cout << count << " real Silero VAD cases; maximum probability error " << maximum << '\n';
}
#endif
}
int main(int argc,char **argv) { try {
#ifdef EZLOCALAI_TEST_VAD_MODEL
    expect(argc == 4,"VAD fixture and modern/legacy model directories"); models(argv[1],argv[2],argv[3]);
#else
    expect(argc == 2,"VAD policy fixture"); policy(argv[1]);
#endif
    return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; } }
