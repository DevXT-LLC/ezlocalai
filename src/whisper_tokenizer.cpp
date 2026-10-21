// Whisper wrapper behavior follows faster-whisper (MIT); byte-level BPE follows
// Hugging Face tokenizers (Apache-2.0). See licenses for notices.
#include "ezlocalai/whisper_tokenizer.hpp"
#include "ezlocalai/whisper_unicode.hpp"
#include "ezlocalai/configuration.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <charconv>
#include <fstream>
#include <queue>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace ezlocalai {
namespace {
struct Character { uint32_t value; size_t begin, end; };
std::vector<Character> characters(const std::string &text,bool lossy = false) {
    std::vector<Character> out;
    for (size_t i = 0; i < text.size();) {
        const auto begin = i; const auto first = static_cast<unsigned char>(text[i]);
        if (first < 128) { out.push_back({first,i,i + 1}); ++i; continue; }
        const size_t length = first >= 0xc2 && first <= 0xdf ? 2 : first >= 0xe0 && first <= 0xef ? 3 : first >= 0xf0 && first <= 0xf4 ? 4 : 0;
        size_t consumed = 1; uint32_t value = length ? first & ((1u << (7 - length)) - 1) : 0;
        bool valid = length != 0;
        for (size_t j = 1; valid && j < length; ++j) {
            if (i + j >= text.size()) { valid = false; break; }
            const auto byte = static_cast<unsigned char>(text[i + j]);
            if (byte < 0x80 || byte >= 0xc0 || (j == 1 && ((first == 0xe0 && byte < 0xa0) || (first == 0xed && byte >= 0xa0) ||
                (first == 0xf0 && byte < 0x90) || (first == 0xf4 && byte >= 0x90)))) { valid = false; break; }
            value = (value << 6) | (byte & 63); ++consumed;
        }
        if (!valid && !lossy) throw std::invalid_argument("Invalid UTF-8 tokenizer input");
        i += consumed; out.push_back({valid ? value : 0xfffd,begin,i});
    }
    return out;
}
std::string utf8(uint32_t c) {
    std::string out;
    if (c < 128) out += static_cast<char>(c);
    else if (c < 2048) { out += static_cast<char>(0xc0 | (c >> 6)); out += static_cast<char>(0x80 | (c & 63)); }
    else if (c < 65536) { out += static_cast<char>(0xe0 | (c >> 12)); out += static_cast<char>(0x80 | ((c >> 6) & 63)); out += static_cast<char>(0x80 | (c & 63)); }
    else { out += static_cast<char>(0xf0 | (c >> 18)); out += static_cast<char>(0x80 | ((c >> 12) & 63)); out += static_cast<char>(0x80 | ((c >> 6) & 63)); out += static_cast<char>(0x80 | (c & 63)); }
    return out;
}
template<size_t N> bool in_ranges(uint32_t c,const whisper_unicode::Range (&ranges)[N]) {
    const auto found = std::lower_bound(std::begin(ranges),std::end(ranges),c,[](const auto &range,uint32_t value) { return range.last < value; });
    return found != std::end(ranges) && found->first <= c;
}
int category(uint32_t c) {
    if (in_ranges(c,whisper_unicode::letters)) return 1;
    if (in_ranges(c,whisper_unicode::numbers)) return 2;
    if (in_ranges(c,whisper_unicode::spaces)) return 3;
    return 0;
}
std::string valid_utf8(const std::string &bytes) {
    std::string out; for (const auto &c : characters(bytes,true)) out += utf8(c.value); return out;
}
uint64_t pair_key(uint32_t a,uint32_t b) { return (static_cast<uint64_t>(a) << 32) | b; }
const std::string languages = "af, am, ar, as, az, ba, be, bg, bn, bo, br, bs, ca, cs, cy, da, de, el, en, es, et, eu, fa, fi, fo, fr, gl, gu, ha, haw, he, hi, hr, ht, hu, hy, id, is, it, ja, jw, ka, kk, km, kn, ko, la, lb, ln, lo, lt, lv, mg, mi, mk, ml, mn, mr, ms, mt, my, ne, nl, nn, no, oc, pa, pl, ps, pt, ro, ru, sa, sd, si, sk, sl, sn, so, sq, sr, su, sv, sw, ta, te, tg, th, tk, tl, tr, tt, uk, ur, uz, vi, yi, yo, zh, yue";
}
std::vector<std::string> whisper_pre_tokenize(const std::string &text) {
    const auto chars = characters(text); std::vector<std::string> out;
    for (size_t at = 0; at < chars.size();) {
        const auto begin = at;
        if (chars[at].value == '\'') {
            bool contraction = false;
            for (const auto *suffix : {"'s","'t","'re","'ve","'m","'ll","'d"}) {
                const auto size = std::char_traits<char>::length(suffix);
                if (text.compare(chars[at].begin,size,suffix) == 0) { at += size; contraction = true; break; }
            }
            if (contraction) { out.push_back(text.substr(chars[begin].begin,chars[at - 1].end - chars[begin].begin)); continue; }
        }
        auto group = category(chars[at].value);
        if (chars[at].value == ' ' && at + 1 < chars.size() && category(chars[at + 1].value) != 3) group = category(chars[++at].value);
        ++at; while (at < chars.size() && category(chars[at].value) == group) ++at;
        // The greedy whitespace branch backtracks one character before a
        // following non-whitespace character; the final branch handles one.
        if (group == 3 && at < chars.size() && at - begin > 1) --at;
        out.push_back(text.substr(chars[begin].begin,chars[at - 1].end - chars[begin].begin));
    }
    return out;
}
struct WhisperTokenizer::Impl {
    struct Merge { uint32_t rank, token; };
    struct Added { std::string text; uint32_t id; bool normalized; };
    std::unordered_map<std::string,uint32_t> vocabulary;
    std::unordered_map<uint32_t,std::string> tokens;
    std::unordered_map<uint64_t,Merge> merges;
    std::unordered_set<uint32_t> special;
    std::vector<Added> added;
    std::array<uint32_t,256> byte_ids{};
    std::unordered_map<uint32_t,unsigned char> decoded_bytes;
    bool prefix_space = false, use_regex = true;
    explicit Impl(const std::string &path) {
        std::ifstream input(path); if (!input) throw std::runtime_error("Cannot open Whisper tokenizer " + path);
        const auto json = nlohmann::json::parse(input); const auto &model = json.at("model"), &pre = json.at("pre_tokenizer");
        if (!json.at("normalizer").is_null() || model.at("type") != "BPE" || pre.at("type") != "ByteLevel" || json.at("decoder").at("type") != "ByteLevel" ||
            !model.value("dropout",nlohmann::json{}).is_null() || !model.value("unk_token",nlohmann::json{}).is_null() ||
            model.value("continuing_subword_prefix",std::string{}) != "" || model.value("end_of_word_suffix",std::string{}) != "" || model.value("byte_fallback",false))
            throw std::invalid_argument("Unsupported Whisper tokenizer configuration");
        prefix_space = pre.value("add_prefix_space",false); use_regex = pre.value("use_regex",true);
        for (const auto &item : model.at("vocab").items()) { const auto id = item.value().get<uint32_t>(); vocabulary[item.key()] = id; tokens[id] = item.key(); }
        uint32_t rank = 0;
        for (const auto &item : model.at("merges")) {
            std::string left,right;
            if (item.is_array()) { left = item.at(0).get<std::string>(); right = item.at(1).get<std::string>(); }
            else { const auto text = item.get<std::string>(); const auto split = text.find(' '); if (split == std::string::npos) throw std::runtime_error("Invalid tokenizer merge"); left = text.substr(0,split); right = text.substr(split + 1); }
            merges[pair_key(vocabulary.at(left),vocabulary.at(right))] = {rank++,vocabulary.at(left + right)};
        }
        unsigned shifted = 256;
        for (unsigned byte = 0; byte < 256; ++byte) {
            const auto c = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || byte >= 174 ? byte : shifted++;
            byte_ids[byte] = vocabulary.at(utf8(c)); decoded_bytes[c] = static_cast<unsigned char>(byte);
        }
        for (const auto &item : json.at("added_tokens")) {
            if (item.value("single_word",false) || item.value("lstrip",false) || item.value("rstrip",false)) throw std::invalid_argument("Unsupported Whisper added-token flags");
            const auto text = item.at("content").get<std::string>(); const auto id = item.at("id").get<uint32_t>();
            if (text.empty()) continue;
            vocabulary[text] = id; tokens[id] = text; added.push_back({text,id,item.value("normalized",true)});
            if (item.value("special",false)) special.insert(id);
        }
    }
    std::vector<uint32_t> bpe(const std::string &word) const {
        struct Node { uint32_t token; int64_t prev, next; bool alive = true; };
        struct Candidate { uint32_t rank, token; size_t position; bool operator<(const Candidate &b) const { return rank != b.rank ? rank > b.rank : position > b.position; } };
        std::vector<Node> nodes; nodes.reserve(word.size());
        for (unsigned char byte : word) { const auto i = static_cast<int64_t>(nodes.size()); nodes.push_back({byte_ids[byte],i - 1,i + 1}); }
        if (nodes.empty()) return {};
        nodes.back().next = -1;
        std::priority_queue<Candidate> queue;
        const auto add = [&](size_t position) {
            const auto &node = nodes[position]; if (node.next < 0) return;
            const auto found = merges.find(pair_key(node.token,nodes[node.next].token));
            if (found != merges.end()) queue.push({found->second.rank,found->second.token,position});
        };
        for (size_t i = 0; i + 1 < nodes.size(); ++i) add(i);
        while (!queue.empty()) {
            const auto top = queue.top(); queue.pop(); auto &left = nodes[top.position];
            if (!left.alive || left.next < 0) continue;
            auto &right = nodes[left.next]; const auto found = merges.find(pair_key(left.token,right.token));
            if (found == merges.end() || found->second.token != top.token) continue;
            left.token = top.token; left.next = right.next; right.alive = false;
            if (right.next >= 0) nodes[right.next].prev = top.position;
            if (left.prev >= 0) add(left.prev);
            add(top.position);
        }
        std::vector<uint32_t> result; for (const auto &node : nodes) if (node.alive) result.push_back(node.token); return result;
    }
    void ordinary(const std::string &raw,std::vector<uint32_t> &output) const {
        if (raw.empty()) return;
        const auto text = prefix_space && raw.front() != ' ' ? " " + raw : raw;
        for (const auto &word : use_regex ? whisper_pre_tokenize(text) : std::vector<std::string>{text}) {
            const auto ids = bpe(word); output.insert(output.end(),ids.begin(),ids.end());
        }
    }
    void encode_part(const std::string &text,bool normalized,std::vector<uint32_t> &output) const {
        size_t at = 0;
        while (at < text.size()) {
            size_t best = std::string::npos; const Added *selected = nullptr;
            for (const auto &token : added) if (token.normalized == normalized) {
                const auto position = text.find(token.text,at);
                if (position < best || (position == best && position != std::string::npos && selected && token.text.size() > selected->text.size())) { best = position; selected = &token; }
            }
            const auto part = text.substr(at,best == std::string::npos ? best : best - at);
            if (normalized) ordinary(part,output); else encode_part(part,true,output);
            if (!selected) break;
            output.push_back(selected->id); at = best + selected->text.size();
        }
    }
    std::string decode_raw(const std::vector<uint32_t> &ids) const {
        std::string bytes;
        for (const auto id : ids) {
            const auto found = tokens.find(id); if (found == tokens.end() || special.count(id)) continue;
            std::string piece; bool all_mapped = true;
            for (const auto &c : characters(found->second)) { const auto byte = decoded_bytes.find(c.value); if (byte == decoded_bytes.end()) { all_mapped = false; break; } piece += static_cast<char>(byte->second); }
            bytes += all_mapped ? piece : found->second;
        }
        return valid_utf8(bytes);
    }
};
WhisperTokenizer::WhisperTokenizer(const std::string &path,bool multilingual,const std::string &task,const std::string &language)
    : impl_(std::make_shared<Impl>(path)) { configure(multilingual,task,language); }
void WhisperTokenizer::configure(bool multilingual,const std::string &task,const std::string &language) {
    task_.reset(); language_.reset(); language_code_ = "en";
    if (!multilingual) return;
    if (task != "transcribe" && task != "translate") throw std::invalid_argument("'" + task + "' is not a valid task (accepted tasks: transcribe, translate)");
    if (language.find(',') != std::string::npos || (", " + languages + ", ").find(", " + language + ", ") == std::string::npos) throw std::invalid_argument("'" + language + "' is not a valid language code (accepted language codes: " + languages + ")");
    task_ = token_id("<|" + task + "|>"); language_ = token_id("<|" + language + "|>"); language_code_ = language;
}
WhisperTokenizer WhisperTokenizer::configured(bool multilingual,const std::string &task,const std::string &language) const { auto copy = *this; copy.configure(multilingual,task,language); return copy; }
std::optional<uint32_t> WhisperTokenizer::token_id(const std::string &text) const { const auto found = impl_->vocabulary.find(text); return found == impl_->vocabulary.end() ? std::optional<uint32_t>{} : found->second; }
std::vector<uint32_t> WhisperTokenizer::encode(const std::string &text) const { characters(text); std::vector<uint32_t> output; impl_->encode_part(text,false,output); return output; }
std::string WhisperTokenizer::decode(const std::vector<uint32_t> &tokens) const { std::vector<uint32_t> text; for (auto token : tokens) if (token < eot()) text.push_back(token); return impl_->decode_raw(text); }
std::string WhisperTokenizer::decode_with_timestamps(const std::vector<uint32_t> &tokens) const {
    std::string output; std::vector<uint32_t> pending;
    for (auto token : tokens) {
        if (token < timestamp_begin()) { pending.push_back(token); continue; }
        output += impl_->decode_raw(pending); pending.clear();
        char number[64]; const auto converted = std::to_chars(number,number + sizeof(number),(token - timestamp_begin()) * 0.02,std::chars_format::fixed,2);
        if (converted.ec != std::errc{}) throw std::runtime_error("Invalid Whisper timestamp token");
        output += "<|" + std::string(number,converted.ptr) + "|>";
    }
    return output + impl_->decode_raw(pending);
}
std::vector<uint32_t> WhisperTokenizer::sot_sequence() const { std::vector<uint32_t> out{sot()}; if (language_) out.push_back(*language_); if (task_) out.push_back(*task_); return out; }
uint32_t WhisperTokenizer::sot() const { return token_id("<|startoftranscript|>").value(); }
uint32_t WhisperTokenizer::sot_lm() const { return token_id("<|startoflm|>").value(); }
uint32_t WhisperTokenizer::sot_prev() const { return token_id("<|startofprev|>").value(); }
uint32_t WhisperTokenizer::eot() const { return token_id("<|endoftext|>").value(); }
uint32_t WhisperTokenizer::transcribe() const { return token_id("<|transcribe|>").value(); }
uint32_t WhisperTokenizer::translate() const { return token_id("<|translate|>").value(); }
uint32_t WhisperTokenizer::no_timestamps() const { return token_id("<|notimestamps|>").value(); }
std::vector<int> WhisperTokenizer::non_speech_tokens() const {
    std::set<int> result{static_cast<int>(encode(" -").at(0)),static_cast<int>(encode(" '").at(0))};
    std::vector<std::string> symbols;
    for (const auto &c : characters("\"#()*+/:;<=>@[\\]^_`{|}~「」『』")) symbols.push_back(utf8(c.value));
    for (const auto *s : {"<<",">>","<<<",">>>","--","---","-(","-[","('","(\"","((","))","(((",")))","[[","]]","{{","}}","♪♪","♪♪♪"}) symbols.emplace_back(s);
    for (const auto &symbol : symbols) for (const auto &text : {symbol," " + symbol}) { const auto ids = encode(text); if (ids.size() == 1) result.insert(ids[0]); }
    for (const auto &c : characters("♩♪♫♬♭♮♯")) for (const auto &text : {utf8(c.value)," " + utf8(c.value)}) result.insert(encode(text).at(0));
    return {result.begin(),result.end()};
}
std::vector<WhisperWordTokens> WhisperTokenizer::split_tokens_on_unicode(const std::vector<uint32_t> &tokens) const {
    const auto full = characters(decode_with_timestamps(tokens)); size_t offset = 0;
    std::vector<WhisperWordTokens> words; std::vector<uint32_t> pending;
    for (auto token : tokens) {
        pending.push_back(token); const auto decoded = decode_with_timestamps(pending); const auto chars = characters(decoded);
        size_t replacement = 0; while (replacement < chars.size() && chars[replacement].value != 0xfffd) ++replacement;
        if (replacement == chars.size() || (replacement + offset < full.size() && full[replacement + offset].value == 0xfffd)) {
            words.push_back({decoded,pending}); pending.clear(); offset += chars.size();
        }
    }
    return words;
}
std::vector<WhisperWordTokens> WhisperTokenizer::split_tokens_on_spaces(const std::vector<uint32_t> &tokens) const {
    static const std::string punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
    std::vector<WhisperWordTokens> words;
    for (auto item : split_tokens_on_unicode(tokens)) {
        if (words.empty() || item.tokens.front() >= eot() || (!item.word.empty() && item.word.front() == ' ') || punctuation.find(strip_config_value(item.word)) != std::string::npos) words.push_back(std::move(item));
        else { words.back().word += item.word; words.back().tokens.insert(words.back().tokens.end(),item.tokens.begin(),item.tokens.end()); }
    }
    return words;
}
std::vector<WhisperWordTokens> WhisperTokenizer::split_to_word_tokens(const std::vector<uint32_t> &tokens) const {
    const bool unicode = language_code_ == "zh" || language_code_ == "ja" || language_code_ == "th" || language_code_ == "lo" || language_code_ == "my" || language_code_ == "yue";
    return unicode ? split_tokens_on_unicode(tokens) : split_tokens_on_spaces(tokens);
}
std::vector<uint32_t> whisper_prompt(const WhisperTokenizer &tokenizer,const std::vector<uint32_t> &previous,bool without_timestamps,const std::string &prefix,const std::string &hotwords,int64_t max_length) {
    const auto half = max_length / 2 - (max_length < 0 && max_length % 2 != 0), limit = half - 1;
    const auto truncate = [&](std::vector<uint32_t> ids) { if (static_cast<int64_t>(ids.size()) >= half) ids.resize(limit < 0 ? std::max<int64_t>(0,static_cast<int64_t>(ids.size()) + limit) : std::min<int64_t>(ids.size(),limit)); return ids; };
    std::vector<uint32_t> out;
    const auto append = [&](const auto &ids) { out.insert(out.end(),ids.begin(),ids.end()); };
    if (!previous.empty() || (!hotwords.empty() && prefix.empty())) {
        out.push_back(tokenizer.sot_prev()); if (!hotwords.empty() && prefix.empty()) append(truncate(tokenizer.encode(" " + strip_config_value(hotwords))));
        if (!previous.empty()) {
            const auto start = limit > 0 ? std::max<int64_t>(0,static_cast<int64_t>(previous.size()) - limit) : std::min<int64_t>(previous.size(),-limit);
            out.insert(out.end(),previous.begin() + start,previous.end());
        }
    }
    append(tokenizer.sot_sequence()); if (without_timestamps) out.push_back(tokenizer.no_timestamps());
    if (!prefix.empty()) { const auto ids = truncate(tokenizer.encode(" " + strip_config_value(prefix))); if (!without_timestamps) out.push_back(tokenizer.timestamp_begin()); append(ids); }
    return out;
}
std::optional<std::vector<int>> whisper_suppressed_tokens(const WhisperTokenizer &tokenizer,std::optional<std::vector<int>> tokens,WhisperTokenizerProfile profile) {
    const bool legacy = profile == WhisperTokenizerProfile::faster_whisper_1_0_3;
    if (!tokens) { if (legacy) return {}; throw std::invalid_argument("argument of type 'NoneType' is not iterable"); }
    const bool defaults = std::find(tokens->begin(),tokens->end(),-1) != tokens->end();
    if (legacy && (tokens->empty() || defaults)) return tokens;
    if (defaults) {
        tokens->erase(std::remove_if(tokens->begin(),tokens->end(),[](int value) { return value < 0; }),tokens->end());
        const auto suppressed = tokenizer.non_speech_tokens(); tokens->insert(tokens->end(),suppressed.begin(),suppressed.end());
    }
    for (auto value : {tokenizer.transcribe(),tokenizer.translate(),tokenizer.sot(),tokenizer.sot_prev(),tokenizer.sot_lm()}) tokens->push_back(value);
    std::sort(tokens->begin(),tokens->end()); tokens->erase(std::unique(tokens->begin(),tokens->end()),tokens->end()); return tokens;
}
}
