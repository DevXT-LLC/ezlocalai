#include "ezlocalai/torch_checkpoint.hpp"
#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace ezlocalai {
namespace {
enum class Kind { mark, none, boolean, integer, floating, string, tuple, list, dictionary, global, storage, tensor };
struct Value;
using Pointer = std::shared_ptr<Value>;
struct Value {
    Kind kind;
    int64_t integer = 0;
    std::string text;
    std::vector<Pointer> sequence;
    std::map<std::string, Pointer> dictionary;
    TorchCheckpointTensor tensor;
    bool ordered = false;
    explicit Value(Kind kind) : kind(kind) {}
};
Pointer make(Kind kind) { return std::make_shared<Value>(kind); }
[[noreturn]] void fail(const std::string &message) { throw std::runtime_error("Invalid torch checkpoint: " + message); }
void require(bool condition, const std::string &message) { if (!condition) fail(message); }
const std::map<std::string, std::string> storage_types = {
    {"torch.FloatStorage", "float32"}, {"torch.DoubleStorage", "float64"},
    {"torch.HalfStorage", "float16"}, {"torch.BFloat16Storage", "bfloat16"},
    {"torch.LongStorage", "int64"}, {"torch.IntStorage", "int32"},
    {"torch.ShortStorage", "int16"}, {"torch.CharStorage", "int8"},
    {"torch.ByteStorage", "uint8"}, {"torch.BoolStorage", "bool"},
    {"torch.ComplexFloatStorage", "complex64"}, {"torch.ComplexDoubleStorage", "complex128"}
};
Pointer global(std::string module, std::string name) {
    auto value = make(Kind::global); value->text = module + "." + name;
    const auto &full = value->text;
    require(storage_types.count(full) || full == "collections.OrderedDict" ||
            full == "torch._utils._rebuild_tensor" || full == "torch._utils._rebuild_tensor_v2" ||
            full == "torch._utils._rebuild_parameter", "unsupported global " + full);
    return value;
}
int64_t integer(const Pointer &value) { require(value->kind == Kind::integer, "integer required"); return value->integer; }
const std::string &string(const Pointer &value) { require(value->kind == Kind::string, "string required"); return value->text; }
std::vector<int64_t> dimensions(const Pointer &value) {
    require(value->kind == Kind::tuple, "tensor dimensions must be a tuple");
    std::vector<int64_t> result;
    for (const auto &item : value->sequence) { const auto n = integer(item); require(n >= 0, "negative tensor dimension or stride"); result.push_back(n); }
    return result;
}
void validate_layout(const TorchCheckpointTensor &tensor) {
    require(tensor.shape.size() == tensor.strides.size(), "shape/stride rank mismatch");
    require(tensor.storage_offset >= 0 && tensor.storage_offset <= tensor.storage_elements, "storage offset outside tensor storage");
    if (std::find(tensor.shape.begin(), tensor.shape.end(), 0) != tensor.shape.end()) return;
    uint64_t last = static_cast<uint64_t>(tensor.storage_offset);
    uint64_t elements = 1;
    for (size_t i = 0; i < tensor.shape.size(); ++i) {
        const auto n = static_cast<uint64_t>(tensor.shape[i]), stride = static_cast<uint64_t>(tensor.strides[i]);
        require(n <= uint64_t(std::numeric_limits<int64_t>::max()) / elements, "tensor element count overflow");
        elements *= n;
        require(stride == 0 || n - 1 <= (uint64_t(std::numeric_limits<int64_t>::max()) - last) / stride, "tensor stride overflow");
        last += (n - 1) * stride;
    }
    require(last < static_cast<uint64_t>(tensor.storage_elements), "tensor layout outside storage");
}
class Reader {
    std::string_view input_;
    size_t position_ = 0;
    std::vector<Pointer> stack_;
    std::unordered_map<uint32_t, Pointer> memo_;
    std::vector<Pointer> values_;
    Pointer make(Kind kind) {
        auto value = std::make_shared<Value>(kind); values_.push_back(value); return value;
    }
    uint8_t byte() { require(position_ < input_.size(), "truncated pickle"); return static_cast<uint8_t>(input_[position_++]); }
    uint64_t unsigned_integer(size_t count) {
        require(count <= 8 && count <= input_.size() - position_, "truncated integer");
        uint64_t value = 0; for (size_t i = 0; i < count; ++i) value |= uint64_t(byte()) << (8 * i); return value;
    }
    std::string bytes(uint64_t count) {
        require(count <= input_.size() - position_, "truncated string");
        auto value = std::string(input_.substr(position_, count)); position_ += count; return value;
    }
    std::string line() {
        const auto end = input_.find('\n', position_); require(end != std::string_view::npos, "unterminated global name");
        const auto start = position_; position_ = end + 1; return std::string(input_.substr(start, end - start));
    }
    Pointer pop() { require(!stack_.empty(), "stack underflow"); auto value = stack_.back(); stack_.pop_back(); return value; }
    Pointer top() { require(!stack_.empty(), "empty stack"); return stack_.back(); }
    std::vector<Pointer> marked() {
        size_t index = stack_.size(); while (index && stack_[index - 1]->kind != Kind::mark) --index;
        require(index != 0, "missing mark");
        std::vector<Pointer> items(stack_.begin() + index, stack_.end()); stack_.resize(index - 1); return items;
    }
    void push_integer(int64_t integer) { auto value = make(Kind::integer); value->integer = integer; stack_.push_back(value); }
    void push_string(uint64_t count) { auto value = make(Kind::string); value->text = bytes(count); stack_.push_back(value); }
    void tuple(size_t count) {
        require(count <= stack_.size(), "tuple stack underflow"); auto value = make(Kind::tuple);
        value->sequence.assign(stack_.end() - count, stack_.end()); stack_.resize(stack_.size() - count); stack_.push_back(value);
    }
    void dictionary_items(const Pointer &dictionary, const std::vector<Pointer> &items) {
        require(dictionary->kind == Kind::dictionary && items.size() % 2 == 0, "invalid dictionary items");
        for (size_t i = 0; i < items.size(); i += 2) dictionary->dictionary[string(items[i])] = items[i + 1];
    }
    void persistent_storage() {
        const auto value = pop(); require(value->kind == Kind::tuple && value->sequence.size() == 5, "invalid persistent storage");
        const auto &items = value->sequence;
        require(string(items[0]) == "storage" && items[1]->kind == Kind::global, "invalid storage descriptor");
        const auto type = storage_types.find(items[1]->text); require(type != storage_types.end(), "unsupported storage type");
        auto storage = make(Kind::storage); storage->tensor.scalar_type = type->second;
        storage->tensor.storage_key = string(items[2]); string(items[3]); // saved device is remapped by the caller
        storage->tensor.storage_elements = integer(items[4]); require(storage->tensor.storage_elements >= 0, "negative storage size");
        const auto &key = storage->tensor.storage_key;
        require(!key.empty() && key != "." && key != ".." && key.find_first_of("/\\") == std::string::npos && key.find('\0') == std::string::npos, "invalid storage key");
        stack_.push_back(storage);
    }
    void reduce() {
        const auto args = pop(), callable = pop();
        require(callable->kind == Kind::global && args->kind == Kind::tuple, "invalid reduction");
        const auto &items = args->sequence;
        if (callable->text == "collections.OrderedDict") {
            auto result = make(Kind::dictionary); result->ordered = true;
            require(items.size() <= 1, "invalid OrderedDict arguments");
            if (!items.empty()) {
                require(items[0]->kind == Kind::list || items[0]->kind == Kind::tuple, "invalid OrderedDict sequence");
                for (const auto &pair : items[0]->sequence) {
                    require((pair->kind == Kind::tuple || pair->kind == Kind::list) && pair->sequence.size() == 2, "invalid OrderedDict pair");
                    dictionary_items(result, pair->sequence);
                }
            }
            stack_.push_back(result); return;
        }
        if (callable->text == "torch._utils._rebuild_parameter") {
            require(items.size() == 3 && items[0]->kind == Kind::tensor && items[1]->kind == Kind::boolean, "invalid parameter reduction");
            stack_.push_back(items[0]); return;
        }
        require(callable->text == "torch._utils._rebuild_tensor" || callable->text == "torch._utils._rebuild_tensor_v2", "unsupported reduction");
        require(items.size() == (callable->text == "torch._utils._rebuild_tensor" ? 4 : 6) ||
                (callable->text == "torch._utils._rebuild_tensor_v2" && items.size() == 7), "invalid tensor arguments");
        require(items[0]->kind == Kind::storage, "tensor requires storage");
        auto result = make(Kind::tensor); result->tensor = items[0]->tensor;
        result->tensor.storage_offset = integer(items[1]); result->tensor.shape = dimensions(items[2]); result->tensor.strides = dimensions(items[3]);
        if (items.size() > 4) require(items[4]->kind == Kind::boolean && (items[5]->kind == Kind::dictionary || items[5]->kind == Kind::none), "invalid tensor state");
        validate_layout(result->tensor); stack_.push_back(result);
    }
public:
    explicit Reader(std::string_view input) : input_(input) {}
    ~Reader() {
        // Pickle can describe cyclic metadata. Break all edges before releasing
        // the arena, including on parse failure; no parsed objects escape.
        for (const auto &value : values_) { value->sequence.clear(); value->dictionary.clear(); }
    }
    std::map<std::string, TorchCheckpointTensor> parse() {
        require(byte() == 0x80, "pickle protocol header required"); const auto protocol = byte(); require(protocol >= 2 && protocol <= 5, "unsupported pickle protocol");
        while (position_ < input_.size()) {
            const auto opcode = byte();
            // Match the pinned weights-only loader, which rejects FRAME,
            // MEMOIZE and STACK_GLOBAL even under protocol 4/5 headers.
            switch (opcode) {
                case '.': {
                    require(stack_.size() == 1 && top()->kind == Kind::dictionary, "checkpoint must be a dictionary");
                    const auto found = top()->dictionary.find("model_state_dict");
                    require(found != top()->dictionary.end() && found->second->kind == Kind::dictionary, "model_state_dict missing");
                    std::map<std::string, TorchCheckpointTensor> result;
                    for (const auto &item : found->second->dictionary) { require(item.second->kind == Kind::tensor, "non-tensor state entry " + item.first); result[item.first] = item.second->tensor; }
                    return result;
                }
                case '(': stack_.push_back(make(Kind::mark)); break;
                case 'N': stack_.push_back(make(Kind::none)); break;
                case 0x88: case 0x89: { auto value = make(Kind::boolean); value->integer = opcode == 0x88; stack_.push_back(value); break; }
                case 'K': push_integer(byte()); break;
                case 'M': push_integer(unsigned_integer(2)); break;
                case 'J': push_integer(static_cast<int32_t>(unsigned_integer(4))); break;
                case 0x8a: {
                    const auto count = byte(); require(count <= 8, "integer exceeds int64");
                    auto value = unsigned_integer(count); if (count && count < 8 && (value & (uint64_t(1) << (count * 8 - 1)))) value |= (~uint64_t(0)) << (count * 8);
                    push_integer(static_cast<int64_t>(value)); break;
                }
                case 'G': bytes(8); stack_.push_back(make(Kind::floating)); break;
                case 'U': { const auto size = byte(); push_string(size); break; }
                case 'X': { const auto size = unsigned_integer(4); push_string(size); break; }
                case ')': stack_.push_back(make(Kind::tuple)); break;
                case ']': stack_.push_back(make(Kind::list)); break;
                case '}': stack_.push_back(make(Kind::dictionary)); break;
                case 't': { const auto items = marked(); auto value = make(Kind::tuple); value->sequence = items; stack_.push_back(value); break; }
                case 0x85: tuple(1); break;
                case 0x86: tuple(2); break;
                case 0x87: tuple(3); break;
                case 's': { const auto value = pop(), key = pop(); dictionary_items(top(), {key, value}); break; }
                case 'u': { const auto items = marked(); dictionary_items(top(), items); break; }
                case 'a': { const auto value = pop(); require(top()->kind == Kind::list, "append requires list"); top()->sequence.push_back(value); break; }
                case 'e': { const auto items = marked(); require(top()->kind == Kind::list, "appends requires list"); top()->sequence.insert(top()->sequence.end(), items.begin(), items.end()); break; }
                case 'c': { auto module = line(); auto name = line(); stack_.push_back(global(std::move(module), std::move(name))); break; }
                case 'Q': persistent_storage(); break;
                case 'R': reduce(); break;
                case 'b': { const auto state = pop(); require(top()->kind == Kind::dictionary && top()->ordered && state->kind == Kind::dictionary, "unsupported object state"); break; }
                case 'q': { const auto index = byte(); memo_[index] = top(); break; }
                case 'r': { const auto index = unsigned_integer(4); memo_[index] = top(); break; }
                case 'h': case 'j': { const auto index = opcode == 'h' ? byte() : unsigned_integer(4); const auto item = memo_.find(index); require(item != memo_.end(), "unknown memo reference"); stack_.push_back(item->second); break; }
                default: fail("unsupported pickle opcode " + std::to_string(opcode));
            }
        }
        fail("missing STOP opcode");
    }
};
}
std::map<std::string, TorchCheckpointTensor> parse_torch_checkpoint_pickle(std::string_view pickle) {
    return Reader(pickle).parse();
}
}
