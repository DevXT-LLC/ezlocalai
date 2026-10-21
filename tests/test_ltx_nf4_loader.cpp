#include "model_loader.h"
#include "ezlocalai/nf4.hpp"
#include "model/common/ggml_block.hpp"
#include "ggml-cpu.h"
#include "json.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstring>
#include <unistd.h>
namespace fs = std::filesystem;
using Json = nlohmann::json;
static void check(bool value, const std::string& message) { if (!value) throw std::runtime_error(message); }
static std::vector<uint16_t> decode(const std::string& hex) {
    std::vector<uint16_t> values;
    for (size_t i = 0; i < hex.size(); i += 4)
        values.push_back(std::stoul(hex.substr(i, 2), nullptr, 16) | (std::stoul(hex.substr(i + 2, 2), nullptr, 16) << 8));
    return values;
}
static std::string contents(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), {}};
}
static void write_weights(const fs::path& path, const std::vector<uint16_t>& input, bool half, bool f32,
                          int64_t columns = 8, int64_t rows = 8) {
    const std::string name = "language_model.model.layers.0.self_attn.q_proj.weight";
    const std::string norm = "language_model.model.layers.0.input_layernorm.weight";
    const std::string embedding = "language_model.model.embed_tokens.weight";
    const size_t matrix_bytes = input.size() * (f32 ? 4 : 2);
    Json header = {
        {name, {{"dtype", f32 ? "F32" : half ? "F16" : "BF16"}, {"shape", {rows, columns}}, {"data_offsets", {0, matrix_bytes}}}},
        {norm, {{"dtype", half ? "F16" : "BF16"}, {"shape", {8}}, {"data_offsets", {matrix_bytes, matrix_bytes + 16}}}},
        {embedding, {{"dtype", half ? "F16" : "BF16"}, {"shape", {8, 8}}, {"data_offsets", {matrix_bytes + 16, matrix_bytes + 144}}}}
    };
    std::string encoded = header.dump();
    encoded.resize((encoded.size() + 7) / 8 * 8, ' ');
    std::ofstream out(path, std::ios::binary);
    const uint64_t size = encoded.size();
    out.write(reinterpret_cast<const char*>(&size), 8);
    out.write(encoded.data(), encoded.size());
    if (f32) {
        for (const auto bits : input) {
            const float value = half ? ezlocalai::nf4_detail::from_f16(bits) : ezlocalai::nf4_detail::from_bf16(bits);
            out.write(reinterpret_cast<const char*>(&value), 4);
        }
    } else out.write(reinterpret_cast<const char*>(input.data()), input.size() * 2);
    out.write(reinterpret_cast<const char*>(input.data()), 16);
    out.write(reinterpret_cast<const char*>(input.data()), 128);
    check(bool(out), "write safetensors fixture");
}
static void run(const fs::path& path, const std::vector<uint16_t>& input, const std::vector<uint16_t>& expected,
                bool half, bool enabled, bool mapped, bool lazy, bool unsupported = false,
                int64_t columns = 8, int64_t rows = 8) {
    const auto original = contents(path);
    ModelLoader loader;
    loader.set_n_threads(3);
    if (enabled && !lazy) loader.set_ltx_nf4(84, 1536);
    check(loader.init_from_file_and_convert_name(path.string(), "ltx_diffusers.text_encoder.", VERSION_LTXAV), "load source");
    if (enabled && lazy) loader.set_ltx_nf4(84, 1536);
    const std::string name = "text_encoders.llm.model.layers.0.self_attn.q_proj.weight";
    const std::string norm_name = "text_encoders.llm.model.layers.0.input_layernorm.weight";
    const std::string embedding_name = "text_encoders.llm.model.embed_tokens.weight";
    ggml_context* ctx = ggml_init({size_t(1024 * 1024 + columns * rows * 6 + columns * 16), nullptr, false});
    check(ctx, "GGML context");
    const auto dtype = half ? GGML_TYPE_F16 : GGML_TYPE_BF16;
    loader.set_wtype_override(dtype);
    check(loader.get_tensor_storage_map().at(name).ltx_nf4_type == (enabled ? dtype : GGML_TYPE_COUNT), "NF4 metadata precision");
    Linear linear(columns, rows, false);
    linear.init(ctx, loader.get_tensor_storage_map(), "text_encoders.llm.model.layers.0.self_attn.q_proj");
    std::map<std::string, ggml_tensor*> projections;
    linear.get_param_tensors(projections, "text_encoders.llm.model.layers.0.self_attn.q_proj");
    auto* tensor = projections.at(name);
    if (enabled) check(tensor->type == GGML_TYPE_I8 && ggml_nbytes(tensor) == ezlocalai::Nf4StorageLayout(columns * rows).bytes,
                       "Linear allocates only packed resident weights");
    if (unsupported) tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 8);
    auto* norm = ggml_new_tensor_1d(ctx, dtype, 8);
    auto* embedding = ggml_new_tensor_2d(ctx, dtype, 8, 8);
    std::map<std::string, ggml_tensor*> tensors{{name, tensor}, {norm_name, norm}, {embedding_name, embedding}};
    std::vector<MmapTensorStore> mappings;
    if (mapped) {
        mappings = loader.mmap_tensors(tensors, {}, false);
        if (enabled) check(tensor->buffer == nullptr, "NF4 destination must not alias source mmap");
        check(norm->buffer != nullptr && embedding->buffer != nullptr, "unconverted weights keep mmap eligibility");
    }
    auto callback = [&](const TensorStorage& source, ggml_tensor** destination) {
        const auto found = tensors.find(source.name);
        *destination = found == tensors.end() ? nullptr : found->second;
        return true;
    };
    // ModelManager marks mapped parameters as loaded and passes only private
    // destinations to a file-reader clone. Re-reading mapped destinations
    // would incorrectly ask the loader to write a read-only source mapping.
    std::set<std::string> pending;
    for (const auto& entry : tensors) if (!entry.second->buffer) pending.insert(entry.first);
    const auto id = loader.get_tensor_storage_map().at(name).file_id;
    check(loader.file_tensors(id, VERSION_LTXAV).at(name).ltx_nf4_type == (enabled ? dtype : GGML_TYPE_COUNT),
          "file-reader clone retains NF4 compute precision");
    const bool loaded = lazy ? loader.load_file_tensors(id, VERSION_LTXAV, callback, pending, mapped)
                             : loader.load_tensors(callback, mapped, &pending, false);
    if (unsupported) check(!loaded, "invalid NF4 destination rejected");
    else {
        check(loaded, "load tensor values");
        const auto& reference = enabled ? expected : input;
        GGMLRunnerContext execution;
        execution.ggml_ctx = ctx;
        auto* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, columns, 3);
        for (int64_t i = 0; i < columns * 3; ++i) static_cast<float*>(x->data)[i] = (i % 37 - 18) / 32.f;
        auto* product = linear.forward(&execution, x);
        auto* w = product->src[0];
        check(enabled ? ezlocalai::is_nf4_ggml(w) : w == tensor, "Linear executes the selected weight representation");
        auto* source = ggml_new_tensor_2d(ctx, dtype, columns, rows);
        memcpy(source->data, reference.data(), reference.size() * sizeof(uint16_t));
        auto* source_product = ggml_mul_mat(ctx, source, x);
        auto* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, product);
        ggml_build_forward_expand(graph, source_product);
        auto* backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(backend, 3);
        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "loaded Linear graph executes");
        ggml_backend_free(backend);
        check(std::memcmp(w->data, reference.data(), reference.size() * 2) == 0, "installed NF4 output bytes");
        check(product->data != source_product->data &&
              std::memcmp(product->data, source_product->data, ggml_nbytes(product)) == 0,
              "loaded packed Linear matches source-expanded matrix product");
        check(std::memcmp(norm->data, input.data(), 16) == 0, "normalization weights unchanged");
        check(std::memcmp(embedding->data, input.data(), 128) == 0, "embedding weights unchanged");
    }
    check(contents(path) == original, "source model bytes unchanged");
    ggml_free(ctx);
}
int main(int argc, char** argv) {
    try {
        check(argc == 3 || (argc == 5 && std::string(argv[3]) == "--weights"), "BF16/FP16 fixtures required; optional --weights manifest");
        char temp[] = "/tmp/ezlocalai-nf4-loader-XXXXXX";
        check(mkdtemp(temp), "temporary directory");
        size_t count = 0;
        for (int format = 0; format < 2; ++format) {
            std::ifstream in(argv[format + 1]); Json fixture; in >> fixture;
            const bool half = format == 1;
            const std::string encoding = half ? "f16" : "bf16";
            const auto row = std::find_if(fixture["cases"].begin(), fixture["cases"].end(), [](const Json& row) {
                return row["size"] == 64 && row["pattern"] == "normal42" && row["nested"] == true;
            });
            check(row != fixture["cases"].end(), "reference case present");
            const auto input = decode(row->at("input_" + encoding + "_hex"));
            const auto expected = decode(row->at("dequantized_" + encoding + "_hex"));
            for (bool f32 : {false, true}) for (bool enabled : {false, true})
                for (bool mapped : {false, true}) for (bool lazy : {false, true}) {
                    const auto path = fs::path(temp) / "model.safetensors";
                    write_weights(path, input, half, f32);
                    run(path, input, expected, half, enabled, mapped, lazy);
                    ++count;
                }
            const auto path = fs::path(temp) / "invalid.safetensors";
            write_weights(path, input, half, true);
            run(path, input, expected, half, true, false, false, true);
        }
        if (argc == 5) {
            const fs::path manifest(argv[4]);
            const auto fixtures = Json::parse(contents(manifest));
            check(fixtures.at("cases").size() == 2, "both full Gemma projections required");
            for (const auto& row : fixtures.at("cases")) {
                auto read = [&](const char* key) {
                    const auto& file = row.at("files").at(key);
                    const auto bytes = contents(manifest.parent_path() / file.at("path").get<std::string>());
                    check(bytes.size() == file.at("bytes") && bytes.size() % 2 == 0, "full weight fixture length");
                    std::vector<uint16_t> values(bytes.size() / 2);
                    memcpy(values.data(), bytes.data(), bytes.size());
                    return values;
                };
                const auto input = read("input"), expected = read("dequantized");
                const int64_t columns = row.at("shape").at(1), rows = row.at("shape").at(0);
                const auto path = fs::path(temp) / "full.safetensors";
                // Real checkpoints store FP32; exercise conversion before NF4 quantization.
                write_weights(path, input, false, true, columns, rows);
                run(path, input, expected, false, true, true, true, false, columns, rows);
                ++count;
                std::cout << row.at("tensor") << " full loader/Linear passed\n";
            }
        }
        fs::remove_all(temp);
        std::cout << count << " actual loader/Linear cases match Python NF4, including mmap, file-reader clones and source preservation\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
