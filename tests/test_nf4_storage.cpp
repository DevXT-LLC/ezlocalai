#include "nf4_ggml.hpp"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#ifdef EZLOCALAI_NF4_CUDA
#include "ggml-cuda.h"
#endif
#include <fstream>
#include <filesystem>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>

using Json = nlohmann::json;
static void check(bool value, const std::string &message) { if (!value) throw std::runtime_error(message); }
static std::vector<uint8_t> hex(const std::string &value) {
    std::vector<uint8_t> result;
    for (size_t i = 0; i < value.size(); i += 2) result.push_back(std::stoul(value.substr(i, 2), nullptr, 16));
    return result;
}
template<class F> static void rejected(F f) {
    bool failed = false; try { f(); } catch (const std::invalid_argument &) { failed = true; }
    check(failed, "invalid NF4 input accepted");
}
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using Allocator = std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>;

static std::vector<uint8_t> read_bytes(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("missing source weight fixture: " + path.string());
    return {std::istreambuf_iterator<char>(input), {}};
}
static void execute(ggml_backend_t backend, const Json &row, bool half, const std::filesystem::path &directory = {}) {
    ezlocalai::Nf4Weights weights;
    std::vector<uint8_t> expected;
    weights.offset = row.at("offset");
    if (directory.empty()) {
        weights.count = row.at("size"); weights.packed = hex(row.at("quantized_hex"));
        weights.nested = row.at("absmax").get<std::vector<uint8_t>>();
        weights.nested_absmax = row.at("nested_absmax").get<std::vector<float>>();
        expected = hex(row.at(half ? "dequantized_f16_hex" : "dequantized_bf16_hex"));
    } else {
        weights.count = row.at("shape").at(0).get<size_t>() * row.at("shape").at(1).get<size_t>();
        auto file = [&](const char *kind) {
            auto data = read_bytes(directory / row.at("files").at(kind).at("path").get<std::string>());
            check(data.size() == row.at("files").at(kind).at("bytes"), "source weight fixture length"); return data;
        };
        weights.packed = file("packed"); weights.nested = file("nested"); expected = file("dequantized");
        const auto maxima = file("nested_absmax"); check(maxima.size() % sizeof(float) == 0, "source scales");
        weights.nested_absmax.resize(maxima.size() / sizeof(float));
        std::memcpy(weights.nested_absmax.data(), maxima.data(), maxima.size());
    }
    const auto packed = ezlocalai::pack_nf4_storage(weights);
    check(expected.size() == weights.count * 2, "source fixture size");
    check(packed.size() == ezlocalai::Nf4StorageLayout(weights.count).bytes, "packed layout size");
    if (weights.count >= 1024) check(packed.size() < expected.size() * .27, "retain double-quantized storage density");
    const bool matmul = weights.count >= 64 && weights.count % 64 == 0;
    const int64_t columns = !directory.empty() ? row.at("shape").at(1).get<int64_t>() : matmul ? 64 : weights.count;
    const int64_t rows = weights.count / columns;
    const auto type = half ? GGML_TYPE_F16 : GGML_TYPE_BF16;
    Context params(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    Context computation(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    check(bool(params) && bool(computation), "GGML contexts");
    const size_t skew = weights.count % 2;
    auto *storage = ggml_new_tensor_1d(params.get(), GGML_TYPE_I8, packed.size() + skew);
    if (skew) storage = ggml_view_1d(params.get(), storage, packed.size(), skew);
    Buffer parameters(ggml_backend_alloc_ctx_tensors(params.get(), backend), ggml_backend_buffer_free);
    check(bool(parameters), "packed parameter buffer");
    check(ggml_backend_buffer_get_size(parameters.get()) <= packed.size() + 512, "packed resident allocation");
    ggml_backend_tensor_set(storage, packed.data(), 0, packed.size());
    auto *expanded = ezlocalai::nf4_ggml_expand(computation.get(), storage, columns, rows, type);
    check(ggml_backend_supports_op(backend, expanded), "backend must execute NF4 expansion without fallback");
    auto *graph = ggml_new_graph(computation.get());
    ggml_tensor *product = nullptr, *reference_product = nullptr, *reference = nullptr, *input = nullptr;
    if (matmul) {
        reference = ggml_new_tensor_2d(computation.get(), type, columns, rows);
        input = ggml_new_tensor_2d(computation.get(), GGML_TYPE_F32, columns, 3);
        ggml_set_input(reference); ggml_set_input(input);
        product = ggml_mul_mat(computation.get(), expanded, input);
        reference_product = ggml_mul_mat(computation.get(), reference, input);
        ggml_set_output(product); ggml_set_output(reference_product);
        ggml_build_forward_expand(graph, product); ggml_build_forward_expand(graph, reference_product);
    } else ggml_build_forward_expand(graph, expanded);
    ggml_set_output(expanded);
    Allocator allocation(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free);
    check(ggml_gallocr_alloc_graph(allocation.get(), graph), "allocate expansion graph");
    if (matmul) {
        check(product->data != reference_product->data, "matrix comparison needs independent output buffers");
        ggml_backend_tensor_set(reference, expected.data(), 0, expected.size());
        std::vector<float> values(columns * 3);
        for (size_t i = 0; i < values.size(); ++i) values[i] = (int(i % 37) - 18) / 32.f;
        ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
    }
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "execute packed weight graph");
    std::vector<uint8_t> output(expected.size()); ggml_backend_tensor_get(expanded, output.data(), 0, output.size());
    check(output == expected, "expanded bytes differ from installed Python NF4");
    std::vector<uint8_t> preserved(packed.size()); ggml_backend_tensor_get(storage, preserved.data(), 0, preserved.size());
    check(preserved == packed, "inference mutated compressed weights");
    if (matmul) {
        std::vector<float> actual(ggml_nelements(product)), reference_output(actual.size());
        ggml_backend_tensor_get(product, actual.data(), 0, actual.size() * sizeof(float));
        ggml_backend_tensor_get(reference_product, reference_output.data(), 0, reference_output.size() * sizeof(float));
        check(actual == reference_output, "packed and source-expanded matrix products differ");
    }
    rejected([&] { ezlocalai::nf4_ggml_expand(computation.get(), storage, 0, 1, type); });
    rejected([&] { ezlocalai::nf4_ggml_expand(computation.get(), storage, INT64_MAX, 2, type); });
    rejected([&] { ezlocalai::nf4_ggml_expand(computation.get(), storage, columns, rows, GGML_TYPE_F32); });
    rejected([&] { ezlocalai::nf4_ggml_expand(computation.get(), input, columns, rows, type); });
    ezlocalai::Nf4CustomParams wrong{};
    std::memcpy(&wrong, expanded->op_params, sizeof(wrong)); wrong.userdata = nullptr;
    std::memcpy(expanded->op_params, &wrong, sizeof(wrong));
    check(!ezlocalai::is_nf4_ggml(expanded), "unrelated custom op accepted as NF4");
    if (std::string(ggml_backend_name(backend)).find("CUDA") == 0)
        check(!ggml_backend_supports_op(backend, expanded), "CUDA must reject unrelated custom ops");
    weights.nested.pop_back(); rejected([&] { ezlocalai::pack_nf4_storage(weights); });
}

int main(int argc, char **argv) {
    try {
        check(argc >= 4, "BF16/FP16 and all-code fixtures required; optional --cuda and --weights manifest");
        bool cuda = false; std::filesystem::path weights;
        for (int i = 4; i < argc; ++i) {
            if (std::string(argv[i]) == "--cuda") cuda = true;
            else if (std::string(argv[i]) == "--weights" && i + 1 < argc) weights = argv[++i];
            else throw std::invalid_argument("invalid argument");
        }
        ggml_backend_t raw = nullptr;
        if (cuda) {
#ifdef EZLOCALAI_NF4_CUDA
            raw = ggml_backend_cuda_init(0);
#else
            throw std::runtime_error("CUDA test backend not linked");
#endif
        } else { raw = ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(raw, 4); }
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(raw, ggml_backend_free);
        check(bool(backend), "backend initialization");
        size_t count = 0;
        for (int format = 0; format < 2; ++format) {
            std::ifstream input(argv[format + 1]); Json fixture; input >> fixture;
            for (const auto &row : fixture.at("cases")) if (row.at("nested").get<bool>()) {
                execute(backend.get(), row, format == 1); ++count;
            }
        }
        check(count == 136, "all nested source cases required");
        { std::ifstream input(argv[3]); Json fixture; input >> fixture;
          check(fixture.at("cases").size() == 8, "all codebook cases required");
          for (const auto &row : fixture.at("cases")) { execute(backend.get(), row, row.at("dtype") == "torch.float16"); ++count; } }
        if (!weights.empty()) {
            std::ifstream input(weights); Json fixture; input >> fixture;
            check(fixture.at("cases").size() == 2, "both full Gemma projections required");
            for (const auto &row : fixture.at("cases")) {
                execute(backend.get(), row, false, weights.parent_path()); ++count;
                std::cout << row.at("tensor") << " compressed inference passed\n";
            }
        }
        rejected([] { ezlocalai::Nf4StorageLayout(0); });
        rejected([] { ezlocalai::Nf4StorageLayout(SIZE_MAX); });
        std::cout << count << " source NF4 cases match compressed storage, exact expansion and matrix products on "
                  << ggml_backend_name(backend.get()) << '\n';
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
