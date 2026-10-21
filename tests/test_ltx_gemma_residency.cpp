#include "model_manager.h"
#include "pipeline/diffusion_engine.h"
#include "core/segment_weight_pipeline.h"
#include "ezlocalai/nf4_storage.hpp"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "json.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <unistd.h>

namespace fs = std::filesystem;
using Json = nlohmann::json;
static void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
static std::string read(const fs::path& path) {
    std::ifstream in(path, std::ios::binary); return {std::istreambuf_iterator<char>(in), {}};
}
static void weights(const fs::path& path) {
    Json header = Json::object();
    size_t offset = 0;
    for (const auto& [name, shape] : std::vector<std::pair<std::string, std::vector<int>>>{
        {"text_encoders.llm.model.layers.0.self_attn.q_proj.weight", {8, 8}},
        {"text_encoders.llm.model.layers.0.input_layernorm.weight", {8}},
        {"text_encoders.llm.model.embed_tokens.weight", {8, 8}},
        {"text_embedding_projection.fixture.weight", {8, 8}}}) {
        size_t count = 1; for (auto n : shape) count *= n;
        header[name] = {{"dtype", "BF16"}, {"shape", shape}, {"data_offsets", {offset, offset + count * 2}}};
        offset += count * 2;
    }
    auto encoded = header.dump(); encoded.resize((encoded.size() + 7) / 8 * 8, ' ');
    const uint64_t length = encoded.size();
    std::ofstream out(path, std::ios::binary); out.write(reinterpret_cast<const char*>(&length), 8); out << encoded;
    for (size_t i = 0; i < offset / 2; ++i) {
        const auto value = ezlocalai::nf4_detail::to_bf16((int(i % 19) - 9) / 32.f);
        out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
    check(bool(out), "write fixture weights");
}

static void run(const fs::path& path, ggml_backend_t cpu, ggml_backend_t gpu, bool quantized, bool mapped, bool disk) {
    const auto original = read(path);
    ModelLoader loader; loader.set_n_threads(2);
    if (quantized) loader.set_ltx_nf4(84, 1536);
    check(loader.init_from_file_and_convert_name(path.string(), "", VERSION_LTXAV), "load fixture metadata");
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    check(bool(context), "parameter metadata context");
    std::map<std::string, ggml_tensor*> tensors;
    std::vector<ggml_tensor*> gemma, all;
    ggml_tensor* projector = nullptr;
    size_t gemma_bytes = 0, projection_bytes = 0;
    for (const auto& [name, source] : loader.get_tensor_storage_map()) {
        auto* tensor = source.ltx_nf4_type != GGML_TYPE_COUNT
            ? ggml_new_tensor_1d(context.get(), GGML_TYPE_I8, ezlocalai::Nf4StorageLayout(source.nelements()).bytes)
            : ggml_new_tensor(context.get(), GGML_TYPE_BF16, source.n_dims, source.ne);
        tensors[name] = tensor; all.push_back(tensor);
        if (name.rfind("text_encoders.llm.", 0) == 0) { gemma.push_back(tensor); gemma_bytes += ggml_nbytes(tensor); }
        else { projector = tensor; projection_bytes += ggml_nbytes(tensor); }
    }
    check(gemma.size() == 3 && all.size() == 4 && projector, "all parameter families are present");
    {
        StableDiffusionGGML engine;
        engine.version = VERSION_LTXAV;
        engine.model_manager = std::make_shared<ModelManager>();
        auto& manager = *engine.model_manager;
        manager.set_n_threads(2); manager.set_enable_mmap(mapped); manager.set_writable_mmap(false);
        check(manager.set_loader(std::move(loader)), "set model source");
        check(manager.register_param_tensors(ModelComponent::Conditioner, tensors,
            disk ? ModelManager::ResidencyMode::Disk : ModelManager::ResidencyMode::ParamBackend,
            gpu, cpu), "register mixed conditioner");
        check(manager.validate_registered_tensors(), "validate packed and normal parameter metadata");
        const auto bytes = manager.registered_params_memory({ModelComponent::Conditioner});
        check(bytes.first == projection_bytes + (quantized ? 0 : gemma_bytes) &&
              bytes.second == (quantized ? gemma_bytes : 0), "registered memory follows individual parameter placement");
        check(manager.load_all_params_eagerly(), "preload parameters");
        for (auto* tensor : gemma) check(ggml_backend_buffer_is_host(tensor->buffer) != quantized,
                                       "whole quantized Gemma stays on compute GPU");
        check(ggml_backend_buffer_is_host(projector->buffer), "projector keeps CPU placement");
        std::vector<void*> resident; for (auto* tensor : gemma) resident.push_back(tensor->data);
        std::vector<std::vector<uint8_t>> values;
        for (auto* tensor : gemma) {
            values.emplace_back(ggml_nbytes(tensor)); ggml_backend_tensor_get(tensor, values.back().data(), 0, values.back().size());
        }
        for (int mode : {1, 2, 0, 2}) {
        check(engine.set_ltx_offload_mode(mode), "change retained execution policy");
        check(manager.segmented_compute_enabled() == (mode != 1) &&
              manager.force_segmented_compute() == (mode == 2) &&
              manager.prefetch_enabled() == (mode != 2), "distinct whole-model, sequential and automatic execution");
        check(!engine.set_ltx_offload_mode(3) && !engine.set_ltx_offload_mode(-1), "invalid policy rejected");
        check(manager.force_segmented_compute() == (mode == 2), "rejected policy preserves current mode");
        check(manager.prepare_params(all), "stage graph parameters");
        manager.release_compute_backend_params(all);
        manager.evict_compute_backend_params(all);
        if (quantized) {
            for (size_t i = 0; i < gemma.size(); ++i) {
                check(gemma[i]->data == resident[i] && !ggml_backend_buffer_is_host(gemma[i]->buffer),
                      "graph eviction preserves Gemma GPU allocation");
                std::vector<uint8_t> after(values[i].size()); ggml_backend_tensor_get(gemma[i], after.data(), 0, after.size());
                check(after == values[i], "graph eviction preserves Gemma weights");
            }
        }
        if (disk) check(projector->data == nullptr, "disk projector is released independently of Gemma");
        else check(ggml_backend_buffer_is_host(projector->buffer), "staged projector returns to CPU parameters");
        auto* graph = ggml_new_graph_custom(context.get(), 16, false);
        ggml_build_forward_expand(graph, ggml_cont(context.get(), projector));
        sd::ggml_graph_cut::Plan plan;
        plan.segments.emplace_back();
        for (int leaf = 0; leaf < sd::ggml_graph_cut::leaf_count(graph); ++leaf) {
            sd::ggml_graph_cut::Segment::InputRef input;
            input.type = sd::ggml_graph_cut::Segment::INPUT_PARAM;
            input.leaf_index = leaf;
            plan.segments.front().input_refs.push_back(input);
        }
        {
            sd::SegmentWeightPipeline pipeline(engine.model_manager, gpu, 1, graph, plan, {projector}, false);
            check(pipeline.segment_start(0, [] { return true; }), "start real projector segment");
            check(manager.inspect_compute_backend_params({projector}).missing_bytes == 0, "segment stages projector on GPU");
            pipeline.segment_end();
            check((manager.inspect_compute_backend_params({projector}).missing_bytes > 0) == (mode == 2),
                  "sequential segment completion evicts staged weights immediately");
        }
        manager.evict_compute_backend_params({projector});
        }
        engine.disable_prefetch = engine.disable_segmented_compute = true;
        check(engine.set_ltx_offload_mode(0) && !manager.prefetch_enabled() && !manager.segmented_compute_enabled(),
              "automatic mode restores original engine flags");
        engine.version = VERSION_SD1;
        check(!engine.set_ltx_offload_mode(2) && engine.set_ltx_offload_mode(0), "LTX override cannot affect other models");
        check(manager.prepare_params({projector}), "projector can be loaded again");
        manager.release_compute_backend_params({projector}); manager.evict_compute_backend_params({projector});
        check(manager.unregister_param_tensors(ModelComponent::Conditioner), "unregister conditioner");
        for (auto* tensor : all) check(tensor->data == nullptr && tensor->buffer == nullptr, "unregister releases every parameter");
        check(manager.registered_params_size({ModelComponent::Conditioner}) == 0, "resource accounting clears after unload");
    }
    check(read(path) == original, "shared model source stays unchanged");
}

int main() {
    try {
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(), ggml_backend_free);
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> gpu(ggml_backend_cuda_init(0), ggml_backend_free);
        check(bool(cpu) && bool(gpu), "CPU and CUDA backend initialization");
        char temp[] = "/tmp/ezlocalai-gemma-residency-XXXXXX"; check(mkdtemp(temp), "temporary directory");
        const auto path = fs::path(temp) / "weights.safetensors"; weights(path);
        for (bool quantized : {false, true}) for (bool mapped : {false, true}) for (bool disk : {false, true})
            run(path, cpu.get(), gpu.get(), quantized, mapped, disk);
        fs::remove_all(temp);
        std::cout << "8 CUDA ModelManager cases preserve complete quantized Gemma residency, independent projector eviction and unload\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
