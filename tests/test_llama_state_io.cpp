#include "llama-state-io.h"
#include "ggml-backend-impl.h"
#include "llama.h"

#include <array>
#include <cstring>
#include <functional>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <openssl/sha.h>
#include <sstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

// Deferred copies deliberately leave destination bytes untouched until the
// backend is synchronized. This exercises host-buffer lifetimes/order without
// requiring a CUDA device or making a GPU hardware-validation claim.
struct fake_backend {
    std::string registry_name;
    bool host;
    ggml_backend_reg registry{};
    ggml_backend_device device{};
    ggml_backend_buffer_type type{}, host_type{};
    ggml_backend backend{};
    int queued = 0, synced = 0, blocking = 0;
    int host_attempts = 0, host_frees = 0;
    size_t pinned_bytes = 0, peak_pinned_bytes = 0;
    bool fail_host = false;
    ggml_backend_buffer_type_t pageable_fallback = nullptr;
    struct host_buffer {
        fake_backend * owner;
        std::vector<uint8_t> data;
        bool pinned;
    };
    std::vector<std::function<void()>> pending;

    static fake_backend & state(ggml_backend_t b) { return *static_cast<fake_backend *>(b->context); }
    static fake_backend & state(ggml_backend_dev_t d) { return *static_cast<fake_backend *>(d->context); }
    explicit fake_backend(const char * name, bool is_host = false) : registry_name(name), host(is_host) {
        registry.api_version = GGML_BACKEND_API_VERSION;
        registry.context = this;
        registry.iface.get_name = [](ggml_backend_reg_t r) { return static_cast<fake_backend *>(r->context)->registry_name.c_str(); };
        device.reg = &registry; device.context = this;
        device.iface.get_name = [](ggml_backend_dev_t d) { return state(d).registry_name.c_str(); };
        device.iface.get_type = [](ggml_backend_dev_t d) { return state(d).host ? GGML_BACKEND_DEVICE_TYPE_CPU : GGML_BACKEND_DEVICE_TYPE_GPU; };
        device.iface.get_buffer_type = [](ggml_backend_dev_t d) { return &state(d).type; };
        device.iface.get_host_buffer_type = [](ggml_backend_dev_t d) { return &state(d).host_type; };
        device.iface.supports_buft = [](ggml_backend_dev_t d, ggml_backend_buffer_type_t t) { return t == &state(d).type; };
        type.device = &device; type.context = this;
        type.iface.get_name = [](ggml_backend_buffer_type_t t) { return static_cast<fake_backend *>(t->context)->registry_name.c_str(); };
        type.iface.get_alignment = [](ggml_backend_buffer_type_t) -> size_t { return 32; };
        type.iface.is_host = [](ggml_backend_buffer_type_t t) { return static_cast<fake_backend *>(t->context)->host; };
        host_type.device = &device; host_type.context = this;
        host_type.iface.get_name = [](ggml_backend_buffer_type_t) { return "CUDA_Host_Test"; };
        host_type.iface.get_alignment = [](ggml_backend_buffer_type_t) -> size_t { return 32; };
        host_type.iface.is_host = [](ggml_backend_buffer_type_t) { return true; };
        host_type.iface.alloc_buffer = [](ggml_backend_buffer_type_t t, size_t size) -> ggml_backend_buffer_t {
            auto & owner = *static_cast<fake_backend *>(t->context);
            ++owner.host_attempts;
            if (owner.fail_host) return nullptr;
            auto * storage = new host_buffer{&owner, std::vector<uint8_t>(size), owner.pageable_fallback == nullptr};
            if (storage->pinned) {
                owner.pinned_bytes += size;
                owner.peak_pinned_bytes = std::max(owner.peak_pinned_bytes, owner.pinned_bytes);
            }
            ggml_backend_buffer_i iface{};
            iface.get_base = [](ggml_backend_buffer_t b) -> void * { return static_cast<host_buffer *>(b->context)->data.data(); };
            iface.free_buffer = [](ggml_backend_buffer_t b) {
                auto * storage = static_cast<host_buffer *>(b->context);
                ++storage->owner->host_frees;
                if (storage->pinned) storage->owner->pinned_bytes -= storage->data.size();
                delete storage;
            };
            return ggml_backend_buffer_init(owner.pageable_fallback ? owner.pageable_fallback : t, iface, storage, size);
        };
        backend.device = &device; backend.context = this;
        backend.iface.get_name = [](ggml_backend_t b) { return state(b).registry_name.c_str(); };
        backend.iface.get_tensor_async = [](ggml_backend_t b, const ggml_tensor * t, void * dst, size_t offset, size_t size) {
            auto & s = state(b); ++s.queued;
            s.pending.push_back([=] { std::memcpy(dst, static_cast<const uint8_t *>(t->data) + offset, size); });
        };
        backend.iface.set_tensor_async = [](ggml_backend_t b, ggml_tensor * t, const void * src, size_t offset, size_t size) {
            auto & s = state(b); ++s.queued;
            s.pending.push_back([=] { std::memcpy(static_cast<uint8_t *>(t->data) + offset, src, size); });
        };
        backend.iface.synchronize = [](ggml_backend_t b) {
            auto & s = state(b); ++s.synced;
            for (auto & copy : s.pending) copy();
            s.pending.clear();
        };
    }
    void reset() { check(pending.empty(), "no deferred copies left behind"); queued = synced = blocking = 0; }
};

struct fake_buffer {
    fake_backend & owner;
    std::array<uint8_t, 128> bytes{};
    ggml_backend_buffer buffer{};
    explicit fake_buffer(fake_backend & backend, ggml_backend_buffer_type_t override_type = nullptr) : owner(backend) {
        buffer.buft = override_type ? override_type : &owner.type;
        buffer.context = this; buffer.size = bytes.size();
        buffer.iface.get_tensor = [](ggml_backend_buffer_t b, const ggml_tensor * t, void * dst, size_t offset, size_t size) {
            ++static_cast<fake_buffer *>(b->context)->owner.blocking;
            std::memcpy(dst, static_cast<const uint8_t *>(t->data) + offset, size);
        };
        buffer.iface.set_tensor = [](ggml_backend_buffer_t b, ggml_tensor * t, const void * src, size_t offset, size_t size) {
            ++static_cast<fake_buffer *>(b->context)->owner.blocking;
            std::memcpy(static_cast<uint8_t *>(t->data) + offset, src, size);
        };
        for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = uint8_t(3 * i + 7);
    }
    void assign(ggml_tensor * t) { t->buffer = &buffer; t->data = bytes.data(); }
};

static void transport_contract() {
    unsetenv("EZLOCALAI_LLAMA_STATE_IO_BATCHING");
    fake_backend first("CUDA"), second("CUDA"), other("Metal"), cpu("CPU", true);
    ggml_backend_t backends[] = {&first.backend, &second.backend, &other.backend, &cpu.backend};
    auto * scheduler = ggml_backend_sched_new(backends, nullptr, 4, 32, false, false);
    struct scheduler_guard { ggml_backend_sched_t p; ~scheduler_guard() { ggml_backend_sched_free(p); } } cleanup{scheduler};
    llama_state_io_pool pool(64);
    auto * context = ggml_init({8192, nullptr, true}); check(context, "tensor context");
    struct context_guard { ggml_context * p; ~context_guard() { ggml_free(p); } } tensors{context};
    auto * a = ggml_new_tensor_1d(context, GGML_TYPE_I8, 64);
    auto * b = ggml_new_tensor_1d(context, GGML_TYPE_I8, 64);
    fake_buffer a_buffer(first), b_buffer(second); a_buffer.assign(a); b_buffer.assign(b);
    auto * view = ggml_view_1d(context, a, 32, 8); view->buffer = nullptr;
    std::array<uint8_t, 160> expected{}, actual{};
    expected.fill(0x55); actual = expected;
    ggml_backend_tensor_get(a, expected.data() + 9, 7, 15);
    ggml_backend_tensor_get(view, expected.data() + 43, 13, 11);
    ggml_backend_tensor_get(b, expected.data() + 91, 4, 16);
    first.reset(); second.reset();
    {
        llama_state_io_batch io(scheduler, &pool);
        io.get(a, actual.data() + 9, 7, 15);
        io.get(view, actual.data() + 43, 13, 11);
        io.get(b, actual.data() + 91, 4, 16);
        check(actual != expected, "transfers remain pending until the synchronization boundary");
        check(first.queued == 2 && second.queued == 1 && first.synced == 0, "queue transfers without per-tensor fences");
        io.finish(); io.finish();
    }
    check(actual == expected, "batched snapshot bytes match synchronous snapshot including metadata gaps");
    check(first.synced == 1 && second.synced == 1, "one synchronization for each used CUDA backend");
    check(other.synced == 0 && cpu.synced == 0, "unused backends are not synchronized");

    // Ordered overlapping writes exercise restore semantics on one stream.
    const std::array<uint8_t, 16> earlier{{1,2,3,4}}, later{{8,7,6,5}};
    auto restored = a_buffer.bytes;
    std::memcpy(restored.data() + 8, earlier.data(), earlier.size());
    std::memcpy(restored.data() + 12, later.data(), later.size());
    first.reset();
    {
        llama_state_io_batch io(scheduler, &pool);
        io.set(a, earlier.data(), 8, earlier.size());
        io.set(a, later.data(), 12, later.size());
        check(a_buffer.bytes != restored, "restore source remains live until completion");
    }
    check(a_buffer.bytes == restored && first.synced == 1, "restore order and destructor completion");

    // The caller's input can be changed as soon as set() has copied it into
    // stable staging, even while the backend still defers the device write.
    std::array<uint8_t, 8> transient{{21,22,23,24,25,26,27,28}};
    const auto captured = transient;
    {
        llama_state_io_batch io(scheduler, &pool);
        io.set(a, transient.data(), 0, transient.size());
        transient.fill(0);
    }
    check(std::memcmp(a_buffer.bytes.data(), captured.data(), captured.size()) == 0,
          "queued uploads retain captured input bytes until completion");

    // A synchronous host/special buffer can alias a prior CUDA allocation.
    // Its copy must not overtake a pending transfer on the CUDA stream.
    auto * alias = ggml_new_tensor_1d(context, GGML_TYPE_I8, 64);
    fake_buffer alias_buffer(cpu); alias_buffer.assign(alias); alias->data = a_buffer.bytes.data();
    restored = a_buffer.bytes;
    std::memcpy(restored.data() + 8, later.data(), later.size());
    std::memcpy(restored.data() + 12, earlier.data(), earlier.size());
    first.reset(); cpu.reset();
    {
        llama_state_io_batch io(scheduler, &pool);
        io.set(a, later.data(), 8, later.size());
        io.set(alias, earlier.data(), 12, earlier.size());
    }
    check(a_buffer.bytes == restored && first.synced == 1 && cpu.blocking == 1,
          "synchronous fallback cannot overtake a pending aliasing CUDA write");

    // CPU, non-CUDA, split/unknown buffers and the opt-out retain direct access.
    auto * fallback = ggml_new_tensor_1d(context, GGML_TYPE_I8, 64);
    ggml_backend_buffer_type split_type = first.type;
    fake_backend absent("CUDA");
    for (auto * backend : {&cpu, &other, &absent}) {
        fake_buffer data(*backend); data.assign(fallback); backend->reset();
        std::array<uint8_t, 8> output{};
        { llama_state_io_batch io(scheduler, &pool); io.get(fallback, output.data(), 3, output.size()); }
        check(std::memcmp(output.data(), data.bytes.data() + 3, output.size()) == 0 && backend->blocking == 1 && backend->queued == 0,
              "unsupported backend retains synchronous data access");
    }
    fake_buffer split(first, &split_type); split.assign(fallback); first.reset();
    { llama_state_io_batch io(scheduler, &pool); io.get(fallback, actual.data(), 0, 8); }
    check(first.blocking == 1 && first.queued == 0, "split buffer does not reach ordinary CUDA async accessor");
    a_buffer.assign(a); first.reset(); setenv("EZLOCALAI_LLAMA_STATE_IO_BATCHING", "0", 1);
    { llama_state_io_batch io(scheduler, &pool); io.get(a, actual.data(), 0, 8); }
    check(first.blocking == 1 && first.queued == 0, "diagnostic opt-out preserves synchronous behavior");
    unsetenv("EZLOCALAI_LLAMA_STATE_IO_BATCHING"); first.reset();
    try {
        llama_state_io_batch io(scheduler, &pool); io.get(a, actual.data(), 0, 8);
        throw std::runtime_error("test unwinding");
    } catch (const std::runtime_error &) {}
    check(first.pending.empty() && first.synced == 1, "unwinding finishes queued transfers");
    { llama_state_io_batch io(nullptr); io.get(a, actual.data(), 0, 8); io.get(fallback, nullptr, 0, 0); }

    check(first.host_attempts == 1 && second.host_attempts == 1, "snapshots reuse the same bounded workspace");
    pool.clear();
    check(first.pinned_bytes == 0 && second.pinned_bytes == 0, "pool releases pinned working memory");
    check(first.peak_pinned_bytes <= 64 && second.peak_pinned_bytes <= 64, "pinned capacity is bounded per backend");

    // Force multi-window copies larger than the workspace. Every deferred
    // source must remain stable until its corresponding fence.
    llama_state_io_pool chunks(16);
    const auto original = a_buffer.bytes;
    std::array<uint8_t, 64> chunked{};
    first.reset();
    {
        llama_state_io_batch io(scheduler, &chunks);
        io.get(a, chunked.data(), 0, chunked.size());
    }
    check(std::memcmp(chunked.data(), original.data(), chunked.size()) == 0 && first.synced == 4,
          "downloads larger than the pinned window preserve all bytes");
    for (size_t i = 0; i < chunked.size(); ++i) chunked[i] ^= uint8_t(0x5a);
    first.reset();
    {
        llama_state_io_batch io(scheduler, &chunks);
        io.set(a, chunked.data(), 0, chunked.size());
    }
    check(std::memcmp(chunked.data(), a_buffer.bytes.data(), chunked.size()) == 0 && first.synced == 4,
          "uploads do not reuse staging before queued copies finish");

    // Direction changes complete downloads before reading their host output.
    std::array<uint8_t, 8> turn{};
    first.reset();
    {
        llama_state_io_batch io(scheduler, &chunks);
        io.get(a, turn.data(), 0, turn.size());
        io.set(b, turn.data(), 0, turn.size());
    }
    check(std::memcmp(a_buffer.bytes.data(), b_buffer.bytes.data(), turn.size()) == 0,
          "download to upload direction change makes host bytes available");
    chunks.clear();
    check(first.pinned_bytes == 0 && second.pinned_bytes == 0, "chunk staging is released");

    // Applying readbacks by backend order would corrupt overlapping host
    // destinations when the serialization order crosses device streams.
    std::array<uint8_t, 24> ordered{}, ordered_expected{};
    ggml_backend_tensor_get(b, ordered_expected.data(), 3, 16);
    ggml_backend_tensor_get(a, ordered_expected.data() + 4, 0, 16);
    {
        llama_state_io_batch io(scheduler, &chunks);
        io.get(b, ordered.data(), 3, 16);
        io.get(a, ordered.data() + 4, 0, 16);
    }
    check(ordered == ordered_expected, "cross-device readbacks preserve global serialization order");

    // A download into imported host memory can alter a later device source.
    const std::array<uint8_t, 8> sequence{{91,92,93,94,95,96,97,98}};
    std::memcpy(a_buffer.bytes.data(), sequence.data(), sequence.size());
    std::memset(b_buffer.bytes.data(), 0, sequence.size());
    {
        llama_state_io_batch io(scheduler, &chunks);
        io.get(a, b_buffer.bytes.data(), 0, sequence.size());
        io.get(b, turn.data(), 0, turn.size());
    }
    check(turn == sequence, "download read-after-host-write aliases are ordered");

    // Uploads must finish a pending device write before copying an aliased
    // host source into staging, including between separate CUDA backends.
    const std::array<uint8_t, 8> replacement{{81,82,83,84,85,86,87,88}};
    {
        llama_state_io_batch io(scheduler, &chunks);
        io.set(a, replacement.data(), 0, replacement.size());
        io.set(b, a_buffer.bytes.data(), 0, replacement.size());
    }
    check(std::memcmp(b_buffer.bytes.data(), replacement.data(), replacement.size()) == 0,
          "upload sources observe prior aliased device writes");
    auto * peer_alias = ggml_new_tensor_1d(context, GGML_TYPE_I8, 64);
    b_buffer.assign(peer_alias); peer_alias->data = a_buffer.bytes.data();
    {
        llama_state_io_batch io(scheduler, &chunks);
        io.set(peer_alias, sequence.data(), 0, sequence.size());
        io.set(a, replacement.data(), 0, replacement.size());
    }
    check(std::memcmp(a_buffer.bytes.data(), replacement.data(), replacement.size()) == 0,
          "cross-device writes to aliased storage preserve order");
    chunks.clear();
    check(first.pinned_bytes == 0 && second.pinned_bytes == 0, "alias test staging is released");

    // A bounded metadata list forces completion before another copy can
    // overrun its records, even when a window is large enough for more data.
    llama_state_io_pool metadata(1024);
    std::array<uint8_t, 300> many{}, many_expected{};
    for (size_t i = 0; i < many.size(); ++i) many_expected[i] = a_buffer.bytes[i % 64];
    first.reset();
    {
        llama_state_io_batch io(scheduler, &metadata);
        for (size_t i = 0; i < many.size(); ++i) io.get(a, many.data() + i, i % 64, 1);
    }
    check(many == many_expected && first.synced == 2, "copy-record metadata remains bounded");
    metadata.clear();

    // A pin-allocation failure, including CUDA's pageable fallback, must use
    // the original synchronous access and avoid retrying for every tensor.
    first.fail_host = true; first.reset();
    const auto attempts = first.host_attempts;
    {
        llama_state_io_batch io(scheduler, &pool);
        io.get(a, turn.data(), 0, turn.size());
        io.get(a, turn.data(), 0, turn.size());
    }
    check(first.queued == 0 && first.blocking == 2 && first.host_attempts == attempts + 1,
          "failed pin allocation falls back once and stays synchronous");
    check(std::memcmp(turn.data(), a_buffer.bytes.data(), turn.size()) == 0, "allocation fallback retains snapshot bytes");
    pool.clear(); first.fail_host = false; first.pageable_fallback = &cpu.type; first.reset();
    {
        llama_state_io_batch io(scheduler, &pool); io.get(a, turn.data(), 0, turn.size());
    }
    check(first.queued == 0 && first.blocking == 1 && first.pinned_bytes == 0,
          "pageable allocation masquerading as a host-buffer fallback is rejected");
    pool.clear(); first.pageable_fallback = nullptr;
    llama_state_io_pool bounded(SIZE_MAX);
    check(bounded.capacity() == 64u * 1024u * 1024u, "workspace requests cannot exceed the fixed 64 MiB cap");
    llama_state_io_pool disabled(0); first.reset();
    { llama_state_io_batch io(scheduler, &disabled); io.get(a, turn.data(), 0, turn.size()); }
    check(first.queued == 0 && first.blocking == 1, "zero-capacity staging retains synchronous IO");
}

static std::vector<uint8_t> snapshot(llama_context * context, const char * mode) {
    setenv("EZLOCALAI_LLAMA_STATE_IO_BATCHING", mode, 1);
    std::vector<uint8_t> bytes(llama_state_seq_get_size_ext(context, 0, 0));
    check(!bytes.empty(), "nonempty sequence state");
    check(llama_state_seq_get_data_ext(context, bytes.data(), bytes.size(), 0, 0) == bytes.size(), "save sequence state");
    return bytes;
}

static void model_contract(const char * path) {
    check(std::filesystem::file_size(path) == 1185376, "pinned tiny CPU model size");
    std::ifstream file(path, std::ios::binary);
    const std::string bytes{std::istreambuf_iterator<char>(file), {}};
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char *>(bytes.data()), bytes.size(), digest);
    std::ostringstream hash; for (auto byte : digest) hash << std::hex << std::setfill('0') << std::setw(2) << unsigned(byte);
    check(hash.str() == "270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d", "pinned CPU model checksum");
    ggml_backend_load_all(); llama_backend_init();
    auto model_params = llama_model_default_params(); model_params.n_gpu_layers = 0;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(path, model_params), llama_model_free);
    check(bool(model), "load CPU smoke model");
    auto params = llama_context_default_params(); params.n_ctx = 128; params.n_batch = 64; params.n_ubatch = 64;
    params.n_threads = 2; params.n_threads_batch = 2;
    std::unique_ptr<llama_context, decltype(&llama_free)> context(llama_init_from_model(model.get(), params), llama_free);
    check(bool(context), "CPU smoke context");
    const auto * vocabulary = llama_model_get_vocab(model.get());
    const std::string text = "Once upon a time there was a small village.";
    std::vector<llama_token> tokens(text.size() + 16);
    const int count = llama_tokenize(vocabulary, text.data(), text.size(), tokens.data(), tokens.size(), true, true);
    check(count > 0, "smoke prompt tokenization"); tokens.resize(count);
    check(llama_decode(context.get(), llama_batch_get_one(tokens.data(), count)) == 0, "smoke prompt decode");
    const auto sync = snapshot(context.get(), "0"), batched = snapshot(context.get(), "1");
    check(sync == batched, "snapshot serialization is byte-identical across modes");
    const auto restore_and_decode = [&](const char * mode) {
        setenv("EZLOCALAI_LLAMA_STATE_IO_BATCHING", mode, 1);
        check(llama_state_seq_set_data_ext(context.get(), batched.data(), batched.size(), 0, 0) == batched.size(), "restore sequence state");
        auto token = tokens.back();
        check(llama_decode(context.get(), llama_batch_get_one(&token, 1)) == 0, "decode after restoring snapshot");
        auto * logits = llama_get_logits(context.get());
        return std::vector<float>(logits, logits + llama_vocab_n_tokens(vocabulary));
    };
    check(restore_and_decode("0") == restore_and_decode("1"), "CPU logits after restore are unchanged across modes");
    const auto expected = restore_and_decode("0");
    // A parse failure must discard deferred tensor writes. A subsequent valid
    // restore must remain usable, with no borrowed pointers left in flight.
    auto truncated = batched; truncated.pop_back();
    setenv("EZLOCALAI_LLAMA_STATE_IO_BATCHING", "1", 1);
    check(llama_state_seq_set_data_ext(context.get(), truncated.data(), truncated.size(), 0, 0) == 0,
          "truncated state restore reports failure");
    check(restore_and_decode("1") == expected, "valid restore recovers after a partial parse exception");
    check(snapshot(context.get(), "0") == snapshot(context.get(), "1"), "continued sequence snapshots remain byte-identical");
    context.reset(); model.reset(); llama_backend_free();
    unsetenv("EZLOCALAI_LLAMA_STATE_IO_BATCHING");
}

int main(int argc, char ** argv) {
    try {
        transport_contract();
        if (argc == 3 && std::string(argv[1]) == "--model") model_contract(argv[2]);
        else check(argc == 1, "usage: test_llama_state_io [--model MODEL]");
        std::cout << "State IO queue order, completion, fallbacks and byte-equivalence checks passed\n";
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
