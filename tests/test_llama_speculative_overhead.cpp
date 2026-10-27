#include "speculative-mtp-state.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

// Compile the actual managed process method and snapshot selector against an
// instrumented backend. This checks which production calls occur, rather than
// a second implementation of either optimization.
namespace {
int checks = 0;
void require(bool ok, const char * message) {
    ++checks;
    if (!ok) throw std::runtime_error(message);
}
#define GGML_ASSERT(value) require(bool(value), "actual engine assertion: " #value)
#define SPC_ERR(...) do {} while (false)

using llama_seq_id = int32_t;
constexpr int32_t LLAMA_TOKEN_NULL = -1;
constexpr int LLAMA_PROCESS_TYPE_DECODE = 1;
constexpr size_t width = 4;
using row = std::array<float, width>;
struct token {
    int32_t id;
    std::array<int32_t, 4> pos;
    llama_seq_id seq_id;
    const float * embd = nullptr;
    const float * state = nullptr;
};
struct common_batch {
    struct state_view { const float * data; size_t rows, width; };
    std::vector<token> tokens;
    bool embedded = false;
    int32_t size() const { return int32_t(tokens.size()); }
    bool has_token() const { return !embedded; }
    bool has_embd() const { return embedded; }
    void clear() { tokens.clear(); }
    int32_t add(int32_t id, int32_t pos, llama_seq_id seq, bool) {
        tokens.push_back({id, {pos, 0, 0, 0}, seq});
        return size() - 1;
    }
    int32_t add_embd(const float * data, const int32_t * pos, llama_seq_id seq, bool) {
        tokens.push_back({LLAMA_TOKEN_NULL, {pos[0], pos[1], pos[2], pos[3]}, seq, data});
        return size() - 1;
    }
    void set_embd_state(int32_t index, state_view value) { tokens.at(index).state = value.data; }
    common_batch * get() { return this; }
};
struct context {
    std::vector<row> hidden;
    std::vector<std::vector<row>> submitted;
    int bulk_reads = 0, row_reads = 0, rollback = 0;
};
float * llama_get_embeddings_nextn(context * ctx) {
    ++ctx->bulk_reads;
    return ctx->hidden.empty() ? nullptr : ctx->hidden[0].data();
}
float * llama_get_embeddings_nextn_ith(context * ctx, size_t index) {
    ++ctx->row_reads;
    return ctx->hidden.at(index).data();
}
context * llama_get_memory(context * ctx) { return ctx; }
void llama_memory_seq_rm(context *, llama_seq_id, int32_t, int32_t) {}
void llama_set_nextn_layer_offset(context *, int) {}
int llama_process(context * ctx, int, common_batch * batch) {
    std::vector<row> rows;
    for (const auto & item : batch->tokens) {
        row copy;
        std::memcpy(copy.data(), item.state, sizeof(copy));
        rows.push_back(copy);
    }
    ctx->submitted.push_back(std::move(rows));
    return 0;
}
struct driver {
    struct { context * ctx_tgt; context * ctx_dft; } params;
    int32_t n_seq, n_embd = int32_t(width), n_mtp_layers = 1;
    bool is_mem_shared = false, chain_heads = false;
    std::vector<int32_t> i_batch_beg, i_batch_end;
    std::vector<common_mtp_state> carry;
    common_batch batch;
    explicit driver(context & target, context & draft, int sequences) :
            params{&target, &draft}, n_seq(sequences), i_batch_beg(sequences), i_batch_end(sequences) {
        for (int i = 0; i < sequences; ++i) carry.emplace_back(width, 9, 4);
    }
#include "llama_mtp_process_under_test.inc"
};

std::vector<uint8_t> saved(const common_mtp_state & state) {
    std::vector<uint8_t> result;
    state.save(result);
    return result;
}
row hidden(size_t index) {
    row result{float(index), -float(index + 1), float(index + 99), float(index + 3)};
    // Retained bytes must preserve signed zero, NaN payload and infinity.
    if (index % 7 == 0) {
        const uint32_t bits[] = {0x80000000, 0x7fc01234, 0xff800000, 0x3f800000};
        std::memcpy(result.data(), bits, sizeof(bits));
    }
    return result;
}
void check_mtp_rows(int sequences, size_t count, bool embedded, bool shared, bool chained) {
    context target, draft;
    common_batch input;
    input.embedded = embedded;
    for (int seq = 0; seq < sequences; ++seq) {
        for (size_t i = 0; i < count; ++i) {
            target.hidden.push_back(hidden(target.hidden.size()));
            input.tokens.push_back({embedded ? LLAMA_TOKEN_NULL : int32_t(i + 10),
                {embedded ? 0 : int32_t(i), 0, 0, 0}, seq, nullptr});
        }
    }
    driver actual(target, draft, sequences);
    actual.is_mem_shared = shared;
    actual.chain_heads = chained;
    actual.n_mtp_layers = chained ? 2 : 1;
    require(actual.process(input), "actual MTP process succeeds");
    require(target.bulk_reads == 1 && target.row_reads == 0,
        "all actual MTP paths synchronize target once without per-row getters");
    require(draft.submitted.size() == size_t(shared ? 0 : (chained ? 2 : 1)),
        "shared KV skips catch-up; all trained heads still run when required");
    for (int seq = 0; seq < sequences; ++seq) {
        common_mtp_state reference(width, 9, 4);
        reference.record(0, count, [&](size_t i) {
            return common_mtp_state::row_view{embedded ? -1 : int32_t(i + 1),
                target.hidden.at(size_t(seq) * count + i).data()};
        });
        require(saved(actual.carry[seq]) == saved(reference),
            "actual MTP CPU snapshots retain exact rows and positional metadata");
        for (const auto & submission : draft.submitted) {
            for (size_t i = 0; i < count; ++i) {
                const row zero{};
                const auto & expected = i == 0 ? zero : target.hidden.at(size_t(seq) * count + i - 1);
                require(std::memcmp(submission.at(size_t(seq) * count + i).data(), expected.data(), sizeof(row)) == 0,
                    "actual draft catch-up receives the same predecessor rows");
            }
        }
    }
}

enum removal { COMMON_CONTEXT_SEQ_RM_TYPE_NO, COMMON_CONTEXT_SEQ_RM_TYPE_PARTIAL,
    COMMON_CONTEXT_SEQ_RM_TYPE_FULL, COMMON_CONTEXT_SEQ_RM_TYPE_RS };
struct sampler { uint32_t rng = 42; };
using common_sampler_ptr = std::unique_ptr<sampler>;
int clone_calls = 0;
sampler * common_sampler_clone(sampler * value) { ++clone_calls; return new sampler(*value); }
int llama_n_rs_seq(context * ctx) { return ctx->rollback; }

void check_sampler_snapshots() {
    for (auto ctx_tgt_seq_rm_type : {COMMON_CONTEXT_SEQ_RM_TYPE_NO, COMMON_CONTEXT_SEQ_RM_TYPE_PARTIAL,
            COMMON_CONTEXT_SEQ_RM_TYPE_FULL, COMMON_CONTEXT_SEQ_RM_TYPE_RS}) {
        for (size_t n_draft = 1; n_draft <= 16; ++n_draft) {
            for (int capacity = 0; capacity <= 17; ++capacity) {
                context target;
                target.rollback = capacity;
                auto * ctx_tgt = &target;
                struct { common_sampler_ptr smpl{new sampler}; } slot;
                clone_calls = 0;
#include "llama_sampler_snapshot_under_test.inc"
                bool replay_possible = false;
                for (size_t accepted = 1; accepted <= n_draft + 1; ++accepted) {
                    const size_t undone = n_draft + 1 - accepted;
                    const bool replay = undone != 0 && (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                        (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && undone > size_t(capacity)));
                    if (replay) {
                        replay_possible = true;
                        require(smpl_save && smpl_save->rng == slot.smpl->rng,
                            "every possible checkpoint replay has exact original sampler state");
                    }
                }
                require(clone_calls == int(replay_possible),
                    "actual snapshot selector clones exactly when a replay is possible");
            }
        }
    }
}
}

int main() {
    try {
        for (bool embedded : {false, true}) for (bool shared : {false, true}) {
            check_mtp_rows(1, 1, embedded, shared, false);
            check_mtp_rows(3, 17, embedded, shared, false);
            check_mtp_rows(2, 4096, embedded, shared, false);
        }
        check_mtp_rows(2, 17, false, false, true);
        context target, draft;
        driver empty(target, draft, 1);
        require(empty.process(common_batch{}) && target.bulk_reads == 0 && target.row_reads == 0,
            "empty input does not touch backend output state");
        check_sampler_snapshots();
        std::cout << checks << " actual speculative-overhead checks passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
