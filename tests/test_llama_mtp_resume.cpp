#include "speculative-mtp-state.h"

#include <array>
#include <cstdlib>
#include <new>
#include <iostream>
#include <stdexcept>
#include <string>

// Fail one actual parser allocation; reset immediately so diagnostics and
// the allocation-free invalidation path can still run normally.
static bool fail_next_allocation = false;
void * operator new(std::size_t size) {
    if (fail_next_allocation) { fail_next_allocation = false; throw std::bad_alloc(); }
    if (void * value = std::malloc(size ? size : 1)) return value;
    throw std::bad_alloc();
}
void operator delete(void * value) noexcept { std::free(value); }
void operator delete(void * value, std::size_t) noexcept { std::free(value); }
void * operator new(std::size_t size, const std::nothrow_t &) noexcept {
    try { return ::operator new(size); } catch (...) { return nullptr; }
}
void operator delete(void * value, const std::nothrow_t &) noexcept { std::free(value); }

namespace {
int checks = 0;
void require(bool value, const char * what) {
    ++checks;
    if (!value) throw std::runtime_error(what);
}
using values = std::array<float, 4>;
values hidden(int conversation, int position) {
    return {float(conversation), float(position), float(conversation * 1000 + position), -float(position)};
}
bool equal(const float * actual, const values & expected) {
    return actual && std::memcmp(actual, expected.data(), sizeof(expected)) == 0;
}
void process(common_mtp_state & state, int conversation, int first, int count) {
    std::vector<values> rows;
    for (int i = 0; i < count; ++i) rows.push_back(hidden(conversation, first + i));
    require(state.prepare(first, false) != nullptr, "production predecessor must exist before draft mutation");
    state.record(first, count, [&](size_t i) { return common_mtp_state::row_view {first + int(i) + 1, rows[i].data()}; });
}
std::vector<uint8_t> snapshot(const common_mtp_state & state) {
    std::vector<uint8_t> bytes;
    state.save(bytes);
    return bytes;
}

void conversation_restore_and_replay() {
    common_mtp_state state(4, 9, 4);
    process(state, 1, 0, 10);
    const auto saved_a = snapshot(state);
    state.reset(); process(state, 2, 0, 10);
    require(equal(state.predecessor(10, false), hidden(2,9)), "B must have a distinct real carryover");
    require(state.restore(saved_a), "restore A CPU payload");
    require(state.rewind(10) && equal(state.predecessor(10,false),hidden(1,9)), "exact append must use A rather than B");
    require(state.restore(saved_a) && state.rewind(9), "identical prompt can replay its final token");
    require(equal(state.predecessor(9,false),hidden(1,8)), "final-token replay needs penultimate hidden row");
    process(state, 1, 9, 1);
    require(equal(state.predecessor(10,false),hidden(1,9)), "single-token replay restores exact append boundary");
    require(state.rewind(9) && equal(state.predecessor(9,false),hidden(1,8)), "single-token process must retain previous predecessor");
    require(!state.rewind(1), "old arbitrary prefix outside bounded history must refuse reuse");
    require(state.rewind(0), "full replay fallback explicitly resets CPU state");
    const values zero{};
    require(equal(state.predecessor(0,false),zero), "fresh position zero cannot leak the previous conversation");
}

void accepted_and_rejected_rows() {
    common_mtp_state state(4, 9, 4);
    process(state, 3, 0, 3);
    const auto checkpoint = snapshot(state);
    process(state, 3, 3, 4);
    const auto verified = snapshot(state);
    require(state.accept(1), "accept one draft after sampled token");
    require(equal(state.predecessor(5,false),hidden(3,4)), "accept n selects actual verification row n, not n-1");
    require(state.find(6)==nullptr && state.find(7)==nullptr, "rejected target rows are not resumable");
    require(state.restore(verified) && state.accept(1), "mid-round bounded verification scratch roundtrips");
    require(equal(state.predecessor(5,false),hidden(3,4)), "restored accept preserves selected row");
    require(state.restore(checkpoint) && state.rewind(3), "speculative checkpoint restores CPU alongside GPU state");
    require(equal(state.predecessor(3,false),hidden(3,2)), "checkpoint replay excludes all unaccepted verification rows");
    process(state, 3, 3, 2);
    require(state.accept(1), "replayed accepted prefix processes normally");

    // MTP can be an 'other' implementation when a longer enabled drafter won.
    common_mtp_state multiple(4, 20, 17);
    process(multiple, 4, 0, 1);
    process(multiple, 4, 1, 17);
    require(multiple.accept(8) && equal(multiple.predecessor(10,false),hidden(4,9)),
        "verification retention uses global enabled draft maximum");
}

void bounded_prefill_copy() {
    constexpr size_t width = 4, count = 4096, history = 9, first = 4;
    common_mtp_state state(width,history,first);
    size_t reads = 0;
    const values row = hidden(1,7);
    state.record(0,count,[&](size_t i) { ++reads; return common_mtp_state::row_view {int32_t(i+1),row.data()}; });
    require(reads==first+history+1, "large prefill only reads verification prefix, bounded tail and final row");
    require(state.verify.size()==first*width && state.history.size()==history, "host scratch does not grow with full prefill");
    require(snapshot(state).size()<1024, "serialized CPU state excludes full 4096-row hidden matrix");
    require(state.rewind(4095) && equal(state.predecessor(4095,false),row), "bounded tail still covers final-token replay");
}

void checkpoint_and_child_copy() {
    common_mtp_state parent(4,9,4), child(4,9,4);
    process(parent,5,0,16);
    const auto before_batch = snapshot(parent);
    require(child.restore(before_batch) && child.rewind(16), "child remaps payload without serialized source sequence ID");
    process(parent,5,16,1);
    require(equal(child.predecessor(16,false),hidden(5,15)), "child owns independent CPU rows");
    process(child,6,16,1);
    require(equal(parent.predecessor(17,false),hidden(5,16)) && equal(child.predecessor(17,false),hidden(6,16)), "copied sequences continue independently");
    require(parent.restore(before_batch) && parent.rewind(15), "checkpoint pos_max replay chooses the prior row");
    require(equal(parent.predecessor(15,false),hidden(5,14)), "checkpoint boundary is not confused with final pending row");
}

void shifts_and_reuse() {
    common_mtp_state state(4,32,17);
    process(state,7,0,10);
    state.erase(3,6); // remove token positions [3,6)
    state.shift(6,10,-3);
    require(equal(state.predecessor(7,false),hidden(7,9)), "context shift remaps the actual pending predecessor");
    require(equal(state.find(3),hidden(7,2)) && equal(state.find(4),hidden(7,6)), "kept prefix and moved tail retain their own rows");
    require(state.rewind(7), "shifted sequence can continue at its new boundary");
    state.shift(0,-1,INT32_MAX);
    require(state.find(7)==nullptr && !state.sealed, "overflow invalidates position metadata instead of wrapping");
}

void multimodal_boundaries() {
    common_mtp_state state(4,9,4);
    process(state,8,0,5);
    require(equal(state.predecessor(5,true),hidden(8,4)), "image begins with preceding text's hidden row");
    const values image_a = hidden(80,1), image_b = hidden(80,2);
    state.record(5,3,[&](size_t) { return common_mtp_state::row_view {-1,image_a.data()}; });
    require(equal(state.predecessor(5,true),image_a), "repeated M-RoPE positions use uninterrupted embedded carryover");
    const auto unsealed = snapshot(state);
    common_mtp_state restored(4,9,4);
    require(restored.restore(unsealed) && !restored.embedded && !restored.rewind(25), "restored unsealed image state cannot guess following text position");
    state.record(5,3,[&](size_t) { return common_mtp_state::row_view {-1,image_b.data()}; });
    common_mtp_state mixed = state;
    require(equal(mixed.prepare(25,false),image_b), "uninterrupted mixed image/text processing binds the actual next boundary");
    state.seal(25); // actual mtmd_input_chunk n_pos, not pos[0]+one
    require(equal(state.predecessor(25,false),image_b), "server seals image carryover to real next text boundary");
    require(restored.restore(snapshot(state)) && restored.rewind(25), "sealed image boundary survives cache restore");
    process(restored,8,25,1);
    require(equal(restored.predecessor(26,false),hidden(8,25)), "text after image uses normal positional tracking");
    restored.reset();
    require(equal(restored.predecessor(0,true),values{}), "fresh image-only request starts from zero predecessor");
}

void payload_validation_and_bits() {
    common_mtp_state state(4,9,4);
    values bits;
    const uint32_t patterns[]={0x80000000,0x7fc01234,0xff800000,0x3f800000};
    std::memcpy(bits.data(),patterns,sizeof(patterns));
    state.record(0,1,[&](size_t) { return common_mtp_state::row_view {1,bits.data()}; });
    auto bytes=snapshot(state);
    common_mtp_state restored(4,9,4);
    require(restored.restore(bytes) && equal(restored.predecessor(1,false),bits), "float bytes including -0/NaN/infinity roundtrip exactly");
    const auto original=bytes;
    fail_next_allocation = true;
    require(!restored.restore(original) && !restored.rewind(1), "allocation failure invalidates old carryover after GPU restore");
    require(!fail_next_allocation && restored.restore(original), "parser allocation failure is recoverable");
    const size_t fields[]={0,4,8,12,16,20,24,28};
    for (size_t offset:fields) {
        bytes=original; bytes[offset]=0xff; bytes[offset+1]=0xff; bytes[offset+2]=0xff; bytes[offset+3]=0x7f;
        require(!restored.restore(bytes) && !restored.rewind(1), "malformed header invalidates old conversation state");
        require(restored.restore(original), "valid state can recover after malformed payload");
    }
    for (size_t length=0;length<original.size();++length) {
        bytes.assign(original.begin(),original.begin()+length);
        require(!restored.restore(bytes) && !restored.rewind(1), "every truncated payload refuses stale reuse");
    }
    bytes=original; bytes.push_back(0);
    require(!restored.restore(bytes), "trailing bytes are not silently accepted");
    require(!restored.restore({}) && !restored.rewind(1), "legacy empty payload requires safe preflight replay");
}
}

int main() {
    try {
        conversation_restore_and_replay(); accepted_and_rejected_rows(); bounded_prefill_copy();
        checkpoint_and_child_copy(); shifts_and_reuse(); multimodal_boundaries(); payload_validation_and_bits();
        std::cout << checks << " production MTP carryover, resume, copy-bound and payload checks passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
