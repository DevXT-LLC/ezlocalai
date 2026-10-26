#include "checkpoint-coverage.h"
#include "server-common.h"
#include "llama-batch.h"
#include "llama-vocab.h"
#include "speculative-mtp-state.h"
#include "speculative.h"

#include <climits>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>

namespace {
int checks = 0;
void require(bool value, const char * what) {
    ++checks;
    if (!value) throw std::runtime_error(what);
}
common_checkpoint_runtime runtime() { return {1,1024,3,true,true}; }
common_checkpoint_plan plan(bool enabled=true, int count=32, bool named=true, bool recurrent=true,
        int64_t tokens=24577, int batch=4096, int ubatch=1024,
        common_checkpoint_runtime rt=runtime()) {
    return common_plan_checkpoints(tokens,batch,ubatch,count,named,recurrent,rt,enabled);
}
std::set<int32_t> offsets(const common_checkpoint_plan & p) {
    return {p.offsets.begin(),p.offsets.begin()+p.count};
}
struct schedule {
    std::vector<std::pair<int32_t,int32_t>> logical;
    std::set<int32_t> snapshots;
};
schedule schedule_prompt(const common_checkpoint_plan & p,int tokens,int start,int batch=4096) {
    schedule out;
    while (start<tokens) {
        const int before=start;
        int added=0;
        while(start<tokens && added<batch) {
            ++start; ++added;
            if(p.break_before_end(tokens-start,added,tokens-before)) break;
        }
        out.logical.emplace_back(before,added);
        if(start==tokens || p.near_end(tokens-start,tokens-before)) out.snapshots.insert(before);
    }
    return out;
}
using group = std::vector<std::tuple<llama_pos,llama_token,int>>;
using groups = std::vector<group>;
// Execute the actual production hybrid/recurrent splitter, including RS-tail
// reservation. Token/position/output rows must retain exactly the old groups.
groups physical_groups(const schedule & s,int tokens,uint32_t ubatch,uint32_t rollback,bool draft=false) {
    groups result;
    llama_vocab vocab;
    for(const auto & chunk:s.logical) {
        llama_batch_ext input(chunk.second,2,2,1,nullptr,64,1);
        bool valid=true;
        for(int i=0;i<chunk.second;++i) {
            const int index=input.add_token(0);
            const llama_pos pos=chunk.first+i;
            valid=valid && index>=0 && input.set_token_id(index,pos%64) && input.set_token_pos(index,&pos) &&
                    input.set_output(index,pos==tokens-1);
        }
        require(valid,"actual input token/position/output row construction succeeds");
        llama_batch_allocr allocator(1);
        require(allocator.init(input,vocab,false),"actual allocator accepts the token batch");
        allocator.split_reset();
        while(allocator.get_n_used()<allocator.get_n_tokens()) {
            const auto physical=draft?allocator.split_simple(ubatch):
                allocator.split_equal(ubatch,true,rollback>0?rollback+1:0);
            require(physical.n_tokens>0 && physical.n_seqs_unq==1 && (draft || physical.n_seqs==1),
                "actual splitter returns rows for one unique sequence");
            group current;
            for(uint32_t i=0;i<physical.n_tokens;++i)
                current.emplace_back(physical.pos[i],physical.token[i],physical.output[i]);
            result.push_back(std::move(current));
        }
    }
    return result;
}

void measured_schedule_and_actual_splitters() {
    const auto on=plan(),off=plan(false);
    require(offsets(on)==std::set<int32_t>({1028,4}) && offsets(on)==offsets(off),
        "both original fixed offsets remain, without a new fixed2052 split");
    require(on.extra_window_end==2052 && on.rollback_tail==4,"extra snapshot window includes target rollback tail");
    const auto before=schedule_prompt(off,24577,75),after=schedule_prompt(on,24577,75);
    require(before.snapshots==std::set<int32_t>({23549,24573}),"observed baseline has the original two checkpoints");
    require(after.snapshots==std::set<int32_t>({22603,23549,24573}),"one added checkpoint is aligned to the original physical groups");
    require(after.snapshots.size()==before.snapshots.size()+1,"only one snapshot is added");
    require(physical_groups(before,24577,1024,3)==physical_groups(after,24577,1024,3),
        "actual target RS splitter keeps identical cold token/position/output groups");
    require(physical_groups(before,24577,1024,0,true)==physical_groups(after,24577,1024,0,true),
        "actual draft splitter keeps identical cold token/position/output groups");
    require(!on.break_before_end(2052,1970,4022),"absolute2052 offset is not a physical boundary");
    require(on.break_before_end(1974,2048,4022),"actual measured full-physical-batch boundary is selected");
    require(on.near_end(1028,1974),"next batch captures the aligned starting checkpoint");
}

void tail_reservation_and_original_start_alignment() {
    for(uint32_t rollback:{0u,1u,3u,7u}) {
        auto rt=runtime();rt.rollback_tokens=rollback;
        for(int length:{1025,1026,1027,1031,2049,2050,2051,2055,2994,3073,3074,3075,3079,4096}) {
            const int start=75,tokens=start+length+1028;
            const auto before=schedule_prompt(plan(false,32,true,true,tokens,4096,1024,rt),tokens,start);
            const auto after=schedule_prompt(plan(true,32,true,true,tokens,4096,1024,rt),tokens,start);
            require(physical_groups(before,tokens,1024,rollback)==physical_groups(after,tokens,1024,rollback),
                "actual target grouping is preserved around short rollback tails");
            require(physical_groups(before,tokens,1024,0,true)==physical_groups(after,tokens,1024,0,true),
                "actual draft grouping is preserved around target rollback tails");
            for(int old:before.snapshots)require(after.snapshots.count(old)==1,"neither original checkpoint disappears");
            require(after.snapshots.size()<=before.snapshots.size()+1,"tail cases add at most one checkpoint");
        }
    }
    require(!plan().break_before_end(1030,3072,4102),"two-row legacy tail cannot split before a four-row RS reservation");
    for(int start:{0,75,128,1234}) {
        const auto before=schedule_prompt(plan(false),24577,start),after=schedule_prompt(plan(),24577,start);
        require(physical_groups(before,24577,1024,3)==physical_groups(after,24577,1024,3),
            "alignment follows each actual/user-start logical base, not absolute token positions");
    }
}

void eligibility_and_other_configs() {
    const auto base=offsets(plan(false));
    for(int count=-2;count<4;++count)require(plan(true,count).extra_window_end==0,"low checkpoint counts retain old behavior");
    require(plan(true,4).extra_window_end>0,"four checkpoint slots qualify");
    require(plan(true,32,false).extra_window_end==0,"unnamed prompts retain old behavior");
    require(plan(true,32,true,false).extra_window_end==0,"non-recurrent paths retain old behavior");
    require(plan(true,32,true,true,4095).extra_window_end==0,"short prompts retain old behavior");
    for(int ubatch:{1,256,512,2048,4096}) {
        auto rt=runtime();rt.draft_ubatch=ubatch;
        const auto on=plan(true,32,true,true,24577,4096,ubatch,rt),off=plan(false,32,true,true,24577,4096,ubatch,rt);
        require(on.extra_window_end==0 && offsets(on)==offsets(off),"unqualified physical sizes retain original scheduler");
    }
    for(int which=0;which<6;++which) {
        auto rt=runtime();
        if(which==0)rt.slots=2;
        if(which==1)rt.draft_ubatch=512;
        if(which==2)rt.draft_ubatch=0;
        if(which==3)rt.rollback_tokens=1023;
        if(which==4)rt.single_mtp_head=false;
        if(which==5)rt.uninterrupted_last_user=false;
        require(plan(true,32,true,true,24577,4096,1024,rt).extra_window_end==0,
            "multiple slots, custom draft UB, excessive RS, multiple heads and future switches disable experiment");
    }
    for(int batch:{1,4,512,1024,2048,4096})for(int ubatch:{1,256,512,1024,2048,4096}) {
        const auto p=plan(true,32,true,true,24577,batch,ubatch),old=plan(false,32,true,true,24577,batch,ubatch);
        require(offsets(p)==offsets(old),"legacy capped/deduplicated fixed offsets are unchanged");
        for(int offset:offsets(p))require(offset>0 && offset<=batch,"all original offsets respect logical cap");
    }
    require(plan(true,32,true,true,24577,0).count==0,"invalid zero batch yields no split");
    require(plan(true,32,true,true,24577,4096,0).count==0,"invalid zero UB yields no split");
    require(offsets(plan(true,32,true,true,INT64_MAX,INT_MAX,INT_MAX))==std::set<int32_t>({INT_MAX,4}),"offset addition cannot overflow");
}

void mtp_boundary_continuity() {
    common_mtp_state state(2,16,4);
    const auto row=[](int pos){return std::array<float,2>{float(pos),float(-pos)};};
    const auto record=[&](int start,int count){state.record(start,count,[&](size_t i){
        static std::array<float,2> value;value=row(start+int(i));
        return common_mtp_state::row_view{start+int32_t(i)+1,value.data()};});};
    state.seal(20555);
    record(20555,2048);
    const auto expected=row(22602);
    const float * predecessor=state.prepare(22603,false);
    require(predecessor && predecessor[0]==expected[0] && predecessor[1]==expected[1],
        "split MTP process gets the exact predecessor h22602 at boundary22603");
    record(22603,946);
    require(state.pending_next==23549 && state.pending==std::vector<float>({23548,-23548}),
        "right logical chunk reaches the same final MTP pending boundary");
}

struct borrowed_context_probe {
    const llama_model * owned_model = nullptr;
    const llama_model * attached_model = nullptr;
    int32_t heads = 1;
    int model_lookups = 0;
    int head_lookups = 0;
};
borrowed_context_probe * context_probe = nullptr;
const llama_model * probe_context_model(const llama_context * ctx) {
    require(ctx == reinterpret_cast<const llama_context *>(context_probe),
        "production accessor queries the actual borrowed context");
    ++context_probe->model_lookups;
    return context_probe->attached_model;
}
int32_t probe_trained_heads(const llama_model * model) {
    require(model == context_probe->attached_model,"head lookup uses the context's attached model");
    ++context_probe->head_lookups;
    return context_probe->heads;
}
void borrowed_mtp_context_accessor() {
    int model_storage=0;
    borrowed_context_probe probe;
    context_probe=&probe;
    probe.attached_model=reinterpret_cast<const llama_model *>(&model_storage);
    const auto * ctx=reinterpret_cast<const llama_context *>(&probe);
    const auto lookup=[&](const llama_context * context) {
        return common_checkpoint_mtp_head_count(context,probe_context_model,probe_trained_heads);
    };
    require(probe.owned_model==nullptr && lookup(ctx)==1,
        "borrowed MTP context with no owned model still exposes one trained head");
    require(probe.model_lookups==1 && probe.head_lookups==1,"both production accessor stages were exercised");
    auto rt=runtime();rt.single_mtp_head=lookup(ctx)==1;
    require(plan(true,32,true,true,24577,4096,1024,rt).extra_window_end==2052,
        "actual borrowed-context accessor result enables aligned planning");
    const int previous_models=probe.model_lookups,previous_heads=probe.head_lookups;
    require(lookup(nullptr)==0 && probe.model_lookups==previous_models && probe.head_lookups==previous_heads,
        "null context invokes neither accessor");
    probe.attached_model=nullptr;
    require(lookup(ctx)==0 && probe.head_lookups==previous_heads,
        "a context without a model never queries trained heads");
    probe.attached_model=reinterpret_cast<const llama_model *>(&model_storage);
    for(int heads:{0,2}) {
        probe.heads=heads;
        rt.single_mtp_head=lookup(ctx)==1;
        require(plan(true,32,true,true,24577,4096,1024,rt).extra_window_end==0,
            "zero or multiple trained heads retain the original scheduler");
    }
    context_probe=nullptr;
}

void actual_draft_batch_parameter_inheritance() {
    common_params configured;
    configured.n_ubatch=1024;
    require(configured.speculative.draft.n_ubatch==0,"actual draft batch default requests target inheritance");
    require(common_base_params_to_speculative(configured).n_ubatch==1024,
        "actual production draft parameter conversion inherits target1024");
    configured.speculative.draft.n_ubatch=512;
    const auto custom=common_base_params_to_speculative(configured);
    require(custom.n_ubatch==512,"actual production conversion preserves an explicit draft override");
    auto rt=runtime();rt.draft_ubatch=custom.n_ubatch;
    require(plan(true,32,true,true,24577,4096,1024,rt).extra_window_end==0,
        "a custom draft physical size retains the original checkpoint scheduler");
}

void environment_optout() {
    const char * existing=std::getenv("EZLOCALAI_LLAMA_CHECKPOINT_COVERAGE");
    const bool present=existing!=nullptr;
    const std::string saved=existing?existing:"";
    unsetenv("EZLOCALAI_LLAMA_CHECKPOINT_COVERAGE");
    require(common_checkpoint_coverage_enabled(),"experimental default applies only after all runtime guards");
    setenv("EZLOCALAI_LLAMA_CHECKPOINT_COVERAGE","0",1);
    require(!common_checkpoint_coverage_enabled(),"explicit zero disables coverage");
    require(common_plan_checkpoints(24577,4096,1024,32,true,true,runtime()).extra_window_end==0,"production planner reads opt-out");
    setenv("EZLOCALAI_LLAMA_CHECKPOINT_COVERAGE","1",1);
    require(common_plan_checkpoints(24577,4096,1024,32,true,true,runtime()).extra_window_end==2052,"explicit one enables qualified alignment");
    if(present)setenv("EZLOCALAI_LLAMA_CHECKPOINT_COVERAGE",saved.c_str(),1);else unsetenv("EZLOCALAI_LLAMA_CHECKPOINT_COVERAGE");
}
void actual_server_tokens_media_classification() {
    const auto enabled_for = [](const server_tokens & tokens) {
        return common_plan_checkpoints(24577,4096,1024,32,
                common_checkpoint_prompt_is_text_only(tokens),true,runtime(),true).extra_window_end == 2052;
    };
    server_tokens ordinary(llama_tokens{1,2,3}, false);
    server_tokens projector_text(llama_tokens{1,2,3}, true);
    server_tokens empty_projector(llama_tokens{}, true);
    require(!ordinary.has_mtmd && projector_text.has_mtmd, "actual token paths reproduce projector-loaded text");
    require(common_checkpoint_prompt_is_text_only(ordinary), "ordinary text has no media");
    require(common_checkpoint_prompt_is_text_only(projector_text), "projector-loaded text is text-only despite has_mtmd");
    require(enabled_for(ordinary) && enabled_for(projector_text), "both actual text paths enable the same production planner");
    require(common_checkpoint_prompt_is_text_only(empty_projector), "empty projector tokens contain no media");

    mtmd::input_chunks chunks(mtmd_test_create_input_chunks());
    require(chunks.ptr != nullptr && chunks.size() == 2, "actual MTMD C fixture supplies text plus image");
    require(mtmd_input_chunk_get_type(chunks[1]) == MTMD_INPUT_CHUNK_TYPE_IMAGE, "fixture's second chunk is image media");
    server_tokens media_later(chunks,true);
    require(media_later[0] != LLAMA_TOKEN_NULL && media_later.find_next_media_chunk(0).first != nullptr,
        "actual media map contains an image after text");
    require(!common_checkpoint_prompt_is_text_only(media_later) && !enabled_for(media_later),
        "actual later image remains outside the coverage policy");

    server_tokens media_at_zero(llama_tokens{},true);
    media_at_zero.push_back(chunks[1]);
    require(media_at_zero[0] == LLAMA_TOKEN_NULL && media_at_zero.find_next_media_chunk(0).first == nullptr,
        "actual upper_bound lookup alone misses media at index zero");
    require(!common_checkpoint_prompt_is_text_only(media_at_zero) && !enabled_for(media_at_zero),
        "image at index zero remains outside coverage");
    auto cloned_text=projector_text.clone();
    auto cloned_media=media_later.clone();
    require(common_checkpoint_prompt_is_text_only(cloned_text) && enabled_for(cloned_text),
        "actual text clone remains eligible");
    require(!common_checkpoint_prompt_is_text_only(cloned_media) && !enabled_for(cloned_media),
        "actual image clone remains excluded");

    const auto roundtrip = [](const server_tokens & tokens) {
        const auto bytes = tokens.serialize();
        require(bytes.size() % sizeof(llama_token) == 0, "serialized tokens retain token alignment");
        llama_tokens packed(bytes.size() / sizeof(llama_token));
        std::memcpy(packed.data(),bytes.data(),bytes.size());
        return server_tokens::deserialize(packed,true);
    };
    auto restored_text=roundtrip(projector_text);
    auto restored_media=roundtrip(media_at_zero);
    require(common_checkpoint_prompt_is_text_only(restored_text) && enabled_for(restored_text),
        "packed text with projector path remains eligible");
    require(!common_checkpoint_prompt_is_text_only(restored_media) && !enabled_for(restored_media),
        "packed media at zero remains excluded");
    media_later.keep_first(5);
    require(common_checkpoint_prompt_is_text_only(media_later) && enabled_for(media_later),
        "actual truncation removing all image chunks restores text classification");
}
}

int main() {
    try {
        measured_schedule_and_actual_splitters(); tail_reservation_and_original_start_alignment();
        eligibility_and_other_configs(); mtp_boundary_continuity(); borrowed_mtp_context_accessor(); environment_optout();
        actual_draft_batch_parameter_inheritance();
        actual_server_tokens_media_classification();
        std::cout << checks << " production checkpoint alignment checks passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
