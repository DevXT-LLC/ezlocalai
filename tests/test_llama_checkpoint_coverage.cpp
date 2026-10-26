#include "checkpoint-coverage.h"

#include <climits>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

namespace {
int checks = 0;
void require(bool value, const char * what) {
    ++checks;
    if (!value) throw std::runtime_error(what);
}
std::set<int32_t> offsets(const common_checkpoint_plan & plan) {
    return {plan.offsets.begin(), plan.offsets.begin() + plan.count};
}
common_checkpoint_plan plan(int count = 32, bool named = true, bool recurrent = true,
        bool enabled = true, int64_t tokens = 23252, int batch = 4096, int ubatch = 1024) {
    return common_plan_checkpoints(tokens,batch,ubatch,count,named,recurrent,enabled);
}

// Drive the production planner's actual break/near-end decisions with the
// server's ordinary monotonic token-fill order. Checkpoints precede each batch.
std::set<int64_t> snapshot_boundaries(const common_checkpoint_plan & p, int64_t tokens, int batch) {
    std::set<int64_t> snapshots;
    int64_t remaining = tokens;
    while (remaining > 0) {
        const int64_t before = remaining;
        int added = 0;
        while (remaining > 0 && added < batch) {
            --remaining; ++added;
            if (p.break_before_end(remaining)) break;
        }
        if (remaining == 0 || p.near_end(remaining,before)) snapshots.insert(before);
    }
    return snapshots;
}

void eligible_and_legacy_boundaries() {
    const auto on = plan(), off = plan(32,true,true,false);
    require(offsets(on)==std::set<int32_t>({2052,1028,4}), "1024 adds exactly the earlier 2K boundary");
    require(offsets(off)==std::set<int32_t>({1028,4}), "opt-out preserves both original offsets");
    require(on.extra_offset==2052 && on.extra_near_span==2048 && on.old_near_span==1024,
        "earlier checkpoint has its own span without changing ordinary near-end behavior");
    require(on.break_before_end(2052) && on.break_before_end(1028) && on.break_before_end(4), "all intended split points remain active");
    require(!off.break_before_end(2052), "baseline has no earlier split");
    require(on.near_end(1028,2052) && !off.near_end(1028,2052),
        "snapshot before the 2052-to-1028 batch is actually permitted");
    require(on.near_end(4,1028) && off.near_end(4,1028), "old near-end microbatch snapshot remains");
    require(on.near_end(0,4) && off.near_end(0,4), "last four-token snapshot remains");
    require(!on.near_end(1500,2400), "expanded coverage cannot add arbitrary intermediate/user-split checkpoints");
    require(on.near_end(1500,2052), "a user split within the extra batch still preserves its designated starting snapshot");
    const auto before = snapshot_boundaries(off,23252,4096);
    const auto after = snapshot_boundaries(on,23252,4096);
    require(before==std::set<int64_t>({1028,4}), "ordinary long prompt retains original two end snapshots");
    require(after==std::set<int64_t>({2052,1028,4}), "production decisions add one checkpoint and lose neither old checkpoint");
    require(after.size()==before.size()+1, "near-end snapshot growth is one, not unbounded");
}

void qualification_guards() {
    const auto base = offsets(plan(32,true,true,false));
    for (int count=-2;count<=3;++count) {
        const auto p=plan(count);
        require(offsets(p)==base && p.extra_offset==0, "checkpoint counts below four retain original behavior");
    }
    require(offsets(plan(4))==offsets(plan(32)), "minimum qualified checkpoint budget supports the same one extra boundary");
    require(offsets(plan(32,false))==base, "unnamed prompts are untouched");
    require(offsets(plan(32,true,false))==base, "non-recurrent checkpoint paths are untouched");
    require(offsets(plan(32,true,true,true,4095))==base, "short prompts are untouched");
    require(offsets(plan(32,true,true,true,4096))==offsets(plan()), "4096-token named prompt reaches the qualification threshold");
    require(offsets(plan(32,true,true,true,0))==base, "empty prompt cannot enable new coverage");
    require(offsets(plan(32,true,true,true,23252,0,1024)).empty(), "invalid zero batch cannot create a boundary");
    require(offsets(plan(32,true,true,true,23252,4096,0)).empty(), "invalid zero physical batch cannot create a boundary");
}

void cap_deduplicate_and_keep_other_cards_unchanged() {
    const auto on2048=plan(32,true,true,true,23252,4096,2048);
    const auto off2048=plan(32,true,true,false,23252,4096,2048);
    require(offsets(on2048)==offsets(off2048) && on2048.extra_offset==0, "2048 already has the desired boundary and remains unchanged");
    require(on2048.old_near_span==off2048.old_near_span, "2048 near-end predicate is unchanged");
    for (int remaining=0;remaining<5000;remaining+=37) {
        require(on2048.break_before_end(remaining)==off2048.break_before_end(remaining), "2048 break decisions are identical");
        require(on2048.near_end(remaining,remaining+500)==off2048.near_end(remaining,remaining+500), "2048 snapshot decisions are identical");
    }
    for (int batch : {1,4,512,1024,2048,4096}) {
        for (int ubatch : {1,256,512,1024,2048,4096}) {
            const auto p=plan(32,true,true,true,23252,batch,ubatch);
            const auto old=plan(32,true,true,false,23252,batch,ubatch);
            require(p.count<=3 && p.count<=old.count+1, "planner never adds more than one distinct boundary");
            for (int offset : offsets(p)) require(offset>0 && offset<=batch, "every boundary respects the logical batch cap");
            for (int offset : offsets(old)) require(offsets(p).count(offset)==1, "deduplication never loses an old boundary");
        }
    }
    const auto max=plan(32,true,true,true,INT64_MAX,INT_MAX,INT_MAX);
    require(offsets(max)==std::set<int32_t>({INT_MAX,4}) && max.extra_offset==0, "large configuration values do not overflow offset arithmetic");
    const auto capped=plan(32,true,true,true,23252,2048,1024);
    require(offsets(capped)==std::set<int32_t>({2048,1028,4}) && capped.extra_near_span==2044,
        "additional boundary is safely capped below requested 2052");
}

void environment_optout() {
    const char * existing=std::getenv("EZLOCALAI_LLAMA_CHECKPOINT_COVERAGE");
    const bool had_existing=existing!=nullptr;
    const std::string saved=existing ? existing : "";
    unsetenv("EZLOCALAI_LLAMA_CHECKPOINT_COVERAGE");
    require(common_checkpoint_coverage_enabled(), "coverage defaults on for qualified cases");
    setenv("EZLOCALAI_LLAMA_CHECKPOINT_COVERAGE","0",1);
    require(!common_checkpoint_coverage_enabled(), "explicit zero disables the experiment in the same binary");
    require(common_plan_checkpoints(23252,4096,1024,32,true,true).extra_offset==0, "default planner reads opt-out");
    setenv("EZLOCALAI_LLAMA_CHECKPOINT_COVERAGE","1",1);
    require(common_plan_checkpoints(23252,4096,1024,32,true,true).extra_offset==2052, "explicit one enables qualified coverage");
    if (had_existing) setenv("EZLOCALAI_LLAMA_CHECKPOINT_COVERAGE",saved.c_str(),1);
    else unsetenv("EZLOCALAI_LLAMA_CHECKPOINT_COVERAGE");
}
}

int main() {
    try {
        eligible_and_legacy_boundaries(); qualification_guards();
        cap_deduplicate_and_keep_other_cards_unchanged(); environment_optout();
        std::cout << checks << " production checkpoint coverage planner checks passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
