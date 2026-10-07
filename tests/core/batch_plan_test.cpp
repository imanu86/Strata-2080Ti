// The rows of a batch window with MTP drafts (include/strata/core/batch_plan.hpp): a slot's kept prefix (equal to
// 0.1.40's --batch-mtp rule with one draft) and the per-slot draft lengths of STRATA_BATCH_MTP_DRAFTS.  Header-only, CPU.
#include "strata/core/batch_plan.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}
constexpr int32_t kEos = 7;
bool is_eos(int32_t y) { return y == kEos; }
}  // namespace

int main() {
    using strata::core::batch_plan_drafts;
    using strata::core::batch_slot_draft_cap;
    using strata::core::batch_slot_keep;
    // two rows (one draft): exactly 0.1.40's rule, keep = 2 iff the pick equals the draft and nothing ends the slot
    // at row 0 and p + 3 <= max_context
    {
        bool same = true;
        for (int match = 0; match < 2; ++match)
            for (int eos0 = 0; eos0 < 2; ++eos0)
                for (int stop = 0; stop < 2; ++stop)
                    for (int64_t produced = 0; produced < 6; ++produced)
                        for (int64_t p = 90; p < 100; ++p) {
                            const int64_t max_new = 4, max_ctx = 96;
                            const int32_t tok[2] = {11, 12};
                            const int32_t out[2] = {eos0 ? kEos : (match ? 12 : 13), 14};
                            // (an end token never equals a draft here unless the draft was the end token itself)
                            const int32_t tok_e[2] = {11, eos0 && match ? kEos : 12};
                            const int legacy = (out[0] == tok_e[1] && !eos0 && !stop && produced + 2 <= max_new &&
                                                p + 3 <= max_ctx) ? 2 : 1;
                            const int k = batch_slot_keep(out, tok_e, 2, is_eos, stop != 0, produced, max_new, p, max_ctx);
                            if (k != legacy) same = false;
                            (void) tok;
                        }
        check(same, "two rows: the 0.1.40 --batch-mtp keep rule");
    }
    // four rows: the accepted prefix, cut where the slot ends
    {
        const int32_t tok[4] = {10, 20, 30, 40};
        const int32_t all[4] = {20, 30, 40, 50};
        check(batch_slot_keep(all, tok, 4, is_eos, false, 0, 100, 0, 1000) == 4, "every draft accepted");
        const int32_t two[4] = {20, 30, 99, 50};
        check(batch_slot_keep(two, tok, 4, is_eos, false, 0, 100, 0, 1000) == 3, "two drafts accepted");
        const int32_t none[4] = {21, 30, 40, 50};
        check(batch_slot_keep(none, tok, 4, is_eos, false, 0, 100, 0, 1000) == 1, "the first draft rejected");
        check(batch_slot_keep(all, tok, 1, is_eos, false, 0, 100, 0, 1000) == 1, "a slot without drafts");
        // an end token picked at row 1 (it was also the draft of row 2): rows 0 and 1 are kept, the slot ends at row 1
        const int32_t tok_e[4] = {10, 20, kEos, 40};
        const int32_t out_e[4] = {20, kEos, 40, 50};
        check(batch_slot_keep(out_e, tok_e, 4, is_eos, false, 0, 100, 0, 1000) == 2, "cut at the end token");
        check(batch_slot_keep(all, tok, 4, is_eos, false, 98, 100, 0, 1000) == 2, "cut at max_new (2 left)");
        check(batch_slot_keep(all, tok, 4, is_eos, false, 99, 100, 0, 1000) == 1, "cut at max_new (1 left)");
        check(batch_slot_keep(all, tok, 4, is_eos, true, 0, 100, 0, 1000) == 1, "a pending stop keeps one row");
        check(batch_slot_keep(all, tok, 4, is_eos, false, 0, 100, 996, 1000) == 3, "cut at the context's end");
    }
    // the cap of a slot's drafts
    {
        check(batch_slot_draft_cap(3, 0, 100, 0, 1000) == 3, "cap: what the drafter proposed");
        check(batch_slot_draft_cap(3, 98, 100, 0, 1000) == 1, "cap: the tokens left");
        check(batch_slot_draft_cap(3, 99, 100, 0, 1000) == 0, "cap: the last token");
        check(batch_slot_draft_cap(3, 0, 100, 997, 1000) == 1, "cap: the context (p + 3 <= max)");
        check(batch_slot_draft_cap(3, 0, 100, 998, 1000) == 0, "cap: the context, none");
    }
    // the plan: a confident drafter verifies more rows, a shaky one none; the bound on rows holds
    {
        const float pa[3] = {0.9f, 0.9f, 0.9f}, pb[3] = {0.3f, 0.3f, 0.3f}, pc[3] = {0.1f, 0.1f, 0.1f};
        const float* prob[3] = {pa, pb, pc};
        const int cap[3] = {3, 3, 3};
        int d[3];
        // 50 ms + 5.5 ms per row (3 slots): a row pays while its chance g * (50 + 5.5 R) > 5.5 E
        int rows = batch_plan_drafts(3, cap, prob, 8, 50.0, 5.5, 0.0, 1.0, d);
        check(d[0] == 3 && d[1] == 0 && d[2] == 0 && rows == 6, "the confident slot takes the rows");
        rows = batch_plan_drafts(3, cap, prob, 4, 50.0, 5.5, 0.0, 1.0, d);
        check(rows == 4 && d[0] == 1 && d[1] == 0 && d[2] == 0, "the bound on rows");
        rows = batch_plan_drafts(3, cap, prob, 8, 50.0, 5.5, 0.95, 1.0, d);
        check(rows == 3 && d[0] == 0, "min_gain above every chance: no drafts");
        const int cap1[3] = {1, 0, 3};
        rows = batch_plan_drafts(3, cap1, prob, 8, 50.0, 5.5, 0.0, 1.0, d);
        check(d[0] == 1 && d[1] == 0 && rows <= 8, "the caps hold");
        // one slot alone, a cheap row: a 0.3 draft pays (0.3 * 55.5 > 5.5)
        const float* one[1] = {pb};
        const int c1[1] = {3};
        rows = batch_plan_drafts(1, c1, one, 8, 50.0, 5.5, 0.0, 1.0, d);
        check(d[0] >= 1 && rows == 1 + d[0], "one slot: a 0.3 draft pays");
        // an expensive row: nothing pays
        rows = batch_plan_drafts(3, cap, prob, 8, 5.0, 50.0, 0.0, 1.0, d);
        check(rows == 3, "rows that cost more than the window: no drafts");
        // every slot equal: the rows spread (the chances of a deeper draft fall below the first ones)
        const float* eq[3] = {pa, pa, pa};
        rows = batch_plan_drafts(3, cap, eq, 6, 50.0, 5.5, 0.0, 1.0, d);
        check(d[0] == 1 && d[1] == 1 && d[2] == 1 && rows == 6, "equal drafters share the rows");
    }
    if (failures != 0) return EXIT_FAILURE;
    std::printf("batch_plan_test: ok\n");
    return EXIT_SUCCESS;
}
