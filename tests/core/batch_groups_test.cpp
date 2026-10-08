// The two slot groups of the pipelined batch windows (include/strata/core/batch_groups.hpp): where a new slot goes,
// how an emptied group takes slots back from the other, and the overlap accounting.  Header-only, CPU.
#include "strata/core/batch_groups.hpp"

#include <cstdio>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}
int sum_of(int n, const int* rows, const int* grp, int g) {
    int s = 0;
    for (int i = 0; i < n; ++i)
        if (grp[i] == g) s += rows[i];
    return s;
}
}  // namespace

int main() {
    using strata::core::batch_group_join;
    using strata::core::batch_group_split;
    using strata::core::BatchOverlap;
    // a new slot joins the lighter group, ties to 0
    check(batch_group_join(0, 0) == 0, "join: both empty -> 0");
    check(batch_group_join(2, 0) == 1, "join: group 1 empty -> 1");
    check(batch_group_join(2, 4) == 0, "join: group 0 lighter -> 0");
    check(batch_group_join(3, 3) == 0, "join: equal -> 0");
    // a split of one group's slots over two
    {
        int grp[8] = {};
        const int one[] = {2};
        check(batch_group_split(1, one, grp) == 0 && grp[0] == 0, "split: one slot stays");
        check(batch_group_split(0, one, grp) == 0, "split: nothing to split");
    }
    {
        int grp[8] = {};
        const int rows[] = {2, 2};
        check(batch_group_split(2, rows, grp) == 1 && grp[0] == 0 && grp[1] == 1, "split: two equal slots, one each");
    }
    {
        int grp[8] = {};
        const int rows[] = {1, 3, 2};   // 3 -> g0, 2 -> g1, 1 -> g1: sums 3 / 3
        const int moved = batch_group_split(3, rows, grp);
        check(moved == 2 && grp[1] == 0 && grp[2] == 1 && grp[0] == 1, "split: three slots, 3 / 2+1");
        check(sum_of(3, rows, grp, 0) == 3 && sum_of(3, rows, grp, 1) == 3, "split: three slots balanced");
    }
    {
        int grp[8] = {};
        const int rows[] = {4, 1, 1, 1};   // 4 -> g0, then 1, 1, 1 -> g1: 4 / 3
        batch_group_split(4, rows, grp);
        check(grp[0] == 0 && grp[1] == 1 && grp[2] == 1 && grp[3] == 1, "split: a heavy slot alone against three light ones");
        check(sum_of(4, rows, grp, 0) >= sum_of(4, rows, grp, 1), "split: group 0 never lighter");
    }
    {
        int grp[8] = {};
        const int rows[] = {1, 1, 1, 1, 1, 1, 1, 1};
        check(batch_group_split(8, rows, grp) == 4, "split: eight equal slots, four each");
        check(sum_of(8, rows, grp, 0) == 4 && sum_of(8, rows, grp, 1) == 4, "split: eight equal slots balanced");
        bool stable = true;   // equal rows keep their index order: slots 0, 2, 4, 6 stay, 1, 3, 5, 7 move
        for (int i = 0; i < 8; ++i) stable = stable && grp[i] == (i & 1);
        check(stable, "split: equal rows alternate by index");
    }
    {
        int grp[8] = {};
        const int rows[] = {0, 0};   // a row count of 0 is taken as 1
        check(batch_group_split(2, rows, grp) == 1, "split: zero rows count as one");
    }
    // the overlap accounting: serial windows never overlap, stages fully overlapped give 1
    {
        BatchOverlap ov;
        ov.begin(0, 0.0);
        ov.end(0, 10.0);
        ov.begin(1, 10.0);
        ov.end(1, 20.0);
        check(ov.busy[0] == 10.0 && ov.busy[1] == 10.0, "overlap: serial busy times");
        check(ov.any == 20.0 && ov.both == 0.0 && ov.overlap() == 0.0, "overlap: serial = 0");
    }
    {
        BatchOverlap ov;
        ov.begin(0, 0.0);
        ov.begin(1, 0.0);
        ov.end(0, 10.0);
        ov.end(1, 10.0);
        check(ov.both == 10.0 && ov.any == 10.0 && ov.overlap() == 1.0, "overlap: full = 1");
    }
    {
        BatchOverlap ov;
        ov.begin(0, 0.0);     // stage 0: 0..10, stage 1: 5..20 -> both 5..10 (5 ms), any 0..20 (20 ms)
        ov.begin(1, 5.0);
        ov.end(0, 10.0);
        ov.end(1, 20.0);
        check(ov.busy[0] == 10.0 && ov.busy[1] == 15.0, "overlap: partial busy times");
        check(ov.both == 5.0 && ov.any == 20.0 && ov.overlap() == 0.25, "overlap: partial = 0.25");
        ov.begin(0, 20.0);    // a double begin or end of a stage changes nothing
        ov.begin(0, 25.0);
        ov.end(0, 30.0);
        ov.end(0, 35.0);
        check(ov.busy[0] == 20.0 && ov.any == 30.0, "overlap: repeated begin/end ignored");
        ov.reset();
        check(ov.busy[0] == 0.0 && ov.any == 0.0 && ov.both == 0.0 && ov.overlap() == 0.0, "overlap: reset");
    }
    if (failures == 0) std::printf("batch_groups_test: ok\n");
    return failures == 0 ? 0 : 1;
}
