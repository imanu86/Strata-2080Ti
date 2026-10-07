#pragma once
// The two slot groups of the pipelined batch windows (STRATA_BATCH_PIPELINE=2, docs/BATCH_PIPELINE.md), as pure
// logic so a CPU test can pin it down:
//   * batch_group_join: which group a newly admitted slot joins (the one with fewer rows);
//   * batch_group_split: when one group has emptied while the other holds two or more slots, how the slots are shared
//     out again so both groups carry about the same rows (a slot's rows: its current token plus its drafts);
//   * BatchOverlap: the stages' busy time and how much of it the two cards spent working at once (the log's
//     "overlap" figure).
//
// A slot is a conversation of its own, so any slot may sit in any group; what the groups balance is the rows each
// window carries (a window's cost is base + row_ms * rows), so that the stage times of the two groups match and each
// card's idle gaps are short.

#include <cstdint>

namespace strata::core {

/// The group a new slot joins: 0 or 1, the one carrying fewer rows (ties go to 0, the even verifiers, whose captured
/// graphs the serial windows share).
inline int batch_group_join(int rows_g0, int rows_g1) { return rows_g1 < rows_g0 ? 1 : 0; }

/// `n` slots of one group with their planned rows `rows[i]` (>= 1 each), shared out over two groups: `grp[i]` = 0 or 1.
/// Greedy by falling rows (a stable order: equal rows keep their index order), each to the lighter group, ties to 0:
/// so group 0 is never lighter than group 1 and, from two slots on, both groups hold at least one.  Returns the slots
/// given to group 1 (0 when n < 2).
inline int batch_group_split(int n, const int* rows, int* grp) {
    if (n > 64) n = 64;
    for (int i = 0; i < n; ++i) grp[i] = 0;
    if (n < 2) return 0;
    int order[64];
    for (int i = 0; i < n; ++i) order[i] = i;
    for (int i = 1; i < n; ++i) {   // insertion sort by falling rows, stable
        const int x = order[i];
        int j = i;
        while (j > 0 && rows[order[j - 1]] < rows[x]) { order[j] = order[j - 1]; --j; }
        order[j] = x;
    }
    int sum[2] = {0, 0}, moved = 0;
    for (int k = 0; k < n; ++k) {
        const int i = order[k], g = sum[1] < sum[0] ? 1 : 0;
        grp[i] = g;
        sum[g] += rows[i] > 0 ? rows[i] : 1;
        moved += g;
    }
    return moved;
}

/// The two stages' busy intervals folded into three sums (milliseconds): each stage's own busy time, the time at
/// least one stage was busy, and the time both were.  `overlap()` is the share of the busy time the two cards
/// worked at once: 0 for serial windows, up to 1 when the stages never wait for each other.
struct BatchOverlap {
    double busy[2] = {0.0, 0.0};
    double any = 0.0, both = 0.0;
    int count = 0;         ///< stages busy now
    double last = 0.0;     ///< the time of the last change
    bool stage_busy[2] = {false, false};

    void begin(int stage, double now) {
        if (stage < 0 || stage > 1 || stage_busy[stage]) return;
        advance(now);
        stage_busy[stage] = true;
        ++count;
    }
    void end(int stage, double now) {
        if (stage < 0 || stage > 1 || !stage_busy[stage]) return;
        advance(now);
        stage_busy[stage] = false;
        --count;
    }
    double overlap() const { return any > 0.0 ? both / any : 0.0; }
    void reset() { busy[0] = busy[1] = any = both = 0.0; }

private:
    void advance(double now) {
        const double dt = now - last;
        last = now;
        if (dt <= 0.0) return;
        for (int s = 0; s < 2; ++s)
            if (stage_busy[s]) busy[s] += dt;
        if (count >= 1) any += dt;
        if (count >= 2) both += dt;
    }
};

}  // namespace strata::core
