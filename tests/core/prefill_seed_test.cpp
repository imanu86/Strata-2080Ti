// tests/core/prefill_seed_test.cpp - the prefill seed's plan, prior and lab key (docs/PREFILL_SEED.md): host only.
#include "strata/core/prefill_seed.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using strata::core::PrefillSeedCfg;
using strata::core::PrefillSeedPlan;
using strata::core::prefill_seed_lab_apply;
using strata::core::prefill_seed_plan;
using strata::core::prefill_seed_prior;

int main() {
    int checked = 0;
    auto check = [&](bool ok) { ++checked; if (!ok) { std::fprintf(stderr, "case %d failed\n", checked); std::exit(1); } };
    const int64_t L = 3, E = 8;
    auto evict_all = [](int64_t, int32_t) { return true; };
    auto enter_all = [](int64_t, int32_t) { return true; };

    // layer 0: residents {0,1,2} with masses 5, 0, 1; outsiders 3 (mass 4), 4 (mass 0.5), 5 (mass 0)
    // layer 1: residents {0,1} both heavy, no outsider heavier: nothing
    // layer 2: residents {0,1} with masses 0, 0; outsiders 2 (3), 3 (2), 4 (2): two swaps, tie on gain by id
    std::vector<float> mass((size_t) (L * E), 0.0f);
    std::vector<int32_t> res((size_t) (L * E), -1);
    res[0] = 10; res[1] = 11; res[2] = 12;
    mass[0] = 5; mass[1] = 0; mass[2] = 1; mass[3] = 4; mass[4] = 0.5f;
    res[E + 0] = 20; res[E + 1] = 21;
    mass[E + 0] = 3; mass[E + 1] = 3; mass[E + 2] = 1;
    res[2 * E + 0] = 30; res[2 * E + 1] = 31;
    mass[2 * E + 2] = 3; mass[2 * E + 3] = 2; mass[2 * E + 4] = 2;

    PrefillSeedCfg cfg;
    cfg.enabled = true;
    cfg.swaps = 0;
    cfg.gain = 1.5f;
    PrefillSeedPlan p = prefill_seed_plan(mass.data(), res.data(), L, E, cfg, evict_all, enter_all);
    // layer 0: 3 (4) takes 1's slot (0): gain 4; 4 (0.5) vs 2 (1): not above -> stop.  layer 2: 2 -> 0 (gain 3),
    // 3 -> 1 (gain 2); 4 has no victim left.
    check(p.swaps.size() == 3);
    check(p.swaps[0].layer == 0 && p.swaps[0].in == 3 && p.swaps[0].out == 1 && p.swaps[0].gain == 4.0f);
    check(p.swaps[1].layer == 2 && p.swaps[1].in == 2 && p.swaps[1].out == 0);
    check(p.swaps[2].layer == 2 && p.swaps[2].in == 3 && p.swaps[2].out == 1);
    check(p.experts_seen == 10);
    check(p.candidates == 6);
    check(p.mass_total == 24.5 && p.mass_resident == 12.0 && p.mass_after == 21.0);

    // the gain threshold: a candidate must be >= gain x victim - layer 0's 4 vs 0 passes any gain, layer 2's 3 vs 0
    // too; with gain 10 and a victim of mass 1 the candidate 4 (0.5) was already out
    cfg.gain = 10.0f;
    p = prefill_seed_plan(mass.data(), res.data(), L, E, cfg, evict_all, enter_all);
    check(p.swaps.size() == 3);

    // the global cap keeps the best gains
    cfg.gain = 1.5f;
    cfg.swaps = 2;
    p = prefill_seed_plan(mass.data(), res.data(), L, E, cfg, evict_all, enter_all);
    check(p.swaps.size() == 2 && p.swaps[0].gain == 4.0f && p.swaps[1].gain == 3.0f);

    // the floor grants layer 2 one swap even when the cap would have gone to layer 0 twice
    mass[5] = 4.5f;   // layer 0: outsider 5 (4.5) now beats resident 2 (1) as well
    cfg.swaps = 2;
    cfg.floor = 1;
    p = prefill_seed_plan(mass.data(), res.data(), L, E, cfg, evict_all, enter_all);
    check(p.swaps.size() == 2);
    {
        bool l0 = false, l2 = false;
        for (const auto& s : p.swaps) { l0 = l0 || s.layer == 0; l2 = l2 || s.layer == 2; }
        check(l0 && l2);
    }
    // without the floor both go to layer 0 (gains 4.5 - 0 = 4.5 and 4 - 1 = 3 vs layer 2's 3 and 2: 4.5, then a
    // tie at 3 broken by the lower layer)
    cfg.floor = 0;
    p = prefill_seed_plan(mass.data(), res.data(), L, E, cfg, evict_all, enter_all);
    check(p.swaps.size() == 2 && p.swaps[0].layer == 0 && p.swaps[1].layer == 0);

    // the eviction and admission rules: no victim in layer 0, no entry for expert 2 of layer 2
    cfg.swaps = 0;
    auto evict_not_l0 = [](int64_t l, int32_t) { return l != 0; };
    auto enter_not_l2e2 = [](int64_t l, int32_t e) { return !(l == 2 && e == 2); };
    p = prefill_seed_plan(mass.data(), res.data(), L, E, cfg, evict_not_l0, enter_not_l2e2);
    check(p.swaps.size() == 2);
    for (const auto& s : p.swaps) check(s.layer == 2 && s.in != 2);

    // the prior: a horizon of 32 tokens over a prompt of 64 tokens with 10 choices per token
    {
        std::vector<float> m(4, 0.0f), c(4, 0.0f), usage(4, 0.0f);
        m[0] = 48.0f; m[1] = 16.0f;   // sum 64 = one unit of mass per token
        c[0] = 400.0f; c[1] = 240.0f; // 640 choices = 10 per token
        usage[1] = 100.0f;            // never lowered
        const int64_t raised = prefill_seed_prior(usage, m.data(), c.data(), 4, 64, 32.0f);
        check(raised == 1);
        // prior[0] = 32 x 48 x (640 / 64) / 64 = 240
        check(usage[0] > 239.9f && usage[0] < 240.1f);
        check(usage[1] == 100.0f && usage[2] == 0.0f);
        check(prefill_seed_prior(usage, m.data(), c.data(), 4, 64, 0.0f) == 0);
        check(prefill_seed_prior(usage, m.data(), c.data(), 3, 64, 1.0f) == 0);   // size mismatch
    }

    // the lab key: only the PREFILL_SEED levers are taken, others are left to their owner, malformed items refuse
    {
        PrefillSeedCfg c;
        std::string err;
        check(prefill_seed_lab_apply("PREFILL_SEED:1", c, err) == false && !err.empty());   // not armed
        c.armed = true;
        err.clear();
        check(prefill_seed_lab_apply("early_l1:1,prefill_seed:1,STRATA_PREFILL_SEED_SWAPS:64,Prefill_Seed_Gain:2.5,,PREFILL_SEED_MIN_TOKENS:0",
                                     c, err));
        check(c.enabled && c.swaps == 64 && c.gain == 2.5f && c.min_tokens == 0 && c.floor == 0);
        check(!prefill_seed_lab_apply("PREFILL_SEED_BOGUS:1", c, err));
        check(!prefill_seed_lab_apply("PREFILL_SEED:x", c, err));
        check(!prefill_seed_lab_apply("PREFILL_SEED", c, err));
        check(prefill_seed_lab_apply("PREFILL_SEED:0,PREFILL_SEED_GAIN:0.2", c, err) && !c.enabled && c.gain == 1.0f);
    }
    std::printf("prefill_seed_test: %d cases ok\n", checked);
    return 0;
}
