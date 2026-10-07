# Gumbel-max picks for coupled drafts (`STRATA_SPEC_GUMBEL=1`, opt-in, lab branch `lab/gumbel`)

Port of upstream PR Niko1221/Strata#1281 (routhjim). Status of this branch: **compiled no, tried no.** Everything below
about behaviour comes from reading the code; everything about speed is a measurement plan, not a result.

## What it does

With sampling (temperature above 0), `STRATA_SPEC_COUPLED=1` makes the MTP draft layer sample its guess with the
target's chain (top_k, top_p, min_p, temperature, penalties) and the target's Philox counter, instead of proposing its
argmax. Both sides then pick by walking one uniform number over their own candidates sorted by probability. If the two
lists differ by one token, every later interval shifts and the two picks land on different tokens even where the
distributions nearly agree.

`STRATA_SPEC_GUMBEL=1` replaces that pick, on both sides, with a Gumbel-max pick: `argmax_i p_i / E_i`, where
`E_i = -log(u_i)` and `u_i` is a hash of `(seed, counter, token id)`. That is `argmax(log p_i + Gumbel_i)`, an exact
sample of `p`. A token gets the same noise in the draft's draw and the target's draw whichever other tokens survived
either cut. Verification is unchanged (exact match against the target's pick), so the text is still a sample of the same
distribution; only the number of accepted drafts changes.

Where it lives:

- `src/kernels/cuda/sampler.cu`: `gumbel_exp`, and the Gumbel pick in both tails (`sampler_kernel`, the legacy path, and
  `sampled_tail_warp`, which serves the split, one-block and coupled-draft paths). The coupled draft keys the noise by the
  real token id through `sub_to_id`, so the `--draft-vocab` subset does not break the coupling.
- `include/strata/kernels/sampler.hpp`: `SamplerParams::gumbel`, in existing padding (`sizeof` stays 64).
- The variable is read once per process in `sampler.cu` (`gumbel_env()`). `sample_tokens` ORs it into the params it was
  given (the target's pick); `coupled_stage_kernel` writes it into the drafter's device copy of the params, so the
  captured coupled round graphs see it.
- No dependency on `fp_lab.hpp` or any hot-lever struct: the environment is the only switch.

## How it fits our engine (read from the code, main at ca43719)

- **Greedy (temperature 0, or `greedy=true`): identical to the default, bit for bit.** `p.gumbel` is read in exactly
  three places: the legacy `sampler_kernel` tail, `sampled_tail_warp`, and `coupled_stage_kernel`. None of them runs for
  a greedy row: `sample_tokens` takes the `sampler_greedy_kernel` / cluster / `argmax_rows` branch (the greedy kernels
  receive the params by value but never read the new byte); the drafter's greedy rounds use the argmax graphs, not
  the coupled ones (`MtpDrafter::set_draft_sampling` sets `coupled_active_` only for sampled requests); the captured
  verify graph records `sample_tokens` with `greedy=true`. The only difference is one byte of struct padding in a kernel
  argument. Remaining caveat for the first run: this is argued from the source, not diffed on a device.
- **MTP drafts (`--spec 4 --mtp ...`, `--spec-min-p 0.5`):** the Gumbel flag has an effect on the drafts only when
  `STRATA_SPEC_COUPLED=1` is also set. Alone it changes only the target's pick (still an exact sample, a different
  stream than the default for the same seed) and the argmax drafts accept as often as before: P(argmax draft equals
  the target's pick) is `p_max` under either rule. So the arm "GUMBEL without COUPLED" is a control, not a candidate.
  The coupled drafter hands `--spec-min-p` the probability of the token it **sampled** (`probs_`), not the top
  probability, so windows are cut by the draw as well as by confidence and can get shorter than with argmax drafts (not measured).
  Judge by tokens per window and output t/s, not by the acceptance ratio alone (fewer low-confidence drafts offered
  lifts the ratio by itself).
- **`--pipeline-windows 2`: no gain, by design of our main.** `generate.cpp` decodes a request serially ("a request
  with coupled draft sampling decodes serially") when `mtp.coupled()` is true, because the pipelined chain teacher-forces
  tokens through the drafter and the coupled graphs skip that forcing. So a sampled request with `STRATA_SPEC_COUPLED=1`
  loses the pipeline. Without COUPLED, the pipeline runs as before and the Gumbel flag only changes the target's pick:
  both the serial window and the pipelined window (`collect` path) call `sample_tokens` with `Philox(seed, pos0 + t)`,
  and the Gumbel noise is keyed by the same counter and the token id, so a replayed speculative window redraws the same
  picks. Not adapted: making coupled drafts work inside the pipeline is a separate design task.
- **Layer split on two GPUs:** the target's sampler runs on the last stage, the card that holds the head
  (`le_ == n_layers` in `Verifier::run` and the pipelined collect); an earlier stage returns before sampling. The MTP
  drafter is loaded on the same card (`OnDevice on_mtp(last_st->dev)`), so the drafter's coupled sampler and the target's
  sampler run on the same GPU, with the same seed and counters. The flag is a host static read once, no per-device state.
  `sub_to_id` is a device pointer of the drafter's card, as before.
- **`--batch-mtp`** slot drafters never use the coupled sampler; `Verifier::sample_rows` picks per slot with
  `sample_tokens`, so the flag changes the target's pick there too and gives no gain. Not tested upstream either.
- **Elastic cache:** unrelated to sampling; no interaction.

## How to measure

Question: does coupled + Gumbel raise accepted drafts and output t/s at temperature 0.7 and 1.0, with our MTP drafts
(`--spec 4 --mtp mtp/rt --spec-min-p 0.5`) on the 3060 + 2080 Ti layout?

1. **One process per arm.** The two switches are read once into process statics, so an A/B inside one server is
   impossible. Start a fresh server for each arm with the same command line and the same environment except:
   - A: default (neither variable);
   - B: `STRATA_SPEC_COUPLED=1` (isolates the coupling);
   - C: `STRATA_SPEC_COUPLED=1 STRATA_SPEC_GUMBEL=1` (the candidate);
   - D (control, optional): `STRATA_SPEC_GUMBEL=1` alone, expected equal to A.
   Run the arms in the order A C B A C B (or ABBA blocks), not all of A and then all of C, to cancel thermal and
   background drift. Keep `STRATA_DECODE_TIMING=1` on in all arms. Do not build or run while
   `D:\ds4_work\strata\lab_locks\bench.lock` exists.
2. **Requests.** Many requests per arm and temperature: at least 30 (the upstream run used 12-13 and a standard error of
   0.6-0.8 t/s). Temperatures 1.0 and 0.7, `top_p 0.95`, `top_k 20`, thinking off, a fresh prefix in every request
   (so the prompt cache does not hide the prompt), the same prompt set across arms, a few hundred output tokens, fixed
   `max_tokens`. Use different seeds per request, the same seed list in every arm. No repetition penalties
   (`penalty_last_n` 0), or the sampled drafts and `--pipeline-windows` behave differently (penalties also serialize the
   pipeline).
3. **What to record per request**: from the serve `DONE` line, drafts accepted and offered; from the stderr line
   `request N, ... accepted X of Y drafts`, windows; from `STRATA_DECODE_TIMING=1`, ms per window and the draft share;
   output tokens per second. Derive tokens per window and accepted drafts per request. Report mean and standard error
   per arm and temperature, and the A-to-C difference with its error.
4. **Greedy check (once)**: temperature 0, the same 5 prompts, arm A against arm C (and C against B): token streams must
   be byte-identical. Any difference is a bug in this port.
5. **Correctness of sampling**: same seed twice on arm C gives the same text; three seeds give three different texts;
   `sampler_parity` (ctest) passes its three new Gumbel checks on the machine's card.
6. **Read the result per temperature.** The coupling pays more where distributions are broad (temperature 1.0); at 0.7
   a small or null gain is plausible. Also compare B against A: upstream saw coupling alone at or slightly under the
   default with one draft head.

## Risks

- **Not compiled, not run here.** The patch applied to main without conflicts, but nothing has been built. First run:
  `sampler_parity` on the 2080 Ti (sm_75) and the 3060 (sm_86), then the greedy byte comparison.
- **Measured only elsewhere.** The +7 points of acceptance and +9% t/s are from a Strix Halo iGPU (ROCm), one model,
  one drafter, 12-13 requests per arm. No NVIDIA card was used by the author. Our drafter, quantisation, `--draft-vocab`
  and `--spec-min-p 0.5` differ; expect another number, possibly zero.
- **The pipeline is lost for sampled requests** if `STRATA_SPEC_COUPLED=1` is on (serial fallback, stderr says so once).
  On the two-card layout that can cost more than the acceptance gain. Compare against the pipelined baseline (arm A with
  `--pipeline-windows 2`), not only against a serial A.
- **Different text for the same seed.** The Gumbel pick is a different stream than the inverse CDF, so a seed no longer
  reproduces the text of the default build, and `STRATA_SPEC_GUMBEL=1` is a separate reproducibility regime. Within the
  flag, text stays a pure function of seed, position and the token ids.
- **Extra cost in the tail.** The pick does one `exp`/`log` per kept candidate in double precision on one lane of the
  warp (up to 64 candidates): upstream measured about 0.3 ms per 64 ms window for the whole coupled draft. On the 2080 Ti
  double precision is slow (1/32 rate); expect a small but visible cost and check ms per window.
- **Hash quality.** The noise is a splitmix64 hash, not Philox; independent enough for sampling, not for cryptography.
  Ties and `u` near 0 or 1 are excluded by construction (open interval), ratios stay finite.
- **Env read once.** Changing the variable needs a server restart; a half-set environment (COUPLED without the drafter
  being able to run coupled, see the `setup_coupled` message on stderr) silently leaves the drafts on argmax.
- **Draft vocab subset.** The coupled draft normalises over its subset while the target normalises over the full
  vocabulary; the noise is the same for shared tokens, but the pick still differs when the target's winner is outside the
  subset, as in the default coupled mode.
