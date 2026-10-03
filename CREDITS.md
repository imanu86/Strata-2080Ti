# Credits and provenance

- **Niko1221 and the Strata contributors**: original engine, server, QSA
  kernels, numerical checks, installer and docs. [Strata](https://github.com/Niko1221/Strata), MIT.
- **jklnvlink and Strata-2080ti contributors**: SM75 HC/GDN FP16 and vector
  dequantization work ported into the daily. [Community fork](https://github.com/jklnvlink/Strata-2080ti).
- **PR315 authors/contributors**: HC read implementation and numerical gate.
  [PR315](https://github.com/Niko1221/Strata/pull/315).
- **Local imanu86/moe-aggressive-commit contributors**: elastic cache,
  checkpoint/grid alignment, grouped expert work, async commit integration,
  combined daily and validation. Author identities and Co-Authored-By trailers
  are preserved in the research history.
- **imanu86 / Codex**: integration, benchmark harnesses, the active-bound
  prefill top-k dispatcher fix and this publication.
- **ggml/llama.cpp contributors**: quantization formats, codebooks and MMQ,
  with vendored license notices preserved under `third_party/`.

Original published snapshot baseline: Strata **0.1.31**, commit
`9259cad4cfa3543cd3b8decab5962672b968c649`, plus the documented
[patches](docs/sm75/PATCHES.md). Upstream history is retained and the tested
snapshot is imported as a new commit; published history is not rewritten.
That historical snapshot and its evidence remain available separately from current main.

The top-k radix/register kernel itself is upstream work. Our fix changes the
prefill dispatch to use the true active block bound while preserving the score
stride. We do not claim authorship of the whole kernel or attribute all model
performance gains to our changes.

[Full research history and evidence](https://github.com/imanu86/moe-aggressive-commit/tree/research/ds4-iq1-subbit-tier-planner/docs/porto/strata_adattivo).

Current main source baseline: Strata0.1.36, `36fa455e579b23a9c909c2c6fe1bddd9e51cb8ca`. The0.1.31 manifest remains
as historical provenance. Community [PR463](https://github.com/Niko1221/Strata/pull/463)
provides the pending-copy ordering correction and [PR500](https://github.com/Niko1221/Strata/pull/500)
the parallel intermediate quantization design; local integration adds opt-in
dispatch and the real-GGUF parity harness. Their published author history and
upstream licenses are preserved. No reported speedup from another machine is
claimed for this fork.

Current0.1.38 integration: upstream `99f3dbd0b21d1401b3769e0c0d963913607f380b`. Original elastic expert-cache design
by Claude Opus; Codex port/guards and validation. Tail checkpoint: Codex,
[PR614](https://github.com/Niko1221/Strata/pull/614). Wide top-k: Chanyeong Lim
(asp345), co-authored by Claude Opus5.5, [PR603](https://github.com/Niko1221/Strata/pull/603),
head6ba96984ac8a5f019edd84d7a83c278e6a1d363d. Their work is credited explicitly;
the old grid is now historical and is absent from the operative source.
