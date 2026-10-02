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

Source baseline: Strata **0.1.31**, commit
`9259cad4cfa3543cd3b8decab5962672b968c649`, plus the documented
[patches](docs/sm75/PATCHES.md). Upstream history is retained and the tested
snapshot is imported as a new commit; published history is not rewritten.
The snapshot's older version is an explicit validation choice.

The top-k radix/register kernel itself is upstream work. Our fix changes the
prefill dispatch to use the true active block bound while preserving the score
stride. We do not claim authorship of the whole kernel or attribute all model
performance gains to our changes.

[Full research history and evidence](https://github.com/imanu86/moe-aggressive-commit/tree/research/ds4-iq1-subbit-tier-planner/docs/porto/strata_adattivo).
