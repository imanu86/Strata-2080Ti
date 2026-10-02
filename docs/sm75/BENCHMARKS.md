# Measured daily:2 October2026

Modified RTX2080Ti22GB, Ryzen7 5800X,96GB RAM, Windows/CUDA12.6, PCIe3.0,
IQ3_XXS, KV262144/int8/resident32768, reserve1393MiB, automatic elastic cache,
automatic prefill, spec4/min-p0.5, MTP. Normal desktop activity remained;
no artificial saturation. JumpConnect sampled CPU was zero in retained runs.

## Same-executable top-k A/B/A

Prompts131248 and131239 per run; the second reused131072. Each generated1024
tokens. Only the dispatch override changed.

| Run | Dispatch | Fresh prefill tokens/s | TTFT s | Aggregate decode tokens/s | Initial slots |
|---|---|---:|---:|---:|---:|
|A1|capacity|831.20|158.509|38.251|6537|
|B|active|886.38|148.664|39.272|6524|
|A2|capacity|828.26|159.092|39.293|6872|

Mean reference829.73, candidate886.38: **+6.83%**, reference spread0.35%.
Mean TTFT158.801→148.664s:10.137s saved. Previous exact daily independently
measured827.53prefill/39.048decode. This is one retained A/B/A; elastic slots
and MTP output vary. B/A2 decode is virtually identical. See `validation.json`.

## Retrieval and deep decode

Needles at10/50/90% were all FOUND, real tokens30742/30742/30743. The tool's
"32k" label estimates characters; use these actual counts.

The exact250k fixture: **250022 tokens**, zero reused, needle at50%, correct
13-token answer,391472.7ms prefill (**638.670tokens/s**). That short answer is
not the decode benchmark.

The follow-up:250091 prompt,250015 reused, **1024 generated to finish length**,
251115 total. Decode26110.5ms = **39.217939tokens/s**;76 fresh prompt tokens in
1317ms. MTP583/944 drafts accepted, expert cache88.3%, KV blocks95.09% fromVRAM.

131k/250k use different prompts and acceptance, so these are observed context
points rather than a controlled sweep of one prompt. The previous250k result
442.4prefill came from another time window: the difference from638.7 cannot
be attributed entirely to the top-k fix, which affects only the initial135k.

## Numerical limits

- 16top-k cases:stride65538, batch1/8/255/256/257, partial tails, zero/tied scores,
  register limit33792/fallback33793, context up to262144. IDs bitwise equal.
- First131k logits:248320 finite values, same argmax1596. Even A/A model logits
  differ: this does not validate general model quality.
- Retrieval/deep-decode logit captures finite. HC/GDN FP16 changes precision;
  numerical checks and retrieval coverage are limited validation.
- Physical DRAM throughput not measured: Nsight Compute reported
  ERR_NVGPUCTRPERM. NVML memory activity is not measured GB/s.

## Reproduction

Original commands, fixture hashes, native logs and tools:
[full research record](https://github.com/imanu86/moe-aggressive-commit/blob/research/ds4-iq1-subbit-tier-planner/docs/porto/strata_adattivo/RISULTATI_TOPK_2_OTTOBRE.md).

`tests/sm75/topk_active.cpp` is the deterministic native bench. Link against
the fork's kernels/core and CUDA runtime; pass context, query count, score mode:

After an explicit engine build, from an x64 VS2022 Developer PowerShell:

```powershell
cl /nologo /std:c++20 /EHsc /O2 /MD /Iinclude /I"$env:CUDA_PATH/include" `
  tests/sm75/topk_active.cpp /Fo:build-sm75/topk_active.obj /Fe:build-sm75/topk_active.exe `
  /link build-sm75/strata_kernels.lib build-sm75/strata_core.lib `
  "$env:CUDA_PATH/lib/x64/cudart.lib" "$env:CUDA_PATH/lib/x64/cudadevrt.lib"
```

Run it from `build-sm75/` with the CUDA runtime on PATH:

```text
topk_active.exe 131072 256 0
topk_active.exe 135164 256 0
topk_active.exe 135168 256 0
topk_active.exe 131072 256 1
```

Mode0=random,1=zero,2=ties,3=bound0,4=bound above stride. Numerical/model tests
consume GPU time and are explicit actions. No new GPU test was run for
publication. The previously installed exact daily passed startup in42.84s at
KV262144, then stopped. Publication CI checks source/config only.
