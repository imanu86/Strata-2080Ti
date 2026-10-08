> **ARCHIVIO / NON ATTIVO:** evidenza storica del fork precedente. Questo documento non descrive il runtime corrente; vedere [baseline verificata](../DAILY_UPSTREAM_BASELINE_2026-10-08.md).

# Physical GPU/CPU expert blocks — research, 4 October 2026

The offline `hybrid_expert_probe` partitions each 640-neuron expert into ten
64-neuron blocks. Blocks 0, 4 and 9 reside on GPU; the other seven run on CPU.
It copies the original quantized gate/up rows and down blocks without
requantization or pruning. The native CPU pool now accepts bounded partial
widths; its normal 640-neuron path remains the operational default.

**The partition preserves the contributions, but did not demonstrate a speedup
at equal active GPU weight bytes.** No server dispatch/cache integration,
Daily deployment, binary release change, or new throughput claim is made.

## Measurements

Modified 22 GB RTX 2080 Ti, Ryzen 5800X, 96 GiB RAM, CUDA 12.6/SM75,
Qwen3.8-Flash-Next IQ3_XXS. Three captured texts, prefixes 86/85/102 tokens,
ten subsequent inputs on layers 0/1/35/36: 120 distinct single-token cases.
No KV cache is allocated in this microbenchmark. This is not 250k-depth testing.

Each case compares three complete GPU experts plus seven CPU experts against
30% of all ten experts on GPU plus their CPU complements. Both read three
expert-equivalents of GPU weights and seven of CPU weights. Scratch/metadata
allocations differ. All weights are preloaded for the case; the full-GPU subset
has the three highest current routing weights. Original online tiers are ignored.

After an initial variable run, the confirmation alternates AB/BA each round,
nine rounds of 60 iterations. Values are means of the per-case median latencies.

| Input | 3 GPU + 7 CPU, us | GPU30/CPU70, us | Hybrid latency change |
|---|---:|---:|---:|
| Code | 947.98 | 970.39 | +2.36% |
| Explanation | 865.64 | 876.67 | +1.27% |
| Reasoning | 720.04 | 730.99 | +1.52% |
| All 120 cases | 844.55 | 859.35 | +1.75% |

The hybrid wins 47/120 cases. Timing remains variable: median per-case
max/min round ratios are 1.48 and 1.56. The small aggregate difference is not
a precise universal slowdown estimate; there is no reproducible acceleration
to promote. CPU/GPU work, input D2H, activation quantization, CPU weighted sum,
partial-result H2D, GPU merge and final synchronization are timed. Packing,
weight uploads, allocation and warm-up are excluded. These warm working sets
do not model the full model's cache pressure or an online migration policy.

Weight reconstruction is bitwise exact. Partition sum versus complete expert
on the **same** backend has maximum relative L2 2.11e-7 on CPU and 9.19e-8 on GPU.
The mixed result differs from all-GPU by 0.8495% mean / 1.2687% maximum, compared
with 1.0330% mean / 1.5474% maximum for all-CPU versus all-GPU. Those are local
backend arithmetic differences, not a guarantee of unchanged generated tokens.
No full conversation quality test was performed with hybrid dispatch.

The pool check passes 192 scenarios across widths 192/448/640, tokens 1..8,
host participation on/off, serial/parallel intermediate quantization, and the
96/97-job boundary. Four-format GPU smoke also passes with parallel quantization.

[Complete raw rounds, provenance and limits](https://github.com/imanu86/moe-aggressive-commit/blob/4574caff1b213efc8a4c9983c17cc7d7603c2b60/docs/porto/strata_adattivo/corse_2080ti/20261004_hybrid_blocks/RESULTS.md).
The probe executable SHA256 is
`daf1015b001ccb58419c4433ea2a51cb97af03d92cee174c5946aab547281199`.

## Explicit reproduction

In an existing CUDA/SM75 build configured using [BUILD.md](BUILD.md), enable
`STRATA_BUILD_NEURON_REPLAY=ON` and `STRATA_PARITY_POOL_QUANT=ON`. Build targets
`hybrid_expert_probe` and `pool_quant_parity`. No engine server is required.

Convert a previously recorded SNTv0001 trace using Python with numpy:

```text
python tools/sm75/hybrid_probe_input.py --trace target.snt --first-position 86 --positions 10 --output code.shb
```

Use the actual first continuation position of that trace, not always 86.
Outputs must not already exist. In the measured environment:

```powershell
$env:STRATA_IQ_MT_MIN = '1'
$env:STRATA_PARALLEL_INTERMEDIATE_QUANT = '0'
hybrid_expert_probe.exe code.shb model-shard1.gguf output.jsonl 60 9 paired
pool_quant_parity.exe model-shard1.gguf 192
pool_quant_parity.exe model-shard1.gguf 448
pool_quant_parity.exe model-shard1.gguf 640
```

The probe uses seven pinned CPU workers plus the host. It bounds inputs to
120 cases and is a standalone diagnostic. `SHBv0001` has little-endian u32
case count/H=2560/K=10/FF=640, then each case's i32 layer/position/token,
2560 f32 input values, ten i32 expert IDs, ten f32 routing weights and ten
i32 original tiers. The converter reads accepted inputs only and reproduces
the three measured datasets byte for byte.
