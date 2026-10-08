> **ARCHIVIO / NON ATTIVO:** evidenza storica del fork precedente. Questo documento non descrive il runtime corrente; vedere [baseline verificata](../DAILY_UPSTREAM_BASELINE_2026-10-08.md).

# Native IQ model validation

Use `tools/sm75/validate_model.py` for a sequential A/B/A comparison of a
reference engine, a candidate, and a second reference run. It downloads
nothing and defaults to a dry run. GPU/model execution requires `--run`.
An existing engine or output directory causes refusal.

The method follows upstream's teacher-forced [cache comparison](https://github.com/Niko1221/Strata/blob/36fa455e579b23a9c909c2c6fe1bddd9e51cb8ca/bench/results/2026-09-27-cache-parity/README.md)
and [reproducibility guidance](https://github.com/Niko1221/Strata/blob/36fa455e579b23a9c909c2c6fe1bddd9e51cb8ca/docs/DETAILS.md).
The public upstream does not provide a single mandatory corpus or universal
quality threshold. Our corpus and gates are explicitly local.

## What is measured

- Frozen code, document and chat texts: 512 / 1024 / 1024 input tokens,
  2,557 scored next-token positions per arm. Token IDs and tokenizer hashes
  are saved before any engine starts.
- Growing prefixes with one generated token each. The prediction is discarded;
  the next input always comes from the fixed reference. The native-IQ path
  bypasses the older per-token `--dump-logits` loop, so this tool uses the
  fork's `STRATA_DUMP_FIRST_LOGITS` first-window diagnostic. Every row must
  exist, have the correct byte count and contain only finite values.
- Top-1 agreement, KL(reference || candidate), perplexity and paired NLL.
  The final input row has no known next token and is excluded from scoring.
  Reference A1/A2 measures baseline variation.
- Six complete JSON answers with programmatic oracles, plus three records
  retrieved from an exact 131,072-token prompt. A correct-looking answer
  truncated at the output cap fails. These are limited checks, not a general
  model capability benchmark.

Teacher prefixes exercise decode; the separate deep oracle exercises batched
prefill. This does not isolate a prefill kernel from all other engine changes.
The three deep-request speeds are diagnostic samples, not a median performance
benchmark with three repetitions per configuration.

## Controlled laboratory settings

KV capacity remains 262,144, int8, with 32,768 resident cells. The laboratory
uses a fixed requested expert budget of 4,600; actual slots must match across
all arms. The expert profile remains the same. `--adapt-swaps 0`,
`--pcie-frac 0`, `STRATA_IQ_MT_MIN=1`, suffix drafting off, MTP maximum 4 and
serial intermediate quantization hold the compute path stable. Prefix reuse
is required for teacher forcing; the daily retains automatic elastic cache.
No daily configuration is written by the tool.
The laboratory fixes prefill chunks at 2,048 tokens. Automatic 8,192-token
chunks left less than the declared VRAM margin at startup even after lowering
the requested cache budget to 4,600 (6,167 actual slots, 1,008 MiB free).
Both startup-only attempts are retained; neither scored a model token.

Local gates are declared in `POLICY` and saved before execution. Candidate NLL
upper bound must not exceed 0.01 nats/token plus the reference-repeat noise
bound. Median/p99 KL floors are 0.001/0.01, or three times reference noise
when larger. Top-1 agreement must stay within one percentage point of the
reference repeat. Each text and the aggregate must pass. The approximate
uncertainty interval uses contiguous 32-token blocks; it does not establish
population-level model quality. Every final-answer oracle must also pass.

## Run

With the dependencies from `requirements.txt` plus NumPy and psutil available:

```powershell
python tools/sm75/test_validate_model.py
python tools/sm75/validate_model.py --config PATH_TO_LOCAL_CONFIG --reference OLD_EXE --candidate NEW_EXE --out NEW_PRIVATE_DIRECTORY
# Review the dry-run plan, then repeat with --run.
```

Raw logits consume about 7.6 GB for all three arms and remain outside Git.
`plan.json`, `comparison.json`, `results.json`, token IDs and engine logs
provide reproducible evidence. The default total deadline is 30 minutes.
Only owned engines are closed. RAM admission requires 16 GiB available,
with an 8 GiB floor checked during execution. No artificial load is generated.
After startup the laboratory requires at least 768 MiB of reported free VRAM.
The initial gate incorrectly reused the 1,393 MiB cache-sizing reserve as a
post-buffer admission threshold. Three startup-only attempts stopped before
any prompt: 5,400/auto (7,249 slots, 1,257 MiB), 4,600/auto (6,167 slots,
1,008 MiB), and 4,600/2048 (6,167 slots, 1,011 MiB). Historical daily runs
already had lower post-start margins. This admission correction is explicit,
precedes all scored tokens, and does not change any numerical quality gate.
Those attempts remain failures of the original startup criterion and are not
completed model comparisons. The configured cache reserve remains 1,393 MiB.

After any failure, retain the output directory and investigate. Do not silently
relax a gate, label a partial run PASS, or promote unrelated performance claims.

## Daily 0.1.39 - 5 ottobre 2026

Upstream v0.1.39 (`6f32ec0`) unito nel fork (`73cb2d4`) con port SM75, cache elastica a due zone, top-k #603, tail #614 e le opzioni di laboratorio di questa sessione spente. Eseguibile shipping validato sulla configurazione del Daily: a 131k L01 51,4 / L02 44,2 t/s contro 38,7 / 38,0 del Daily 0.1.38; fedelta' in teacher forcing KL 0,0004 / 0,0020 / 0,0001 (A/A 0,0004 / 0,0005 / 0,0001), perplessita' -0,16% / -2,10% / +1,07% senza spostamento sistematico. CTest 8/8, pool CPU 64 casi bitwise, test server 262/271 (7 saltati, 2 errori non del fork). Dettagli: `validation.json`.

## Daily 0.1.39-r2 - 6 ottobre 2026

Sorgente del fork `14433b0` (upstream v0.1.39 `6f32ec0` + port SM75): PR #904, Encapsulate, merge v2 (#910), kernel Q2_0 esatto, #930, #949, #965 e diagnostica PLE, tutti identici bit per bit a posizionamento fisso (teacher forcing dell'eseguibile shipping uguale a `TFX-enc`). Configurazione: nucleo elastico adattivo, DMA, `--pool-tasks 96`, tabella PLE su D:, cache di righe PLE 16M. A 131k L01 53,1 / L02 45,8 t/s (0.1.39 r1: 51,4 / 44,2). Benchmark a turni fino a ~130k: costo per finestra -10% (45,4 contro 50,5 ms), +3,4% di t/s sui turni a testo normale, nessun blocco del DMA. CTest 11/11, pool CPU bitwise sul GGUF reale, test server 276/278 (2 errori non del fork). Dettagli: `validation.json`.
