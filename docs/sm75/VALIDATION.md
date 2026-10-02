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
uses a fixed requested expert budget of 5,400; actual slots must match across
all arms. The expert profile remains the same. `--adapt-swaps 0`,
`--pcie-frac 0`, `STRATA_IQ_MT_MIN=1`, suffix drafting off, MTP maximum 4 and
serial intermediate quantization hold the compute path stable. Prefix reuse
is required for teacher forcing; the daily retains automatic elastic cache.
No daily configuration is written by the tool.

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

After any failure, retain the output directory and investigate. Do not silently
relax a gate, label a partial run PASS, or promote unrelated performance claims.
