# Reproduction and provenance

The patches target Strata 0.1.38 commit `99f3dbd0b21d1401b3769e0c0d963913607f380b` and GGML pin `3cf03257f219afbe7334045ff7c6a06ac68c627d`. Apply with `git -c core.autocrlf=false apply`. `elastic/elastic.patch` and `tail/checkpoint-tail.patch` are independent shipping contributions; QA readout patches are diagnostics only.

The full elastic numeric gate used `elastic-full-gate-v1.patch`; subsequent CLI/help, supported-mode and opt-in wait refinements are recorded separately and checked on the final pure build. The initial incorrect unit-test allocation label is preserved and explained in `full-gate-v1.json`.

Scripts preserve the workstation paths and run folders used for the recorded tests. For reproduction, adapt paths, install NumPy/psutil, obtain the same model/tokenizer/pack/profile locally, recreate the pinned plain source exports and build artifacts, and use a fresh output folder (runners deliberately refuse overwriting evidence). `validate_model.py` supplies the corpus and local numerical gate; its CLI is not used by these runners. Token ID fixtures and complete result JSON are included. The original grid results are retained as a failed reference, not as a passing contribution.

Operational results contain an initial malformed dialogue attempt whose 131k/250k guide outputs stopped after 48 tokens. The corrected `dialogue-r1` includes assistant turn delimiters, stronger long-output instructions and distinct subsequent user markers; both long outputs reach 1024 tokens. This does not establish that missing delimiters alone caused the first result.

No weights, executables or multi-GB raw logits are included. Raw dump hashes and sizes remain in the result metadata; `manifest.json` attests the exact published bytes. This is local regression evidence on a modified RTX 2080 Ti 22 GB, not a universal upstream quality standard or a Daily speed record.
