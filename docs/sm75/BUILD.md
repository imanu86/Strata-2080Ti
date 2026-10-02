# Build the SM75 snapshot

The measured daily used Windows, CUDA **12.6**, MSVC **14.44.35207** (VS2022
Community), Python3.12, Ninja, Release, CUDA targets`75;86`, shared CUDA runtime
and `STRATA_PORTABLE=OFF`.

Open an **x64 Native Tools / Developer PowerShell for VS2022**, put CUDA12.6's
`bin` on PATH, install `requirements.txt` in `.venv`, then:

```powershell
.\.venv\Scripts\python.exe tools/sm75/build.py --jobs 2
```

This produces `build-sm75/strata.exe`. Generate your config with
`tools/sm75/configure.py --exe build-sm75/strata.exe ...`.
`--ggml-dir` selects an existing pinned llama.cpp source directory; otherwise
CMake fetches `3cf03257f219afbe7334045ff7c6a06ac68c627d`, recorded in
`third_party/ggml/VERSION.txt`. `--arch` changes the CUDA architecture list.

The builder never downloads model weights or starts the engine. Calling it
does compile CUDA code and use CPU/RAM; it is an explicit user action.
It was **not run for publication**. The release engine is the exact previously
tested daily, assembled from compatible cached base/HC/vector/PR315/top-k
objects. Their modified source is present here without publication-only CUDA
or C++ changes. Fresh toolchains/dependencies/linking can change the binary hash;
portability beyond the measured machine remains unvalidated.

## CPU-only source check

```powershell
python tools/sm75/check_source.py
```

It checks the source file hashes from `docs/sm75/source-manifest.json` and
the absence of CH16/BPW switches. Publication overrides cover README, launcher,
ignore rules and byte-preserving Git attributes; original README is kept in
`docs/UPSTREAM_README.md`. These checks never launch CUDA or the model.

See [BENCHMARKS.md](BENCHMARKS.md) for numerical targets and model evidence.
GPU checks are explicit and are not part of the publication CI workflow.
