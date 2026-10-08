# Working on Strata-2080Ti

Work on main and preserve local changes. Use README.md and docs/sm75 for this
validated snapshot. The inherited installer downloads upstream engines; this
fork builds the verified Daily/upstream baseline with the CMake instructions in
[docs/DAILY_UPSTREAM_BASELINE_2026-10-08.md](docs/DAILY_UPSTREAM_BASELINE_2026-10-08.md).

Do not launch GPU/model tests or CUDA builds for documentation/publication work.
Publication checks use CPU only and do not start the engine. Measurements must
name hardware, actual context, cache and validation limits. The measured card
is a modified 22 GB RTX 2080 Ti; stock 11 GB is unvalidated.

Keep credentials in environment and local configs out of Git. Keep the server
on loopback unless the user explicitly configures network access and authentication.
The source manifest describes the original tested daily. Later core changes
need new documented validation; do not silently replace provenance hashes.
Preserve upstream MIT notices and community attribution. Keep claims measured
and plain. Engine architecture and settings are documented in docs/DETAILS.md.
Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**: check the PC, pick the model by RAM, run setup non-interactively,
start and verify the server, and connect the user's apps. Never expose the server beyond `127.0.0.1` without
`--api-key`. As an alternative to shell commands, Strata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md))
offers the same steps as tools.

## Working on the code

- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md).
- Setup's own tests run without a GPU or downloads: `python tools/test_setup_<name>.py` (for example
  `tools/test_setup_amd.py`, `tools/test_setup_choices.py`).
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claims without a
  measurement.

## Contributing a change or report

- Search the open issues and pull requests first, and add to a thread that already covers your point.
- Open an issue with the form that fits (bug report, feature request or question).
- One change per pull request. Say what it changes and what it leaves alone.
- A new feature is opt-in, and the default path stays byte-identical to the last release. Say how you checked.
- Build every backend a file touches (CUDA, HIP, SYCL) before asking for review.
- Change a default only where you measured it faster, and show the numbers with what they were measured on.
- A report from hardware the maintainers do not have is welcome. Follow
  [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md), compare against a same-day run of the build you are
  testing, and say what you did not test.
- Open test requests and the hardware that is wanted are listed in [docs/TEST_REQUESTS.md](docs/TEST_REQUESTS.md).
