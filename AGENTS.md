# Working on Strata-2080Ti

Work on main and preserve local changes. Use README.md and docs/sm75 for this
validated snapshot. The inherited installer downloads upstream engines; this
fork uses its release or explicit tools/sm75/build.py builder.

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
