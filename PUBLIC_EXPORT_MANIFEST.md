# Public export manifest

`public_export/manifest.toml` is the sole allowlist. `scripts/build_public_export.py` expands approved patterns into sorted individual source/destination mappings, validates every file, and defaults to a no-write dry run. Writing requires `--write`, a fresh empty destination outside the private source, and subsequent checksum verification. It never invokes Git or rsync.

The release includes the production package architecture, scientific nodes and R runtimes, schemas, adapters, planning, reasoning, reporting, plotting, reproducibility APIs, resource acquisition tooling, public documentation, and synthetic tests. It excludes obsolete applications, private history and infrastructure, generated outputs, staging data, downloaded TF resources, biological/model binaries, and unreviewed assets.

`src/cellstate/data_interchange/reconstruct_seurat.R` is temporarily excluded as `DEFERRED_NFS_REVIEW`; the safe Python data-interchange API remains included. Software licensing remains an owner decision and blocks publication. Upstream TF data licensing is separate and redistribution is not asserted.
