# L3 archive

The 11 historical L3 run directories previously in
`/workspace/upenn/block-stack-world-model/analysis/L3` now live in
`/workspace/artifacts/L3`. The old analysis/L3 link and historical L3 run aliases have been removed.
The local `runs` symlink points to this machine-local archive; checkpoints
and logs are not committed to Git. Existing organization/run manifests retain
their historical paths and provenance.

`index.json` maps each run to its configuration, archive directory, source
revision, and recorded source hashes. `source/` contains the complete source
file set recorded by each run, reconstructed from its clean Git revision and
verified against every recorded SHA-256. The two source versions are historical
snapshots, not a newly combined implementation. They include shared L2/L4 code
because the same trainer implements all rungs. Source copies also live in the
external archive. Datasets, environments, assets, and checkpoints are external
dependencies; these snapshots are not standalone installations.

Before relocation, a /proc scan found no process command, working directory,
or open file descriptor referencing the L3 archive. All 2,931 existing files
retained their inode and size after the same-filesystem rename, and were checked again on relocation to `/workspace/artifacts/L3`.
Old run paths are intentionally no longer available; `relocation.json` in the
external archive records removed aliases. No training processes were stopped or
restarted. Active source worktrees and environments were left in place.
