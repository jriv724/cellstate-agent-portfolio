# GZMB CD8 T-cell evidence snapshot

This directory contains the complete 31-artifact evidence
snapshot for the NBM versus NDMM CellState Agent demonstration.

- Source EvidenceBundle: `e70fcedaf8e1250049a18c0560d9572dd80f8e1c4abe9705b1c517b23e3d09a5`
- Deterministic values and result tables are preserved.
- Biological replicate identifiers are replaced consistently with
  snapshot-specific `BIOREP_` hashes.
- Private cluster paths are removed.
- The original atlas and cell-level expression matrix are not included.
- The pseudobulk input matrix is included.
- Normalized regulon edges are included for reproducibility; redistribution
  licensing must be confirmed before making the repository public.
- Cache manifests are retained as historical provenance. Their original
  file hashes refer to the unsanitized private artifacts.

The public artifact manifest records both source and exported checksums.
