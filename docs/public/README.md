# CellState Agent

CellState Agent turns a scientific request into a schema-validated plan, presents an explicit approval gate, routes the approved plan through a capability registry, and executes deterministic scientific code. It supports replicate-aware aggregation, pseudobulk differential expression with DESeq2, design-estimability checks that block invalid comparisons, opt-in exploratory leave-one-dataset-out robustness analysis, signed TF activity and cross-resource consensus, and regulatory-network analysis.

Each run produces provenance-bearing EvidenceBundles. A Critic independently checks evidence and limitations; a constrained Interpreter synthesizes only exact evidence references. Structured-output validation permits bounded repair retries. Authoritative JSON reports retain the complete audit record, while PDF reports and figures are presentation layers. Cache identity covers scientific inputs and configuration so restored results remain attributable.

The architecture is extensible through capability specifications, schemas, adapters, deterministic nodes, evidence adapters, and presentation components. Public tests use synthetic, non-biological fixtures to validate planning, approval, routing, replication safeguards, confounding and estimability failures, DESeq2 integration, LODO directionality, TF agreement/discordance, plotting, reasoning, cache restoration, and reporting.

The project distributes no research or patient data. Users provide their own readable atlas through `CELLSTATE_ATLAS_PATH` and obtain TF resources from their upstream sources. External runtime requirements are R, DESeq2, and Matrix; apeglm is optional. Seurat is needed only for reconstruction, and dorothea is needed only for local resource preparation when that workflow is used.

This software is research tooling. It makes no clinical, mechanistic, or biological-performance claim.
