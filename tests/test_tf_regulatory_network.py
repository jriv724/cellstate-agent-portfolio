import json
import tempfile
import unittest
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from cellstate.cache import load_complete_manifest
from cellstate.capability_specs import CAPABILITY_SPECS_BY_ID
from cellstate.context import AnalysisContext
from cellstate.nodes.tf_regulatory_network import (
    CACHE_SCHEMA_VERSION, CAPABILITY_ID, IMPLEMENTATION_VERSION, NODE_VERSION,
    _bh, _build_network, calculate_tf_regulatory_network, file_hash, normalize_resource,
    run_tf_regulatory_network, validate_feature_programs,
)
from cellstate.schemas.common import CapabilityStatus
from cellstate.schemas.tf_regulatory_network import TFRegulatoryNetworkInput


FIXTURE = Path(__file__).parent / "fixtures/tf_regulatory_network"


def program_rows(source, source_hash, program_id="P1", cell_state="CD8 T",
                 contrast="A_vs_B", feature_direction="increased", genes=None):
    genes = genes or ["G1", "G2", "G3", "G4", "G5"]
    rows = []
    for gene in genes:
        rows.append({
            "program_id": program_id, "feature_set_id": f"FS-{program_id}",
            "feature_id": gene, "feature_type": "gene", "cell_state": cell_state,
            "condition_a": "A", "condition_b": "B", "contrast": contrast,
            "contrast_direction": "A relative to B", "feature_direction": feature_direction,
            "source_capability_id": "CAP-UP-001", "source_capability_version": "1.2.0",
            "source_cache_key": "upstream-key", "source_artifact_path": str(source),
            "source_artifact_hash": source_hash, "feature_selection_method": "frozen fixture",
            "feature_selection_thresholds": '{"fdr":0.1}',
            "feature_selection_provenance": '{"approved":true}',
            "effect_size": 1.0, "adjusted_p_value": 0.01,
        })
    return rows


class TFRegulatoryNetworkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "upstream.tsv"
        self.source.write_text("feature_id\nG1\nG2\nG3\nG4\nG5\n")
        self.program = self.root / "programs.tsv"
        pd.DataFrame(program_rows(self.source, file_hash(self.source))).to_csv(
            self.program, sep="\t", index=False)
        self.request = TFRegulatoryNetworkInput(
            self.program, FIXTURE / "background.tsv", FIXTURE / "dorothea.tsv",
            FIXTURE / "collectri.tsv")

    def tearDown(self):
        self.tmp.cleanup()

    def prepared(self, request=None):
        request = request or self.request
        programs, excluded, background = validate_feature_programs(request)
        resources, qcs = [], []
        for name, path in (("DoRothEA", request.dorothea_path),
                           ("CollecTRI", request.collectri_path)):
            table = pd.read_csv(path, sep="\t")
            normalized, qc = normalize_resource(table, name, request)
            resources.append(normalized); qcs.append(qc)
        tables, warnings, summary = calculate_tf_regulatory_network(
            programs, background, pd.concat(resources), pd.DataFrame(qcs),
            excluded, request)
        return programs, tables, warnings, summary

    def test_valid_single_program_exact_fisher_and_bh(self):
        programs, tables, warnings, _ = self.prepared()
        self.assertEqual(programs.feature_id.tolist(), ["G1", "G2", "G3", "G4", "G5"])
        row = tables["tf_enrichment_by_database"].query(
            "database == 'DoRothEA' and tf == 'TFBOTH'").iloc[0]
        odds, p = fisher_exact([[5, 0], [0, 15]], alternative="greater")
        self.assertEqual(row.odds_ratio, odds)
        self.assertAlmostEqual(row.p_value, p)
        self.assertTrue(row.significant)
        np.testing.assert_allclose(_bh(pd.Series([.01, .04, .03])), [.03, .04, .04])
        self.assertTrue(any(x.code == "TF_CONVERGENCE_NOT_APPLICABLE" for x in warnings))
        self.assertTrue(tables["tf_regulatory_convergence"].empty)

    def test_consensus_and_network_exact_semantics(self):
        _, tables, _, _ = self.prepared()
        consensus = tables["consensus_tf_hits"].set_index("tf")
        self.assertTrue(bool(consensus.loc["TFBOTH", "consensus_status"]))
        self.assertFalse(bool(consensus.loc["TFDOR", "consensus_status"]))
        self.assertFalse(bool(consensus.loc["TFCOL", "consensus_status"]))
        self.assertEqual(consensus.loc["TFBOTH", "number_of_supporting_databases"], 2)
        self.assertEqual(consensus.loc["TFBOTH", "supporting_databases"], "CollecTRI;DoRothEA")
        self.assertEqual(consensus.loc["TFBOTH", "union_overlap_genes"], "G1;G2;G3;G4;G5")
        nodes = tables["tf_target_program_network_nodes"]
        edges = tables["tf_target_program_network_edges"]
        self.assertEqual(set(nodes.node_type), {"TF", "target_gene", "program"})
        self.assertIn("TF::TFBOTH", set(nodes.node_id))
        self.assertIn("GENE::G1", set(nodes.node_id))
        self.assertIn("PROGRAM::P1", set(nodes.node_id))
        self.assertEqual(len(edges.query("edge_type == 'TF_to_target'")), 5)
        self.assertEqual(len(edges.query("edge_type == 'target_to_program'")), 5)
        self.assertNotIn("GENE::G6", set(edges.target))
        self.assertTrue(edges.edge_id.is_unique)

    def test_tf_node_cross_program_aggregation_is_order_invariant(self):
        programs = pd.DataFrame([
            {"program_id": "P1", "source_artifact_hash": "h1"},
            {"program_id": "P2", "source_artifact_hash": "h2"},
            {"program_id": "P3", "source_artifact_hash": "h3"},
        ])
        common = {
            "tf": "TFX", "condition_a": "A", "condition_b": "B",
            "contrast_direction": "A relative to B", "feature_direction": "increased",
            "source_capability_id": "CAP-UP-001", "source_cache_key": "key",
            "number_of_supporting_databases": 1, "total_overlap_count": 1,
            "consensus_status": True,
        }
        consensus = pd.DataFrame([
            {**common, "program_id": "P1", "feature_set_id": "FS-P1",
             "cell_state": "CD8 T", "contrast": "A_vs_B",
             "supporting_databases": "DoRothEA", "minimum_adjusted_p_value": .04,
             "maximum_odds_ratio": 3.0, "union_overlap_genes": "G1",
             "per_resource_overlap_details": '{"DoRothEA":{}}'},
            {**common, "program_id": "P2", "feature_set_id": "FS-P2",
             "cell_state": "B cell", "contrast": "C_vs_D",
             "supporting_databases": "CollecTRI", "minimum_adjusted_p_value": .01,
             "maximum_odds_ratio": 8.0, "union_overlap_genes": "G2",
             "per_resource_overlap_details": '{"CollecTRI":{}}'},
        ])
        nodes, edges = _build_network(consensus, programs, pd.DataFrame())
        reversed_nodes, reversed_edges = _build_network(
            consensus.iloc[::-1].reset_index(drop=True), programs, pd.DataFrame())
        pd.testing.assert_frame_equal(nodes, reversed_nodes)
        pd.testing.assert_frame_equal(edges, reversed_edges)
        tf = nodes.set_index("node_id").loc["TF::TFX"]
        self.assertEqual(tf.number_of_represented_programs, 2)
        self.assertEqual(tf.represented_program_ids, "P1;P2")
        self.assertEqual(tf.number_of_represented_cell_states, 2)
        self.assertEqual(tf.represented_cell_states, "B cell;CD8 T")
        self.assertEqual(tf.supporting_databases, "CollecTRI;DoRothEA")
        self.assertEqual(tf.number_of_supporting_databases, 2)
        self.assertEqual(tf.maximum_odds_ratio, 8.0)
        self.assertEqual(tf.strongest_odds_ratio, 8.0)
        self.assertEqual(tf.minimum_adjusted_p_value, .01)
        self.assertEqual(tf.convergence_score, 2 / 3)
        self.assertEqual(tf.convergence_denominator, 3)
        self.assertEqual(tf.program_prevalence, 2 / 3)
        p1_edge = edges.query(
            "edge_type == 'TF_to_target' and program_id == 'P1'").iloc[0]
        self.assertEqual(p1_edge.minimum_adjusted_p_value, .04)
        self.assertEqual(p1_edge.maximum_odds_ratio, 3.0)
        self.assertEqual(p1_edge.supporting_databases, "DoRothEA")

    def test_tf_prevalence_denominator_includes_program_without_consensus(self):
        programs = pd.DataFrame([
            {"program_id": program_id, "source_artifact_hash": f"h-{program_id}"}
            for program_id in ("P1", "P2", "P3")
        ])
        consensus = pd.DataFrame([{
            "tf": "TFX", "program_id": "P1", "feature_set_id": "FS-P1",
            "cell_state": "CD8 T", "condition_a": "A", "condition_b": "B",
            "contrast": "A_vs_B", "contrast_direction": "A relative to B",
            "feature_direction": "increased", "source_capability_id": "CAP-UP-001",
            "source_cache_key": "key", "number_of_supporting_databases": 2,
            "supporting_databases": "CollecTRI;DoRothEA",
            "minimum_adjusted_p_value": .01, "maximum_odds_ratio": 5.0,
            "union_overlap_genes": "G1", "per_resource_overlap_details": "{}",
        }])
        nodes, _ = _build_network(consensus, programs, pd.DataFrame())
        tf = nodes.set_index("node_id").loc["TF::TFX"]
        self.assertEqual(tf.program_prevalence, 1 / 3)
        self.assertEqual(tf.convergence_score, 1 / 3)
        self.assertEqual(tf.convergence_denominator, 3)

    def test_multi_program_convergence_and_separate_families(self):
        rows = program_rows(self.source, file_hash(self.source))
        rows += program_rows(self.source, file_hash(self.source), "P2", "B cell",
                             "C_vs_D", "decreased")
        pd.DataFrame(rows).to_csv(self.program, sep="\t", index=False)
        _, tables, warnings, _ = self.prepared()
        convergence = tables["tf_regulatory_convergence"].set_index("tf")
        self.assertEqual(convergence.loc["TFBOTH", "convergence_denominator"], 2)
        self.assertEqual(convergence.loc["TFBOTH", "convergence_score"], 1.0)
        self.assertEqual(convergence.loc["TFBOTH", "number_of_represented_cell_states"], 2)
        self.assertFalse(any(x.code == "TF_CONVERGENCE_NOT_APPLICABLE" for x in warnings))
        enrichment = tables["tf_enrichment_by_database"]
        for _, group in enrichment.groupby(["database", "program_id"]):
            np.testing.assert_allclose(group.adjusted_p_value, _bh(group.p_value))

    def test_validation_duplicates_exclusions_hashes_and_errors(self):
        rows = program_rows(self.source, file_hash(self.source), genes=["g1", " G1 ", "G2", "OUT"])
        pd.DataFrame(rows).to_csv(self.program, sep="\t", index=False)
        request = replace(self.request, minimum_query_features=2)
        programs, excluded, background = validate_feature_programs(request)
        self.assertEqual(programs.feature_id.tolist(), ["G1", "G2"])
        reasons = excluded.groupby("exclusion_reason").feature_id.apply(list).to_dict()
        self.assertEqual(reasons["duplicate_program_feature"], ["G1"])
        self.assertEqual(reasons["query_feature_not_in_background"], ["OUT"])
        self.assertEqual(len(background), 20)
        bad = pd.DataFrame(rows).drop(columns=["contrast"])
        bad.to_csv(self.program, sep="\t", index=False)
        with self.assertRaisesRegex(ValueError, "missing required"):
            validate_feature_programs(request)
        rows[1]["cell_state"] = "other"
        pd.DataFrame(rows).to_csv(self.program, sep="\t", index=False)
        with self.assertRaisesRegex(ValueError, "contradictory"):
            validate_feature_programs(request)
        rows[1]["cell_state"] = "CD8 T"
        for row in rows:
            row["source_artifact_hash"] = "bad"
        pd.DataFrame(rows).to_csv(self.program, sep="\t", index=False)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_feature_programs(request)

    def test_resource_unmatched_query_feature_is_retained_and_reported(self):
        rows = program_rows(
            self.source, file_hash(self.source), genes=["G1", "G2", "G20"])
        pd.DataFrame(rows).to_csv(self.program, sep="\t", index=False)
        request = replace(self.request, minimum_query_features=3)
        programs, tables, _, _ = self.prepared(request)
        self.assertEqual(programs.feature_id.tolist(), ["G1", "G2", "G20"])
        unmatched = tables["excluded_or_unmatched_features"].query(
            "exclusion_reason == 'query_feature_not_in_any_regulon'")
        self.assertEqual(unmatched.feature_id.tolist(), ["G20"])
        self.assertTrue(
            tables["tf_enrichment_by_database"].query_n.eq(3).all())

    def test_resource_filtering_duplicates_conflicts_and_malformed(self):
        table = pd.DataFrame({
            "source": ["X"] * 4, "target": ["G1", "G1", "G1", "G2"],
            "weight": [1, 1, -1, 1], "confidence": ["A", "A", "A", "D"]})
        normalized, qc = normalize_resource(table, "DoRothEA", self.request)
        self.assertEqual(len(normalized), 2)
        self.assertEqual(qc["exact_duplicate_rows_removed"], 1)
        self.assertEqual(qc["conflicting_tf_target_entries_preserved"], 1)
        self.assertNotIn("G2", set(normalized.target_gene))
        with self.assertRaisesRegex(ValueError, "malformed"):
            normalize_resource(pd.DataFrame({"source": ["X"]}), "CollecTRI", self.request)
        changed = table.copy(); changed.loc[0, "weight"] = 2
        other, _ = normalize_resource(changed, "DoRothEA", self.request)
        self.assertNotEqual(normalized.resource_hash.iloc[0], other.resource_hash.iloc[0])

    def test_runner_cache_provenance_artifacts_and_completion_safety(self):
        context = AnalysisContext((self.root / "out").resolve(),
                                  (self.root / "cache").resolve(), "fixture-v1")
        first = run_tf_regulatory_network(self.request, context)
        second = run_tf_regulatory_network(self.request, context)
        self.assertFalse(first.cache_hit); self.assertTrue(second.cache_hit)
        self.assertEqual(first.status, CapabilityStatus.COMPLETED_WITH_WARNINGS)
        common = first.to_capability_result()
        common.validate_required_artifacts(require_exists=True)
        names = {x.logical_name for x in common.artifacts}
        self.assertIn("tf_enrichment_by_database", names)
        self.assertIn("cache_manifest", names)
        provenance = json.loads(Path(first.provenance_path).read_text())
        self.assertEqual(provenance["parameters"]["implementation_version"], IMPLEMENTATION_VERSION)
        self.assertIn("resource_hashes", provenance["parameters"])
        self.assertIn("correction_family_definition", provenance["parameters"])
        self.assertEqual(provenance["parameters"]["resource_resolution_mode"],
                         "caller_supplied_local")
        self.assertEqual(provenance["parameters"]["resource_provider"],
                         "decoupler_or_local_injected_table")
        self.assertIsNone(provenance["parameters"]["decoupler_version"])
        compact = provenance["parameters"]["feature_program_metadata"]
        self.assertEqual(len(compact), 1)
        self.assertEqual(compact[0]["feature_count"], 5)
        self.assertNotIn("feature_id", compact[0])
        self.assertIn("normalized_feature_program_hash", provenance["parameters"])
        self.assertNotIn("decoupler", provenance["software_versions"])
        manifest = json.loads(Path(first.cache_manifest_path).read_text())
        self.assertEqual(manifest["completion_status"], "complete")
        manifest["completion_status"] = "incomplete"
        Path(first.cache_manifest_path).write_text(json.dumps(manifest))
        self.assertIsNone(load_complete_manifest(Path(first.cache_manifest_path), first.cache_key))

    def test_cache_identity_changes_for_scientific_inputs(self):
        context = AnalysisContext((self.root / "out").resolve(),
                                  (self.root / "cache").resolve(), "fixture-v1")
        base = run_tf_regulatory_network(self.request, context)
        for changed in (
            replace(self.request, fdr_cutoff=.2),
            replace(self.request, minimum_query_target_overlap=3),
            replace(self.request, dorothea_confidence_levels=("A", "B")),
        ):
            self.assertNotEqual(base.cache_key, run_tf_regulatory_network(changed, context).cache_key)
        rows = program_rows(self.source, file_hash(self.source), feature_direction="decreased")
        pd.DataFrame(rows).to_csv(self.program, sep="\t", index=False)
        self.assertNotEqual(base.cache_key, run_tf_regulatory_network(self.request, context).cache_key)

    def test_schema_capability_versions_and_security(self):
        self.assertEqual(CAPABILITY_ID, "CAP-TF-001")
        self.assertEqual((IMPLEMENTATION_VERSION, NODE_VERSION, CACHE_SCHEMA_VERSION),
                         ("1.0.0", "1.0.0", 1))
        self.assertIn(CAPABILITY_ID, CAPABILITY_SPECS_BY_ID)
        spec = CAPABILITY_SPECS_BY_ID[CAPABILITY_ID]
        self.assertEqual(spec.name, "Consensus TF Regulatory Network")
        self.assertEqual(spec.input_schema,
            "cellstate.schemas.tf_regulatory_network.TFRegulatoryNetworkInput")
        self.assertEqual(spec.output_schema,
            "cellstate.schemas.tf_regulatory_network.TFRegulatoryNetworkOutput")
        source_paths = [
            Path("src/cellstate/nodes/tf_regulatory_network.py"),
            Path("src/cellstate/schemas/tf_regulatory_network.py"),
            Path("src/cellstate/plotting/tf_regulatory_network.py"),
        ]
        forbidden = ("graphistry.register", "password=", "api_key", "access_token")
        joined = "\n".join(x.read_text().lower() for x in source_paths)
        self.assertFalse(any(token in joined for token in forbidden))


if __name__ == "__main__":
    unittest.main()
