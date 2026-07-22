import importlib.util
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

from cellstate.cache import load_complete_manifest
from cellstate.capability_specs import CAPABILITY_SPECS_BY_ID
from cellstate.context import AnalysisContext
from cellstate.nodes.tf_activity import (
    CACHE_SCHEMA_VERSION, CAPABILITY_ID, IMPLEMENTATION_VERSION, MODEL_DEFINITION,
    NODE_VERSION, _bh, calculate_tf_activity, file_hash,
    normalize_signed_resource, run_tf_activity, ulm_equivalent,
    validate_signed_gene_statistics,
)
from cellstate.schemas.common import CapabilityStatus
from cellstate.schemas.tf_activity import TFActivityInput


FIXTURE = Path(__file__).parent / "fixtures/tf_activity"


def program_rows(source, source_hash, program_id="P1", statistics=None):
    statistics = statistics or [-2, -1, 1, 2, 3, 0, 0, 0, 0, 0, 4]
    rows = []
    for index, statistic in enumerate(statistics, 1):
        rows.append({
            "program_id": program_id,
            "feature_set_id": f"FS-{program_id}",
            "feature_id": f"G{index}",
            "feature_type": "gene",
            "cell_state": "CD8 T" if program_id == "P1" else "B cell",
            "condition_a": "SMM",
            "condition_b": "NBM",
            "contrast": "SMM_vs_NBM",
            "contrast_direction": "SMM relative to NBM",
            "feature_direction": "bidirectional_complete",
            "signed_statistic": statistic,
            "signed_statistic_name": "delta_median",
            "statistic_orientation": "condition_a_minus_condition_b",
            "source_capability_id": "CAP-LODO-001",
            "source_capability_version": "1.0.0",
            "source_cache_key": "lodo-key",
            "source_artifact_path": str(source),
            "source_artifact_hash": source_hash,
            "upstream_analysis_method": "atlas_lodo_ols",
            "upstream_analysis_parameters": '{"minimum_cells":100}',
            "upstream_provenance": '{"approved":true}',
            "raw_p_value": .5,
            "adjusted_p_value": .8,
        })
    return rows


class TFActivityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "upstream.tsv"
        self.source.write_text("complete signed upstream artifact\n")
        self.program = self.root / "programs.tsv"
        pd.DataFrame(program_rows(
            self.source, file_hash(self.source))).to_csv(
                self.program, sep="\t", index=False)
        self.request = TFActivityInput(
            self.program, FIXTURE / "dorothea.tsv", FIXTURE / "collectri.tsv")

    def tearDown(self):
        self.tmp.cleanup()

    def prepared(self, request=None):
        request = request or self.request
        programs, excluded = validate_signed_gene_statistics(request)
        resources, qcs = [], []
        for database, path in (
            ("DoRothEA", request.dorothea_path),
            ("CollecTRI", request.collectri_path),
        ):
            normalized, qc = normalize_signed_resource(
                pd.read_csv(path, sep="\t"), database, request)
            resources.append(normalized)
            qcs.append(qc)
        tables, warnings, summary = calculate_tf_activity(
            programs, pd.concat(resources, ignore_index=True),
            pd.DataFrame(qcs), excluded, request)
        return programs, tables, warnings, summary

    def test_input_validation_duplicates_complete_universe_and_hashes(self):
        rows = program_rows(self.source, file_hash(self.source))
        duplicate = dict(rows[0])
        duplicate["feature_id"] = " g1 "
        rows.append(duplicate)
        pd.DataFrame(rows).to_csv(self.program, sep="\t", index=False)
        programs, excluded = validate_signed_gene_statistics(self.request)
        self.assertEqual(len(programs), 11)
        self.assertEqual(
            programs.feature_id.tolist(),
            sorted(f"G{x}" for x in range(1, 12)))
        self.assertEqual(excluded.exclusion_reason.tolist(), ["duplicate_program_gene"])
        rows[-1]["signed_statistic"] = 9
        pd.DataFrame(rows).to_csv(self.program, sep="\t", index=False)
        with self.assertRaisesRegex(ValueError, "contradictory duplicate"):
            validate_signed_gene_statistics(self.request)
        for invalid in ("bad", np.nan, np.inf):
            bad = program_rows(self.source, file_hash(self.source))
            bad[0]["signed_statistic"] = invalid
            pd.DataFrame(bad).to_csv(self.program, sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "numeric and finite"):
                validate_signed_gene_statistics(self.request)
        bad = pd.DataFrame(program_rows(self.source, file_hash(self.source))).drop(
            columns=["contrast"])
        bad.to_csv(self.program, sep="\t", index=False)
        with self.assertRaisesRegex(ValueError, "missing required"):
            validate_signed_gene_statistics(self.request)
        rows = program_rows(self.source, file_hash(self.source))
        rows[0]["statistic_orientation"] = "condition_b_minus_condition_a"
        pd.DataFrame(rows).to_csv(self.program, sep="\t", index=False)
        with self.assertRaisesRegex(ValueError, "orientation"):
            validate_signed_gene_statistics(self.request)
        rows = program_rows(self.source, "bad")
        pd.DataFrame(rows).to_csv(self.program, sep="\t", index=False)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_signed_gene_statistics(self.request)

    def test_resource_normalization_filter_conflict_and_hash(self):
        table = pd.DataFrame({
            "source": ["X", "X", "X", "X"],
            "target": ["G1", "G1", "G2", "G3"],
            "weight": [1, 1, -1, 1],
            "confidence": ["A", "A", "B", "D"],
            "organism": ["human"] * 4,
        })
        normalized, qc = normalize_signed_resource(table, "DoRothEA", self.request)
        self.assertEqual(len(normalized), 2)
        self.assertEqual(qc["exact_duplicate_rows_removed"], 1)
        self.assertNotIn("G3", set(normalized.target_gene))
        changed = table.copy()
        changed.loc[2, "weight"] = -2
        other, _ = normalize_signed_resource(changed, "DoRothEA", self.request)
        self.assertNotEqual(
            normalized.resource_hash.iloc[0], other.resource_hash.iloc[0])
        conflict = pd.concat([table, pd.DataFrame([{
            "source": "X", "target": "G1", "weight": -1,
            "confidence": "A", "organism": "human"}])])
        with self.assertRaisesRegex(ValueError, "conflicting signed"):
            normalize_signed_resource(conflict, "DoRothEA", self.request)
        for invalid in (0, "bad", np.nan, np.inf):
            malformed = table.copy().astype({"weight": object})
            malformed.loc[0, "weight"] = invalid
            with self.assertRaisesRegex(ValueError, "weights"):
                normalize_signed_resource(malformed, "DoRothEA", self.request)
        wrong = table.copy()
        wrong["organism"] = "mouse"
        with self.assertRaisesRegex(ValueError, "organism"):
            normalize_signed_resource(wrong, "DoRothEA", self.request)
        with self.assertRaisesRegex(ValueError, "missing"):
            normalize_signed_resource(
                pd.DataFrame({"source": ["X"], "target": ["G1"]}),
                "CollecTRI", self.request)

    def test_closed_form_ulm_positive_negative_df_pvalue_and_nonestimable(self):
        x = np.array([0., -2, -1, 1, 2, 3, 0, 0])
        y = np.array([0., -2, -1, 1, 2, 3, 0, 0])
        result = ulm_equivalent(y, x)
        reference = linregress(x, y)
        self.assertEqual(result["degrees_of_freedom"], 6)
        self.assertGreater(result["activity_score"], 0)
        self.assertAlmostEqual(result["fitted_coefficient"], reference.slope)
        self.assertAlmostEqual(result["p_value"], reference.pvalue, places=6)
        negative = ulm_equivalent(-y, x)
        self.assertLess(negative["activity_score"], 0)
        self.assertEqual(abs(result["activity_score"]),
                         abs(negative["activity_score"]))
        with self.assertRaisesRegex(ValueError, "not estimable"):
            ulm_equivalent(y, np.ones(len(y)))

    if importlib.util.find_spec("decoupler") is not None:
        def test_parity_with_installed_decoupler(self):
            import decoupler as dc
            data = pd.DataFrame(
                [[0., -2, -1, 1, 2, 3, 0, 0]],
                index=["P1"], columns=[f"G{x}" for x in range(1, 9)])
            net = pd.DataFrame({
                "source": ["TFX"] * 5,
                "target": [f"G{x}" for x in range(2, 7)],
                "weight": [-2., -1., 1., 2., 3.],
            })
            scores, pvalues = dc.mt.ulm(data=data, net=net, tmin=5)
            x = np.array([0., -2, -1, 1, 2, 3, 0, 0])
            expected = ulm_equivalent(data.iloc[0].to_numpy(), x)
            self.assertAlmostEqual(float(scores.loc["P1", "TFX"]),
                                   expected["activity_score"], places=10)
            self.assertAlmostEqual(float(pvalues.loc["P1", "TFX"]),
                                   expected["p_value"], places=10)

    def test_activity_estimability_bh_consensus_discordance_and_qc(self):
        programs, tables, warnings, summary = self.prepared()
        self.assertEqual(len(programs), 11)
        unmatched = tables["excluded_or_unmatched_gene_statistics"].query(
            "exclusion_reason == 'gene_not_in_any_signed_regulon'")
        self.assertEqual(unmatched.feature_id.tolist(), ["G11"])
        universe = tables["program_gene_universe_qc"].iloc[0]
        self.assertEqual(universe.eligible_gene_count, 11)
        self.assertTrue(bool(universe.complete_gene_universe_retained))
        estimability = tables["tf_activity_estimability"]
        self.assertFalse(bool(estimability.query(
            "database == 'CollecTRI' and tf == 'TFNONEST'").iloc[0].estimable))
        activity = tables["tf_activity_by_resource"]
        for _, group in activity.groupby(["database", "program_id"]):
            np.testing.assert_allclose(group.adjusted_p_value, _bh(group.p_value))
        self.assertTrue((activity.query("tf == 'TFUP'").activity_score > 0).all())
        self.assertTrue((activity.query("tf == 'TFDOWN'").activity_score < 0).all())
        consensus = tables["tf_activity_consensus"].set_index("tf")
        self.assertTrue(bool(consensus.loc["TFUP", "directional_consensus_status"]))
        self.assertEqual(consensus.loc["TFUP", "consensus_direction"], "increased")
        supporting_scores = activity.query("tf == 'TFUP'").activity_score.to_numpy()
        self.assertEqual(len(supporting_scores), 2)
        self.assertNotEqual(supporting_scores[0], supporting_scores[1])
        self.assertAlmostEqual(
            consensus.loc["TFUP", "median_consensus_activity_score"],
            float(np.median(supporting_scores)))
        self.assertAlmostEqual(
            consensus.loc["TFUP", "mean_consensus_activity_score"],
            float(np.mean(supporting_scores)))
        self.assertAlmostEqual(
            consensus.loc["TFUP", "minimum_absolute_supporting_score"],
            float(np.min(np.abs(supporting_scores))))
        heatmap_tfup = tables["heatmap_source_table"].query("tf == 'TFUP'").iloc[0]
        self.assertAlmostEqual(
            heatmap_tfup.display_activity_score,
            consensus.loc["TFUP", "median_consensus_activity_score"])
        self.assertNotEqual(
            heatmap_tfup.display_activity_score,
            consensus.loc["TFUP", "strongest_absolute_activity_score"])
        self.assertTrue(bool(consensus.loc["TFDOWN", "directional_consensus_status"]))
        self.assertEqual(consensus.loc["TFDOWN", "consensus_direction"], "decreased")
        self.assertFalse(bool(consensus.loc["TFDIS", "directional_consensus_status"]))
        self.assertTrue(bool(
            consensus.loc["TFDIS", "directional_discordance_status"]))
        self.assertTrue(bool(
            consensus.loc["TFDIS", "any_resource_discordance_status"]))
        discordance = tables["tf_activity_discordance"].set_index("tf")
        self.assertIn("significant_opposite_directions",
                      discordance.loc["TFDIS", "resource_discordance_reasons"])
        self.assertTrue(bool(
            discordance.loc["TFDIS", "directional_discordance_status"]))
        self.assertIn("resource_estimability_difference",
                      discordance.loc["TFNONEST", "resource_discordance_reasons"])
        self.assertIn("major_regulon_coverage_difference",
                      discordance.loc["TFNONEST", "resource_discordance_reasons"])
        self.assertFalse(bool(
            discordance.loc["TFNONEST", "directional_discordance_status"]))
        self.assertIn("single_resource_significant_support",
                      discordance.loc["TFSINGLE", "resource_discordance_reasons"])
        for tf, row in discordance.iterrows():
            self.assertEqual(
                bool(row.any_resource_discordance_status),
                bool(consensus.loc[tf, "any_resource_discordance_status"]))
            self.assertEqual(
                row.resource_discordance_reasons,
                consensus.loc[tf, "resource_discordance_reasons"])
        self.assertEqual(summary["eligible_gene_counts"]["P1"], 11)
        self.assertTrue(any(
            warning.code == "TF_ACTIVITY_RESOURCE_DISCORDANCE"
            for warning in warnings))

    def test_resource_program_coverage_is_program_specific(self):
        rows = program_rows(self.source, file_hash(self.source))
        second = program_rows(
            self.source, file_hash(self.source), program_id="P2")
        for index, row in enumerate(second, 20):
            row["feature_id"] = f"G{index}"
        rows += second
        pd.DataFrame(rows).to_csv(self.program, sep="\t", index=False)
        _, tables, warnings, _ = self.prepared()
        coverage = tables["resource_program_coverage_qc"].set_index(
            ["database", "program_id"])
        for database in ("CollecTRI", "DoRothEA"):
            self.assertGreater(
                coverage.loc[(database, "P1"), "resource_target_overlap"], 0)
            self.assertEqual(
                coverage.loc[(database, "P2"), "resource_target_overlap"], 0)
            self.assertEqual(
                coverage.loc[(database, "P2"),
                             "resource_target_coverage_fraction"], 0)
            self.assertEqual(
                coverage.loc[(database, "P2"), "number_of_estimable_tfs"], 0)
            self.assertGreater(
                coverage.loc[(database, "P2"),
                             "number_of_nonestimable_tfs"], 0)
        warning = next(
            item for item in warnings
            if item.code == "TF_ACTIVITY_RESOURCE_LOW_TARGET_COVERAGE")
        pairs = {
            (item["database"], item["program_id"])
            for item in warning.context["resource_program_pairs"]}
        self.assertIn(("DoRothEA", "P2"), pairs)
        self.assertIn(("CollecTRI", "P2"), pairs)

    def test_bh_families_are_separate_across_resources_and_programs(self):
        rows = program_rows(self.source, file_hash(self.source))
        rows += program_rows(
            self.source, file_hash(self.source), program_id="P2",
            statistics=[2, 1, -1, -2, -3, 0, 0, 0, 0, 0, 4])
        pd.DataFrame(rows).to_csv(self.program, sep="\t", index=False)
        _, tables, _, _ = self.prepared()
        activity = tables["tf_activity_by_resource"]
        self.assertEqual(set(activity.program_id), {"P1", "P2"})
        for _, group in activity.groupby(["database", "program_id"]):
            np.testing.assert_allclose(group.adjusted_p_value, _bh(group.p_value))

    def test_runner_cache_provenance_and_scientific_identity(self):
        context = AnalysisContext(
            (self.root / "out").resolve(), (self.root / "cache").resolve(),
            "fixture-v1")
        first = run_tf_activity(self.request, context)
        second = run_tf_activity(self.request, context)
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(first.status, CapabilityStatus.COMPLETED_WITH_WARNINGS)
        first.to_capability_result().validate_required_artifacts(require_exists=True)
        provenance = json.loads(Path(first.provenance_path).read_text())
        parameters = provenance["parameters"]
        self.assertEqual(parameters["implementation_version"], IMPLEMENTATION_VERSION)
        self.assertEqual(parameters["ulm_implementation_mode"],
                         "documented_decoupler_v2_ulm_formula")
        self.assertEqual(parameters["model_definition"], MODEL_DEFINITION)
        self.assertIsNone(parameters["decoupler_version"])
        self.assertNotIn("decoupler", provenance["software_versions"])
        self.assertIn("correction_family_definition", parameters)
        self.assertEqual(len(parameters["compact_program_metadata"]), 1)
        self.assertEqual(parameters["compact_program_metadata"][0]["feature_count"], 11)
        self.assertNotIn("signed_statistic", parameters["compact_program_metadata"][0])
        self.assertIn("normalized_signed_gene_statistics_hash", parameters)
        self.assertIsNone(provenance["reference_group"])
        compact_program = parameters["compact_program_metadata"][0]
        self.assertEqual(
            compact_program["statistic_orientation"],
            "condition_a_minus_condition_b")
        self.assertEqual(compact_program["condition_a"], "SMM")
        self.assertEqual(compact_program["condition_b"], "NBM")
        self.assertEqual(compact_program["contrast"], "SMM_vs_NBM")
        self.assertEqual(
            compact_program["contrast_direction"], "SMM relative to NBM")
        manifest = json.loads(Path(first.cache_manifest_path).read_text())
        manifest["completion_status"] = "incomplete"
        Path(first.cache_manifest_path).write_text(json.dumps(manifest))
        self.assertIsNone(load_complete_manifest(
            Path(first.cache_manifest_path), first.cache_key))

        baseline_key = first.cache_key
        for changed in (
            replace(self.request, fdr_cutoff=.2),
            replace(self.request, minimum_overlapping_targets=4),
            replace(self.request, dorothea_confidence_levels=("A", "B")),
        ):
            self.assertNotEqual(
                baseline_key, run_tf_activity(changed, context).cache_key)
        rows = program_rows(self.source, file_hash(self.source))
        rows[0]["signed_statistic"] = -3
        pd.DataFrame(rows).to_csv(self.program, sep="\t", index=False)
        self.assertNotEqual(
            baseline_key, run_tf_activity(self.request, context).cache_key)

        rows = program_rows(self.source, file_hash(self.source))[:-1]
        pd.DataFrame(rows).to_csv(self.program, sep="\t", index=False)
        membership_key = run_tf_activity(self.request, context).cache_key
        self.assertNotEqual(baseline_key, membership_key)

        resource = self.root / "changed_collectri.tsv"
        changed_resource = pd.read_csv(FIXTURE / "collectri.tsv", sep="\t")
        changed_resource.loc[
            (changed_resource.source == "TFUP")
            & (changed_resource.target == "G1"), "weight"] = -3
        changed_resource.to_csv(resource, sep="\t", index=False)
        resource_key = run_tf_activity(
            replace(self.request, collectri_path=resource), context).cache_key
        self.assertNotEqual(membership_key, resource_key)

        self.source.write_text("changed upstream source identity\n")
        rows = program_rows(self.source, file_hash(self.source))[:-1]
        pd.DataFrame(rows).to_csv(self.program, sep="\t", index=False)
        source_key = run_tf_activity(self.request, context).cache_key
        self.assertNotEqual(membership_key, source_key)

    def test_schema_registration_versions_and_scope_security(self):
        self.assertEqual(CAPABILITY_ID, "CAP-TF-002")
        self.assertEqual(
            (IMPLEMENTATION_VERSION, NODE_VERSION, CACHE_SCHEMA_VERSION),
            ("1.0.0", "1.0.0", 1))
        spec = CAPABILITY_SPECS_BY_ID[CAPABILITY_ID]
        self.assertEqual(spec.name, "Signed TF Activity Inference")
        self.assertEqual(
            spec.input_schema, "cellstate.schemas.tf_activity.TFActivityInput")
        self.assertEqual(
            spec.output_schema, "cellstate.schemas.tf_activity.TFActivityOutput")
        self.assertEqual(spec.unit_of_inference, "resource_x_feature_program_x_tf")
        source_paths = [
            Path("src/cellstate/nodes/tf_activity.py"),
            Path("src/cellstate/schemas/tf_activity.py"),
            Path("src/cellstate/plotting/tf_activity.py"),
        ]
        joined = "\n".join(path.read_text().lower() for path in source_paths)
        forbidden = (
            "graphistry", "password", "api_key", "access_token", "bearer",
            "ligand-receptor", "pathway analysis", "perturbation prediction",
            "cap-tf-001",
        )
        self.assertFalse(any(token in joined for token in forbidden))


if __name__ == "__main__":
    unittest.main()
