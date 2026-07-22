import importlib
import tempfile
import unittest
from pathlib import Path

from cellstate.capability_specs import CAPABILITY_SPECS, CAPABILITY_SPECS_BY_ID
from cellstate.cache import CacheManifest, build_cache_manifest, load_complete_manifest, write_manifest_atomic
from cellstate.schemas.abundance import AbundanceOutput
from cellstate.schemas.age_association import AgeAssociationOutput
from cellstate.schemas.progression import ProgressionOutput
from cellstate.schemas.pseudobulk_de import PseudobulkOutput
from cellstate.schemas.deseq2 import DifferentialExpressionOutput
from cellstate.schemas.arbitrary_two_group_de import (
    ArbitraryTwoGroupDEOutput, DEArtifactReference, TwoGroupDesignAssessment)
from cellstate.schemas.atlas_lodo import AtlasLODOOutput
from cellstate.schemas.tf_activity import TFActivityOutput
from cellstate.schemas.tf_regulatory_network import TFRegulatoryNetworkOutput
from cellstate.schemas.common import (AnalysisProvenance, ArtifactCategory, ArtifactReference,
    CapabilityResult, CapabilityStatus, StructuredWarning, WarningSeverity, capability_status_from_warnings)

def provenance(capability_id):
    return AnalysisProvenance(capability_id, "1.0.0", 1, ("source",), ("cell",), "signature", {}, None,
        None, (), "biological_sample", None, {}, (), (), "2026-01-01T00:00:00+00:00")

class CapabilityContractTests(unittest.TestCase):
    def test_all_family_outputs_convert(self):
        common = ("1.0.0", "key", False)
        outputs = [
            AbundanceOutput("CAP-COMP-001", *common, "sample.csv", "summary.csv", "prov.json", "manifest.json", (), provenance("CAP-COMP-001")),
            AgeAssociationOutput("CAP-STAT-001", *common, "analysis.csv", ("stats.csv",), "prov.json", "manifest.json", (), provenance("CAP-STAT-001")),
            AgeAssociationOutput("CAP-STAT-002", *common, "analysis.csv", ("stats.csv",), "prov.json", "manifest.json", (), provenance("CAP-STAT-002")),
            ProgressionOutput("CAP-COMP-002", *common, ("desc.csv",), ("infer.csv",), "prov.json", "manifest.json", (), provenance("CAP-COMP-002")),
            ProgressionOutput("CAP-STAT-003", *common, ("desc.csv",), ("infer.csv",), "prov.json", "manifest.json", (), provenance("CAP-STAT-003")),
            ProgressionOutput("CAP-STAT-004", *common, ("desc.csv",), ("infer.csv",), "prov.json", "manifest.json", (), provenance("CAP-STAT-004")),
            PseudobulkOutput("CAP-DESEQ-001", *common, ("counts.csv",), "meta.csv", "qc.csv", "prov.json", "manifest.json", (), provenance("CAP-DESEQ-001")),
            DifferentialExpressionOutput("CAP-DESEQ-002", *common, ("result.csv",), (), "qc.csv", "prov.json", "manifest.json", (), provenance("CAP-DESEQ-002")),
            ArbitraryTwoGroupDEOutput(
                terminal_status="completed", cache_key="key",
                comparison_direction="A - B",
                evidence_class="adjusted_inference",
                design_assessment=TwoGroupDesignAssessment(
                    group_replicate_counts={"A": 3, "B": 3},
                    represented_datasets=["D"], shared_datasets=["D"],
                    design_formula="~ group", design_columns=["Intercept", "group"],
                    design_rank=2, design_column_count=2,
                    residual_degrees_of_freedom=4, full_rank=True,
                    group_coefficient="A - B", estimable=True),
                input_gene_count=1, retained_gene_count=1, tested_gene_count=1,
                significant_gene_count=0, upregulated_in_group_a_count=0,
                upregulated_in_group_b_count=0,
                group_replicate_counts={"A": 3, "B": 3},
                artifacts=[
                    DEArtifactReference(logical_name="deseq2_results",
                        path="result.tsv", category="inferential",
                        media_type="text/tab-separated-values"),
                    DEArtifactReference(logical_name="provenance",
                        path="prov.json", category="provenance",
                        media_type="application/json")],
                provenance_path="prov.json", cache_manifest_path="manifest.json"),
            AtlasLODOOutput("CAP-LODO-001", "1.0.0", "completed", "key", False,
                (("fold_results", "fold.csv", "inferential"),), "prov.json", "manifest.json", (), provenance("CAP-LODO-001")),
            TFRegulatoryNetworkOutput("CAP-TF-001", "1.0.0", "1.0.0", "completed", "key", False,
                (("tf_enrichment_by_database", "tf.tsv", "inferential", "text/tab-separated-values"),),
                "prov.json", "manifest.json", (), provenance("CAP-TF-001")),
            TFActivityOutput("CAP-TF-002", "1.0.0", "1.0.0", "completed", "key", False,
                (("tf_activity_by_resource", "activity.tsv", "inferential", "text/tab-separated-values"),),
                "prov.json", "manifest.json", (), provenance("CAP-TF-002")),
        ]
        results = [output.to_capability_result() for output in outputs]
        self.assertEqual({result.capability_id for result in results}, {spec.capability_id for spec in CAPABILITY_SPECS})
        self.assertTrue(all(result.status is CapabilityStatus.COMPLETED for result in results))
        self.assertTrue(all(any(a.category is ArtifactCategory.PROVENANCE for a in result.artifacts) for result in results))

    def test_exact_status_mapping_and_cache_rules(self):
        warning = lambda code: StructuredWarning(code, "message", WarningSeverity.ERROR)
        self.assertEqual(capability_status_from_warnings(()), CapabilityStatus.COMPLETED)
        self.assertEqual(capability_status_from_warnings((warning("NOTICE"),)), CapabilityStatus.COMPLETED_WITH_WARNINGS)
        self.assertEqual(capability_status_from_warnings((warning("NON_ESTIMABLE_ADJUSTED_DESIGN"),)), CapabilityStatus.NOT_ESTIMABLE)
        self.assertEqual(capability_status_from_warnings((warning("INSUFFICIENT_PAIRED_REPLICATION"),)), CapabilityStatus.INSUFFICIENT_REPLICATION)
        self.assertEqual(capability_status_from_warnings((warning("DESEQ2_RUNTIME_FAILURE"),)), CapabilityStatus.FAILED_EXECUTION)
        self.assertEqual(capability_status_from_warnings((), outcome=CapabilityStatus.INVALID_INPUT), CapabilityStatus.INVALID_INPUT)
        self.assertEqual(capability_status_from_warnings((), outcome=CapabilityStatus.BLOCKED_SCIENTIFIC_DECISION), CapabilityStatus.BLOCKED_SCIENTIFIC_DECISION)
        artifact = (ArtifactReference("qc", "qc.csv", ArtifactCategory.QC),)
        with self.assertRaisesRegex(ValueError, "cannot be cache hits"):
            CapabilityResult("CAP-X", "1", CapabilityStatus.FAILED_EXECUTION, "key", True, artifact, (), "p", "m")
        CapabilityResult("CAP-X", "1", CapabilityStatus.COMPLETED_WITH_WARNINGS, "key", True, artifact,
                         (warning("NOTICE"),), "p", "m")

    def test_required_artifact_validation(self):
        with self.assertRaisesRegex(ValueError, "at least one required"):
            CapabilityResult("CAP-X", "1", "completed", "key", False,
                             (ArtifactReference("optional", "x", "optional", required=False),), (), "p", "m")
        with tempfile.TemporaryDirectory() as tmp:
            result = CapabilityResult("CAP-X", "1", "completed", "key", False,
                (ArtifactReference("required", str(Path(tmp) / "missing"), "QC"),), (), "p", "m")
            with self.assertRaisesRegex(ValueError, "do not exist"): result.validate_required_artifacts(require_exists=True)

    def test_terminal_results_never_become_cache_hits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); output = root / "result.csv"; output.write_text("x\n1\n")
            for code in ("INSUFFICIENT_GROUPS", "NON_ESTIMABLE_ADJUSTED_DESIGN", "DESEQ2_RUNTIME_FAILURE"):
                manifest = build_cache_manifest(cache_key=code, capability_id="CAP-X", node_version="1",
                    cache_schema_version=1, input_signature="input", source_dataset_signature="dataset",
                    parameters={}, output_files=(str(output),), completion_status="complete",
                    warnings=(StructuredWarning(code, "terminal", WarningSeverity.ERROR),), software_versions={})
                self.assertEqual(manifest.completion_status, "failed")
                path = root / f"{code}.json"; write_manifest_atomic(manifest, path)
                self.assertIsNone(load_complete_manifest(path, code))

            with self.assertRaisesRegex(ValueError, "cannot have complete manifests"):
                CacheManifest("direct", "CAP-X", "1", 1, "now", "input", "dataset", {},
                    (str(output),), "complete", (StructuredWarning("DESEQ2_RUNTIME_FAILURE", "terminal"),), {})

            # Reject an unsafe legacy manifest that was previously marked complete.
            legacy = build_cache_manifest(cache_key="legacy", capability_id="CAP-X", node_version="1",
                cache_schema_version=1, input_signature="input", source_dataset_signature="dataset",
                parameters={}, output_files=(str(output),), completion_status="complete", warnings=(), software_versions={})
            object.__setattr__(legacy, "warnings", (StructuredWarning("INSUFFICIENT_GROUPS", "terminal", WarningSeverity.ERROR),))
            write_manifest_atomic(legacy, root / "legacy.json")
            self.assertIsNone(load_complete_manifest(root / "legacy.json", "legacy"))

    def test_specs_complete_unique_and_deseq2_compatibility(self):
        self.assertEqual(len({x.capability_id for x in CAPABILITY_SPECS}), len(CAPABILITY_SPECS))
        self.assertEqual(set(CAPABILITY_SPECS_BY_ID), {x.capability_id for x in CAPABILITY_SPECS})
        self.assertTrue(all(x.required_metadata_columns and x.input_schema and x.output_schema for x in CAPABILITY_SPECS))
        for spec in CAPABILITY_SPECS:
            for schema_path in (spec.input_schema, spec.output_schema):
                module_name, object_name = schema_path.rsplit(".", 1)
                self.assertTrue(hasattr(importlib.import_module(module_name), object_name), schema_path)
            self.assertTrue(spec.accepted_data_representation)
            self.assertTrue(spec.name)
        tf_spec = CAPABILITY_SPECS_BY_ID["CAP-TF-001"]
        self.assertEqual(tf_spec.name, "Consensus TF Regulatory Network")
        self.assertEqual(tf_spec.input_schema,
            "cellstate.schemas.tf_regulatory_network.TFRegulatoryNetworkInput")
        self.assertEqual(tf_spec.output_schema,
            "cellstate.schemas.tf_regulatory_network.TFRegulatoryNetworkOutput")
        self.assertEqual(tf_spec.unit_of_inference, "feature_program")
        self.assertIn("validated_feature_program_table", tf_spec.accepted_data_representation)
        activity_spec = CAPABILITY_SPECS_BY_ID["CAP-TF-002"]
        self.assertEqual(activity_spec.name, "Signed TF Activity Inference")
        self.assertEqual(activity_spec.input_schema,
            "cellstate.schemas.tf_activity.TFActivityInput")
        self.assertEqual(activity_spec.output_schema,
            "cellstate.schemas.tf_activity.TFActivityOutput")
        self.assertEqual(activity_spec.unit_of_inference,
                         "resource_x_feature_program_x_tf")
        self.assertIn("complete_signed_gene_statistic_table",
                      activity_spec.accepted_data_representation)
        from cellstate.schemas.pseudobulk_de import DifferentialExpressionInput as legacy_schema
        from cellstate.schemas.deseq2 import DifferentialExpressionInput as new_schema
        from cellstate.nodes import pseudobulk_de as legacy_node
        from cellstate.nodes import deseq2 as new_node
        self.assertIs(legacy_schema, new_schema)
        self.assertIsNot(legacy_node.run_deseq2_differential_expression, new_node.run_deseq2_differential_expression)
        self.assertFalse(hasattr(new_node, "construct_raw_pseudobulk"))
        self.assertFalse(hasattr(legacy_node, "DE_CAPABILITY_ID"))

if __name__ == "__main__": unittest.main()
