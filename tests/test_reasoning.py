from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cellstate.evidence import build_evidence_bundle, write_evidence_bundle_atomic
from cellstate.reasoning.critic import run_critic
from cellstate.reasoning.engine import ReasoningEngine
from cellstate.reasoning.exceptions import (
    ReasoningAPIError,
    ReasoningConfigurationError,
    ReasoningValidationError,
)
from cellstate.reasoning.interpreter import run_interpreter
from cellstate.reasoning.openai_client import OpenAIReasoningClient
from cellstate.reasoning.report import assemble_scientific_report, write_model_atomic
from cellstate.schemas.evidence import EvidenceArtifact
from cellstate.schemas.reasoning import (
    CRITIC_PROMPT_VERSION,
    INTERPRETER_PROMPT_VERSION,
    REASONING_SCHEMA_VERSION,
    CriticReport,
    InterpretationReport,
)


def assessment(status="pass", score=8, refs=None):
    return {
        "status": status,
        "score": score,
        "summary": "Assessment based on the supplied deterministic bundle.",
        "evidence_refs": refs or ["design_assessment.ready"],
    }


def critic_payload(bundle_id, *, confidence="low", limitations=None):
    return {
        "schema_version": REASONING_SCHEMA_VERSION,
        "evidence_bundle_id": bundle_id,
        "replication_assessment": assessment(),
        "statistical_support_assessment": assessment(
            "not_assessable", None, ["deterministic_evidence"]
        ),
        "confounding_assessment": assessment(),
        "design_validity_assessment": assessment(),
        "assumption_risk_assessment": assessment(),
        "generalizability_assessment": assessment(),
        "strengths": ["The unit of inference is explicit."],
        "limitations": limitations or ["No inferential biological result is present."],
        "recommended_follow_up": ["Run a validated inferential capability."],
        "overall_confidence": confidence,
        "overall_confidence_score": 3,
        "reasoning_summary": "The bundle supports design review but not biology.",
        "created_at_utc": "2026-07-18T12:00:00+00:00",
    }


def interpretation_payload(
    bundle_id,
    *,
    confidence="insufficient",
    limitations=None,
    observations=None,
    programs=None,
    regulators=None,
    hypotheses=None,
):
    return {
        "schema_version": REASONING_SCHEMA_VERSION,
        "evidence_bundle_id": bundle_id,
        "observations": observations or [],
        "biological_programs": programs or [],
        "candidate_regulators": regulators or [],
        "hypotheses": hypotheses or [],
        "critic_limitations_referenced": limitations
        or ["No inferential biological result is present."],
        "experimental_follow_up": ["Run differential-expression inference."],
        "interpretation_confidence": confidence,
        "summary": "Biological interpretation is not supported by this bundle.",
        "created_at_utc": "2026-07-18T12:01:00+00:00",
    }


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.model_name = "fake-openai-model"

    def generate_structured(self, *, system_prompt, payload, response_model):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "payload": payload,
                "response_model": response_model,
            }
        )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response_model.model_validate(response)


class ReasoningTests(unittest.TestCase):
    def make_bundle(self, root: Path, *, blocked=False, inferential=False):
        (root / "design_assessment.json").write_text("{}\n", encoding="utf-8")
        design = {
            "group_sample_counts": {"NDMM": 4, "NBM": 5},
            "shared_datasets": 2,
            "design_rank": 3,
            "design_columns": 3,
            "full_rank": True,
            "ready": not blocked,
        }
        if blocked:
            design.update(shared_datasets=0, full_rank=False, ready=False)
        bundle = build_evidence_bundle(
            state={
                "question": "Compare NDMM with NBM",
                "analysis_type": "pseudobulk_de",
                "cell_type": "CD8 T",
                "group_a": "NDMM",
                "group_b": "NBM",
            },
            execution_status="blocked" if blocked else "completed",
            output_directory=root,
            design_assessment=design,
            deterministic_evidence={"n_samples": 9, "n_genes": 100},
        )
        if inferential:
            bundle = replace(
                bundle,
                artifacts=(
                    EvidenceArtifact(
                        logical_name="inferential_results",
                        path=str(root / "must_not_be_opened.tsv"),
                        category="inferential",
                        media_type="text/tab-separated-values",
                    ),
                ),
            )
        return bundle

    def test_critic_receives_only_serialized_bundle_and_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self.make_bundle(Path(tmp))
            client = FakeClient([critic_payload(bundle.bundle_id)])
            report = run_critic(bundle, client=client)
            self.assertIsInstance(report, CriticReport)
            self.assertEqual(
                set(client.calls[0]["payload"]),
                {"evidence_bundle", "allowed_evidence_references"},
            )
            self.assertTrue(
                client.calls[0]["payload"]["allowed_evidence_references"]
            )
            json.dumps(client.calls[0]["payload"])
            self.assertNotIn("biological_programs", CriticReport.model_fields)
            self.assertNotIn("candidate_regulators", CriticReport.model_fields)
            self.assertEqual(len(client.calls), 1)

    def test_malformed_critic_output_retries_then_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self.make_bundle(Path(tmp))
            malformed = critic_payload(bundle.bundle_id)
            malformed.pop("reasoning_summary")
            client = FakeClient([malformed, critic_payload(bundle.bundle_id)])
            report = run_critic(bundle, client=client)
            self.assertEqual(report.evidence_bundle_id, bundle.bundle_id)
            self.assertEqual(len(client.calls), 2)
            retry = client.calls[1]["payload"]
            self.assertIn("invalid CriticReport", retry["validation_feedback"])
            self.assertEqual(
                retry["allowed_evidence_references"],
                client.calls[0]["payload"]["allowed_evidence_references"],
            )

    def test_invalid_critic_reference_retries_with_exact_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self.make_bundle(Path(tmp))
            invalid = critic_payload(bundle.bundle_id)
            invalid["replication_assessment"]["evidence_refs"] = ["atlas.raw.X"]
            corrected = critic_payload(bundle.bundle_id)
            client = FakeClient([invalid, corrected])
            report = run_critic(bundle, client=client)
            self.assertEqual(
                report.replication_assessment.evidence_refs,
                ["design_assessment.ready"],
            )
            self.assertIn("unknown evidence", client.calls[1]["payload"]["validation_feedback"])
            self.assertIn("allowed_evidence_references", client.calls[1]["payload"]["repair_instruction"])

    def test_critic_rejects_identity_and_fabricated_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self.make_bundle(Path(tmp))
            mismatch = critic_payload("wrong-id")
            with self.assertRaisesRegex(ReasoningValidationError, "identity mismatch"):
                run_critic(bundle, client=FakeClient([mismatch, mismatch]))
            invalid = critic_payload(bundle.bundle_id)
            invalid["replication_assessment"]["evidence_refs"] = ["atlas.raw.X"]
            with self.assertRaisesRegex(ReasoningValidationError, "unknown evidence"):
                run_critic(bundle, client=FakeClient([invalid, invalid]))

    def test_interpreter_receives_only_bundle_and_critic(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self.make_bundle(Path(tmp))
            critic = CriticReport.model_validate(critic_payload(bundle.bundle_id))
            client = FakeClient([interpretation_payload(bundle.bundle_id)])
            report = run_interpreter(bundle, critic, client=client)
            self.assertIsInstance(report, InterpretationReport)
            self.assertEqual(
                set(client.calls[0]["payload"]),
                {
                    "evidence_bundle",
                    "critic_report",
                    "allowed_critic_limitations",
                },
            )
            self.assertEqual(
                client.calls[0]["payload"]["allowed_critic_limitations"],
                list(critic.limitations),
            )
            json.dumps(client.calls[0]["payload"])
            self.assertEqual(len(client.calls), 1)

    def test_malformed_interpreter_output_retries_then_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self.make_bundle(Path(tmp))
            critic = CriticReport.model_validate(critic_payload(bundle.bundle_id))
            malformed = interpretation_payload(bundle.bundle_id)
            malformed.pop("summary")
            client = FakeClient([malformed, interpretation_payload(bundle.bundle_id)])
            report = run_interpreter(bundle, critic, client=client)
            self.assertEqual(report.evidence_bundle_id, bundle.bundle_id)
            self.assertEqual(len(client.calls), 2)
            self.assertIn("invalid InterpretationReport", client.calls[1]["payload"]["validation_feedback"])
            self.assertEqual(
                client.calls[1]["payload"]["allowed_critic_limitations"],
                list(critic.limitations),
            )

    def test_paraphrased_limitation_retries_with_exact_limitation(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self.make_bundle(Path(tmp), inferential=True)
            limitation = "Cohort scope limits generalizability."
            critic = CriticReport.model_validate(critic_payload(
                bundle.bundle_id, confidence="moderate", limitations=[limitation]
            ))
            paraphrased = interpretation_payload(
                bundle.bundle_id, confidence="low",
                limitations=["The cohort has limited generalizability."],
            )
            corrected = interpretation_payload(
                bundle.bundle_id, confidence="low", limitations=[limitation]
            )
            client = FakeClient([paraphrased, corrected])
            report = run_interpreter(bundle, critic, client=client)
            self.assertEqual(report.critic_limitations_referenced, [limitation])
            self.assertIn("not present", client.calls[1]["payload"]["validation_feedback"])

    def test_interpreter_separates_observations_and_hypotheses(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self.make_bundle(Path(tmp), inferential=True)
            limitation = "Cohort scope limits generalizability."
            critic = CriticReport.model_validate(
                critic_payload(
                    bundle.bundle_id,
                    confidence="moderate",
                    limitations=[limitation],
                )
            )
            payload = interpretation_payload(
                bundle.bundle_id,
                confidence="low",
                limitations=[limitation],
                observations=["A deterministic effect estimate was reported."],
                hypotheses=["Hypothesis: the reported feature merits validation."],
            )
            report = run_interpreter(bundle, critic, client=FakeClient([payload]))
            self.assertTrue(report.observations)
            self.assertTrue(report.hypotheses[0].startswith("Hypothesis:"))
            self.assertEqual(report.critic_limitations_referenced, [limitation])

    def test_confidence_cannot_exceed_critic(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self.make_bundle(Path(tmp), inferential=True)
            critic = CriticReport.model_validate(
                critic_payload(bundle.bundle_id, confidence="low")
            )
            payload = interpretation_payload(bundle.bundle_id, confidence="high")
            with self.assertRaisesRegex(ReasoningValidationError, "cannot exceed"):
                run_interpreter(bundle, critic, client=FakeClient([payload, payload]))

    def test_interpreter_requires_real_critic_limitations(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self.make_bundle(Path(tmp), inferential=True)
            critic = CriticReport.model_validate(critic_payload(bundle.bundle_id))
            payload = interpretation_payload(
                bundle.bundle_id, limitations=["Fabricated limitation"]
            )
            with self.assertRaisesRegex(ReasoningValidationError, "not present"):
                run_interpreter(bundle, critic, client=FakeClient([payload, payload]))

    def test_retry_exhaustion_preserves_informative_contract_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self.make_bundle(Path(tmp), inferential=True)
            critic = CriticReport.model_validate(critic_payload(bundle.bundle_id))
            fabricated = interpretation_payload(
                bundle.bundle_id, limitations=["Fabricated limitation"]
            )
            client = FakeClient([fabricated, fabricated])
            with self.assertRaisesRegex(
                ReasoningValidationError,
                "not present.*Retry exhausted after 2 model attempts",
            ):
                run_interpreter(bundle, critic, client=client)
            self.assertEqual(len(client.calls), 2)

    def test_blocked_and_aggregation_only_bundles_forbid_biology(self):
        for blocked in (True, False):
            with self.subTest(blocked=blocked), tempfile.TemporaryDirectory() as tmp:
                bundle = self.make_bundle(Path(tmp), blocked=blocked)
                critic = CriticReport.model_validate(critic_payload(bundle.bundle_id))
                valid = interpretation_payload(bundle.bundle_id)
                report = run_interpreter(bundle, critic, client=FakeClient([valid]))
                self.assertEqual(report.interpretation_confidence, "insufficient")
                invalid = interpretation_payload(
                    bundle.bundle_id,
                    observations=["Unsupported direction"],
                    programs=["Unsupported program"],
                    hypotheses=["Hypothesis: unsupported claim"],
                )
                with self.assertRaisesRegex(
                    ReasoningValidationError, "cannot contain biological claims"
                ):
                    run_interpreter(bundle, critic, client=FakeClient([invalid, invalid]))

    def test_interpreter_identity_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self.make_bundle(Path(tmp))
            critic = CriticReport.model_validate(critic_payload(bundle.bundle_id))
            payload = interpretation_payload("wrong-id")
            with self.assertRaisesRegex(ReasoningValidationError, "identity mismatch"):
                run_interpreter(bundle, critic, client=FakeClient([payload, payload]))

    def test_engine_writes_reports_and_preserves_bundle_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = self.make_bundle(root)
            bundle_path = root / "evidence_bundle.json"
            write_evidence_bundle_atomic(bundle, bundle_path)
            before = bundle_path.read_bytes()
            client = FakeClient(
                [
                    critic_payload(bundle.bundle_id),
                    interpretation_payload(bundle.bundle_id),
                ]
            )
            stages = []
            result = ReasoningEngine(
                client=client, progress_callback=stages.append
            ).run(bundle, root)
            self.assertEqual(stages, ["critic", "interpreter", "report"])
            self.assertEqual(bundle_path.read_bytes(), before)
            for path in (
                result.critic_report_path,
                result.interpretation_report_path,
                result.scientific_report_path,
            ):
                self.assertTrue(path.exists())
                json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(result.scientific_report.critic_report, result.critic_report)
            self.assertEqual(
                result.scientific_report.interpretation_report,
                result.interpretation_report,
            )
            provenance = result.scientific_report.provenance
            self.assertEqual(provenance["critic_prompt_version"], CRITIC_PROMPT_VERSION)
            self.assertEqual(
                provenance["interpreter_prompt_version"],
                INTERPRETER_PROMPT_VERSION,
            )
            self.assertEqual(provenance["openai_model"], "fake-openai-model")
            self.assertFalse(any(root.glob(".*.tmp")))

    def test_presentation_failure_preserves_successful_json_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = self.make_bundle(root)
            client = FakeClient([
                critic_payload(bundle.bundle_id),
                interpretation_payload(bundle.bundle_id),
            ])
            with patch(
                "cellstate.reasoning.engine.generate_presentation",
                side_effect=RuntimeError("renderer unavailable"),
            ):
                result = ReasoningEngine(client=client).run(bundle, root)
            self.assertTrue(result.scientific_report_path.exists())
            json.loads(result.scientific_report_path.read_text(encoding="utf-8"))
            self.assertIsNone(result.presentation.pdf_path)
            self.assertEqual(
                result.presentation.warnings[0].code,
                "PRESENTATION_GENERATION_FAILED",
            )

    def test_report_assembly_is_deterministic_for_fixed_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self.make_bundle(Path(tmp))
            critic = CriticReport.model_validate(critic_payload(bundle.bundle_id))
            interpreter = InterpretationReport.model_validate(
                interpretation_payload(bundle.bundle_id)
            )
            kwargs = {
                "model_name": "model",
                "report_paths": {"scientific_report": "scientific_report.json"},
                "created_at_utc": "2026-07-18T12:02:00+00:00",
            }
            first = assemble_scientific_report(bundle, critic, interpreter, **kwargs)
            second = assemble_scientific_report(bundle, critic, interpreter, **kwargs)
            self.assertEqual(first, second)

    def test_atomic_write_failure_cleans_temporary_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = self.make_bundle(root)
            critic = CriticReport.model_validate(critic_payload(bundle.bundle_id))
            destination = root / "critic_report.json"
            with patch("cellstate.reasoning.report.os.replace", side_effect=OSError):
                with self.assertRaises(ReasoningValidationError):
                    write_model_atomic(critic, destination)
            self.assertFalse(any(root.glob(".critic_report.json.*.tmp")))
            self.assertFalse(destination.exists())

    def test_missing_api_key_and_lazy_initialization(self):
        with patch.dict(os.environ, {}, clear=True):
            client = OpenAIReasoningClient()
            self.assertIsNone(client._client)
            with self.assertRaisesRegex(
                ReasoningConfigurationError, "OPENAI_API_KEY is not configured"
            ):
                client._get_client()

    def test_api_failure_preserves_deterministic_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = self.make_bundle(root)
            bundle_path = root / "evidence_bundle.json"
            write_evidence_bundle_atomic(bundle, bundle_path)
            before = bundle_path.read_bytes()
            client = FakeClient([ReasoningAPIError("service unavailable")])
            with self.assertRaises(ReasoningAPIError):
                ReasoningEngine(client=client).run(bundle, root)
            self.assertEqual(bundle_path.read_bytes(), before)
            self.assertFalse((root / "critic_report.json").exists())

    def test_referenced_large_artifact_is_never_opened(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = self.make_bundle(root, inferential=True)
            critic = CriticReport.model_validate(
                critic_payload(bundle.bundle_id, confidence="moderate")
            )
            payload = interpretation_payload(bundle.bundle_id, confidence="low")
            missing_artifact = Path(bundle.artifacts[0].path)
            self.assertFalse(missing_artifact.exists())
            run_interpreter(bundle, critic, client=FakeClient([payload]))
            self.assertFalse(missing_artifact.exists())

    def test_prototype_failure_policy_does_not_mutate_run_status(self):
        source = (
            Path(__file__).resolve().parents[1] / "prototype_agent.py"
        ).read_text(encoding="utf-8")
        helper = source[source.index("def run_reasoning_cli"):source.index(
            "\ndef execute_analysis", source.index("def run_reasoning_cli")
        )]
        self.assertIn("except ReasoningError", helper)
        self.assertIn("Evidence bundle preserved at:", helper)
        self.assertNotIn("run_status.json", helper)


if __name__ == "__main__":
    unittest.main()
