"""Two-stage OpenAI reasoning orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from cellstate.schemas.evidence import EvidenceBundle
from cellstate.schemas.reasoning import CriticReport, InterpretationReport, ScientificReport

from .critic import run_critic
from .exceptions import ReasoningError, ReasoningValidationError
from .interpreter import run_interpreter
from .openai_client import OpenAIReasoningClient
from .presentation import PresentationResult, PresentationWarning, generate_presentation
from .report import assemble_scientific_report, write_model_atomic


@dataclass(frozen=True)
class ReasoningResult:
    critic_report: CriticReport
    interpretation_report: InterpretationReport
    scientific_report: ScientificReport
    critic_report_path: Path
    interpretation_report_path: Path
    scientific_report_path: Path
    presentation: PresentationResult

    @property
    def report_artifact_paths(self) -> tuple[Path, ...]:
        return (
            self.critic_report_path,
            self.interpretation_report_path,
            self.scientific_report_path,
            *self.presentation.artifact_paths,
        )


class ReasoningEngine:
    def __init__(
        self,
        *,
        client: OpenAIReasoningClient | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._client = client
        self._progress_callback = progress_callback

    @property
    def client(self) -> OpenAIReasoningClient:
        if self._client is None:
            self._client = OpenAIReasoningClient()
        return self._client

    def run(self, bundle: EvidenceBundle, run_dir: Path) -> ReasoningResult:
        try:
            if self._progress_callback:
                self._progress_callback("critic")
            critic = run_critic(bundle, client=self.client)
            if self._progress_callback:
                self._progress_callback("interpreter")
            interpretation = run_interpreter(bundle, critic, client=self.client)
            if self._progress_callback:
                self._progress_callback("report")
            paths = {
                "critic_report": str(run_dir / "critic_report.json"),
                "interpretation_report": str(run_dir / "interpretation_report.json"),
                "scientific_report": str(run_dir / "scientific_report.json"),
            }
            scientific = assemble_scientific_report(
                bundle,
                critic,
                interpretation,
                model_name=self.client.model_name,
                report_paths=paths,
            )
            write_model_atomic(critic, Path(paths["critic_report"]))
            write_model_atomic(
                interpretation, Path(paths["interpretation_report"])
            )
            write_model_atomic(scientific, Path(paths["scientific_report"]))
            try:
                presentation = generate_presentation(scientific, bundle, run_dir)
            except Exception as exc:  # presentation is an optional isolation boundary
                presentation = PresentationResult(
                    pdf_path=None,
                    figure_paths=(),
                    captions=(),
                    warnings=(PresentationWarning(
                        code="PRESENTATION_GENERATION_FAILED",
                        message=f"{type(exc).__name__}: {exc}",
                    ),),
                )
            return ReasoningResult(
                critic_report=critic,
                interpretation_report=interpretation,
                scientific_report=scientific,
                critic_report_path=Path(paths["critic_report"]),
                interpretation_report_path=Path(paths["interpretation_report"]),
                scientific_report_path=Path(paths["scientific_report"]),
                presentation=presentation,
            )
        except ReasoningError:
            raise
        except Exception as exc:
            raise ReasoningValidationError(
                f"ReasoningEngine failed: {type(exc).__name__}."
            ) from exc
