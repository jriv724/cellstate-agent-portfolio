"""Rich terminal presentation with a dependency-free plain fallback."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from .capability_registry import all_capability_rows, build_capability_registry
from .labels import disease_group_label, expand_disease_group_ids
from .models import AnalysisPlan, ApplicationRunResult, AtlasSummary

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    from rich.tree import Tree
    RICH_AVAILABLE = True
except Exception:  # pragma: no cover - exercised through forced fallback
    Console = Any
    RICH_AVAILABLE = False


class TerminalUI:
    STATUS_PRESENTATION = {
        "connected": ("✓", "connected"),
        "connected but configuration required": ("◐", "configuration required"),
        "invalid resource configuration": ("◐", "invalid resource configuration"),
        "requires explicit caller inputs": ("ⓘ", "requires explicit inputs"),
        "backend implemented but not yet exposed": ("•", "backend only"),
    }
    GROUP_ORDER = (
        "Available now",
        "Configuration required",
        "Advanced / explicit-input workflows",
        "Backend capabilities not yet exposed",
    )
    def __init__(self, console: Any | None = None, *, force_plain: bool = False):
        self.rich = bool(RICH_AVAILABLE and not force_plain)
        if self.rich:
            try:
                self.console = console or Console()
            except Exception:
                self.rich = False
                self.console = None
        else:
            self.console = console

    def print(self, value: Any = "") -> None:
        if self.rich:
            self.console.print(value)
        elif self.console is not None and hasattr(self.console, "print"):
            self.console.print(str(value))
        else:
            print(value)

    def startup(self, atlas: AtlasSummary) -> None:
        registry = build_capability_registry()
        if self.rich:
            self.print(Panel.fit(
                "[bold cyan]CELLSTATE AGENT[/bold cyan]\n"
                "Auditable immune-atlas discovery system",
                border_style="cyan",
            ))
            table = Table.grid(padding=(0, 2))
            table.add_row("[bold]Atlas[/bold]", f"{atlas.n_cells:,} cells")
            table.add_row("", f"{len(atlas.cell_states)} cell states · {len(atlas.groups)} groups")
            table.add_row("[bold]Ready[/bold]", "✓ CAP-DESEQ-003")
            tf_symbol = self.STATUS_PRESENTATION[registry["CAP-TF-002"].status][0]
            table.add_row("[bold]Optional[/bold]", f"{tf_symbol} CAP-TF-002 · ⓘ CAP-TF-001")
            table.add_row("[bold]Reasoning[/bold]", "✓ EvidenceBundle · ✓ Critic · ✓ Interpreter")
            self.print(Panel(table, title="Workspace", border_style="cyan"))
        else:
            self.print("CELLSTATE AGENT — Auditable immune-atlas discovery system")
            self.print(
                f"Atlas: {atlas.n_cells:,} cells | {len(atlas.cell_states)} cell states "
                f"| {len(atlas.groups)} groups"
            )
            self.print("Available now: ✓ CAP-DESEQ-003")
            tf_symbol, tf_label = self.STATUS_PRESENTATION[registry["CAP-TF-002"].status]
            self.print(f"Optional: {tf_symbol} CAP-TF-002 ({tf_label}); ⓘ CAP-TF-001 (requires explicit inputs)")

    def plan(self, plan: AnalysisPlan, atlas: AtlasSummary) -> None:
        a = atlas.sample_counts.get((plan.cell_state, plan.group_a), 0)
        b = atlas.sample_counts.get((plan.cell_state, plan.group_b), 0)
        registry = build_capability_registry()
        workflow = ["Atlas adapter", "CAP-DESEQ-003"]
        if "CAP-TF-002" in plan.requested_capabilities:
            workflow.append("CAP-TF-002")
        workflow.extend(["EvidenceBundle", "Critic", "Interpreter"])
        values = [
            ("Question", expand_disease_group_ids(plan.question)),
            ("Population", plan.cell_state),
            ("Contrast", plan.display_contrast),
            ("Effect direction", plan.display_effect_direction),
            ("Capabilities", ", ".join(plan.requested_capabilities)),
            ("Replicates", f"{disease_group_label(plan.group_a)}: {a} · {disease_group_label(plan.group_b)}: {b}"),
            ("Cell threshold", "100 cells per sample/cell state"),
            ("Replicate threshold", "3 independent replicates per group"),
            ("Design", "exploratory ~ group" if plan.confounded_design_policy == "exploratory_lodo" else "~ dataset + group when estimable"),
            ("Robustness", "dataset-level LODO" if plan.confounded_design_policy == "exploratory_lodo" else "not applicable"),
            ("Confounding policy", plan.confounded_design_policy),
            ("TF readiness", registry["CAP-TF-002"].status),
            ("Workflow", " → ".join(workflow)),
            ("Stop conditions", "invalid raw counts; low replication; non-estimable design"),
        ]
        if plan.confounded_design_policy == "exploratory_lodo":
            values.insert(-3, ("Warning", "group and dataset are not independently identifiable"))
        if self.rich:
            table = Table(show_header=False, box=None)
            for key, value in values:
                table.add_row(f"[bold]{key}[/bold]", value)
            self.print(Panel(table, title="Analysis plan", border_style="blue"))
        else:
            self.print("Analysis plan")
            for key, value in values:
                self.print(f"{key}: {value}")

    @classmethod
    def _group_name(cls, status: str) -> str:
        return {
            "connected": "Available now",
            "connected but configuration required": "Configuration required",
            "invalid resource configuration": "Configuration required",
            "requires explicit caller inputs": "Advanced / explicit-input workflows",
            "backend implemented but not yet exposed": "Backend capabilities not yet exposed",
        }[status]

    def capabilities(self, *, connected_only: bool = False, details: bool = False) -> None:
        rows = all_capability_rows()
        if connected_only:
            rows = [row for row in rows if row.executable]
        grouped = {name: [] for name in self.GROUP_ORDER}
        for row in rows:
            grouped[self._group_name(row.status)].append(row)
        for group_name in self.GROUP_ORDER:
            group = grouped[group_name]
            if not group:
                continue
            if self.rich:
                table = Table("", "Capability", "Title", box=None, pad_edge=False)
                for row in group:
                    symbol, label = self.STATUS_PRESENTATION[row.status]
                    table.add_row(symbol, row.capability_id, row.title)
                    if details:
                        table.add_row("", "Canonical inputs", ", ".join(row.required_inputs))
                        table.add_row("", "Status", label)
                        table.add_row("", "Remaining work", row.detail)
                self.print(Panel(table, title=group_name, border_style="blue"))
            else:
                self.print(group_name)
                for row in group:
                    symbol, label = self.STATUS_PRESENTATION[row.status]
                    self.print(f"  {symbol} {row.capability_id} — {row.title}")
                    if details:
                        self.print(f"    Canonical inputs: {', '.join(row.required_inputs)}")
                        self.print(f"    Status: {label}")
                        self.print(f"    Remaining work: {row.detail}")

    def capability(self, capability_id: str) -> bool:
        normalized = capability_id.strip().upper()
        row = next((item for item in all_capability_rows()
                    if item.capability_id.upper() == normalized), None)
        if row is None:
            self.error("validation", f"Unknown capability ID: {capability_id}")
            return False
        symbol, label = self.STATUS_PRESENTATION[row.status]
        values = (
            ("Title", row.title),
            ("Status", f"{symbol} {label}"),
            ("Canonical inputs", ", ".join(row.required_inputs)),
            ("Backend module", row.node_module),
            ("Remaining work", row.detail),
        )
        if self.rich:
            table = Table.grid(padding=(0, 2))
            for key, value in values:
                table.add_row(f"[bold]{key}[/bold]", value)
            self.print(Panel(table, title=row.capability_id, border_style="blue"))
        else:
            self.print(row.capability_id)
            for key, value in values:
                self.print(f"{key}: {value}")
        return True

    def atlas(self, atlas: AtlasSummary) -> None:
        values = (
            ("Path", str(atlas.path)),
            ("Dimensions", f"{atlas.n_cells:,} cells × {atlas.n_genes:,} genes"),
            ("Cell states", str(len(atlas.cell_states))),
            ("Groups", f"{len(atlas.groups)} available"),
            ("Group labels", ", ".join(disease_group_label(group) for group in atlas.groups)),
            ("Metadata", f"{len(atlas.metadata_columns)} columns"),
        )
        if self.rich:
            table = Table.grid(padding=(0, 2))
            for key, value in values:
                table.add_row(f"[bold]{key}[/bold]", value)
            self.print(Panel(table, title="Atlas", border_style="cyan"))
        else:
            self.print("Atlas")
            for key, value in values:
                self.print(f"{key}: {value}")

    def status(self, values: dict[str, str]) -> None:
        for name, value in values.items():
            self.print(f"{name}: {value}")

    def progress(self, stage: str) -> None:
        labels = {
            "critic": "OpenAI Critic evaluating evidence",
            "interpreter": "OpenAI Interpreter synthesizing findings",
            "report": "ScientificReport assembly",
        }
        self.print(f"◐ {labels.get(stage, stage)}")

    def error(self, kind: str, message: str, output_dir: Path | None = None) -> None:
        body = expand_disease_group_ids(message) + (f"\nPartial output: {output_dir}" if output_dir else "")
        if self.rich:
            colors = {
                "validation": "yellow", "scientifically blocked": "magenta",
                "missing configuration": "yellow", "runtime": "red",
                "reasoning": "yellow",
            }
            self.print(Panel(body, title=kind.title(),
                             border_style=colors.get(kind, "red")))
        else:
            self.print(f"{kind.upper()}: {body}")

    def final(self, result: ApplicationRunResult) -> None:
        de = result.de
        lines = [
            f"Cell state: {result.plan.cell_state}",
            f"Contrast: {result.plan.display_contrast}",
            f"Overall status: {result.overall_status}",
            f"Output directory: {result.run_dir}",
            f"Elapsed: {result.elapsed_seconds:.1f}s",
        ]
        if de is not None:
            lines.extend([
                "",
                "Differential expression",
                f"Tested genes: {de.tested_gene_count}",
                f"Significant genes: {de.significant_gene_count}",
                f"Higher in {disease_group_label(result.plan.group_a)}: {de.upregulated_in_group_a_count}",
                f"Higher in {disease_group_label(result.plan.group_b)}: {de.upregulated_in_group_b_count}",
                f"Design: {de.design_assessment.design_formula}",
                f"Cache: {'restored' if de.cache_hit else 'new'}",
            ])
        consensus_regulators = result.tf_summary.get(
            "directional_consensus_regulators", []
        )
        consensus_count = result.tf_summary.get(
            "directional_consensus_count", 0
        )
        consensus_display = str(consensus_count)
        if consensus_regulators:
            consensus_display += f" ({', '.join(consensus_regulators)})"

        lines.extend([
            "",
            "TF activity",
            f"Status: {result.tf_status}",
            f"Resources: {', '.join(result.tf_resources) or 'not configured'}",
            (
                "Estimable TF-resource models: "
                f"{result.tf_summary.get('estimable_model_count', 0)}"
            ),
            (
                "Significant resource-level results: "
                f"{result.tf_summary.get('significant_activity_count', 0)}"
            ),
            (
                "Significant in ≥1 resource: "
                f"{result.tf_summary.get('significant_regulator_count', 0)}"
            ),
            f"Cross-resource consensus: {consensus_display}",
        ])
        if result.reasoning is not None:
            critic = result.reasoning.critic_report
            lines.extend([
                "",
                "Evidence quality",
                f"Confidence: {critic.overall_confidence}",
                f"Evidence sufficiency: {critic.statistical_support_assessment.summary}",
                "Major limitations: " + "; ".join(critic.limitations[:3]),
            ])
        artifact_paths = []
        for output in (result.de, result.tf):
            if output is None:
                continue
            artifacts = (
                output.artifacts if hasattr(output, "artifacts")
                else output.to_capability_result().artifacts
            )
            for artifact in artifacts:
                if Path(artifact.path).exists() and artifact.logical_name in {
                    "deseq2_results", "summary", "provenance",
                    "tf_activity_by_resource", "tf_activity_consensus",
                    "heatmap_source_table",
                }:
                    artifact_paths.append(
                        f"{output.capability_id} {artifact.logical_name}: {artifact.path}"
                    )
        if result.evidence_bundle_path:
            artifact_paths.append(f"EvidenceBundle: {result.evidence_bundle_path}")
        if result.reasoning is not None:
            artifact_paths.extend([
                f"CriticReport: {result.reasoning.critic_report_path}",
                f"InterpretationReport: {result.reasoning.interpretation_report_path}",
                f"ScientificReport: {result.reasoning.scientific_report_path}",
            ])
            presentation = getattr(result.reasoning, "presentation", None)
            if presentation is not None and presentation.pdf_path:
                artifact_paths.append(f"Scientific report PDF: {presentation.pdf_path}")
            if presentation is not None:
                artifact_paths.extend(
                    f"Report figure: {path}" for path in presentation.figure_paths
                )
            if presentation is not None and presentation.warnings:
                lines.extend([
                    "",
                    "Presentation warnings",
                    *(f"{warning.code}: {warning.message}" for warning in presentation.warnings),
                ])
        if artifact_paths:
            lines.extend(["", "Artifacts", *artifact_paths])
        title = "CellState Agent results"
        if self.rich:
            self.print(Panel("\n".join(lines), title=title, border_style="green"))
            if result.reasoning is not None:
                report = result.reasoning.scientific_report
                preview = (
                    f"## Scientific report\n\n{report.deterministic_executive_summary}\n\n"
                    f"{report.interpretation_report.summary}"
                )
                self.print(Markdown(preview))
        else:
            self.print(title)
            self.print("\n".join(lines))
