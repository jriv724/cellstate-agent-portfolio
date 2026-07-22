"""Deterministic semantic intake for canonical atlas comparisons."""

from __future__ import annotations

import re
import json
import os
from pathlib import Path
from typing import Callable

from .models import AnalysisPlan, AtlasSummary


def _normalized(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.casefold())).strip()


GROUP_ALIASES = {
    "normal bone marrow": "NBM",
    "newly diagnosed multiple myeloma": "NDMM",
    "newly diagnosed myeloma": "NDMM",
    "relapsed refractory multiple myeloma": "RRMM",
    "relapsed refractory myeloma": "RRMM",
    "smoldering multiple myeloma": "SMM",
}


class SemanticPlanner:
    def __init__(
        self,
        atlas: AtlasSummary,
        *,
        gemini_parser: Callable[[str, AtlasSummary], AnalysisPlan] | None = None,
        semantic_cache_path: Path | None = None,
    ) -> None:
        self.atlas = atlas
        self.gemini_parser = gemini_parser
        self.semantic_cache_path = semantic_cache_path

    def _cached(self, question: str) -> AnalysisPlan | None:
        if self.semantic_cache_path is None or not self.semantic_cache_path.is_file():
            return None
        try:
            payload = json.loads(self.semantic_cache_path.read_text())
            item = payload.get(_normalized(question))
            if item is None:
                return None
            return AnalysisPlan(
                question=question,
                cell_state=item["cell_state"],
                group_a=item["group_a"],
                group_b=item["group_b"],
                requested_capabilities=tuple(item["requested_capabilities"]),
                reasoning_requested=bool(item.get("reasoning_requested", True)),
                assumptions=tuple(item.get("assumptions", ())),
                confounded_design_policy=item.get("confounded_design_policy", "block"),
            )
        except Exception:
            return None

    def _store(self, plan: AnalysisPlan) -> None:
        if self.semantic_cache_path is None:
            return
        try:
            payload = (
                json.loads(self.semantic_cache_path.read_text())
                if self.semantic_cache_path.is_file() else {}
            )
            payload[_normalized(plan.question)] = {
                "cell_state": plan.cell_state,
                "group_a": plan.group_a,
                "group_b": plan.group_b,
                "requested_capabilities": list(plan.requested_capabilities),
                "reasoning_requested": plan.reasoning_requested,
                "assumptions": list(plan.assumptions),
                "confounded_design_policy": plan.confounded_design_policy,
            }
            self.semantic_cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.semantic_cache_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
            os.replace(temporary, self.semantic_cache_path)
        except OSError:
            pass

    @staticmethod
    def requested_capabilities(question: str) -> tuple[str, ...]:
        text = _normalized(question)
        requested = ["CAP-DESEQ-003"]
        tf_terms = (
            "signed tf activity", "tf activity", "transcription factor activity",
            "transcription factor", "regulator activity",
        )
        if any(term in text for term in tf_terms):
            requested.append("CAP-TF-002")
        if "tf regulatory enrichment" in text or "regulatory enrichment" in text:
            requested.append("CAP-TF-001")
        return tuple(requested)

    @staticmethod
    def reasoning_requested(question: str) -> bool:
        # Reasoning is the production default; explicit language only reinforces it.
        return True

    def _ordered_matches(self, question: str, values: tuple[str, ...]) -> list[str]:
        normalized = _normalized(question)
        matches = []
        for value in values:
            phrase = _normalized(value)
            position = normalized.find(phrase)
            if position >= 0:
                matches.append((position, value))
        for phrase, canonical in GROUP_ALIASES.items():
            position = normalized.find(phrase)
            if position >= 0 and canonical in values:
                matches.append((position, canonical))
        ordered = []
        for _, value in sorted(matches):
            if value not in ordered:
                ordered.append(value)
        return ordered

    def parse(self, question: str) -> AnalysisPlan:
        cached = self._cached(question)
        if cached is not None:
            return cached
        groups = self._ordered_matches(question, self.atlas.groups)
        states = self._ordered_matches(question, self.atlas.cell_states)
        if len(states) == 1 and len(groups) >= 2:
            plan = AnalysisPlan(
                question=question,
                cell_state=states[0],
                group_a=groups[0],
                group_b=groups[1],
                requested_capabilities=self.requested_capabilities(question),
                reasoning_requested=self.reasoning_requested(question),
            )
            self._store(plan)
            return plan
        if self.gemini_parser is not None:
            plan = self.gemini_parser(question, self.atlas)
            self._store(plan)
            return plan
        missing = []
        if len(states) != 1:
            missing.append("one atlas cell state")
        if len(groups) < 2:
            missing.append("two ordered atlas groups")
        raise ValueError("Please clarify " + " and ".join(missing) + ".")
