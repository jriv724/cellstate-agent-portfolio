"""Presentation-only disease-group labels; canonical IDs remain unchanged."""

from __future__ import annotations

import re


DISEASE_GROUP_LABELS: dict[str, str] = {
    "NBM": "Normal Bone Marrow (NBM)",
    "MGUS": "Monoclonal Gammopathy of Undetermined Significance (MGUS)",
    "SMM": "Smoldering Multiple Myeloma (SMM)",
    "NDMM": "Newly Diagnosed Multiple Myeloma (NDMM)",
    "RRMM": "Relapsed/Refractory Multiple Myeloma (RRMM)",
    "MM-Remission": "Multiple Myeloma in Remission (MM-Remission)",
}


def disease_group_label(canonical_id: str) -> str:
    return DISEASE_GROUP_LABELS.get(canonical_id, canonical_id)


def disease_contrast_label(group_a: str, group_b: str) -> str:
    return f"{disease_group_label(group_a)} − {disease_group_label(group_b)}"


def expand_disease_group_ids(text: str) -> str:
    """Expand standalone canonical IDs in user-facing free text."""
    rendered = str(text)
    for canonical_id in sorted(DISEASE_GROUP_LABELS, key=len, reverse=True):
        label = DISEASE_GROUP_LABELS[canonical_id]
        rendered = re.sub(
            rf"(?<![A-Za-z0-9-]){re.escape(canonical_id)}(?![A-Za-z0-9-]|\))",
            label,
            rendered,
        )
    return rendered
