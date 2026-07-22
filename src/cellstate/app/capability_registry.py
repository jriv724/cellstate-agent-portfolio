"""Application readiness layered over canonical capability specifications."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from cellstate.capability_specs import CAPABILITY_SPECS_BY_ID
from .tf_resource_validation import validate_tf_resource_pair


@dataclass(frozen=True)
class CapabilityConnection:
    capability_id: str
    title: str
    status: str
    detail: str
    executable: bool
    resources: tuple[Path, ...] = ()
    node_module: str = ""
    required_inputs: tuple[str, ...] = ()


def tf_resource_paths() -> tuple[Path | None, Path | None]:
    dorothea = os.getenv("CELLSTATE_DOROTHEA_PATH")
    collectri = os.getenv("CELLSTATE_COLLECTRI_PATH")
    return (
        Path(dorothea).expanduser() if dorothea else None,
        Path(collectri).expanduser() if collectri else None,
    )


def build_capability_registry() -> dict[str, CapabilityConnection]:
    de = CAPABILITY_SPECS_BY_ID["CAP-DESEQ-003"]
    tf = CAPABILITY_SPECS_BY_ID["CAP-TF-002"]
    net = CAPABILITY_SPECS_BY_ID["CAP-TF-001"]
    dorothea, collectri = tf_resource_paths()
    validations = validate_tf_resource_pair(dorothea, collectri)
    configured = all(item.valid for item in validations)
    any_configured = bool(dorothea or collectri)
    resources_exist = all(item.exists for item in validations)
    invalid = any_configured and resources_exist and not configured
    tf_resources = tuple(
        path for path in (dorothea, collectri) if path is not None
    )
    return {
        de.capability_id: CapabilityConnection(
            de.capability_id,
            de.name,
            "connected",
            "Ordered sample-level contrasts; CAP-DESEQ-001 adapter prerequisite.",
            True,
            node_module="arbitrary_two_group_de",
            required_inputs=de.accepted_data_representation,
        ),
        tf.capability_id: CapabilityConnection(
            tf.capability_id,
            tf.name,
            "connected" if configured else (
                "invalid resource configuration" if invalid
                else "connected but configuration required"
            ),
            (
                "Production DoRothEA and CollecTRI resources are ready."
                if configured else
                "Configured TF resources failed schema validation."
                if invalid else
                "Requires valid DoRothEA and CollecTRI resource paths."
            ),
            configured,
            tf_resources,
            "tf_activity",
            tf.accepted_data_representation,
        ),
        net.capability_id: CapabilityConnection(
            net.capability_id,
            net.name,
            "requires explicit caller inputs",
            "Requires an explicit feature program and tested-feature background.",
            False,
            node_module="tf_regulatory_network",
            required_inputs=net.accepted_data_representation,
        ),
    }


def all_capability_rows() -> list[CapabilityConnection]:
    connected = build_capability_registry()
    rows = list(connected.values())
    for capability_id, spec in CAPABILITY_SPECS_BY_ID.items():
        if capability_id not in connected:
            module = {
                "CAP-COMP-001": "abundance",
                "CAP-COMP-002": "progression",
                "CAP-STAT-001": "age_association",
                "CAP-STAT-002": "age_association",
                "CAP-STAT-003": "progression",
                "CAP-STAT-004": "progression",
                "CAP-DESEQ-001": "pseudobulk_de",
                "CAP-DESEQ-002": "deseq2",
                "CAP-LODO-001": "atlas_lodo",
            }[capability_id]
            status = (
                "requires explicit caller inputs"
                if capability_id in {"CAP-STAT-001", "CAP-STAT-002",
                                     "CAP-COMP-002", "CAP-STAT-003",
                                     "CAP-STAT-004", "CAP-DESEQ-002",
                                     "CAP-LODO-001"}
                else "backend implemented but not yet exposed"
            )
            rows.append(CapabilityConnection(
                capability_id, spec.name, status,
                "Canonical backend exists; this CLI has no safe request adapter yet.",
                False, node_module=module,
                required_inputs=spec.accepted_data_representation,
            ))
    return rows
