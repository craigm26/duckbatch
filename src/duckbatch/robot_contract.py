"""What a real Microduck will install, checked with Pollen's own code before anything leaves here.

    duckbatch check records/<batch>/policies/finetuned/policy.onnx [--manifest manifest.json]

WHY POLLEN'S CODE AND NOT A COPY OF IT. `mjlab_microduck.publish.manifest` is what Pollen runs
before publishing a policy, and its checks are the ones the robot's daemon applies at load:
one input and one output, 61 in and 14 out, fifty steps of finite, non-constant actions, and a
manifest whose `kind`, `command.encoding` and `duration_s` are ones the daemon can drive. A
second implementation here would be a second opinion about a contract that already has an owner.

WHY EVERY FINE-TUNE NOW WRITES A MANIFEST. Without one a network is a file nobody can install
with `robotctl`; with one built by Pollen's `build_manifest`, the published repo is exactly the
shape `robotctl policy load walk <repo>` expects. A walking fine-tune is a gait: `kind:
perpetual`, `slot: walk`, the twist fed from the sticks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mjlab_microduck.publish import manifest as pollen


def check(policy: str | Path, manifest: dict[str, Any] | None = None) -> list[tuple[str, bool, str]]:
    """Run Pollen's checks. Each row is (check, passed, detail); nothing raises."""
    rows: list[tuple[str, bool, str]] = []
    try:
        shape = pollen.check_onnx(Path(policy))
        rows.append(("onnx shape (1 in, 1 out, 61 -> 14)", True, str(shape)))
    except Exception as e:  # noqa: BLE001 — every refusal is reported, not raised
        rows.append(("onnx shape (1 in, 1 out, 61 -> 14)", False, str(e)))
    try:
        pollen.smoke_run_onnx(Path(policy))
        rows.append(("50-step smoke run (finite, not constant)", True, "ok"))
    except Exception as e:  # noqa: BLE001
        rows.append(("50-step smoke run (finite, not constant)", False, str(e)))
    if manifest is not None:
        try:
            pollen.validate_manifest(manifest)
            rows.append(("manifest (daemon would load it)", True, "ok"))
        except Exception as e:  # noqa: BLE001
            rows.append(("manifest (daemon would load it)", False, str(e)))
    return rows


def gait_manifest(name: str, description: str, training: dict[str, Any]) -> dict[str, Any]:
    """A walking fine-tune's manifest: a gait for the `walk` slot, built by Pollen's builder."""
    return pollen.build_manifest(
        name=name, kind="perpetual", slot="walk", description=description,
        command_help={"twist": "vx, vy, wz from the sticks", "head": "zeros", "body": "zeros"},
        training=training)


def write_manifest(path: Path, manifest: dict[str, Any]) -> Path:
    path.write_text(pollen.dump_manifest(manifest) if hasattr(pollen, "dump_manifest")
                    else json.dumps(manifest, indent=2))
    return path
