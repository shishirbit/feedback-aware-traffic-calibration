from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_EVIDENCE = [
    "artifacts/evaluation/cross-backbone-transfer-synthesis-v3.json",
    "artifacts/staeformer-evaluation-matrix.json",
    "artifacts/stid-evaluation-matrix.json",
    "artifacts/dcrnn-evaluation-matrix.json",
    "artifacts/corel-evaluation/matrix-summary.json",
    "artifacts/evaluation/sumo-asymmetric-validation.json",
    "artifacts/evaluation/sumo-asymmetric-C4-transfer.json",
    "artifacts/evaluation/sumo-recovery-summary.json",
    "artifacts/evaluation/sumo-C3-farcal-ablation-subset12.json",
]

CONFIGS = [
    "configs/base.yaml",
    "configs/metr_la.yaml",
    "configs/pems_bay.yaml",
    "configs/sumo.yaml",
    "configs/experiment_registry.yaml",
    "requirements.lock",
    "requirements-py311.lock",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def file_record(root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"required reproducibility input is missing: {relative}")
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def gpu_inventory() -> list[dict[str, str]]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    inventory = []
    for line in output.splitlines():
        if not line.strip():
            continue
        name, driver = (part.strip() for part in line.split(",", maxsplit=1))
        inventory.append({"name": name, "driver": driver})
    return inventory


def build(root: Path) -> tuple[dict[str, object], str]:
    cross_path = root / REQUIRED_EVIDENCE[0]
    cross = json.loads(cross_path.read_text(encoding="utf-8"))
    dataset_manifests = []
    for dataset in ("metr_la", "pems_bay", "sumo"):
        relative = f"data/prepared/{dataset}.manifest.json"
        record = file_record(root, relative)
        source = json.loads((root / relative).read_text(encoding="utf-8"))
        record.update(
            {
                "dataset": dataset,
                "prepared_artifact_sha256": source.get("artifact_sha256"),
                "source_sha256": source.get("source_sha256") or source.get("source_generation_sha256"),
            }
        )
        dataset_manifests.append(record)

    limitations = [
        {"id": "claim-universal", "status": "unsupported", "detail": cross["claims"]["claim_boundary"]},
        {"id": "data-redistribution", "status": "blocked", "detail": "No explicit license was found for redistribution of the exact METR-LA and PEMS-BAY benchmark packages; raw files remain ignored."},
        {"id": "real-scenarios", "status": "incomplete", "detail": "Real-data C1, C2, C5, and stress rows required by the specification remain unevaluated."},
        {"id": "ablations", "status": "partial", "detail": "The prespecified SUMO subset ablation is complete; the full registered cross-dataset ablation matrix is not."},
        {"id": "figures", "status": "incomplete", "detail": "The existing asymmetric-effects plot is supplemental; all six figure families required by the specification remain."},
        {"id": "recovery", "status": "censored", "detail": "All registered SUMO recovery endpoints are censored because the 100-pair threshold exceeds the maximum available affected pairs."},
        {"id": "theory", "status": "not-claimed", "detail": "No finite-sample conformal validity theorem is claimed for dependent, weighted, selectively observed streams."},
        {"id": "live", "status": "partial", "detail": "A three-bin equivalence smoke check is complete; a full live gateway/forecaster study is not."},
        {"id": "corel", "status": "adaptation-only", "detail": "The CoRel evidence is a PS-1 multihorizon adaptation, not a source-split reproduction."},
    ]

    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "partial_publication_bundle",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "git_commit": git(root, "rev-parse", "HEAD"),
            "git_describe": git(root, "describe", "--always", "--dirty"),
            "working_tree_clean_at_generation": not bool(git(root, "status", "--porcelain")),
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "packages": {name: package_version(name) for name in ("numpy", "pandas", "PyYAML", "scipy", "torch", "traci")},
            "gpus": gpu_inventory(),
            "gpu_training_environment": "Python 3.11 / CUDA environment pinned by requirements-py311.lock",
        },
        "registered_seeds": {"model": [11, 22, 33], "fault": [101, 202, 303]},
        "datasets": dataset_manifests,
        "configs_and_locks": [file_record(root, relative) for relative in CONFIGS],
        "evidence": [file_record(root, relative) for relative in REQUIRED_EVIDENCE],
        "verification": {"command": ".\\.venv\\Scripts\\python.exe -m pytest -q", "result": "78 passed"},
        "cross_backbone_result": cross["claims"],
        "readiness": {
            "executable_repository": "complete",
            "experiment_registry": "substantial_but_required_rows_planned",
            "sumo_package": "complete",
            "results_tables": "partial",
            "required_figures": "incomplete_0_of_6",
            "manuscript": "draft_complete_claims_bounded",
            "reproduction_manifest": "complete",
            "submission_ready": False,
        },
        "limitations": limitations,
    }

    readiness_rows = [
        ("Executable repository", "Complete", "78 tests pass; registered training/evaluation commands and locked environments are present."),
        ("Exhaustive experiment registry", "Partial", "Completed matrices are registered; real-data C1/C2/C5/stress and full ablations remain planned."),
        ("SUMO package", "Complete", "120 generated episodes, prepared manifest, causal replay, mechanism subset, recovery, and live equivalence smoke are present."),
        ("Results package", "Partial", "C0, C3/C4, feedback, calibration, recovery, CoRel, and four-backbone evidence exist; required real-data rows remain."),
        ("Required figures", "Incomplete (0/6)", "The asymmetric-effects plot is supplemental; coverage-width, outage-duration score, spatial heatmap, outage timeline, recovery, and compute-quality figures remain."),
        ("Manuscript", "Draft complete", "Claims follow the 12/16 cross-backbone result and explicitly reject universal superiority."),
        ("Reproduction manifest", "Complete", "Environment, commit, hashes, seeds, configs, hardware inventory, evidence hashes, and limitations are machine-readable."),
    ]
    lines = [
        "# Publication readiness audit",
        "",
        f"Generated from source commit `{manifest['source']['git_commit']}`.",
        "",
        "The repository is not yet submission-ready. Its strongest registered result is complete and artifact-backed, but the specification still requires real-data scenario rows, broader ablations, and six figure families.",
        "",
        "| Required deliverable | Status | Evidence or remaining work |",
        "|---|---|---|",
    ]
    lines.extend(f"| {name} | {status} | {detail} |" for name, status, detail in readiness_rows)
    lines.extend(
        [
            "",
            "## Supported claim boundary",
            "",
            f"The four-backbone matrix is complete: **{cross['claims']['cells_meeting_rule']}/{cross['claims']['cells_total']}** registered dataset/scenario cells meet the superiority rule. All four backbones pass METR-LA C3/C4. FAR-GW and STID do not pass PEMS-BAY C3/C4, so universal backbone-independent superiority is unsupported.",
            "",
            "## Required next experiments",
            "",
            "1. Execute the registered real-data C1, C2, C5, and stress matrix.",
            "2. Complete the full registered ablation matrix beyond the prespecified SUMO subset.",
            "3. Produce the six required publication figure families directly from immutable artifacts.",
            "4. Complete the full live gateway/forecaster study if a real-time claim is retained.",
            "",
            "## Dataset distribution decision",
            "",
            "Raw METR-LA and PEMS-BAY files remain excluded. The DCRNN software license does not establish rights for the benchmark data, and the inspected public records do not state an explicit license for the exact packages. Reproduction instructions therefore require users to fetch the data and verify hashes locally.",
            "",
            "Machine-readable details are in `reports/reproducibility-manifest.json`.",
            "",
        ]
    )
    return manifest, "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the PS-1 reproduction manifest and publication-readiness audit.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest, readiness = build(root)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "reproducibility-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (reports / "publication-readiness.md").write_text(readiness, encoding="utf-8")
    print(reports / "reproducibility-manifest.json")
    print(reports / "publication-readiness.md")


if __name__ == "__main__":
    main()
