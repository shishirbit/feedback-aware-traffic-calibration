"""Run the registered 2 dataset x 2 scenario x 3 x 3 CoRel audit matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--datasets", nargs="+", default=["metr_la", "pems_bay"])
    parser.add_argument("--scenarios", nargs="+", default=["C3", "C4"])
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--fault-seeds", nargs="+", type=int, default=[101, 202, 303])
    args = parser.parse_args()

    script = Path(__file__).with_name("evaluate_corel_ps1.py").resolve()
    root = args.artifacts.resolve()
    matrix_root = root / "corel-evaluation"
    matrix_root.mkdir(parents=True, exist_ok=True)
    runs = []
    started = time.perf_counter()
    total = len(args.datasets) * len(args.scenarios) * len(args.model_seeds) * len(args.fault_seeds)
    for dataset in args.datasets:
        for scenario in args.scenarios:
            for model_seed in args.model_seeds:
                for fault_seed in args.fault_seeds:
                    name = f"{dataset}-seed{model_seed}-{scenario}-fault{fault_seed}"
                    output = matrix_root / name
                    manifest = output / "manifest.json"
                    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")).get("status") == "completed":
                        status = "existing"
                    else:
                        command = [sys.executable, str(script),
                            "--calibration-cache", str(root / "predictions" / f"{dataset}-seed{model_seed}" / "calibration"),
                            "--test-cache", str(root / "predictions" / f"{dataset}-seed{model_seed}" / f"{scenario}-fault{fault_seed}"),
                            "--fault-dir", str(root / "faults" / dataset / f"{scenario}-seed{fault_seed}"),
                            "--checkpoint", str(root / "corel" / f"{dataset}-seed{model_seed}-full" / "model.pt"),
                            "--output", str(output)]
                        subprocess.run(command, check=True)
                        status = "completed"
                    runs.append({"run": name, "status": status})
                    (matrix_root / "progress.json").write_text(json.dumps({
                        "status": "running", "completed": len(runs), "total": total,
                        "last_run": name, "runtime_seconds": time.perf_counter() - started,
                    }, indent=2) + "\n", encoding="utf-8")
    summary = {"status": "completed", "runs": runs, "total": total,
               "runtime_seconds": time.perf_counter() - started}
    (matrix_root / "matrix-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "total": total,
                      "runtime_seconds": summary["runtime_seconds"]}))


if __name__ == "__main__":
    main()
