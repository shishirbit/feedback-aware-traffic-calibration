"""Synthesize registered FAR-Cal transfer evidence across frozen backbones."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ps1.utils.artifacts import sha256_file, write_json


AUDITS = {
    ("far_gw", "metr_la", "C3"): "metr_la-asymmetric-transfer-audit.json",
    ("far_gw", "metr_la", "C4"): "metr_la-asymmetric-C4-transfer-audit.json",
    ("far_gw", "pems_bay", "C3"): "pems_bay-asymmetric-transfer-audit.json",
    ("far_gw", "pems_bay", "C4"): "pems_bay-asymmetric-C4-transfer-audit.json",
    ("staeformer", "metr_la", "C3"): "staeformer-metr_la-asymmetric-C3-transfer-audit.json",
    ("staeformer", "metr_la", "C4"): "staeformer-metr_la-asymmetric-C4-transfer-audit.json",
    ("staeformer", "pems_bay", "C3"): "staeformer-pems_bay-asymmetric-C3-transfer-audit.json",
    ("staeformer", "pems_bay", "C4"): "staeformer-pems_bay-asymmetric-C4-transfer-audit.json",
    ("stid", "metr_la", "C3"): "stid-metr_la-asymmetric-C3-transfer-audit.json",
    ("stid", "metr_la", "C4"): "stid-metr_la-asymmetric-C4-transfer-audit.json",
    ("stid", "pems_bay", "C3"): "stid-pems_bay-asymmetric-C3-transfer-audit.json",
    ("stid", "pems_bay", "C4"): "stid-pems_bay-asymmetric-C4-transfer-audit.json",
    ("dcrnn", "metr_la", "C3"): "dcrnn-metr_la-asymmetric-C3-transfer-audit.json",
    ("dcrnn", "metr_la", "C4"): "dcrnn-metr_la-asymmetric-C4-transfer-audit.json",
    ("dcrnn", "pems_bay", "C3"): "dcrnn-pems_bay-asymmetric-C3-transfer-audit.json",
    ("dcrnn", "pems_bay", "C4"): "dcrnn-pems_bay-asymmetric-C4-transfer-audit.json",
}


def synthesize(evaluation_dir: Path) -> dict:
    cells = []
    sources = []
    for (backbone, dataset, scenario), name in AUDITS.items():
        path = evaluation_dir / name
        audit = json.loads(path.read_text(encoding="utf-8"))
        if audit["dataset"] != dataset or audit["scenario"] != scenario:
            raise ValueError(f"audit identity mismatch: {path}")
        comparisons = audit["primary_comparisons"]
        cells.append({
            "backbone": backbone,
            "dataset": dataset,
            "scenario": scenario,
            "far_cal_interval_score": audit["far_cal_interval_score"],
            "far_cal_coverage": audit["far_cal_coverage"],
            "rolling_minus_far_cal": comparisons["rolling_minus_far_cal"],
            "aci_minus_far_cal": comparisons["aci_minus_far_cal"],
            "superiority_rule_met": bool(audit["superiority_rule_met"]),
        })
        sources.append({"path": str(path), "sha256": sha256_file(path)})

    def rules(**wanted):
        return [cell["superiority_rule_met"] for cell in cells
                if all(cell[key] == value for key, value in wanted.items())]

    far_gw = rules(backbone="far_gw")
    staeformer = rules(backbone="staeformer")
    stid = rules(backbone="stid")
    dcrnn = rules(backbone="dcrnn")
    metr_la = rules(dataset="metr_la")
    all_rules = [cell["superiority_rule_met"] for cell in cells]
    return {
        "schema_version": 3,
        "status": "completed",
        "cells": cells,
        "claims": {
            "four_backbone_matrix_complete": len(far_gw) == len(staeformer) == len(stid) == len(dcrnn) == 4,
            "staeformer_superiority_all_registered_cells": all(staeformer),
            "stid_superiority_all_registered_cells": all(stid),
            "dcrnn_superiority_all_registered_cells": all(dcrnn),
            "metr_la_superiority_across_tested_backbones": all(metr_la),
            "universal_backbone_independent_superiority": all(all_rules),
            "cells_meeting_rule": sum(all_rules),
            "cells_total": len(all_rules),
            "far_gw_cells_meeting_rule": sum(far_gw),
            "staeformer_cells_meeting_rule": sum(staeformer),
            "stid_cells_meeting_rule": sum(stid),
            "dcrnn_cells_meeting_rule": sum(dcrnn),
            "claim_boundary": (
                "The result shows consistent METR-LA superiority across four tested backbones, "
                "but does not establish universal backbone-independent superiority because FAR-GW "
                "and STID fail the registered rule on PEMS-BAY C3 and C4."
            ),
        },
        "sources": sources,
    }


def effect_text(effect: dict) -> str:
    return f"{effect['estimate']:.4f} [{effect['lower']:.4f}, {effect['upper']:.4f}]"


def render_markdown(result: dict) -> str:
    rows = []
    labels = {"far_gw": "FAR-GW", "staeformer": "STAEformer", "stid": "STID", "dcrnn": "DCRNN",
              "metr_la": "METR-LA", "pems_bay": "PEMS-BAY"}
    for cell in result["cells"]:
        rows.append(
            f"| {labels[cell['backbone']]} | {labels[cell['dataset']]} | {cell['scenario']} | "
            f"{effect_text(cell['rolling_minus_far_cal'])} | "
            f"{effect_text(cell['aci_minus_far_cal'])} | {cell['far_cal_coverage']:.4f} | "
            f"{'Met' if cell['superiority_rule_met'] else 'Not met'} |"
        )
    claims = result["claims"]
    return "\n".join([
        "# Cross-backbone FAR-Cal transfer synthesis",
        "",
        "| Frozen backbone | Dataset | Condition | Rolling − FAR-Cal (95% CI) | ACI − FAR-Cal (95% CI) | Coverage | Rule |",
        "|---|---|---|---:|---:|---:|---:|",
        *rows,
        "",
        f"The registered superiority rule was met in **{claims['cells_meeting_rule']} of "
        f"{claims['cells_total']}** backbone/dataset/condition cells: "
        f"{claims['far_gw_cells_meeting_rule']}/4 with FAR-GW, "
        f"{claims['staeformer_cells_meeting_rule']}/4 with STAEformer, "
        f"{claims['stid_cells_meeting_rule']}/4 with STID, and "
        f"{claims['dcrnn_cells_meeting_rule']}/4 with DCRNN. "
        "All four tested backbones met the rule on METR-LA C3 and C4. STAEformer and DCRNN also "
        "met it on PEMS-BAY C3 and C4, whereas FAR-GW and STID did not.",
        "",
        "This is evidence that the method can transfer across multiple frozen forecasting "
        "backbones on METR-LA. It does not support a universal backbone-independent superiority claim: "
        "the PEMS-BAY result changes with the frozen backbone, and the audits do not include "
        "a formal interaction test between backbone and calibrator effects.",
        "",
        "The direct CoRel adaptation remains a separate calibration-method comparison on "
        "the FAR-GW prediction caches; it is not evidence from an additional point backbone.",
        "",
        "Machine-readable evidence: `artifacts/evaluation/cross-backbone-transfer-synthesis-v3.json`.",
        "",
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    result = synthesize(root / "artifacts/evaluation")
    write_json(root / "artifacts/evaluation/cross-backbone-transfer-synthesis-v3.json", result)
    report = root / "reports/cross-backbone-transfer-synthesis.md"
    report.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result["claims"], indent=2))


if __name__ == "__main__":
    main()
