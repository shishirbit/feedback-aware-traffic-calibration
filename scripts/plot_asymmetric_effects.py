"""Render the artifact-backed asymmetric FAR-Cal comparison figure and source table."""
from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np


ROOT=Path(__file__).resolve().parents[1]
SOURCES=(
    ("SUMO C3", "sumo-asymmetric-validation.json"),
    ("SUMO C4", "sumo-asymmetric-C4-transfer.json"),
    ("METR-LA C3", "metr_la-asymmetric-transfer-audit.json"),
    ("METR-LA C4", "metr_la-asymmetric-C4-transfer-audit.json"),
    ("PEMS-BAY C3", "pems_bay-asymmetric-transfer-audit.json"),
    ("PEMS-BAY C4", "pems_bay-asymmetric-C4-transfer-audit.json"),
)


def main():
    rows=[]
    for condition,name in SOURCES:
        result=json.loads((ROOT/"artifacts/evaluation"/name).read_text(encoding="utf-8"))
        for comparator in ("rolling","aci"):
            effect=result["primary_comparisons"][f"{comparator}_minus_far_cal"]
            rows.append({"condition":condition,"comparator":comparator.upper(),"estimate":effect["estimate"],
                "lower":effect["lower"],"upper":effect["upper"],"coverage":result["far_cal_coverage"],
                "rule_met":result["superiority_rule_met"]})
    output=ROOT/"reports/figures"; output.mkdir(parents=True,exist_ok=True)
    with (output/"asymmetric-effects.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    fig,axis=plt.subplots(figsize=(10,6)); y=np.arange(len(SOURCES)); colors={"ROLLING":"#2673b8","ACI":"#d05a45"}
    for comparator,offset in (("ROLLING",-.16),("ACI",.16)):
        subset=[row for row in rows if row["comparator"]==comparator]; estimate=np.asarray([row["estimate"] for row in subset])
        lower=np.asarray([row["lower"] for row in subset]); upper=np.asarray([row["upper"] for row in subset])
        axis.errorbar(estimate,y+offset,xerr=np.vstack((estimate-lower,upper-estimate)),fmt="o",capsize=4,label=comparator.title(),color=colors[comparator])
    axis.axvline(0,color="#333333",linewidth=1); axis.set_yticks(y,labels=[x[0] for x in SOURCES]); axis.invert_yaxis()
    axis.set_xlabel("Comparator - asymmetric FAR-Cal mean interval score (95% CI)")
    axis.set_title("Asymmetric FAR-Cal effects under correlated feedback impairment")
    axis.grid(axis="x",alpha=.25); axis.legend(frameon=False,ncols=2); fig.tight_layout()
    fig.savefig(output/"asymmetric-effects.png",dpi=200); fig.savefig(output/"asymmetric-effects.pdf")


if __name__=="__main__": main()
