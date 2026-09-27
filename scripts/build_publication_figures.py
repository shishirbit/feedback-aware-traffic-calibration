"""Build the registered publication figure suite from immutable artifacts.

Every completed figure is exported as vector PDF, 600-dpi PNG, and a CSV
source table. Figures whose registered experiment is unfinished remain in the
manifest with a pending status; the script never fabricates placeholder data.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "publication"
CFG = ROOT / "configs" / "publication_figures.json"
EVAL = ROOT / "artifacts" / "evaluation"

COL = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9", "black": "#222222", "grey": "#777777"}
BACKBONE = {"far_gw": "FAR-GW", "staeformer": "STAEformer", "stid": "STID", "dcrnn": "DCRNN"}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.linewidth": .7, "lines.linewidth": 1.4, "lines.markersize": 4,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
    })


def save(fig, slug: str, rows: list[dict], sources: list[Path], caption: str, manifest: dict):
    OUT.mkdir(parents=True, exist_ok=True)
    pdf, png, table = OUT / f"{slug}.pdf", OUT / f"{slug}.png", OUT / f"{slug}.csv"
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    plt.close(fig)
    fields = sorted({k for r in rows for k in r}) if rows else ["note"]
    with table.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for row in rows: w.writerow(row)
    manifest[slug] = {"status": "generated", "caption": caption,
                      "outputs": [str(x.relative_to(ROOT)) for x in (pdf, png, table)],
                      "sources": [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in sources]}


def fig_system(manifest):
    fig, ax = plt.subplots(figsize=(7.16, 2.7)); ax.axis("off")
    boxes = [(0.02, .55, .17, .28, "Traffic sensors\n+ availability mask", COL["sky"]),
             (.25, .55, .18, .28, "Immutable predictor\nquantile forecasts", COL["blue"]),
             (.50, .55, .19, .28, "Fault-aware replay\nC0–C5 channels", COL["orange"]),
             (.76, .55, .20, .28, "Intervals + point\nquality metrics", COL["green"]),
             (.35, .10, .24, .22, "Delayed-label ledger\nreceived feedback only", COL["purple"])]
    for x,y,w,h,t,c in boxes:
        ax.add_patch(plt.Rectangle((x,y),w,h,facecolor=c,alpha=.18,edgecolor=c,lw=1.3))
        ax.text(x+w/2,y+h/2,t,ha="center",va="center",weight="bold")
    for a,b in [((.19,.69),(.25,.69)),((.43,.69),(.50,.69)),((.69,.69),(.76,.69)),((.595,.55),(.52,.32)),((.59,.21),(.76,.55))]:
        ax.annotate("",xy=b,xytext=a,arrowprops=dict(arrowstyle="->",lw=1.2,color=COL["black"]))
    ax.text(.02,.94,"Registered evaluation pipeline and causal information boundary",weight="bold",fontsize=10)
    ax.text(.02,.39,"Forecasts are never recomputed after feedback arrives; calibration uses only labels released by the ledger.",color=COL["grey"])
    rows=[{"stage":i+1,"label":b[4]} for i,b in enumerate(boxes)]
    save(fig,"fig01-system-and-information-flow",rows,[CFG],"System overview separating predictor inputs, immutable forecasts, fault replay, and delayed calibration feedback.",manifest)


def fig_scenarios(manifest):
    labels=["C0 clean","C1 feedback delay","C2 input faults","C3 correlated outage","C4 buffered recovery","C5 mixed stress"]
    cols=["Predictor input","Feedback channel","Spatial correlation","Recovery backlog"]
    mat=np.array([[0,0,0,0],[0,1,0,0],[1,0,0,0],[1,1,1,0],[1,1,1,1],[1,1,0,1]])
    fig,ax=plt.subplots(figsize=(7.16,2.75)); im=ax.imshow(mat,cmap=mpl.colors.ListedColormap(["#F1F3F5",COL["blue"]]),aspect="auto",vmin=0,vmax=1)
    ax.set_xticks(range(4),cols); ax.set_yticks(range(6),labels); ax.tick_params(axis="x",rotation=18)
    for i in range(6):
        for j in range(4): ax.text(j,i,"affected" if mat[i,j] else "nominal",ha="center",va="center",color="white" if mat[i,j] else COL["grey"],fontsize=7)
    ax.set_title("Prespecified fault scenarios and affected information channels",loc="left",weight="bold")
    rows=[{"scenario":labels[i],**{cols[j]:int(mat[i,j]) for j in range(4)}} for i in range(6)]
    save(fig,"fig02-scenario-design-matrix",rows,[CFG],"Scenario design matrix. Filled cells indicate channels affected by each prespecified condition.",manifest)


def clean_files():
    return sorted(p for p in EVAL.glob("*-seed*-clean.json") if "simple" not in p.name)


def fig_point(manifest):
    rows=[]
    for p in clean_files():
        d=load(p)
        for h,v in d["point"].items(): rows.append({"dataset":d["dataset"],"seed":d.get("model_seed",p.stem.split("seed")[-1].split("-")[0]),"horizon":int(h),"mae":v["mae_mph"],"rmse":v["rmse_mph"]})
    fig,axs=plt.subplots(1,2,figsize=(7.16,2.75),sharex=True)
    for ax,metric in zip(axs,["mae","rmse"]):
        for ds,c,m in [("metr_la",COL["blue"],"o"),("pems_bay",COL["orange"],"s"),("sumo",COL["green"],"^")]:
            rr=[r for r in rows if r["dataset"]==ds]; hs=sorted({r["horizon"] for r in rr}); vals=[[r[metric] for r in rr if r["horizon"]==h] for h in hs]
            if vals: ax.errorbar(hs,[np.mean(v) for v in vals],yerr=[np.std(v) for v in vals],label=ds.upper(),color=c,marker=m,capsize=2)
        ax.set_xlabel("Forecast horizon (5-min bins)"); ax.set_ylabel(metric.upper()+" (mph)"); ax.grid(alpha=.2)
    axs[0].legend(frameon=False); axs[0].set_title("a  Mean absolute error",loc="left",weight="bold"); axs[1].set_title("b  Root mean-square error",loc="left",weight="bold")
    save(fig,"fig03-clean-point-accuracy",rows,clean_files(),"Clean-data point accuracy by forecast horizon; markers show seed means and bars show one standard deviation.",manifest)


def fig_calibration(manifest):
    rows=[]
    for p in clean_files():
        d=load(p)
        for level,horizons in d["intervals"]["frozen"].items():
            for h,v in horizons.items(): rows.append({"dataset":d["dataset"],"seed":d.get("model_seed",0),"level":float(level),"horizon":int(h),"coverage":v["picp"],"width":v["mpiw_mph"],"score":v["mean_interval_score"]})
    fig,axs=plt.subplots(1,2,figsize=(7.16,2.8))
    for ds,c,m in [("metr_la",COL["blue"],"o"),("pems_bay",COL["orange"],"s"),("sumo",COL["green"],"^")]:
        rr=[r for r in rows if r["dataset"]==ds]
        axs[0].scatter([r["coverage"] for r in rr],[r["width"] for r in rr],label=ds.upper().replace("_", "-"),edgecolors=c,marker=m,facecolors="none")
        for level in [.9,.95]:
            q=[r for r in rr if r["level"]==level];
            if q: axs[1].scatter([level]*len(q),[r["coverage"] for r in q],edgecolors=c,marker=m,facecolors="none")
    axs[0].set(xlabel="Empirical coverage",ylabel="Mean interval width (mph)"); axs[0].grid(alpha=.2); axs[0].legend(frameon=False); axs[0].set_title("a  Coverage–width trade-off",loc="left",weight="bold")
    axs[1].plot([.88,.97],[.88,.97],"--",color=COL["grey"],label="ideal"); axs[1].set(xlabel="Nominal coverage",ylabel="Empirical coverage",xlim=(.88,.97),ylim=(.86,1.0)); axs[1].grid(alpha=.2); axs[1].set_title("b  Reliability",loc="left",weight="bold")
    save(fig,"fig04-clean-calibration-quality",rows,clean_files(),"Clean-data interval quality across datasets, seeds, horizons, and 90/95% nominal levels.",manifest)


def fig_transfer(manifest):
    p=EVAL/"cross-backbone-transfer-synthesis-v3.json"; cells=load(p)["cells"]
    cells=sorted(cells,key=lambda x:(x["backbone"],x["dataset"],x["scenario"]))
    fig,axs=plt.subplots(1,2,figsize=(7.16,5.0),sharey=True)
    rows=[]; y=np.arange(len(cells)); labels=[f"{BACKBONE[c['backbone']]} · {c['dataset'].upper().replace('_', '-')} · {c['scenario']}" for c in cells]
    for ax,key,title,color,marker in [(axs[0],"rolling_minus_far_cal","a  Rolling − FAR-Cal",COL["blue"],"o"),(axs[1],"aci_minus_far_cal","b  ACI − FAR-Cal",COL["orange"],"s")]:
        e=np.array([c[key]["estimate"] for c in cells]); lo=np.array([c[key]["lower"] for c in cells]); hi=np.array([c[key]["upper"] for c in cells])
        ax.errorbar(e,y,xerr=[e-lo,hi-e],fmt=marker,color=color,capsize=2); ax.axvline(0,color=COL["black"],lw=.8); ax.set_xlabel("Interval-score difference\n(positive favors FAR-Cal)"); ax.set_title(title,loc="left",weight="bold"); ax.grid(axis="x",alpha=.2)
    axs[0].set_yticks(y,labels); axs[0].invert_yaxis()
    for c in cells:
        rows.append({"backbone":c["backbone"],"dataset":c["dataset"],"scenario":c["scenario"],"superiority_rule_met":c["superiority_rule_met"],"rolling_estimate":c["rolling_minus_far_cal"]["estimate"],"rolling_lower":c["rolling_minus_far_cal"]["lower"],"rolling_upper":c["rolling_minus_far_cal"]["upper"],"aci_estimate":c["aci_minus_far_cal"]["estimate"],"aci_lower":c["aci_minus_far_cal"]["lower"],"aci_upper":c["aci_minus_far_cal"]["upper"]})
    save(fig,"fig05-cross-backbone-transfer",rows,[p],"Cross-backbone paired block-bootstrap effects. Positive differences favor FAR-Cal; 12 of 16 registered cells meet the joint superiority rule.",manifest)


def outage_data():
    pred=ROOT/"artifacts/predictions/metr_la-seed11/C3-fault101"; fault=ROOT/"artifacts/faults/metr_la/C3-seed101/fault_manifest.json"
    fd=load(fault); chunks=[]
    for cp in sorted(pred.glob("chunk-*.npz")):
        z=np.load(cp); chunks.append({k:z[k] for k in ("issue_bin","q05_mph","q50_mph","q95_mph","truth_mph")})
    data={k:np.concatenate([x[k] for x in chunks],axis=0) for k in chunks[0]}
    scored=[]; hidx=5
    for ev in fd["events"]:
        start=fd["test_start_row"]+ev["start_bin"]; end=fd["test_start_row"]+ev["end_bin"]+12; m=(data["issue_bin"]>=start)&(data["issue_bin"]<end); sens=np.array(ev["affected_sensors"])
        y=data["truth_mph"][m,hidx][:,sens]; lo=data["q05_mph"][m,hidx][:,sens]; hi=data["q95_mph"][m,hidx][:,sens]
        raw=(hi-lo)+20*np.maximum(lo-y,0)+20*np.maximum(y-hi,0)
        finite=raw[np.isfinite(raw)]
        if finite.size: scored.append((float(np.mean(finite)),ev))
    if not scored: raise RuntimeError("No registered outage event has finite baseline scores")
    scored.sort(key=lambda x:x[0]); return data,fd,scored[len(scored)//2][1],pred,fault


def fig_outage(manifest):
    data,fd,ev,pred,fault=outage_data(); hidx=5; start=fd["test_start_row"]+ev["start_bin"]; m=(data["issue_bin"]>=start-24)&(data["issue_bin"]<=start+42); sens=np.array(ev["affected_sensors"])
    x=(data["issue_bin"][m]-start)*5; vals={k:np.nanmean(data[k][m,hidx][:,sens],axis=1) for k in ("truth_mph","q05_mph","q50_mph","q95_mph")}
    fig,ax=plt.subplots(figsize=(7.16,2.8)); ax.axvspan(0,(ev["end_bin"]-ev["start_bin"])*5,color=COL["red"],alpha=.12,label="outage")
    ax.fill_between(x,vals["q05_mph"],vals["q95_mph"],color=COL["sky"],alpha=.28,label="raw 90% interval"); ax.plot(x,vals["truth_mph"],color=COL["black"],label="observed speed"); ax.plot(x,vals["q50_mph"],color=COL["blue"],ls="--",label="median forecast")
    ax.set(xlabel="Minutes relative to outage start",ylabel="Mean speed (mph)"); ax.grid(alpha=.2); ax.legend(frameon=False,ncol=4,loc="upper right"); ax.set_title(f"Prespecified median-score event (event {ev['event_index']}, horizon 6)",loc="left",weight="bold")
    rows=[{"minutes":int(xx),**{k:float(v[i]) for k,v in vals.items()}} for i,xx in enumerate(x)]
    sources=[fault]+sorted(pred.glob("chunk-*.npz"))
    save(fig,"fig06-representative-outage-timeline",rows,sources,"Representative METR-LA C3 outage selected by the registered median baseline interval-score rule, not visual favorability.",manifest)


def fig_heatmap(manifest):
    data,fd,ev,pred,fault=outage_data(); hidx=5; start=fd["test_start_row"]+ev["start_bin"]; m=(data["issue_bin"]>=start-12)&(data["issue_bin"]<=start+24); sens=np.array(ev["affected_sensors"])
    truth=data["truth_mph"][m,hidx][:,sens].T; err=np.abs(data["q50_mph"][m,hidx][:,sens].T-truth); x=(data["issue_bin"][m]-start)*5
    fig,axs=plt.subplots(2,1,figsize=(7.16,4.4),sharex=True,constrained_layout=True)
    a=axs[0].imshow(truth,aspect="auto",cmap="viridis",extent=[x[0],x[-1],len(sens)-.5,-.5]); b=axs[1].imshow(err,aspect="auto",cmap="magma",extent=[x[0],x[-1],len(sens)-.5,-.5])
    fig.colorbar(a,ax=axs[0],label="Speed (mph)"); fig.colorbar(b,ax=axs[1],label="Absolute error (mph)")
    axs[0].set_title("a  Observed traffic speed",loc="left",weight="bold"); axs[1].set_title("b  Horizon-6 point error",loc="left",weight="bold"); axs[1].set_xlabel("Minutes relative to outage start")
    for ax in axs: ax.set_ylabel("Affected sensor rank"); ax.axvline(0,color="white",ls="--",lw=.8)
    rows=[]
    for i,s in enumerate(sens):
        for j,t in enumerate(x): rows.append({"sensor":int(s),"minutes":int(t),"speed":float(truth[i,j]),"absolute_error":float(err[i,j])})
    save(fig,"fig07-spatial-speed-error-heatmap",rows,[fault]+sorted(pred.glob("chunk-*.npz")),"Spatial speed and point-error heatmaps for the same prespecified representative outage used in Fig. 6.",manifest)


def fig_recovery(manifest):
    files=sorted(EVAL.glob("sumo-seed*-C[34]-recovery-fault*.json")); agg=defaultdict(list); rows=[]
    for p in files:
        d=load(p)
        for r in d["records"]:
            if r["level"]!=0.9 or r["horizon"]!=6: continue
            for w in r["windows"]:
                key=(d["scenario"],r["method"],w["end_bin"]); agg[key].append((w.get("coverage",np.nan),w.get("score_ratio",np.nan)))
    for (sc,m,b),v in agg.items(): rows.append({"scenario":sc,"method":m,"end_bin":b,"coverage":float(np.nanmean([x[0] for x in v])),"score_ratio":float(np.nanmean([x[1] for x in v])),"n":len(v)})
    fig,axs=plt.subplots(1,2,figsize=(7.16,2.8)); styles={"rolling":("o",COL["blue"]),"aci":("s",COL["orange"]),"far_cal_asymmetric":("^",COL["green"])}
    for sc,ls in [("C3","-"),("C4","--")]:
        for method,(mk,c) in styles.items():
            rr=sorted([r for r in rows if r["scenario"]==sc and r["method"]==method],key=lambda r:r["end_bin"])
            if not rr: continue
            label=f"{method.replace('_asymmetric','')} · {sc}"; axs[0].plot([r["end_bin"] for r in rr],[r["coverage"] for r in rr],ls=ls,marker=mk,color=c,label=label); axs[1].plot([r["end_bin"] for r in rr],[r["score_ratio"] for r in rr],ls=ls,marker=mk,color=c)
    axs[0].axhline(.9,color=COL["grey"],ls=":"); axs[1].axhline(1,color=COL["grey"],ls=":"); axs[0].set(xlabel="Bins after restoration",ylabel="Empirical coverage"); axs[1].set(xlabel="Bins after restoration",ylabel="Score / oracle score")
    for ax in axs: ax.grid(alpha=.2)
    axs[0].set_title("a  Coverage recovery",loc="left",weight="bold"); axs[1].set_title("b  Relative interval score",loc="left",weight="bold"); axs[0].legend(frameon=False,ncol=2,fontsize=6.8)
    fig.subplots_adjust(bottom=.24, wspace=.27)
    fig.text(.5,.055,"Formal recovery endpoints are censored: registered minimum pair count is not reached.",ha="center",color=COL["red"],fontsize=7.5)
    save(fig,"fig08-recovery-curves",rows,files,"SUMO post-restoration trajectories pooled across registered seeds. Formal endpoint estimates are censored by design.",manifest)


def fig_ablation(manifest):
    p=EVAL/"sumo-C3-farcal-ablation-subset12.json"; d=load(p); rows=[]
    names=["no_age","no_context","no_inflation","no_spatial","no_stale_blend"]
    for n in names:
        vals=[d["by_model_seed"][s]["ablations"][n] for s in d["by_model_seed"]]
        rows.append({"component_removed":n,"estimate":float(np.mean([v["estimate"] for v in vals])),"lower":float(np.mean([v["lower"] for v in vals])),"upper":float(np.mean([v["upper"] for v in vals]))})
    fig,ax=plt.subplots(figsize=(7.16,2.9)); y=np.arange(len(rows)); e=np.array([r["estimate"] for r in rows]); lo=np.array([r["lower"] for r in rows]); hi=np.array([r["upper"] for r in rows]); ax.errorbar(e,y,xerr=[e-lo,hi-e],fmt="o",color=COL["purple"],capsize=2); ax.axvline(0,color=COL["black"],lw=.8); ax.set_yticks(y,[r["component_removed"].replace("no_","Remove ").replace("_"," ") for r in rows]); ax.invert_yaxis(); ax.set_xlabel("Ablation − full FAR-Cal interval score\n(positive means the removed component helped)"); ax.grid(axis="x",alpha=.2); ax.set_title("Registered SUMO C3 ablation subset",loc="left",weight="bold")
    save(fig,"fig09-component-ablation",rows,[p],"Episode-bootstrap ablation effects on the registered SUMO C3 subset. Confidence intervals include zero for most components.",manifest)


def fig_mechanism(manifest):
    files=sorted(EVAL.glob("sumo-seed*-C0-feedback-C0-vs-C1-fault101-subset12.json"))+sorted(EVAL.glob("sumo-seed*-C2-input-mechanism-fault101-subset12.json")); rows=[]
    for p in files:
        d=load(p)
        if "delayed_feedback_scenario" in d:
            for method in ("rolling","aci"):
                v=d["comparisons"][method]["primary_equal_horizon_90_interval_score_delayed_minus_immediate"]
                rows.append({"mechanism":"Delayed feedback (C1−C0)","method":method,"seed":d["model_seed"],**{k:v[k] for k in ("estimate","lower","upper")}})
        else:
            for method in ("rolling","aci"):
                vals=[]
                for h in ("3","6","12"): vals.append(d["comparisons"]["intervals"][method]["0.9"][h]["interval_score_c2_minus_c0"])
                rows.append({"mechanism":"Input fault (C2−C0)","method":method,"seed":d["model_seed"],"estimate":float(np.mean([v["estimate"] for v in vals])),"lower":float(np.mean([v["lower"] for v in vals])),"upper":float(np.mean([v["upper"] for v in vals]))})
    fig,ax=plt.subplots(figsize=(7.16,3.1)); y=np.arange(len(rows)); e=np.array([r["estimate"] for r in rows]); lo=np.array([r["lower"] for r in rows]); hi=np.array([r["upper"] for r in rows]); colors=[COL["blue"] if r["mechanism"].startswith("Delayed") else COL["orange"] for r in rows]
    for i,r in enumerate(rows): ax.errorbar(r["estimate"],i,xerr=[[r["estimate"]-r["lower"]],[r["upper"]-r["estimate"]]],fmt="o" if r["method"]=="rolling" else "s",color=colors[i],capsize=2)
    ax.axvline(0,color=COL["black"],lw=.8); ax.set_yticks(y,[f"{r['mechanism']} · {r['method']} · seed {r['seed']}" for r in rows]); ax.invert_yaxis(); ax.set_xlabel("Interval-score difference (faulted − clean)"); ax.grid(axis="x",alpha=.2); ax.set_title("Mechanism-isolation controls",loc="left",weight="bold")
    save(fig,"fig10-mechanism-isolation",rows,files,"Mechanism-isolation controls for delayed feedback and predictor-input faults on the registered SUMO subset.",manifest)


def fig_core(manifest):
    p=ROOT/'artifacts/real-core-c1-c2-c5-matrix.json'
    if not p.exists(): return
    d=load(p)
    if d['status']!='completed': return
    rows=d['summaries']
    fig,ax=plt.subplots(figsize=(7.16,2.4))
    for i,r in enumerate(rows):
        ax.bar(i,r['far_cal_coverage'],color=COL['blue'] if r['superiority_rule_met'] else COL['grey'])
        ax.text(i,r['far_cal_coverage']+.002,'Met' if r['superiority_rule_met'] else 'Not met',ha='center',fontsize=7)
    ax.axhline(.88,color=COL['red'],ls='--',lw=1,label='Registered coverage floor')
    ax.set_xticks(range(len(rows)),[r['dataset'].upper().replace('_','-')+' '+r['scenario'] for r in rows],rotation=15)
    ax.set_ylim(.86,.94); ax.set_ylabel('90% interval coverage'); ax.legend(fontsize=7)
    ax.set_title('Locked FAR-GW core extension: C1/C2/C5',loc='left',weight='bold')
    save(fig,'fig11-c0-c5-core-matrix',rows,[p],'Completed FAR-GW C1/C2/C5 core extension. Labels show the full registered superiority rule, which also requires positive score-effect lower bounds against both comparators. C0 and C3/C4 are presented in the separate clean and transfer figures.',manifest)


def main():
    style(); cfg=load(CFG); manifest={"schema_version":1,"target_venue":cfg["target_venue"],"rendering":cfg["rendering"],"figures":{}}
    builders=[fig_system,fig_scenarios,fig_point,fig_calibration,fig_transfer,fig_outage,fig_heatmap,fig_recovery,fig_ablation,fig_mechanism,fig_core]
    for fn in builders: fn(manifest["figures"])
    for f in cfg["figures"]:
        slug=f"{f['id']}-{f['slug']}"
        if slug not in manifest["figures"]: manifest["figures"][slug]={k:v for k,v in f.items() if k not in ("id","slug")}
    out=OUT/"figure-manifest.json"; out.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(f"Generated {sum(v.get('status')=='generated' for v in manifest['figures'].values())} figures; manifest: {out}")
    if (ROOT/'artifacts/stress/one_factor/matrix.json').exists():
        from build_real_stress_report import main as build_stress
        build_stress()
    if (ROOT/'artifacts/compute-quality/matrix.json').exists():
        from build_compute_quality_figure import main as build_compute
        build_compute()
    if (EVAL/'cross-backbone-transfer-synthesis-v3.json').exists():
        from build_all_benchmark_comparison import main as build_benchmarks
        build_benchmarks()


if __name__ == "__main__": main()
