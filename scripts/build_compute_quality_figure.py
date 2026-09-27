"""Export completed matched archive-cap benchmark measurements."""
import json
from pathlib import Path
import statistics
from build_publication_figures import ROOT, OUT, style, save, COL
import matplotlib.pyplot as plt

def main():
    source=ROOT/'artifacts/compute-quality/matrix.json'
    d=json.loads(source.read_text()); assert d['status']=='completed'
    import hashlib
    for r in d['rows']:
        path=source.parent/f"{r['dataset']}-cap{r['cap']}-repeat{r['repeat']}"/'manifest.json'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r['manifest_sha256']
        if r['cap']==256:
            reference=ROOT/'artifacts/stress/one_factor/duration03/farcal'/f"{r['dataset']}-seed11-C3-fault101-far_cal_asymmetric-asym100"/'manifest.json'
            assert json.loads(path.read_text())['metrics']==json.loads(reference.read_text())['metrics']
    style(); fig,axes=plt.subplots(1,2,figsize=(7.16,3))
    rows=[]
    for ax,dataset in zip(axes,('metr_la','pems_bay')):
        for cap,color,marker in zip((64,128,256),(COL['blue'],COL['orange'],COL['green']),('o','s','^')):
            subset=[r for r in d['rows'] if r['dataset']==dataset and r['cap']==cap]
            assert len(subset)==3
            row={'dataset':dataset,'cap':cap,'runtime_median_seconds':statistics.median(r['runtime_seconds'] for r in subset),'runtime_min_seconds':min(r['runtime_seconds'] for r in subset),'runtime_max_seconds':max(r['runtime_seconds'] for r in subset),'cpu_median_seconds':statistics.median(r['cpu_seconds'] for r in subset),'interval_score':statistics.mean(r['interval_score'] for r in subset),'coverage':statistics.mean(r['coverage'] for r in subset)}
            rows.append(row)
            ax.errorbar(row['runtime_median_seconds'],row['interval_score'],xerr=[[row['runtime_median_seconds']-row['runtime_min_seconds']],[row['runtime_max_seconds']-row['runtime_median_seconds']]],fmt=marker,color=color,capsize=2,label=f'{cap} records/sensor')
        ax.set_title(dataset.upper().replace('_','-')); ax.set_xlabel('CPU replay wall time (seconds)'); ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean 90% interval score (mph)'); axes[1].legend(fontsize=6)
    mp=OUT/'figure-manifest.json'; manifest=json.loads(mp.read_text())
    save(fig,'fig13-compute-quality-tradeoff',rows,[source,source.parent/'protocol.json'],'Matched FAR-Cal archive-cap sensitivity on C3 duration03, model seed11/fault seed101. Three timed replays per setting on the same host; markers show median time and bars the observed range. Includes loading/serialization, excludes predictor inference. Descriptive single-seed resource sensitivity, not cross-method ranking or confirmatory superiority. Cold and warm filesystem cache effects are mixed.',manifest['figures'])
    mp.write_text(json.dumps(manifest,indent=2)+'\n')
    report=['# Matched compute–quality benchmark','','This is a descriptive FAR-Cal archive-cap sensitivity study, not a cross-method runtime benchmark. Identical saved forecast/fault inputs are replayed at caps 64/128/256, using C3 duration03 and fixed model/fault seeds 11/101. Three sequential repetitions use rotated cap order. Runtime includes CPU calibration, loading and output serialization, but excludes training and prediction. Filesystem cache is not flushed.','','| Dataset | Cap | Median seconds (range) | Interval score | Coverage |','|---|---:|---:|---:|---:|']
    for r in rows: report.append(f"| {r['dataset']} | {r['cap']} | {r['runtime_median_seconds']:.2f} ({r['runtime_min_seconds']:.2f}–{r['runtime_max_seconds']:.2f}) | {r['interval_score']:.4f} | {r['coverage']:.4f} |")
    report += ['', 'These quality metrics are equal-horizon averages of pooled valid sensor-target metrics on this single run; the full stress summaries instead average origin-level scores and bootstrap over model/fault seeds. They are distinct estimands and should not be substituted for one another. The cap256 benchmark is checked against the archived same-run output. Timing ranges are observed repetition ranges, not confidence intervals. Other user workloads may share the host; median process CPU seconds are retained in the CSV alongside wall time. Hardware details and the protocol are saved with the benchmark artifacts.']
    (ROOT/'reports/compute-quality-benchmark.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    manuscript=ROOT/'reports/manuscript-draft.md'
    text=manuscript.read_text(encoding='utf-8')
    if '### 3.6 Descriptive compute' not in text:
        section='''### 3.6 Descriptive compute–quality sensitivity

An additional matched replay study varied the FAR-Cal archive cap over 64, 128 and 256 records per sensor, holding saved forecasts, fault schedules, hourly origins and evaluation masks fixed within each dataset. It used C3 duration03 with model seed11/fault seed101 and three sequential repetitions with rotated cap order. Timings include CPU calibration, loading and serialization, and exclude point-model training and inference. The 256-record setting reproduced the archived same-run quality metrics exactly. Median times and observed ranges are reported in [compute-quality-benchmark.md](compute-quality-benchmark.md).

![Matched archive-cap compute–quality sensitivity](figures/publication/fig13-compute-quality-tradeoff.png)

This single-run archive-cap study is descriptive. It is not a cross-method runtime comparison, a basis for retuning the locked method on test outcomes, or a replacement for the multi-seed primary effects. Filesystem cache and other host workloads were not isolated; process CPU time is retained alongside wall time. Its pooled sensor-target quality metric differs from the origin-averaged estimand in the full stress summaries.

'''
        manuscript.write_text(text.replace('## 4. Discussion',section+'## 4. Discussion'),encoding='utf-8')
    registry=ROOT/'configs/experiment_registry.yaml'
    text=registry.read_text(encoding='utf-8').replace('id: farcal_archive_cap_compute_quality\n    status: running','id: farcal_archive_cap_compute_quality\n    status: completed')
    registry.write_text(text,encoding='utf-8')
    cfg=ROOT/'configs/publication_figures.json'; c=json.loads(cfg.read_text())
    for f in c['figures']:
        if f['id']=='fig13': f.update(status='available',dependency='artifacts/compute-quality/matrix.json')
    cfg.write_text(json.dumps(c,indent=2)+'\n')
    p=ROOT/'reports/publication-readiness.md'; text=p.read_text(encoding='utf-8')
    text=text.replace('A matched FAR-Cal archive-cap compute–quality benchmark is running;','The matched FAR-Cal archive-cap compute–quality benchmark is complete (18 timed replays);')
    text=text.replace('full ablations and runtime evidence remain.','full ablations remain.')
    text=text.replace('Partial (5/6)','Generated (6/6)').replace('Compute-quality is running.','The descriptive archive-cap compute-quality figure is generated.')
    text=text.replace('3. Finish the matched archive-cap runtime benchmark and export its compute–quality figure.','3. Completed: matched archive-cap runtime benchmark and compute–quality figure.')
    p.write_text(text,encoding='utf-8')
    p=ROOT/'reports/publication-figure-suite.md'; text=p.read_text(encoding='utf-8').replace('## Registered, pending evidence','## Completed extension evidence')
    text=text.replace('Matched archive-cap benchmark running on identical saved C3 inputs; interrupted evaluation-session runtimes are excluded.','Completed 18-replay matched archive-cap benchmark; PDF, PNG and CSV generated. Single-seed descriptive sensitivity; interrupted evaluation-session runtimes are excluded.')
    p.write_text(text,encoding='utf-8')
    print('Compute-quality figure and report generated')

if __name__=='__main__':main()
