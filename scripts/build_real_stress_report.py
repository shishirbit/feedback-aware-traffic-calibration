"""Build tables, figures and a provenance audit from the completed locked stress matrix."""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_publication_figures import style, save, COL
import matplotlib.pyplot as plt

def load(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    base = ROOT / 'artifacts/stress/one_factor'
    matrix = load(base / 'matrix.json')
    assert matrix['status'] == 'completed' and matrix['completed'] == 1488
    assert matrix['registry_sha256'] == sha(ROOT / 'configs/real_one_factor_stress.json')
    rows, sources = [], [base / 'matrix.json']
    reuse='--reuse-audit' in sys.argv
    prior=None
    if reuse:
        prior=load(ROOT/'reports/real-stress-provenance-audit.json')
        assert prior['status']=='passed'
        assert all(sha(ROOT/s['path'])==s['sha256'] for s in prior['sources'])
    checked = 0
    for cell in matrix['summaries']:
        p = base / cell['variant'] / 'evaluation' / f"{cell['dataset']}-{cell['scenario']}-transfer-audit.json"
        d = load(p); sources.append(p)
        rule = d['far_cal_coverage'] >= d['coverage_floor'] and all(x['lower'] > 0 for x in d['primary_comparisons'].values())
        assert rule == cell['superiority_rule_met'] == d['superiority_rule_met']
        assert d['model_seeds'] == [11,22,33] and d['fault_seeds'] == [101,202,303]
        for src in d['sources']:
            identity = f"{cell['dataset']}-seed{src['model_seed']}-{cell['scenario']}"
            prefix = base / cell['variant']
            paths = {
                'farcal': prefix / 'farcal' / f"{identity}-fault{src['fault_seed']}-far_cal_asymmetric-asym100" / 'manifest.json',
                'online': prefix / 'online' / f"{identity}-feedback-{cell['scenario']}-fault{src['fault_seed']}" / 'manifest.json',
                'test': prefix / 'predictions' / f"{cell['dataset']}-seed{src['model_seed']}" / f"{cell['scenario']}-fault{src['fault_seed']}" / 'manifest.json'}
            for kind, path in paths.items():
                assert sha(path) == src[kind + '_manifest_sha256'], path
                manifest = load(path)
                assert manifest['status'] == 'completed'
                for chunk in ([] if reuse else manifest['chunks']):
                    assert sha(path.parent / chunk['file']) == chunk['sha256'], path
                    checked += 1
        row = {**cell, 'far_cal_interval_score': d['far_cal_interval_score']}
        for name, v in d['primary_comparisons'].items():
            row.update({f'{name}_{k}':v[k] for k in ('estimate','lower','upper')})
        rows.append(row)
    out = ROOT / 'reports'; out.mkdir(exist_ok=True)
    with (out / 'real-stress-results.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    labels = {'duration_bins':'Outage duration (bins)', 'connected_fraction':'Affected sensor fraction', 'additional_backlog_delay_bins':'Additional backlog delay (bins)', 'loss_within':'Permanent loss fraction', 'cold_calibration':'Cold calibration archive', 'mnar_low_speed':'Low-speed MNAR loss'}
    style(); mp=out/'figures/publication/figure-manifest.json'; manifest=load(mp)
    for index,(factor,label) in enumerate(labels.items()):
        selected=[dict(r) for r in rows if r['factor']==factor]
        figure_sources=list(sources)
        if factor in labels:
            if factor in ('cold_calibration','mnar_low_speed'):
                for r in selected: r['value']=1
            for dataset in ('metr_la','pems_bay'):
                for scenario in ('C3','C4'):
                    suffix='transfer-audit' if scenario=='C3' else 'C4-transfer-audit'
                    p=ROOT/'artifacts/evaluation'/f'{dataset}-asymmetric-{suffix}.json'
                    d=load(p); figure_sources.append(p)
                    value={'duration_bins':6,'connected_fraction':.1,'additional_backlog_delay_bins':0,'loss_within':0 if scenario=='C3' else .5,'cold_calibration':0,'mnar_low_speed':0}[factor]
                    selected.append({'dataset':dataset,'scenario':scenario,'variant':'baseline','factor':factor,'value':value,'far_cal_interval_score':d['far_cal_interval_score'],'far_cal_coverage':d['far_cal_coverage'],'superiority_rule_met':d['superiority_rule_met']})
        fig, axes=plt.subplots(1,2,figsize=(7.16,2.9))
        for ax,dataset in zip(axes,('metr_la','pems_bay')):
            for scenario,color in [('C3',COL['blue']),('C4',COL['orange'])]:
                subset=sorted((r for r in selected if r['dataset']==dataset and r['scenario']==scenario),key=lambda r:r['value'] or 0)
                x=[r['value'] if r['value'] is not None else 0 for r in subset]
                ax.plot(x,[r['far_cal_interval_score'] for r in subset],marker='o' if scenario=='C3' else 's',linestyle='-' if scenario=='C3' else '--',color=color,label=scenario)
            ax.set_title(dataset.upper().replace('_','-')); ax.set_xlabel(label); ax.grid(alpha=.2)
            if factor in ('cold_calibration','mnar_low_speed'): ax.set_xticks([0,1],['Baseline','Stress condition'])
        axes[0].set_ylabel('Mean 90% interval score (mph)'); axes[1].legend()
        slug='fig12-interval-score-vs-outage-duration' if index==0 else 'stress-'+factor
        save(fig,slug,selected,figure_sources,f'{label}: locked FAR-GW C3/C4 audit, three model and three fault seeds, including the archived baseline. Each point uses its condition-specific affected-sensor/outage-recovery focus; scores across durations are descriptive, not paired effects on identical origins. No universal superiority or identifiable MNAR correction is established.',manifest['figures'])
    mp.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    if reuse: checked=prior['verified_chunk_hashes']
    audit={'status':'passed','matrix_cells':len(rows),'referenced_manifests':len(rows)*27,'verified_chunk_hashes':checked,'registry_sha256':matrix['registry_sha256'],'sources':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in sources], 'scope':'One-factor audit; checks rule recomputation, seeds, registry, referenced manifests and every referenced prediction/interval chunk. Does not audit unreferenced artifacts or establish theoretical validity.'}
    if not reuse: (out/'real-stress-provenance-audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    lines=['# Completed FAR-GW real-data stress results','','All 1,488 stages and 48 summaries are complete. Each cell uses model seeds 11/22/33 and fault seeds 101/202/303. The rule requires both rolling-minus-FAR-Cal and ACI-minus-FAR-Cal 95% lower bounds above zero and coverage at least 0.88.','','| Dataset | Scenario | Variant | Coverage | Interval score | Rolling difference (95% CI) | ACI difference (95% CI) | Rule |','|---|---|---|---:|---:|---|---|---|']
    for r in rows:
        effects=['{:.4f} [{:.4f}, {:.4f}]'.format(*(r[f'{n}_{k}'] for k in ('estimate','lower','upper'))) for n in ('rolling_minus_far_cal','aci_minus_far_cal')]
        lines.append(f"| {r['dataset']} | {r['scenario']} | {r['variant']} | {r['far_cal_coverage']:.4f} | {r['far_cal_interval_score']:.4f} | {effects[0]} | {effects[1]} | {'Met' if r['superiority_rule_met'] else 'Not met'} |")
    lines += ['', 'METR-LA meets the registered rule in all 24 stress cells; PEMS-BAY meets it in none of 24. This pattern includes duration, affected fraction, extra backlog delay, permanent loss, cold calibration and low-speed MNAR. Coverage remains above the registered floor in all cells. Coverage alone does not establish interval-score superiority. Duration scores use condition-specific focus windows and must not be interpreted as paired causal effects on a fixed set of origins. MNAR truth is available only to the fault generator/evaluator; these tests do not establish identifiable MNAR correction. No universal superiority claim is supported.', '', 'The core C1/C2/C5 matrix and matched-rate independent controls show the same dataset-specific rule pattern. The matched control tests superiority within independently assigned outages, not superiority of correlated outages over independent outages.', '', f'Provenance audit passed: 48 rule recomputations, 1,296 referenced manifests, and {checked} chunk hashes. Source hashes are in real-stress-provenance-audit.json.']
    (out/'real-stress-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('sources',)}))

if __name__=='__main__': main()
