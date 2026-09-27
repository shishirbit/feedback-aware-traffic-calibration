"""Consolidate core, independent-control and stress endpoints without mixing estimands."""
import csv
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    rows=[]; sources=[]
    for dataset in ('metr_la','pems_bay'):
        for scenario in ('C1','C2','C5'):
            sources.append(('core',ROOT/'artifacts/evaluation'/f'{dataset}-asymmetric-{scenario}-transfer-audit.json'))
        for scenario in ('C3','C4'):
            sources.append(('independent_control',ROOT/'artifacts/stress/matched_control/evaluation'/f'{dataset}-{scenario}-transfer-audit.json'))
    for stage,p in sources:
        d=json.loads(p.read_text())
        rule=d['far_cal_coverage']>=d['coverage_floor'] and all(v['lower']>0 for v in d['primary_comparisons'].values())
        assert rule==d['superiority_rule_met']
        row={'stage':stage,'dataset':d['dataset'],'scenario':d['scenario'],'coverage':d['far_cal_coverage'],'interval_score':d['far_cal_interval_score'],'superiority_rule_met':rule,'source_path':str(p.relative_to(ROOT)),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        for method,v in d['primary_comparisons'].items():
            row.update({f'{method}_{k}':v[k] for k in ('estimate','lower','upper')})
        rows.append(row)
    with (ROOT/'reports/real-core-and-control-results.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lines=['# FAR-GW core extension and matched independent controls','','Positive differences favor FAR-Cal. The rule requires both 95% lower confidence bounds above zero and coverage at least 0.88. Independent-control results test the method within independently assigned outages; they are not correlated-minus-independent effects.','','| Stage | Dataset | Scenario | Coverage | Score | Rolling difference (95% CI) | ACI difference (95% CI) | Rule |','|---|---|---|---:|---:|---|---|---|']
    for r in rows:
        effects=['{:.4f} [{:.4f}, {:.4f}]'.format(*(r[f'{n}_{k}'] for k in ('estimate','lower','upper'))) for n in ('rolling_minus_far_cal','aci_minus_far_cal')]
        lines.append(f"| {r['stage']} | {r['dataset']} | {r['scenario']} | {r['coverage']:.4f} | {r['interval_score']:.4f} | {effects[0]} | {effects[1]} | {'Met' if r['superiority_rule_met'] else 'Not met'} |")
    (ROOT/'reports/real-core-and-control-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('10 core/control rule recomputations passed; tables saved')

if __name__=='__main__':main()
