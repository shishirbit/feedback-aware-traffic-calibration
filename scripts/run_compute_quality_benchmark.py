"""Matched FAR-Cal archive-cap benchmark; timing is descriptive on this host."""
import json
import os
from pathlib import Path
import platform
import sys
import time
import statistics

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ps1.evaluation.farcal_online import evaluate_continuous_farcal
from ps1.utils.artifacts import sha256_file

def main():
    stage=ROOT/'artifacts/compute-quality'; stage.mkdir(exist_ok=True)
    protocol={'schema_version':1,'datasets':['metr_la','pems_bay'],'scenario':'C3','variant':'duration03','model_seed':11,'fault_seed':101,'record_caps':[64,128,256],'repetitions':3,'scope':'FAR-Cal archive-cap sensitivity only; identical hourly fault-focused inputs per dataset. End-to-end CPU calibration replay includes loading and output serialization, excludes predictor inference and training. Same process/host; filesystem cache is not controlled. Other user workloads may share the host; process CPU time is recorded alongside wall time. No cross-method runtime ranking or confirmatory superiority claim.','hardware':{'platform':platform.platform(),'processor':platform.processor(),'logical_cpus':os.cpu_count()}}
    protocol_path=stage/'protocol.json'
    if protocol_path.exists(): assert json.loads(protocol_path.read_text())==protocol
    else: protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    rows=[]
    for repeat in range(3):
        for dataset in protocol['datasets']:
            # Rotate order to reduce systematic order effects.
            for cap in ([64,128,256][repeat:]+[64,128,256][:repeat]):
                base=ROOT/'artifacts/stress/one_factor/duration03'
                archived=base/'farcal'/f'{dataset}-seed11-C3-fault101-far_cal_asymmetric-asym100'/'manifest.json'
                reference=json.loads(archived.read_text())
                parameters={**reference['parameters'],'records_per_sensor_cap':cap}
                output=stage/f'{dataset}-cap{cap}-repeat{repeat}'
                timing=output/'timing.json'
                if timing.exists(): row=json.loads(timing.read_text())
                else:
                    started=time.perf_counter(); cpu_started=time.process_time()
                    result=evaluate_continuous_farcal(ROOT/'artifacts/predictions'/f'{dataset}-seed11'/'calibration',base/'predictions'/f'{dataset}-seed11'/'C3-fault101',base/'faults'/dataset/'C3-seed101',output,{'prepared_path':ROOT/f'data/prepared/{dataset}.npz','parameters':parameters,'candidate_id':'asym100','evaluation_stride':12,'focus_fault_events':True,'focus_recovery_bins':12})
                    elapsed=time.perf_counter()-started
                    values=list(result['metrics']['far_cal_asymmetric']['0.9'].values())
                    row={'dataset':dataset,'cap':cap,'repeat':repeat,'runtime_seconds':elapsed,'cpu_seconds':time.process_time()-cpu_started,'interval_score':statistics.mean(v['mean_interval_score'] for v in values),'coverage':statistics.mean(v['picp'] for v in values),'manifest_sha256':sha256_file(output/'manifest.json'),'reference_manifest_sha256':sha256_file(archived)}
                    timing.write_text(json.dumps(row,indent=2)+'\n')
                rows.append(row)
                (stage/'progress.json').write_text(json.dumps({'completed':len(rows),'total':18,'last':row,'status':'running'},indent=2)+'\n')
                print(f"{len(rows)}/18 {dataset} cap={cap} repeat={repeat}",flush=True)
    result={'status':'completed','protocol_sha256':sha256_file(protocol_path),'protocol':protocol,'rows':rows}
    (stage/'matrix.json').write_text(json.dumps(result,indent=2)+'\n')
    (stage/'progress.json').write_text(json.dumps({'completed':18,'total':18,'status':'completed'})+'\n')

if __name__=='__main__':main()
