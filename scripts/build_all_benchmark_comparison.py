"""Publication comparison of all completed paired calibrator/backbone audits."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from build_publication_figures import ROOT, OUT, EVAL, BACKBONE, COL, load, save, style

SLUG='fig14-all-benchmark-comparison'

def main():
    source=EVAL/'cross-backbone-transfer-synthesis-v3.json'
    synthesis=load(source)
    cells={(r['dataset'],r['scenario'],r['backbone']):r for r in synthesis['cells']}
    assert len(cells)==16
    order=[('far_gw','rolling'),('far_gw','aci'),('far_gw','corel'),('staeformer','rolling'),('staeformer','aci'),('stid','rolling'),('stid','aci'),('dcrnn','rolling'),('dcrnn','aci')]
    ypos=np.array([0,1,2,3.6,4.6,6.2,7.2,8.8,9.8])
    labels=[f"{BACKBONE[b]} · {m.upper() if m=='aci' else 'CoRel' if m=='corel' else 'Rolling'}" for b,m in order]
    rows=[]; sources=[source]
    style()
    fig,axes=plt.subplots(2,2,figsize=(7.16,6.15),sharex=True,sharey=True)
    methods={'rolling':(COL['blue'],'o'),'aci':(COL['orange'],'s'),'corel':(COL['purple'],'D')}
    panels=[('metr_la','C3'),('metr_la','C4'),('pems_bay','C3'),('pems_bay','C4')]
    for letter,ax,(dataset,scenario) in zip('abcd',axes.flat,panels):
        cp=EVAL/f'{dataset}-corel-{scenario}-transfer-audit.json'
        corel=load(cp); sources.append(cp)
        assert corel['status']=='completed'
        for (backbone,method),y in zip(order,ypos):
            cell=cells[(dataset,scenario,backbone)]
            effect=corel['primary_comparisons']['corel_minus_far_cal'] if method=='corel' else cell[f'{method}_minus_far_cal']
            assert effect['lower']<=effect['estimate']<=effect['upper']
            assert effect['resamples']==2000
            if method=='corel':
                assert abs(corel['aggregate_score']['far_cal']-cell['far_cal_interval_score'])<1e-5
            color,marker=methods[method]
            e,lo,hi=(effect[k] for k in ('estimate','lower','upper'))
            ax.errorbar(e,y,xerr=[[e-lo],[hi-e]],fmt=marker,color=color,capsize=2,elinewidth=1.1,markersize=4.2)
            rows.append({'dataset':dataset,'scenario':scenario,'backbone':backbone,'comparator':method,'proposed_method':'FAR-Cal asymmetric (asym100)','estimate_mph':e,'lower_95_mph':lo,'upper_95_mph':hi,'far_cal_coverage':cell['far_cal_coverage'],'joint_rule_met_vs_rolling_and_aci':cell['superiority_rule_met'],'bootstrap_resamples':2000,'comparison_scope':'same frozen backbone and forecast cache'})
        ax.axvline(0,color=COL['black'],linestyle='--',linewidth=1,zorder=0)
        for y in (2.8,5.4,8.0): ax.axhline(y,color='#D8D8D8',linewidth=.5,zorder=0)
        passed=sum(cells[(dataset,scenario,b)]['superiority_rule_met'] for b in BACKBONE)
        ax.set_title(f"{letter}  {dataset.upper().replace('_','-')} · {scenario}",loc='left',weight='bold',pad=16)
        ax.text(0,1.02,f'Joint score-and-coverage rule: {passed}/4 backbones',transform=ax.transAxes,fontsize=6.8,color=COL['grey'])
        ax.set_yticks(ypos,labels); ax.set_ylim(10.6,-.7)
        ax.grid(axis='x',alpha=.17)
        ax.spines[['top','right']].set_visible(False)
    lo=min(r['lower_95_mph'] for r in rows); hi=max(r['upper_95_mph'] for r in rows)
    for ax in axes.flat: ax.set_xlim(min(-.55,lo-.12),hi+.16)
    for ax in axes[1]: ax.set_xlabel('Comparator − proposed FAR-Cal\n90% interval score difference (mph)')
    handles=[Line2D([0],[0],color=c,marker=m,linestyle='none',markersize=5,label={'rolling':'Rolling','aci':'ACI (project adaptation)','corel':'CoRel (project adaptation)'}[name]) for name,(c,m) in methods.items()]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.035),ncol=3,frameon=False,fontsize=7)
    fig.text(.5,.012,'Left of zero: comparator better     |     Zero: proposed-method reference     |     Right: FAR-Cal better',ha='center',fontsize=6.8)
    fig.subplots_adjust(left=.225,right=.99,top=.95,bottom=.17,hspace=.29,wspace=.14)
    caption=('Paired comparison of Rolling, the project ACI adaptation and the direct CoRel adaptation with proposed asymmetric FAR-Cal (asym100), across FAR-GW, STAEformer, STID and DCRNN frozen backbones on METR-LA/PEMS-BAY C3/C4. Points show comparator-minus-FAR-Cal mean 90% equal-horizon interval-score differences at 15/30/60 minutes; bars show 95% one-day moving-block bootstrap intervals (2,000 resamples), stratified by fault seed after averaging three model seeds. Positive effects favor FAR-Cal. The zero line is the proposed-method reference, not an independently estimated effect. Joint-rule counts require both Rolling and ACI lower bounds above zero and coverage at least 0.88. CoRel was evaluated only on FAR-GW; other CoRel/backbone combinations are not available. Each comparison holds forecasts and evaluation masks fixed within a backbone; it is not a direct ranking of point-prediction backbones. Tests support no universal superiority or identifiable MNAR correction claim.')
    mp=OUT/'figure-manifest.json'; manifest=load(mp)
    save(fig,SLUG,rows,sources,caption,manifest['figures'])
    mp.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    (OUT/f'{SLUG}-caption.md').write_text(caption+'\n',encoding='utf-8')
    cfg=ROOT/'configs/publication_figures.json'; d=load(cfg)
    if not any(f['id']=='fig14' for f in d['figures']):
        d['figures'].append({'id':'fig14','slug':'all-benchmark-comparison','role':'main','status':'available','dependency':str(source.relative_to(ROOT))})
    cfg.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
    print(f'Generated {SLUG}: {len(rows)} paired comparisons, {len(sources)} source audits')

if __name__=='__main__':main()
