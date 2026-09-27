from pathlib import Path
import runpy,re,json
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
R=Path(__file__).resolve().parents[1];M=R/'manuscript';F=M/'figures';k=.44704
base=runpy.run_path(str(R/'scripts/build_manuscript_assets.py'));table=base['table'];save=base['save'];name=base['name'];B=base['B'];A=base['A'];S=base['S'];Q=base['Q']
# Rewrite table sequence in order of first mention.
def wrapped(label,caption,heads,rows,widths,note=''):
 return '\\begin{table}[p]\\caption{'+caption+'}\\label{tab:'+label+'}\\centering\\small\n\\begin{tabular}{'+''.join('>{\\raggedright\\arraybackslash}p{'+str(w)+'cm}' for w in widths)+'}\\toprule\n'+' & '.join(heads)+' \\\\\\midrule\n'+'\n'.join(' & '.join(r)+' \\\\' for r in rows)+'\n\\bottomrule\\end{tabular}\\par\\smallskip\\footnotesize '+note+'\\end{table}\\clearpage\n'
related=wrapped('related-models','Forecasting and input-reconstruction literature: role in this study.',['Reference','Representation or objective','Numerical study role'],[[r'Graph WaveNet \\citep{wu2019}','Graph supports and temporal convolutions','FAR-GW causal quantile adapter'],[r'DCRNN \\citep{li2018}','Diffusion recurrent forecasting','Causal quantile adapter; data provenance'],[r'STID \\citep{shao2022}','Spatial and temporal identity','Causal quantile adapter'],[r'STAEformer \\citep{liu2023}','Adaptive embeddings and Transformer','Independent paper-based adapter'],[r'GRIN \\citep{cini2022}','Graph-based missing-value reconstruction','Literature context; not evaluated']],[3.0,4.3,5.0],'Representation choices do not determine how missing outcome feedback should be handled. Source-paper performance values are not used as study results.').replace('\\\\citep','\\citep')
related+=wrapped('related-calibration','Calibration literature and feedback scope.',['Reference','Relevant objective','Scope here'],[[r'CQR \\citep{romano2019}','Calibrated quantile regression','Literature; predictor quantiles distinguished from calibrated bounds'],[r'Beyond exchangeability \\citep{barber2023}','Weighted inference under drift','Literature; no theorem imported'],[r'ACI \\citep{gibbs2021}','Sequential coverage adaptation','Horizon-specific spatial-batch adapter'],[r'SPCI \\citep{xu2023}','Sequential residual-based intervals','Literature; not evaluated'],[r'Multistep ACI \\citep{hallberg2024}','Ordinary forecast-target delay','Starting point for ACI adaptation'],[r'CoRel \\citep{cini2025}','Relational series calibration','Official architecture; FAR-GW only'],[r'Incomplete series \\citep{chen2026}','Post-training missing-input adaptation','Published 2026; literature comparison'],[r'Delayed ACI \\citep{elhalabi2026}','Delay-dependent recursion and guarantees','2026 preprint; not evaluated'],[r'Corrupted feedback \\citep{wang2026}','Binary-flip feedback robustness','2026 preprint; different corruption model']],[3.6,4.0,4.7],'Entries summarise reviewed objectives, not universal capability claims or a numerical ranking. General CP background is supplied by \\citet{angelopoulos2021}.').replace('\\\\citep','\\citep')
old=(M/'tables.tex').read_text(encoding='utf8');blocks={re.search(r'\\label\{tab:([^}]+)\}',b).group(1):b+'\\clearpage\n' for b in re.findall(r'\\begin\{table\}.*?\\end\{table\}',old,re.S)}
rows=[['METR-LA','207','34,272','1 Mar--27 Jun 2012','Hourly fault audit'],['PEMS-BAY','325','52,128','1 Jan--30 Jun 2017','Hourly fault audit'],['SUMO original','20','11,520','120 eight-hour episodes','Full episode audit'],['SUMO fresh C3','20','3,840','40 fresh episodes','Full episode audit']]
blocks['data']=table('data','Dataset dimensions, periods and evaluation sampling.',['Data','Sensors','Rows','Acquired/prepared period','Test sampling'],rows,'All rows use five-minute bins. PEMS-BAY has 52,116 raw rows plus twelve inserted wholly invalid rows. Real zero speeds follow the benchmark missingness convention; physical SUMO zeros remain valid.')
stats=[];train_profiles={}
for d in ['metr_la','pems_bay','sumo']:
 with np.load(R/'data/prepared'/f'{d}.npz') as z:
  values=z['values_mph'];valid=z['original_valid'];observed=values[valid]*k
  if d=='sumo':tr=z['partition_code']==0
  else:
   man=json.loads((R/'data/prepared'/f'{d}.manifest.json').read_text());cut=np.datetime64(man['boundary_timestamps'][1],'ns').astype('int64');tr=z['timestamp_ns']<cut
  obs=values[tr][valid[tr]]*k;train_profiles[d]=(obs,float(valid[tr].mean()))
  stats.append([name[d],f'{100*valid.mean():.2f}',f'{np.median(observed):.2f}',f'{np.quantile(observed,.05):.2f}',f'{np.quantile(observed,.95):.2f}'])
blocks['data-statistics']=table('data-statistics','Descriptive prepared-data validity and observed speeds.',['Data','Valid (\\%)','Median','5th percentile','95th percentile'],stats,'Speed summaries are in $\\ms$ across originally valid prepared measurements before reporting interventions. They are not distributional estimates for the selectively received feedback stream.')
blocks['partitions']=wrapped('partitions','Chronological boundaries and episode separation.',['Data','Train/tune boundary','Tune/calibration boundary','Calibration/test boundary'],[['METR-LA','11 May 2012 09:35','23 May 2012 07:10','4 Jun 2012 04:45'],['PEMS-BAY','19 Apr 2017 14:45','7 May 2017 17:05','25 May 2017 19:20'],['SUMO original','60 training episodes','15 tuning episodes','15 calibration; 30 test episodes'],['SUMO fresh C3','No predictor retraining','Candidate locked before generation','40 new validation episodes']],[2.3,3.3,3.3,3.4],'Real proportions are 60/10/10/20 percent. Targets must remain inside their assigned partition. Timestamp boundaries precede PEMS-BAY gap reindexing.')
blocks['architecture-contract']=wrapped('architecture-contract','Architecture interfaces and information boundary.',['Module','Consumes','Produces / restriction'],[['Input store','Released speed, validity and measurement time','Past-only seven-channel history; no future packets'],['Frozen predictor','Available history and fitted parameters','Median and floored scale; no test-time training'],['Issue ledger','Point, scale, context and interval','Immutable sensor/target/horizon records'],['Feedback matcher','Released valid outcomes and stored records','Residual once per forecast--horizon key'],['Calibration kernel','Received live and valid frozen archives','Weighted signed-tail bounds; no hidden truth'],['Evaluator','Original validity, truth and issued bounds','Scores; no return edge to service']],[2.8,4.0,5.5],'The diagram separates service data flow from evaluator knowledge. Recorded software/artifact checks establish replay properties, not operational latency.')
blocks['notation']=table('notation','Principal mathematical notation.',['Symbol','Meaning','Units'],[['$G=(V,E)$','Sensor graph','--'],['$t,u,h$','Issue bin, target bin, forecast horizon','Five-minute bins'],['$a^x,a^f$','Input and feedback release times','Bins or infinity'],['$y,m,s$','Target speed, point centre, scale','$\\ms$'],['$r$','Signed normalised forecast residual','Dimensionless'],['$c,z$','Issue context and missing fraction','Dimensionless'],['$R_h,A_h$','Live and frozen residual archives','Records'],['$B,\\tau,d$','Retention, age and stale-decay lengths','Bins'],['$G_i,C$','Local group and group-cap multiplier','Sensors / records'],['$b,n_0$','Context bandwidth and support reference','Dimensionless'],['$n_{\\rm eff},g$','Timestamp-block support and freshness gap','Blocks / bins'],['$\\eta,\\lambda,\\kappa$','Archive mixture, local mixture, inflation','Dimensionless'],['$L,U,\\mathrm{IS}$','Interval bounds and interval score','$\\ms$'],['$\\alpha$','Nominal miscoverage','Probability']],'SI conversion uses exactly 0.44704 metres per second per mph. Dimensionless mixture weights and coverage remain unchanged.')
blocks['settings']=blocks['settings'].replace('256 records per group node','256 times group size')
rows=[]
for (b,d,c),g in B.groupby(['backbone','dataset','scenario'],sort=False):
 a=g[g.comparator=='aci'].iloc[0];r=g[g.comparator=='rolling'].iloc[0]
 def ci(v):return f'{v.estimate_mph*k:.3f} [{v.lower_95_mph*k:.3f}, {v.upper_95_mph*k:.3f}]'
 rows.append([name[b],name[d],c,ci(r),ci(a),f'{a.far_cal_coverage:.4f}','Yes' if a.joint_rule_met_vs_rolling_and_aci else 'No'])
blocks['backbones']=table('backbones','All sixteen paired backbone comparisons.',['Backbone','Data','Cell','Rolling minus FAR','ACI minus FAR','Coverage','Pass'],rows,'Effects and 95\\% paired intervals are in $\\ms$. Every comparison conditions on one frozen backbone. Joint rule requires both lower bounds above zero and FAR-Cal coverage at least 0.88.')
rows=[]
# Direct CoRel summaries are retained at original report precision; effects use exact CSV values.
corel_scores={('metr_la','C3'):(20.3370,22.0330,.8743),('metr_la','C4'):(23.4332,25.6099,.8725),('pems_bay','C3'):(10.3271,12.5817,.8837),('pems_bay','C4'):(10.5644,12.8094,.8743)}
for _,r in B[B.comparator=='corel'].iterrows():
 f,c,cv=corel_scores[(r.dataset,r.scenario)];rows.append([name[r.dataset],r.scenario,f'{f*k:.3f}',f'{c*k:.3f}',ci(r),f'{r.far_cal_coverage:.4f}',f'{cv:.4f}'])
blocks['corel']=table('corel','Direct CoRel adaptation comparison on FAR-GW caches.',['Data','Cell','FAR score','CoRel score','CoRel minus FAR','FAR cov.','CoRel cov.'],rows,'Dimensional scores and paired intervals are in $\\ms$. Absolute scores and CoRel coverage retain the report\'s rounding; paired effects use full saved precision. This is not a comparison on other backbones.')
for d,key in [('metr_la','metr'),('pems_bay','pems'),('sumo','sumo')]:
 rows=[]
 for _,r in A[A.dataset==d].iterrows():rows.append([r.scenario,r.ablation.replace('_',' '),f'{r.difference*k:.3f} [{r.lower*k:.3f}, {r.upper*k:.3f}]',f'{r.coverage:.4f}',f'{r.width_mph*k:.3f}'])
 blocks['ablation-'+key]=table('ablation-'+key,'Full '+name[d]+' component ablations.',['Cell','Removal / diagnostic','Removal minus full (95\\% CI)','Coverage','Width'],rows,'Effects and widths are in $\\ms$. Positive differences mean removal worsens score. Intervals are exploratory and unadjusted. The frozen-only and arrival-freshness rows are diagnostic alternatives.')
sequence=['data','data-statistics','partitions','architecture-contract','notation','settings','conditions','primary','backbones','corel','controls','stress','ablations','ablation-metr','ablation-pems','ablation-sumo','runtime']
for key in ['backbones','corel']:
 blocks[key]=blocks[key].replace('\\centering\\small','\\centering\\footnotesize\n\\setlength{\\tabcolsep}{4pt}')
(M/'tables.tex').write_text(related+''.join(blocks[x] for x in sequence),encoding='utf8')
# Dataset profile, before window construction.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
fig,axs=plt.subplots(1,3,figsize=(9,3))
for ax,(d,(v,vr)) in zip(axs,train_profiles.items()):
 ax.hist(v,bins=np.linspace(0,40,65),density=True,color='#176087',alpha=.85);ax.set_title(name[d]);ax.set_xlabel('Training observed speed (m/s)');ax.set_ylabel('Density');ax.text(.04,.94,f'Original valid: {vr:.1%}',transform=ax.transAxes,va='top',fontsize=9);ax.grid(axis='y',alpha=.15)
fig.tight_layout();save(fig,'data-profile')
# Architecture: separate lanes; only intentional junctions; no connector crosses a box.
fig,ax=plt.subplots(figsize=(9,6.8));ax.set(xlim=(0,13.5),ylim=(0,10.8));ax.axis('off')
positions={}
def box(key,x,y,text,color='#e7eff6',w=2.8,h=1.15):
 positions[key]=(x,y,w,h);ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.04',fc=color,ec='#365569',lw=1.2));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
def arr(a,b,color='#365569'):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.5,'color':color,'shrinkA':0,'shrinkB':0})
def route(points):
 for a,b in zip(points[:-2],points[1:-1]):ax.plot([a[0],b[0]],[a[1],b[1]],color='#365569',lw=1.5)
 arr(points[-2],points[-1])
xs=[.25,3.55,6.85,10.15]
ax.text(.25,10.25,'(a) Input release and frozen forecasting',fontsize=12,weight='bold')
for key,x,txt in zip(['source','store','model','ledger'],xs,['Physical source\nSpeed, time, validity','Input release queue\nCausal input store\nSeven channels','Frozen backbone\nMedian and scale','Immutable ledger\nIssue, target, horizon']):box(key,x,8.5,txt)
for x in xs[:-1]:arr((x+2.8,9.075),(x+3.3,9.075))
ax.text(13.0,6.6,'(b) Calibration',fontsize=12,weight='bold',ha='right')
for key,x,txt in zip(['queue','matcher','kernel','interval'],xs,['Feedback queue\nReleased outcomes only','Residual matcher\nLive archive\nOnce per forecast key','Calibration kernel\nAge / context / support\nLocal or local + global','Published intervals\nSigned tails + inflation\nSaved without revision']):box(key,x,4.85,txt)
for x in xs[:-1]:arr((x+2.8,5.425),(x+3.3,5.425))
arr((1.65,8.5),(1.65,6.0));ax.text(.2,7.55,'Separate\nschedules',fontsize=9,ha='left')
route([(11.55,8.5),(11.55,7.55),(4.95,7.55),(4.95,6.0)])
arr((8.25,7.55),(8.25,6.0));ax.plot(8.25,7.55,'o',ms=3,color='#365569');ax.text(6.25,7.78,'Stored issue records',fontsize=10,ha='center',bbox={'fc':'white','ec':'none','pad':1})
box('archive',6.85,2.8,'Frozen archive\nCalibration residuals',color='#e5f1e8');arr((8.25,3.95),(8.25,4.85))
ax.text(.25,3.1,'No unreleased packets,\nfuture arrivals or loss flags\nare visible to the service.',fontsize=11,color='#365569')
ax.text(.25,1.95,'(c) Evaluator only',fontsize=12,weight='bold',color='#955524')
box('truth',.25,.5,'Hidden truth\nOriginal validity',color='#faecdf');box('score',10.15,.5,'Evaluator scores\nCoverage, width, score',color='#faecdf');arr((3.05,1.075),(10.15,1.075),color='#955524');ax.text(6.6,1.3,'No return edge to service',fontsize=10,ha='center',color='#955524');arr((11.55,4.85),(11.55,1.65),color='#955524')
fig.tight_layout();save(fig,'architecture')
# Coverage and interval score from the final FAR-GW study, not old symmetric evidence.
quality=[]
for d,c in [('metr_la','C3'),('metr_la','C4'),('pems_bay','C3'),('pems_bay','C4')]:
 f,_,_=corel_scores[(d,c)];r=B[(B.dataset==d)&(B.scenario==c)&(B.backbone=='far_gw')].iloc[0];quality.append((name[d]+' '+c,f*k,r.far_cal_coverage))
for c in ['C3','C4']:
 path='sumo-asymmetric-validation.json' if c=='C3' else 'sumo-asymmetric-C4-transfer.json';j=json.loads((R/'artifacts/evaluation'/path).read_text())
 print(path,[(x,j[x]) for x in j if 'far_cal' in x and not isinstance(j[x],dict)])
quality += [('SUMO C3',7.603801727294922*k,.889002732240437),('SUMO C4',11.294840812683105*k,.8793351548269582)]
fig,axs=plt.subplots(1,2,figsize=(8,3.5));labs=[x[0] for x in quality];ys=np.arange(len(quality))
axs[0].barh(ys,[x[1] for x in quality],color='#176087');axs[1].scatter([x[2] for x in quality],ys,color='#176087',s=35);axs[1].axvline(.88,color='#bb4b31',ls='--',label='Decision floor');axs[1].axvline(.90,color='.45',ls=':',label='Nominal coverage')
for ax in axs:ax.set_yticks(ys,labs);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
axs[0].set_xlabel('FAR-Cal interval score (m/s)');axs[1].set_xlabel('FAR-Cal coverage');axs[1].set_xlim(.872,.918);axs[1].legend(frameon=False,fontsize=8);fig.tight_layout();save(fig,'quality')
# Five additional stress factors: actual paired effects, not smoothed extrapolation.
factors=['connected_fraction','additional_backlog_delay_bins','loss_within','cold_calibration','mnar_low_speed']
fig,axs=plt.subplots(2,3,figsize=(10,6.2));col={'metr_la':'#176087','pems_bay':'#c56926'}
for ax,f in zip(axs.flat,factors):
 g=S[S.factor==f];categorical=f in ['cold_calibration','mnar_low_speed']
 for d in ['metr_la','pems_bay']:
  for ci,c in enumerate(['C3','C4']):
   z=g[(g.dataset==d)&(g.scenario==c)].sort_values('value');v=z.aci_minus_far_cal_estimate.to_numpy()*k;lo=z.aci_minus_far_cal_lower.to_numpy()*k;hi=z.aci_minus_far_cal_upper.to_numpy()*k
   if categorical:x=np.array([ci+(0 if d=='metr_la' else .18)])
   else:x=z.value.to_numpy()*(5 if f=='additional_backlog_delay_bins' else 1)
   ax.errorbar(x,v,yerr=[v-lo,hi-v],fmt='o-' if c=='C3' else 's--',capsize=2,ms=4,color=col[d],label=name[d]+' '+c)
 ax.axhline(0,color='.4',lw=.8);ax.set_title({'connected_fraction':'Affected fraction','additional_backlog_delay_bins':'Additional backlog delay (min)','loss_within':'Permanent-loss fraction','cold_calibration':'Cold calibration','mnar_low_speed':'MNAR low-speed stress'}[f],fontsize=10);ax.set_ylabel('ACI minus FAR score (m/s)',fontsize=8);ax.grid(alpha=.15)
 if categorical:ax.set_xticks([.09,1.09],['C3','C4'])
 else:ax.set_xlabel('Tested factor value',fontsize=8)
axs.flat[-1].axis('off');handles,labels=axs.flat[0].get_legend_handles_labels();axs.flat[-1].legend(handles,labels,loc='center',frameon=False);fig.tight_layout();save(fig,'stress')
fig,axs=plt.subplots(1,2,figsize=(8,3.5))
for ax,(d,g) in zip(axs,Q.groupby('dataset',sort=False)):
 for _,r in g.iterrows():ax.errorbar(r.runtime_median_seconds,r.interval_score*k,xerr=[[r.runtime_median_seconds-r.runtime_min_seconds],[r.runtime_max_seconds-r.runtime_median_seconds]],fmt='o',capsize=3,label='Cap '+str(int(r.cap)))
 ax.set_title(name[d]);ax.set_xlabel('Replay wall time (s)');ax.set_ylabel('Pooled interval score (m/s)');ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=8)
fig.tight_layout();save(fig,'compute')
# Figure sequence follows first mention in the text.
caps={
'data-profile':'Training-partition observed-speed distributions before forecast-window construction. Only originally valid prepared measurements are plotted; labels report original valid fractions. SUMO includes its prepared training-partition measurements. The distributions are descriptive, not an explanation of calibration transfer.',
'architecture':'Causal architecture with separate input, feedback and evaluator lanes. Stored issue records feed both residual matching and the calibration query; the junction is intentional. Hidden truth feeds scoring only. All module routes avoid text and preserve the service information boundary.',
'benchmarks':base['caps']['benchmarks'],
'quality':'Final FAR-GW interval score and coverage for the six primary cells. Dashed and dotted lines mark the 0.88 decision floor and 0.90 nominal coverage. Absolute network scores are not a paired between-network effect.',
'duration':base['caps']['duration'],
'stress':'Additional one-factor stress: ACI-minus-FAR-Cal effects and paired 95% intervals for affected fraction, backlog delay, permanent loss, cold calibration and MNAR. Positive values favour FAR-Cal. Categorical panels contain only their registered conditions; continuous lines connect tested values without extrapolation.',
'ablations':base['caps']['ablations'],
'compute':'Descriptive archive-cap quality and wall-time sensitivity, with three observed runtime repetitions per cap and network. Horizontal bars are observed ranges, not confidence intervals. These fixed-run pooled scores differ from the origin-level paired inference estimand.'}
# Order quality before duration by relocating results subsection at build time if necessary.
(M/'figures.tex').write_text('\n'.join('\\begin{figure}[p]\\centering\\includegraphics[width=\\linewidth]{figures/'+n+'.pdf}\\caption{'+c.replace('%','\\%')+'}\\label{fig:'+n+'}\\end{figure}\\clearpage' for n,c in caps.items()),encoding='utf8')
(M/'captions.tex').write_text('\n\n'.join('Figure '+str(i+1)+'. '+c.replace('%','\\%') for i,(n,c) in enumerate(caps.items())),encoding='utf8')
(M/'figure-alt-text.txt').write_text('\n\n'.join('Figure '+str(i+1)+': '+c for i,(n,c) in enumerate(caps.items())),encoding='utf8')
# Counts exclude section titles, citation strings, equations, tables and captions.
def prose(s):
 s=re.sub(r'\\begin\{(?:equation|align)\}.*?\\end\{(?:equation|align)\}',' ',s,flags=re.S)
 s=re.sub(r'\\(?:section|subsection|subsubsection)\*?\{[^}]*\}(?:\\label\{[^}]*\})?','',s)
 s=re.sub(r'\\(?:citep|citet|ref|eqref|label|input)\{[^}]*\}','',s)
 s=s.replace('~',' ');s=re.sub(r'\\[A-Za-z]+',' ',s);s=s.replace('{',' ').replace('}',' ');return s
counts={n:len(prose((M/'sections'/f'{n}.tex').read_text(encoding='utf8')).split()) for n in ['introduction','related-work']}
alltext=(M/'main.tex').read_text(encoding='utf8')+'\n'+'\n'.join(p.read_text(encoding='utf8') for p in (M/'sections').glob('*.tex'))+(M/'reproducibility.tex').read_text(encoding='utf8')
wc=len(prose(alltext).split());(M/'word-count.tex').write_text('\\noindent Approximate manuscript prose word count: '+str(wc)+' (excluding references, tables, captions and displayed equations).\\par\n',encoding='utf8')
counts.update({'approximate_prose_words':wc,'table_count':19,'figure_count':8});(M/'section-word-counts.json').write_text(json.dumps(counts,indent=2),encoding='utf8');print(counts)
