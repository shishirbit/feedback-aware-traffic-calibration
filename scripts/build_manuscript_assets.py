from pathlib import Path
import pandas as pd,numpy as np,re,shutil
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
R=Path(__file__).resolve().parents[1]; M=R/'manuscript'; F=M/'figures'; F.mkdir(exist_ok=True)
k=.44704
for n in ['interact.cls','tfcad.bst','natbib.sty']:
 if (M/'template/interactcadlatex'/n).exists(): shutil.copy2(M/'template/interactcadlatex'/n,M/n)
s=(M/'main.tex').read_text(encoding='utf8').replace('log observation age','capped, scaled observation age').replace('\\begin{figure}[ht]\n\\fbox','\\begin{center}\n\\refstepcounter{algorithm}\n\\fbox').replace('\\end{minipage}}\n\\end{figure}','\\end{minipage}}\n\\end{center}')
if '\\newcounter{algorithm}' not in s: s=s.replace('\\newcommand{\\ms}', '\\newcounter{algorithm}\n\\newcommand{\\ms}')
(M/'main.tex').write_text(s,encoding='utf8')
abstract=s.split('\\begin{abstract}')[1].split('\\end{abstract}')[0]; print('Abstract words',len(abstract.split()))
words=len(re.findall(r"\b[\w]+(?:[-'][\w]+)*\b",re.sub(r'\\[a-zA-Z]+(?:\[[^]]*\])?', ' ',s)))
(M/'word-count.tex').write_text('\\noindent Manuscript word count (approximate, including declarations; excluding references and separate tables/captions): '+str(words)+'.\\par\n',encoding='utf8')
name={'metr_la':'METR-LA','pems_bay':'PEMS-BAY','sumo':'SUMO','far_gw':'FAR-GW','staeformer':'STAEformer','stid':'STID','dcrnn':'DCRNN'}
def table(label,caption,headers,rows,note=''):
 return '\\begin{table}[p]\n\\caption{'+caption+'}\\label{tab:'+label+'}\n\\centering\\small\n\\begin{tabular}{'+'l'*len(headers)+'}\\toprule\n'+' & '.join(headers)+' \\\\\\midrule\n'+'\n'.join(' & '.join(map(str,r))+' \\\\' for r in rows)+'\n\\bottomrule\\end{tabular}\n\\par\\smallskip\\footnotesize '+note+'\n\\end{table}\\clearpage\n'
def ci(r,p): return f"{r[p+'_estimate']*k:.3f} [{r[p+'_lower']*k:.3f}, {r[p+'_upper']*k:.3f}]"
t=''
t+=table('data','Data and evaluation design.',['Data','Sensors','Partitions','Test sampling'],[['METR-LA','207','60/10/10/20\\%','Hourly origins'],['PEMS-BAY','325','60/10/10/20\\%','Hourly origins'],['SUMO','20','60/15/15/30 episodes','Full episode origins']],'Real partitions are chronological. SUMO C3 additionally uses 40 fresh validation episodes. All histories span 60 min; reported horizons are 15, 30 and 60 min.')
t+=table('settings','Shared protocol and locked asymmetric settings.',['Quantity','Setting'],[['Nominal coverage / decision floor','0.90 / 0.88'],['History / forecast window','12 / 12 five-minute bins'],['Training / fault seeds','11, 22, 33 / 101, 202, 303'],['Scale floor','0.44704 $\\ms$'],['Buffer / age decay / stale decay','2016 / 288 / 288 bins'],['Context bandwidth / support reference','2 / 50'],['Inflation coefficients / cap','0.05, 0.05 / 3'],['Real local archive cap','256 records per group node'],['Maximum local neighbours','2'],['Bootstrap resamples','2000']],'Per-backbone training configurations are supplied in the release. These locked calibration settings supersede exploratory defaults.')
t+=table('conditions','Reporting conditions.',['Condition','Predictor input','Calibration feedback'],[['C0','Available','Available'],['C1','Available','Correlated delay'],['C2','Correlated outage','Available'],['C3','Correlated outage','Delay and backlog'],['C4','Correlated outage','Permanent within-outage loss'],['C5','Independent packet loss','Independent packet loss']],'C3/C4 baseline: 10\\% connected sensors, 30-min outage, two daily events, 15-min backlog; C4 permanently loses half of affected feedback. Schedules are separate from physical validity.')
B=pd.read_csv(R/'reports/figures/publication/fig14-all-benchmark-comparison.csv')
primary=[['SUMO','C3','0.266 [0.105, 0.450]','0.259 [0.104, 0.427]','0.8890','Yes'],['SUMO','C4','0.460 [0.272, 0.682]','0.422 [0.256, 0.614]','0.8793','No']]
for (d,c),g in B[B.backbone=='far_gw'].groupby(['dataset','scenario'],sort=False):
 a=g[g.comparator=='aci'].iloc[0]; b=g[g.comparator=='rolling'].iloc[0]
 def bc(r):return f"{r.estimate_mph*k:.3f} [{r.lower_95_mph*k:.3f}, {r.upper_95_mph*k:.3f}]"
 primary.append([name[d],c,bc(b),bc(a),f'{a.far_cal_coverage:.4f}','Yes' if a.joint_rule_met_vs_rolling_and_aci else 'No'])
t+=table('primary','Final FAR-GW paired score effects and joint decisions.',['Data','Cell','Rolling minus FAR','ACI minus FAR','Coverage','Pass'],primary,'Effects and 95\\% paired intervals are in $\\ms$. Positive differences favour FAR-Cal. SUMO values are rounded from saved validation summaries.')
rows=[]
for (b,d,c),g in B.groupby(['backbone','dataset','scenario'],sort=False):
 r=g.iloc[0];rows.append([name[b],name[d],c,f'{r.far_cal_coverage:.4f}','Yes' if r.joint_rule_met_vs_rolling_and_aci else 'No'])
t+=table('backbones','Backbone transfer decisions.',['Backbone','Data','Cell','Coverage','Joint rule'],rows,'Joint rule compares rolling and ACI on identical frozen forecasts. CoRel was evaluated only with FAR-GW.')
C=pd.read_csv(R/'reports/real-core-and-control-results.csv');rows=[]
for _,r in C.iterrows():rows.append([name[r.dataset],r.scenario,'Core' if r.stage=='core' else 'Independent',ci(r,'rolling_minus_far_cal'),ci(r,'aci_minus_far_cal'),'Yes' if r.superiority_rule_met else 'No'])
t+=table('controls','Core and independent matched-rate controls.',['Data','Cell','Schedule','Rolling effect','ACI effect','Pass'],rows,'Effects and paired 95\\% intervals are in $\\ms$; the independent controls randomise sensor identities at matched severity.')
S=pd.read_csv(R/'reports/real-stress-results.csv');rows=[]
for (d,f),g in S.groupby(['dataset','factor'],sort=False):rows.append([name[d],f.replace('_','\\_'),len(g),int(g.superiority_rule_met.sum()),f'{g.far_cal_coverage.min():.4f}'])
t+=table('stress','One-factor stress summary by network and factor.',['Data','Factor','Cells','Passes','Min coverage'],rows,'Cells comprise separately evaluated values and C3/C4 conditions; the release retains every effect estimate and interval. MNAR is a sensitivity experiment, not an identification result.')
A=pd.read_csv(R/'reports/figures/publication/fig15-full-component-ablations.csv');rows=[]
for a,g in A.groupby('ablation',sort=False): rows.append([a.replace('_',' '),int((g.lower>0).sum()),int((g.upper<0).sum()),int(((g.lower<=0)&(g.upper>=0)).sum())])
t+=table('ablations','Exploratory paired ablation classifications across six cells.',['Removal/diagnostic','Worsens','Improves','Inconclusive'],rows,'Worsens: lower 95\\% bound above zero; improves: upper bound below zero. The intervals are exploratory and unadjusted for multiple comparisons. Counts refer to removals, not full-method superiority.')
Q=pd.read_csv(R/'reports/figures/publication/fig13-compute-quality-tradeoff.csv');rows=[]
for _,r in Q.iterrows():rows.append([name[r.dataset],int(r.cap),f'{r.runtime_median_seconds:.2f} [{r.runtime_min_seconds:.2f}, {r.runtime_max_seconds:.2f}]',f'{r.interval_score*k:.4f}',f'{r.coverage:.4f}'])
t+=table('runtime','Descriptive CPU archive-cap replays.',['Data','Cap','Median seconds [range]','Score ($\\ms$)','Coverage'],rows,'Three repetitions per cap and network; identical C3 duration03 inputs, seeds 11/101. Ranges are observed timing ranges, not confidence intervals. The pooled quality estimand differs from origin-level paired inference.')
(M/'tables.tex').write_text(t,encoding='utf8')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
def save(fig,n):
 fig.savefig(F/(n+'.pdf'),bbox_inches='tight');fig.savefig(F/(n+'.png'),dpi=220,bbox_inches='tight');fig.savefig(F/(n+'.tiff'),dpi=600,bbox_inches='tight',pil_kwargs={'compression':'tiff_lzw'});plt.close(fig)
fig,ax=plt.subplots(figsize=(9,6));ax.set(xlim=(0,10),ylim=(0,8));ax.axis('off')
def box(x,y,w,h,text,col='#e6eef6'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',fc=col,ec='#34495e'));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9)
def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#34495e','lw':1.4})
box(.2,6.7,2.5,.9,'Traffic source\nSpeed, validity, graph');box(3.4,6.7,3,.9,'Evaluator fault schedules\nSeparate input / feedback release');arrow((2.7,7.15),(3.4,7.15))
box(.2,4.7,2.6,1.1,'Service: causal input store\nPast fill; 7 input channels');box(3.4,4.7,3,1.1,'Frozen predictor\nNoncrossing quantile head');box(7,4.7,2.6,1.1,'Immutable issue ledger\nCentre, scale, context\nTarget and horizon');arrow((2.8,5.2),(3.4,5.2));arrow((6.4,5.2),(7,5.2));arrow((4.9,6.7),(1.5,5.8))
box(.2,2.5,2.6,1.1,'Released feedback only\nMatch once to ledger\nSigned residual');box(3.4,2.5,3,1.1,'Calibration pools\nTarget age; context; support\nFrozen + live archive');box(7,2.5,2.6,1.1,'Signed-tail intervals\nStale blend and inflation\nSaved at issuance');arrow((4.9,6.7),(1.5,3.6));arrow((2.8,3.05),(3.4,3.05));arrow((6.4,3.05),(7,3.05));arrow((8.3,4.7),(8.3,3.6))
box(3.4,.3,6.2,1.1,'Evaluator-only hidden truth + originally valid targets\nScore saved intervals; never update from unavailable outcomes','#fcebdc');arrow((8.3,2.5),(8.3,1.4));ax.text(.3,.8,'No future arrivals or\nloss flags visible\nto the service',fontsize=9)
save(fig,'architecture')
fig,axs=plt.subplots(1,2,figsize=(10,8),sharex=False)
for ax,(d,g) in zip(axs,B.groupby('dataset',sort=False)):
 g=g.copy();ys=np.arange(len(g));v=g.estimate_mph.to_numpy()*k;lo=g.lower_95_mph.to_numpy()*k;hi=g.upper_95_mph.to_numpy()*k
 for j,(_,r) in enumerate(g.iterrows()):ax.errorbar(v[j],j,xerr=[[v[j]-lo[j]],[hi[j]-v[j]]],fmt='o',color={'rolling':'#176087','aci':'#c56926','corel':'#55803b'}[r.comparator],capsize=2,ms=4)
 ax.set_yticks(ys,[name[r.backbone]+' '+r.scenario+' / '+r.comparator.upper() for _,r in g.iterrows()],fontsize=8);ax.invert_yaxis();ax.axvline(0,color='.4',lw=1);ax.set_title(name[d]);ax.set_xlabel('Comparator minus FAR-Cal score (m/s)');ax.grid(axis='x',alpha=.2)
fig.tight_layout();save(fig,'benchmarks')
fig,axs=plt.subplots(2,3,figsize=(11,6.5))
for ax,((d,c),g) in zip(axs.flat,A.groupby(['dataset','scenario'],sort=False)):
 v=g.difference.to_numpy()*k;lo=g.lower.to_numpy()*k;hi=g.upper.to_numpy()*k
 ax.errorbar(v,np.arange(7),xerr=[v-lo,hi-v],fmt='o',color='#176087',capsize=2,ms=4);ax.set_yticks(range(7),[a.replace('_',' ') for a in g.ablation],fontsize=7);ax.invert_yaxis();ax.axvline(0,color='.5');ax.set_title(name[d]+' '+c);ax.set_xlabel('Removal minus full score (m/s)',fontsize=8);ax.grid(axis='x',alpha=.2)
ax=axs.flat[3]
# PEMS-BAY C4 has one much larger effect; label it explicitly outside the zoomed axis.
ax.set_xlim(-.22,.25);ax.text(-.18,3,'4.220 [3.760, 4.699] →',fontsize=7,color='#176087',va='center')
fig.tight_layout();save(fig,'ablations')
D=pd.read_csv(R/'reports/figures/publication/fig12-interval-score-vs-outage-duration.csv');fig,axs=plt.subplots(1,2,figsize=(8,3.3))
for ax,(d,g) in zip(axs,D.groupby('dataset',sort=False)):
 for c,z in g.groupby('scenario'):z=z.sort_values('value');ax.plot(z.value*5,z.far_cal_interval_score*k,'o-',label=c)
 ax.set_title(name[d]);ax.set_xlabel('Outage duration (min)');ax.set_ylabel('FAR-Cal interval score (m/s)');ax.legend(frameon=False);ax.grid(alpha=.2)
fig.tight_layout();save(fig,'duration')
caps={'architecture':'Causal system architecture. The service observes only released packets. Evaluator truth is reserved for scoring immutable issued intervals.','benchmarks':'Paired comparator-minus-FAR-Cal interval-score effects and 95% intervals in SI units. Positive values favour FAR-Cal. Comparisons share frozen forecasts within each backbone; CoRel is restricted to FAR-GW.','duration':'Interval score across the registered outage-duration grid. Lines connect descriptive scores, not interpolation guarantees; C3 and C4 are distinct feedback conditions.','ablations':'Exploratory removal-minus-full interval-score effects and 95% paired intervals. Positive values indicate deterioration after removal. Panel scales differ; the off-scale PEMS-BAY C4 stale-blend failure is explicitly labelled with its estimate and bounds.'}
(M/'figures.tex').write_text('\n'.join('\\begin{figure}[p]\\centering\\includegraphics[width=\\linewidth]{figures/'+n+'.pdf}\\caption{'+c.replace('%','\\%')+'}\\label{fig:'+n+'}\\end{figure}\\clearpage' for n,c in caps.items()),encoding='utf8')
(M/'captions.tex').write_text('\n\n'.join('Figure '+str(i+1)+'. '+c.replace('%','\\%') for i,(n,c) in enumerate(caps.items())),encoding='utf8')
print('Generated tables and four SI figures')
