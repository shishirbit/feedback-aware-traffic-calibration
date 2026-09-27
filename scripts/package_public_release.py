from pathlib import Path
import shutil,json,hashlib,re
R=Path.cwd(); P=R/'artifacts/public-release/feedback-aware-traffic-calibration';P.mkdir(parents=True,exist_ok=True)
def copy(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
for folder in ['src','scripts','configs','tests','docs']:
 for f in (R/folder).rglob('*'):
  if f.is_file() and f.name != 'redraft_manuscript.py' and not any(x in f.parts for x in ['__pycache__','.git']) and f.suffix in ['.py','.ps1','.yaml','.json','.md','.csv']:copy(f,P/f.relative_to(R))
for f in R.glob('requirements*.lock'):copy(f,P/f.name)
for n in ['README.md','pyproject.toml','pytest.ini']:copy(R/n,P/n)
for f in (R/'reports').glob('*'):
 if f.is_file() and f.suffix in ['.md','.json','.csv']:copy(f,P/f.relative_to(R))
for f in (R/'reports/figures/publication').glob('*.csv'):copy(f,P/f.relative_to(R))
for f in (R/'data/prepared').glob('sumo*'):copy(f,P/f.relative_to(R))
for f in (R/'data/prepared').glob('*.manifest.json'):copy(f,P/f.relative_to(R))
for f in (R/'manuscript').rglob('*'):
 if f.is_file() and 'template' not in f.parts and f.suffix in ['.tex','.bib','.cls','.bst','.sty','.pdf','.png']:copy(f,P/f.relative_to(R))
for folder in ['Graph-WaveNet','DCRNN','stid','corel']:
 copy(R/'third_party'/folder/'LICENSE',P/'third_party_notices'/folder/'LICENSE')
# CoRel's unchanged architecture is needed by its adapter; preserve licensed sources.
for f in (R/'third_party/corel').rglob('*'):
 if f.is_file() and '.git' not in f.parts and '__pycache__' not in f.parts and (f.suffix in ['.py','.yaml','.yml','.toml'] or f.name in ['LICENSE','README.md']):copy(f,P/f.relative_to(R))
for path in ['artifacts/ablations/full_registered/matrix.json','artifacts/ablations/full_registered/protocol.json','artifacts/ablations/full_registered/completion-audit.json','artifacts/stress/one_factor/matrix.json','artifacts/stress/matched_control/matrix.json','artifacts/real-core-c1-c2-c5-matrix.json']:
 f=R/path
 if f.exists():copy(f,P/path)
spec=R.parent/'PS1_Codex_Research_and_SUMO_Specification.md'
if spec.exists():copy(spec,P/'docs/research-specification.md')
for folder in ['docs','scripts','src']:
 for f in (P/folder).rglob('*'):
  if f.is_file() and f.suffix in ['.md','.py','.ps1']:
   s=f.read_text(encoding='utf-8-sig');s=s.replace('docs/research-specification.md','docs/research-specification.md');f.write_text(s,encoding='utf8')
(P/'.gitignore').write_text('__pycache__/\n.venv*/\n*.egg-info/\n.pytest_cache/\n*.log\n*.aux\n*.blg\n*.xdv\ndata/raw/\n',encoding='utf8')
(P/'THIRD_PARTY_NOTICES.md').write_text('Project adapters derive from MIT Graph WaveNet (Zonghan Wu, 2019), MIT DCRNN (Yaguang Li, 2017), and Apache-2.0 STID. Their complete licences are retained in third_party_notices/. CoRel source is MIT (Andrea Cini, 2025), pinned to 4504c4edf76128bfe657762a41ec9eb043038ca2, with its licence retained. STAEformer is independently implemented; its unlicensed source is not distributed. The Taylor & Francis Interact template retains publisher provenance and is supplied solely for manuscript preparation. Generated SUMO arrays are project-produced research data; underlying SUMO software is obtained separately. Exact METR-LA/PEMS-BAY packages are excluded pending redistribution-rights verification.\n',encoding='utf8')
read=(P/'README.md').read_text(encoding='utf8').replace('docs/research-specification.md','docs/research-specification.md')
read='''# Feedback-aware traffic calibration: reproducibility release

This release accompanies the Transportmetrica B manuscript by Shishir Singh Chauhan, Manipal University Jaipur. It contains code, configurations, tests, generated SUMO prepared arrays, evidence summaries and the supplied-template manuscript.

The registered evaluations and all 432 ablation stages are complete. Findings are conditional: no universal superiority, no identifiable MNAR correction and no production-service claim. Negative and inconclusive results are retained.

## Reproduce

Install the relevant locked environment (Python 3.11 backbone/CoRel locks or the main lock), install this package with `pip install -e .`, and run `python -m unittest discover -s tests`. The latest local verification passed 77 tests. SUMO/TraCI 1.27.1 is required for simulation regeneration. Raw real benchmark packages must be acquired separately using `docs/sources.md` and checked against the recorded hashes; graphs are obtained from pinned DCRNN sources. Training checkpoints and bulky prediction/residual caches are not bundled: regenerate these before invoking resumable evaluation runners. The original protocols, candidate locks, inference units and limitations are in docs and reports.

Generated SUMO prepared arrays cover original, replication, confirmation and fresh asymmetric-validation sets; original episode XML can be regenerated from registered plans. Prepared manifest provenance paths describe the original host and are not portable runtime paths. This is an evidence/source release rather than a precomputed-cache image. Study scripts use repository-relative paths unless explicitly documented as original-host launch helpers.

Build manuscript tables/figures with `python scripts/build_revised_manuscript_assets.py`. Compile `manuscript/main.tex` with Tectonic or a LaTeX/BibTeX toolchain. All manuscript effects are converted from native mph artifacts to SI using 0.44704 exactly.

## Original workflow and command inventory

'''+read
(P/'README.md').write_text(read,encoding='utf8')
files=list(P.rglob('*')); largest=sorted([(f.stat().st_size,str(f.relative_to(P))) for f in files if f.is_file()],reverse=True)[:5]
print('Release files',sum(f.is_file() for f in files),'largest',largest)
# Scan for secret-shaped literals without printing matching values.
patterns=[r'gh[pousr]_[A-Za-z0-9]{20,}',r'github_pat_[A-Za-z0-9_]{30,}',r'AKIA[0-9A-Z]{16}',r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----']
hits=[]
for f in files:
 if f.is_file() and f.suffix in ['.py','.ps1','.json','.md','.yaml','.tex']:
  text=f.read_text(encoding='utf8',errors='replace')
  if any(re.search(p,text) for p in patterns):hits.append(str(f.relative_to(P)))
if hits:raise RuntimeError('Secret scan failed for files: '+str(hits))
print('Credential-shaped literal scan clean')

