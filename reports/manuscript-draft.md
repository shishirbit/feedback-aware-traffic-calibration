# Feedback-aware uncertainty calibration for traffic forecasting under correlated sensor outages

## Abstract

Traffic forecasting intervals are commonly updated from recently observed forecast errors. Correlated sensor outages impair both predictor inputs and the outcome feedback needed for calibration, leaving residual archives stale and spatially unrepresentative. We implement a timestamp-correct replay benchmark that separates input availability from calibration-feedback availability on METR-LA, PEMS-BAY, and controlled SUMO traffic. We evaluate rolling calibration, a spatial batch adaptation of adaptive conformal inference, a direct CoRel adaptation, and feedback-aware residual calibration. The original symmetric FAR-Cal formulation failed confirmatory testing. A subsequently locked asymmetric signed-tail redesign improved 90% mean interval score over rolling and ACI on 40 fresh SUMO C3 episodes while achieving 0.889 coverage. In real-data audits, the registered rule was met in twelve of sixteen backbone/dataset/condition cells: two of four with FAR-GW, four of four with STAEformer, two of four with STID, and four of four with DCRNN. FAR-Cal also significantly outperformed the CoRel adaptation in all FAR-GW real-data C3/C4 audits. Under SUMO C4 it improved score but achieved 0.879 coverage, narrowly below the registered 0.88 floor. These findings support multi-backbone transfer on METR-LA and a direct relational-comparator advantage while rejecting universal superiority.

## 1. Introduction

Multihorizon traffic forecasts support routing, operations, and traveler information, but point predictions alone do not express operational uncertainty. Online interval methods can adapt to changing errors when outcomes arrive normally. A reporting outage changes that setting twice: recent speed inputs may be unavailable to the predictor, and the same missing reports may prevent the calibration service from observing realized forecast errors. A delayed backlog can then produce a concentrated update after service resumes, while permanently lost reports leave an irrecoverable hole in the residual history.

This work studies calibration-feedback impairment as a distinct mechanism. Its main engineering contribution is a causal replay system with separate input and feedback release times, immutable issued forecasts, per-forecast feedback matching, and explicit recovery evaluation. Its methodological contribution is FAR-Cal, an empirical adaptive interval method that weights signed residual tails using spatial proximity, issue-time missingness context, residual age, and stale-feedback inflation. A focused review of close 2026 methods found substantial overlap on incomplete-input and relational adaptive calibration, but did not find an explicit separately delayed or permanently lost calibration-feedback channel or outage-recovery protocol in the reviewed texts. The contribution is therefore positioned around that mechanism and benchmark within the bounded search scope; weighted adaptive intervals are not claimed to inherit split-conformal finite-sample validity.

The experiments address whether missing feedback harms interval quality when point predictions are held fixed, whether feedback-aware residual use improves interval score without materially lowering coverage, and whether effects transfer across two real traffic networks and independently generated SUMO episodes.

## 2. Methods

### 2.1 Data and splits

METR-LA and PEMS-BAY contain five-minute traffic speeds on 207 and 325 sensors. Each dataset uses chronological 60% training, 10% tuning, 10% initial calibration, and 20% test boundaries defined before window construction. Input histories contain 12 completed bins and targets contain the next 12 bins. All target timestamps must belong to one split. Scaling, graph processing, fallback statistics, and context normalization use training data only.

SUMO provides controlled 20-detector freeway episodes from four registered demand/event families. Training, tuning, calibration, test, replication, confirmation, and asymmetric-validation episode sets remain disjoint. Each evaluation episode resets the received residual state and begins from the specified historical archive.

### 2.2 Causal observation and feedback replay

Each observation has separate predictor-input and calibration-feedback arrival times. At a bin boundary, the engine releases due events, updates available input history, ingests feedback for previously issued forecasts, and then creates new forecasts. Forecasts retain their original point, scale, intervals, target, horizon, issue context, and model/calibrator versions. An outcome can score forecasts from several horizons, while each forecast-horizon residual is ingested at most once. Unreleased packets reveal neither future arrival nor permanent loss to the predictor.

### 2.3 Predictor

The frozen FAR-GW predictor adapts Graph WaveNet with observation-mask and observation-age channels and a noncrossing 0.05/0.50/0.95 quantile head. Additional audits use an independently implemented STAEformer-style predictor, an Apache-2.0 STID adaptation with learned spatial/temporal identities and residual MLP blocks, and an MIT-licensed DCRNN adaptation with dual-random-walk diffusion recurrent layers. All receive the same seven causal inputs and noncrossing quantile head. The calibration center is the median prediction, and scale is half the outer-quantile span with a 1 mph floor. Point-model weights remain fixed throughout calibration and test replay.

### 2.4 Calibration methods

Rolling calibration retains causally received normalized residuals by sensor and horizon. The ACI comparator uses a horizon-specific significance control updated once per same-arrival spatial batch; this project adaptation does not inherit the source theorem under missing or delayed outcomes.

Symmetric FAR-Cal models absolute normalized residuals. Its initial and regularized forms did not establish confirmatory superiority. The asymmetric redesign models signed normalized residual tails separately. It pools the sensor with up to two graph neighbors, weights residuals by target age and issue-time context similarity, blends toward a frozen archive as fresh support declines, and applies bounded inflation from feedback staleness and current missingness. Candidate `asym100` was selected with split-safe tuning and locked before generating its fresh validation episodes.

The real-data transfer audits use a registered resource-bounded implementation: one systematically selected hourly origin, fault-affected sensors through 12 post-event bins, and at most 256 recent records per sensor in the local pool. These audits support transfer assessment but are not interchangeable with the full five-minute SUMO validation.

### 2.5 Conditions and inference

C3 represents correlated outages with delayed feedback and backlog release. C4 represents correlated outages with permanent feedback loss. The primary endpoint is the equal-horizon average 90% interval score at 15, 30, and 60 minutes. Positive comparator-minus-method differences favor FAR-Cal. The registered rule requires both rolling-minus-method and ACI-minus-method 95% lower confidence bounds above zero and aggregate 90% coverage of at least 0.88.

SUMO inference averages paired effects across model and fault seeds within each episode and bootstraps complete episodes. Real-data inference averages three model seeds within each fault seed and uses one-day moving blocks stratified across three fault seeds.

## 3. Results

### 3.1 Symmetric FAR-Cal

The original tuning-locked symmetric method was worse than rolling and ACI in the held-out SUMO C3 matrix. A regularized redesign produced favorable point estimates in a 30-episode replication, but both intervals crossed zero. In a preregistered 160-episode confirmation, its effects reversed slightly: rolling and ACI each scored about 0.027 better, with both intervals crossing zero. These results do not support symmetric FAR-Cal superiority.

### 3.2 Asymmetric FAR-Cal

| Dataset and condition | Rolling − method (95% CI) | ACI − method (95% CI) | Coverage | Rule |
|---|---:|---:|---:|---:|
| SUMO C3, fresh validation | 0.5947 [0.2356, 1.0065] | 0.5788 [0.2319, 0.9544] | 0.8890 | Met |
| SUMO C4, scenario transfer | 1.0285 [0.6083, 1.5249] | 0.9435 [0.5728, 1.3726] | 0.8793 | Not met |
| METR-LA C3, 3 × 3 audit | 0.7932 [0.4293, 1.2330] | 0.7723 [0.4141, 1.2039] | 0.8985 | Met |
| METR-LA C4, 3 × 3 audit | 0.9507 [0.5331, 1.4261] | 0.9512 [0.5656, 1.3808] | 0.8958 | Met |
| PEMS-BAY C3, 3 × 3 audit | -0.0786 [-0.2260, 0.1046] | -0.0936 [-0.2487, 0.0951] | 0.9006 | Not met |
| PEMS-BAY C4, 3 × 3 audit | -0.2651 [-0.4350, -0.0955] | -0.2908 [-0.4839, -0.1051] | 0.9068 | Not met |

The fresh SUMO C3 validation met the complete rule. METR-LA effects were positive under both delayed and permanently lost feedback. PEMS-BAY C3 estimates were small and uncertain, while its C4 intervals were entirely below zero, favoring both comparators. SUMO C4 showed large score gains but missed the coverage floor by 0.0007.

### 3.3 Direct CoRel adaptation

Across the complete real-data matrix, CoRel-minus-FAR-Cal interval-score effects
were 1.6960 [1.1594, 2.3049] on METR-LA C3, 2.1767 [1.3784, 3.1390] on
METR-LA C4, 2.2547 [1.8212, 2.7181] on PEMS-BAY C3, and 2.2450 [1.7615,
2.7452] on PEMS-BAY C4. All intervals exclude zero in favor of FAR-Cal. CoRel
coverage was 0.8743, 0.8725, 0.8837, and 0.8743, respectively. This comparison
uses the official CoRel architecture with PS-1 splits, multihorizon residuals,
identical frozen point forecasts, and causal feedback replay; it is a project
adaptation rather than a reproduction of the paper's source-split scores.

![All evaluated benchmark comparisons with proposed FAR-Cal](figures/publication/fig14-all-benchmark-comparison.png)

**Figure 14.** Paired comparator-minus-FAR-Cal differences in mean 90% interval score across four frozen forecasting backbones, two datasets and C3/C4. Points are estimates and bars are 95% one-day moving-block bootstrap intervals, stratified by fault seed after averaging three model seeds. Positive differences favor FAR-Cal; the zero line is the proposed-method reference. Comparisons hold forecasts and evaluation masks fixed within a backbone. CoRel was evaluated only with FAR-GW. Panel counts report the joint rule against Rolling and ACI, including the 0.88 coverage floor. This is a calibration comparison conditional on each backbone, not a direct ranking of forecasting backbones.

The complete registered visual evidence suite is listed in [publication-figure-suite.md](publication-figure-suite.md). Figures whose source experiments are unfinished remain explicitly pending rather than being populated with provisional or synthetic values.

### 3.4 Multi-backbone audit

The complete STAEformer matrix met the registered rule on METR-LA C3 and C4
and PEMS-BAY C3 and C4. Rolling-minus-FAR-Cal effects were 1.4972 [0.7436,
2.3621], 1.7914 [0.9847, 2.7599], 0.2531 [0.0849, 0.4478], and 0.2598
[0.0983, 0.4390], respectively; all ACI comparisons also had lower bounds above
zero, and coverage ranged from 0.8916 to 0.9101. DCRNN also met all four rules,
with coverage ranging from 0.8891 to 0.9144. Across all four backbones,
twelve of sixteen registered cells met the rule. STID met both
METR-LA rules but neither PEMS-BAY rule. This demonstrates multi-backbone
transfer on METR-LA, but the conflicting PEMS-BAY results do not support a
universal backbone-independent superiority claim.

### 3.5 Locked FAR-GW core, independent controls and one-factor stress

The completed 168-stage C1/C2/C5 core matrix met the registered superiority rule for METR-LA in all three scenarios and for PEMS-BAY in none. The 120-stage matched-rate independent-control matrix likewise met the rule for METR-LA C3/C4 and did not meet it for PEMS-BAY C3/C4. These controls preserve event severity while randomizing affected sensor identities; they assess method superiority within independent outages and do not alone establish an advantage caused by spatial correlation. Paired effects and confidence intervals are in [real-core-and-control-results.md](real-core-and-control-results.md).

The locked one-factor stress matrix completed all 1,488 stages and 48 dataset/scenario/variant summaries. Across C3/C4, METR-LA met the rule in all 24 conditions, including duration, affected fraction, additional backlog delay, permanent loss, a faulted initial calibration archive, and low-speed-dependent MNAR loss. PEMS-BAY met it in none of its 24 conditions. FAR-Cal coverage ranged from 0.8908 to 0.9072 on METR-LA and 0.8981 to 0.9102 on PEMS-BAY, so coverage above the floor does not imply interval-score superiority. Complete paired estimates and block-bootstrap intervals are in [real-stress-results.md](real-stress-results.md), with machine-readable values in `real-stress-results.csv`.

![Interval score versus outage duration](figures/publication/fig12-interval-score-vs-outage-duration.png)

Each stress condition uses its own affected-sensor and outage/recovery focus window. The duration plot therefore describes condition-specific quality rather than a paired causal duration effect on identical origins. The cold-archive and MNAR results extend the robustness audit but do not identify or correct a missingness mechanism. True low speed is exposed only to the evaluator-side generator; no identifiable MNAR correction is claimed.

### 3.6 Descriptive compute–quality sensitivity

An additional matched replay study varied the FAR-Cal archive cap over 64, 128 and 256 records per sensor, holding saved forecasts, fault schedules, hourly origins and evaluation masks fixed within each dataset. It used C3 duration03 with model seed11/fault seed101 and three sequential repetitions with rotated cap order. Timings include CPU calibration, loading and serialization, and exclude point-model training and inference. The 256-record setting reproduced the archived same-run quality metrics exactly. Median times and observed ranges are reported in [compute-quality-benchmark.md](compute-quality-benchmark.md).

![Matched archive-cap compute–quality sensitivity](figures/publication/fig13-compute-quality-tradeoff.png)

This single-run archive-cap study is descriptive. It is not a cross-method runtime comparison, a basis for retuning the locked method on test outcomes, or a replacement for the multi-seed primary effects. Filesystem cache and other host workloads were not isolated; process CPU time is retained alongside wall time. Its pooled sensor-target quality metric differs from the origin-averaged estimand in the full stress summaries.

## 4. Discussion

The results support two conclusions. First, tail asymmetry matters: the symmetric method failed repeated confirmation, whereas the signed-tail redesign succeeded on fresh SUMO C3 episodes. Second, transfer depends jointly on network and frozen predictor. METR-LA effects replicate across FAR-GW, STAEformer, STID, and DCRNN, while PEMS-BAY succeeds with STAEformer and DCRNN but not FAR-GW or STID. Differences in graph neighborhoods, residual-tail shape, sensor coverage, or predictor-scale quality may explain this heterogeneity; the present experiments do not isolate which factor dominates.

The PEMS-BAY result must remain in the principal findings. Retuning on its evaluated test outcomes would invalidate a confirmatory claim. Future method development should use training/tuning simulations or new external networks, then test a newly locked rule on untouched data.

The completed FAR-GW extension preserves this network-specific pattern across isolated core faults, matched independent outages and all six stress families. Passing the METR-LA stress grid provides evidence over the tested conditions, not over arbitrary missingness mechanisms or unseen networks. PEMS-BAY remains a negative transfer result despite coverage above the registered floor. The MNAR audit is a sensitivity experiment using evaluator-generated low-speed loss, not an identification result.

## 5. Limitations

FAR-Cal is an empirical adaptive method without a claimed finite-sample coverage theorem for dependent streams and selectively observed outcomes. The real-data audits are hourly, fault focused, and use capped local pooling for tractable execution. The CoRel result is a documented PS-1 adaptation and does not reproduce its published source split. The STAEformer-style model is an independent paper-based implementation because the audited official repository had no license file; STID and DCRNN are documented project adaptations of their licensed reference implementations. Dataset redistribution-rights verification is pending. The focused literature review cannot establish universal absence of closer work, so novelty remains bounded by its documented search scope.

## 6. Conclusion

Separating predictor-input availability from calibration-feedback availability enables causal evaluation of delayed and permanently lost outcomes. Asymmetric feedback-aware residual calibration can improve interval efficiency under correlated outages while preserving near-nominal coverage, but the effect is dataset dependent. The evidence supports a conditional method and benchmark contribution and rejects universal superiority.

## Reproducibility statement

The repository stores resolved configurations, dataset and artifact hashes, immutable prediction caches, fault manifests, episode- or block-bootstrap outputs, and machine-readable result summaries. The current automated suite contains 78 passing tests under Python 3.12.14; the earlier 73-test suite also passed under Python 3.11.9, which executed the GPU backbone matrices. Exact environments are recorded in `requirements.lock`, `requirements-py311.lock`, and `requirements-staeformer.lock`.

## References used for positioning

- Graph WaveNet (IJCAI 2019): https://arxiv.org/abs/1906.00121
- STAEformer (CIKM 2023): https://arxiv.org/abs/2308.10425
- DCRNN (ICLR 2018): https://arxiv.org/abs/1707.01926
- Adaptive conformal inference for multi-step time-series forecasting (COPA 2024): https://proceedings.mlr.press/v230/hallberg-szabadvary24a.html
- Relational Conformal Prediction / CoRel (ICML 2025): https://proceedings.mlr.press/v267/cini25a.html
- MAS-Mamba (2026): https://ietresearch.onlinelibrary.wiley.com/doi/pdfdirect/10.1049/itr2.70304
- Post-training adaptive conformal prediction for incomplete time series (TMLR 2026): https://openreview.net/forum?id=KMBU4wx79B
- Adaptive conformal prediction for spatio-temporal hypergraph neural networks (Applied Soft Computing 2026): https://www.sciencedirect.com/science/article/pii/S1568494626017047
