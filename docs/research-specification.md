# PS-1: Codex-ready research and implementation specification

**Working title:** Feedback-Aware Uncertainty Calibration for Traffic Speed Forecasting under Correlated Sensor Outages: METR-LA, PEMS-BAY, and SUMO Evaluation

**Proposed system:** FAR-GW — Feedback-Aware Reliability with Graph WaveNet.

**Core proposed component:** FAR-Cal — Feedback-Aware Residual Calibration. Both names are working labels, not claims of unique registered names.

**Status:** Research proposal and implementation contract, 8 September 2026. No experiments have been executed for this draft. Hypotheses, algorithm choices, and numerical defaults below are proposed and must be tested. This file is self-contained and can be given directly to Codex.

## 1. Instructions to the implementing Codex agent

Implement a reproducible research repository for the selected PS-1. Build the data pipeline, causal online replay engine, proposed calibrator, comparison methods, SUMO scenario generator, evaluation, and report generation described below. Start with a small end-to-end smoke run, then complete the registered experiments within available compute. Preserve working checkpoints and exact configurations. Do not substitute generated numbers for experiment outputs.

Use this document as the default specification. Resolve ordinary implementation choices autonomously and record them in `docs/decisions.md`. If data, a required dependency, full-text evidence, or compute is unavailable, complete all independent work, identify the specific blocker, and mark affected outputs as pending. Do not claim that a planned command, placeholder table, unavailable baseline, or smoke test is a completed research experiment.

The repository must support three distinct activities:

1. Real-data traffic forecasting and uncertainty experiments on METR-LA and PEMS-BAY.
2. Controlled SUMO traffic generation followed by the same observation-failure and forecasting evaluation.
3. A live TraCI integration check demonstrating that the forecaster only sees observations released by the reporting gateway.

The central contribution is the treatment of missing and delayed **calibration feedback**. A new graph architecture, traffic-signal controller, reinforcement-learning policy, or digital twin is outside the core scope.

## 2. Finalized problem statement

Traffic forecasting systems need both accurate future speed predictions and useful prediction intervals. A sensor outage can remove recent inputs and also prevent the system from observing the outcomes needed to update those intervals. Spatially correlated outages therefore produce stale, geographically unrepresentative calibration residuals. Delayed backlogs can further distort online updates after service resumes.

**PS-1:** How can a multihorizon traffic speed forecasting system maintain useful empirical uncertainty calibration when spatially correlated sensor outages delay or permanently suppress the outcome observations required for calibration, and how quickly can reliability recover after reporting resumes?

Investigate two separate mechanisms: degradation of the predictor because inputs are unavailable, and degradation of the uncertainty estimator because calibration outcomes are unavailable. Isolate these mechanisms experimentally before evaluating their combination.

### Research questions

- **RQ1:** How much additional uncertainty degradation results from missing or delayed feedback when the input stream and point predictions are held fixed?
- **RQ2:** Can feedback-aware residual selection, spatial pooling, and an explicit stale-feedback fallback improve the coverage–width trade-off over strong received-feedback-only methods?
- **RQ3:** How do outage duration, spatial extent, delay, and permanent loss affect reliability and recovery time?
- **RQ4:** Do the same conclusions hold for controlled SUMO congestion and recovery episodes with a separately modeled reporting channel?
- **RQ5:** Which components are necessary, and how much online memory and latency do they require?

### Research objectives

| ID | Objective | Concrete output |
|---|---|---|
| O1 | Formalize the input-observation and outcome-feedback processes with causal timestamps | Schemas, replay engine, leakage tests |
| O2 | Develop a mask-aware multihorizon speed predictor with a modular uncertainty interface | Reproducible FAR-GW backbone and checkpoints |
| O3 | Develop and test feedback-aware residual calibration | FAR-Cal implementation, ablations, diagnostics |
| O4 | Benchmark reliability under independent and corridor-correlated reporting failures | Paired METR-LA and PEMS-BAY experiments |
| O5 | Test robustness and recovery under controlled traffic dynamics and reporting failures in SUMO | Network, demand, event schedules, detector streams, replay and live check |
| O6 | Quantify accuracy, uncertainty quality, recovery, and computational cost | Tables, figures, confidence intervals, reproducibility manifest |

### Falsifiable hypotheses

- H1: Delaying or suppressing feedback worsens interval quality even with identical point forecasts and identical inputs.
- H2: FAR-Cal improves mean interval score under the primary correlated-outage condition relative to the strongest validation-selected feasible comparator, without materially worsening empirical coverage.
- H3: Pooling and staleness handling reduce recovery delay after reporting resumes; either component may fail or prove unnecessary.
- H4: Improvements persist across both real datasets and independently generated SUMO test episodes.

Do not promise an accuracy gain, a percentage improvement, or a statistical guarantee in advance. A well-supported negative result is an acceptable scientific outcome.

## 3. Literature positioning and novelty boundary

The following sources define the minimum related-work and comparison scope. “Strong comparison” does not mean a verified universal SOTA ranking. Published scores from different splits and preprocessing pipelines must not be placed into one ranking.

| Work | Established capability | Consequence for PS-1 |
|---|---|---|
| [DCRNN, ICLR 2018](https://arxiv.org/abs/1707.01926) | Directed graph-based traffic forecasting | Foundational point-forecast baseline |
| [Graph WaveNet, IJCAI 2019](https://arxiv.org/abs/1906.00121) | Adaptive graph with temporal convolutions | Reusable backbone; not a novel contribution here |
| [STID, CIKM 2022](https://arxiv.org/abs/2208.05233) | Strong simple spatial/temporal identity baseline | Tests whether added complexity is necessary |
| [STAEformer, CIKM 2023](https://arxiv.org/abs/2308.10425) | Strong supervised spatiotemporal forecasting | Modern point reference and second calibration backbone |
| [TESTAM, ICLR 2024](https://arxiv.org/abs/2403.02600) | Routing among temporal and graph experts | Optional broader point comparison |
| [Graph-based Forecasting with Missing Data through Spatiotemporal Downsampling, ICML 2024](https://proceedings.mlr.press/v235/marisca24a.html) | Forecasting conditioned on structured missing observations | Missing-input competitor |
| [Relational Conformal Prediction / CoRel, ICML 2025](https://proceedings.mlr.press/v267/cini25a.html) | Calibration exploiting relationships between series | Direct methodological competitor; graph residual pooling is already established |
| [Adaptive Conformal Inference for Multi-Step Ahead Time-Series Forecasting Online, COPA 2024](https://proceedings.mlr.press/v230/hallberg-szabadvary24a.html) | Horizon-specific online conformal inference | Ordinary forecast-horizon delay is already addressed; author is Johan Hallberg Szabadváry |
| [MAS-Mamba, 2026](https://ietresearch.onlinelibrary.wiley.com/doi/pdfdirect/10.1049/itr2.70304) | Missingness-aware traffic forecasting with uncertainty calibration | Mask/age encoding plus conformal calibration alone is insufficient novelty |
| [Post-Training Adaptive Conformal Prediction for Incomplete Time Series, TMLR 2026](https://openreview.net/forum?id=KMBU4wx79B) | Closely related incomplete-series calibration | Full text remains an explicit overlap check; do not infer an absent feature from inaccessible text |
| [Adaptive conformal prediction for spatio-temporal hypergraph neural networks, 2026](https://www.sciencedirect.com/science/article/abs/pii/S1568494626017047) | Closely related adaptive graph uncertainty work | Full-text comparison required before claiming novelty |

**Candidate contribution:** a timestamp-correct evaluation of jointly impaired inputs and feedback, plus a calibration mechanism that explicitly tracks residual freshness, available spatial support, and recovery from reporting backlogs.

Before manuscript submission, create `docs/novelty_matrix.csv` with columns: work, version, missing inputs, delayed feedback, permanently lost feedback, correlated outages, stale-residual treatment, recovery evaluation, code availability, evidence location. Retrieve the two unresolved full texts where possible and perform a refreshed primary-source search. If a close paper already implements the proposed mechanism, revise the methodological claim and retain the benchmark only if it adds a defensible contribution. Implementation can proceed while this evidence check remains open.

**Statistical boundary:** FAR-Cal as specified below is an empirical adaptive interval method. Its weighted pooling and fallback do not automatically inherit split-conformal finite-sample validity. Arbitrary outcome-dependent missing feedback is not identifiable from the observed stream without further assumptions. Pointwise sensor/horizon coverage is not simultaneous network or trajectory coverage.

## 4. Data and temporal protocol

### 4.1 Real datasets

Use the [original DCRNN repository](https://github.com/liyaguang/DCRNN) as the provenance entry point. Expected benchmark dimensions are 207 sensors for METR-LA and 325 for PEMS-BAY, sampled every five minutes. Inspect actual files and timestamps instead of hard-coding total lengths or date ranges.

Record source URL, retrieval date, license/access terms, SHA-256, sensor order, time zone, timestamp gaps, target units, original missing-value convention, and adjacency provenance. Never silently replace METR-LA/PEMS-BAY speed with a different PEMS flow dataset.

Use speed as the prediction target. Other benchmark-derived channels are observed mask, observation age, time-of-day sine/cosine, and day-of-week encoding. No weather, accident labels, future congestion labels, or external features are required.

### 4.2 Canonical split

Primary uncertainty experiment: chronological **60% training / 10% tuning / 10% initial calibration / 20% test**, applied separately to each dataset. This deliberately differs from the common 70/10/20 point benchmark to reserve untouched initial calibration data. Rerun all comparisons under the primary split.

Define split boundaries on raw timestamps before generating windows. All target timestamps of a window must belong to its assigned split. Past input context may cross a boundary if those observations were already available. Drop windows whose targets straddle a boundary. Fit scaling, graph learning, historical fallback statistics, and congestion thresholds using training data only.

Train model weights on training data. Select checkpoints and all calibration hyperparameters on tuning data. For tuning online calibrators, use the first third of the tuning partition as its temporary residual warm-up and score only the remaining two thirds. Then discard that tuning state. Build the final initial residual store on the separate calibration partition using the frozen chosen model. Neither calibration nor test labels train the predictor.

The primary test starts with a **clean historical calibration archive**. This is an explicit deployment assumption. Add a secondary experiment with corrupted calibration history to test sensitivity to that assumption. In test, label access is governed entirely by the event gateway.

### 4.3 Windows and masks

- History: 12 completed bins, or 60 minutes.
- Targets: all 12 future bins, or 5–60 minutes.
- Primary reporting horizons: 3, 6, and 12 bins, corresponding to 15, 30, and 60 minutes. Also export all horizons.
- Original validity mask and injected reporting mask are separate arrays.
- Score artificially hidden values only if the original value was valid.
- Naturally missing targets have no known truth and never enter accuracy or coverage denominators.
- Audit whether zeros are used as missing sentinels in the specific release. Do not automatically discard physically valid stationary speeds in SUMO.
- Scale with observed training values. Preserve real-unit outputs in mph; retain source units in metadata.
- At a forecast origin, fill an unavailable historical slot using the most recent causally received value with an observation timestamp no later than that slot; otherwise use a training-only sensor median. Keep its mask at zero. Never interpolate using future observations.
- Age is the elapsed number of bins since the latest observation usable for that slot; clip at 288 and divide by 288 for the backbone. Preserve uncapped age for diagnostics.

## 5. Observation and feedback semantics

Let `t` be the end of a completed five-minute bin. A forecast issued at `t` predicts `y[i,t+h]`. Its target does not yet exist at issue time. Its normal earliest feedback time is `t+h`; any reporting delay is additional to this forecast horizon.

Maintain separate release times:

- `a_input(i,u)`: when the observation for bin `u` becomes available to the predictor.
- `a_feedback(i,u)`: when the same outcome is available to the calibration service.
- Infinity means permanently unavailable in that channel.

In the realistic **coupled channel**, both releases are equal. In diagnostic experiments, independently restrict either channel to distinguish causes. Describe input-only and feedback-only settings as controlled service architectures, not universal real-sensor behavior.

| Entity | Required fields |
|---|---|
| Observation | episode_id, sensor_id, observed_bin, value, original_valid |
| Release event | event_id, channel, observed_bin, arrival_bin, sensor_id, value, original_valid |
| Forecast | forecast_id, issue_bin, target_bin, horizon, sensor_id, point, scale, interval bounds at each level, issue-context features, model/calibrator version |
| Feedback match | forecast_id, event_id, arrival_bin, immutable target, normalized residual, miss indicator for originally issued interval |
| Fault manifest | scenario_id, seed, affected sensors, start/end, fault type, delay/loss parameters; evaluator access only |

A single outcome observation can score earlier forecasts at several horizons. Within a given horizon, ingest a forecast residual at most once. Backlog arrival order must not change deduplication. Store original forecast intervals permanently; never recompute them when labels arrive.

At each bin boundary: release due events, update input history, match and ingest eligible calibration outcomes, and then issue the new forecasts. This permits a zero-additional-delay observation for the just-completed bin to inform the next forecast. Release events cannot have `arrival_bin < observed_bin`.

The predictor must not infer that an unreleased packet is permanently lost just because the offline manifest says so. It only knows that it has not arrived. Future fault schedules and fault-generator probabilities are private to the simulator/evaluator.

## 6. Proposed FAR-GW predictor

Use a maintained, auditable Graph WaveNet implementation with author-source attribution. Retain its temporal convolution and graph backbone. Add mask/age channels and a noncrossing three-quantile head. This is an engineering adaptation, not an architectural novelty claim.

Input tensor: `[batch, history=12, sensors=N, features]`. Output tensors: `[batch, horizon=12, sensors=N]` for each of `q05`, `q50`, `q95`.

Parameterize `q50 = m`, `q05 = m - softplus(d_low)`, and `q95 = m + softplus(d_high)`. Train with equally weighted masked pinball losses at quantiles 0.05, 0.50, and 0.95:

`rho_tau(e) = max(tau*e, (tau-1)*e)`, where `e = y - q_tau`.

Normalize the loss by the number of valid target entries, not by all tensor cells. Skip a batch with no valid targets and log the event. Do not train against filled targets.

For FAR-Cal use `mu = q50` and `s = max((q95-q05)/2, 1 mph)`. Compute these in real units. Train point and scale jointly once, then freeze them throughout test. The calibrated intervals are symmetric around `mu`; the underlying quantile head is also reported as an uncalibrated baseline.

Initial training defaults: Adam, learning rate 0.001, batch size 32, maximum 100 epochs, early stopping patience 15, gradient clipping norm 5. Adopt source architecture widths initially and serialize the actual resolved architecture. Tune only a small registered budget; do not create undocumented architecture advantages.

Training augmentation: equal probability of clean history, independent missing history, or a connected-block missing history. Draw independent missing rates from 0.1–0.3; block length from 3–12 bins and spatial fraction from 0.05–0.20. Use only training observations and causal filling. Apply the same augmentation policy to all mask-aware backbone comparisons. Add a clean-trained control to measure augmentation effects.

## 7. Proposed FAR-Cal algorithm

This is the initial method to implement and evaluate, not an established theorem. Keep it modular so that improvements or failures can be traced to specific choices.

### 7.1 Residuals and immutable context

For a forecast `j` whose genuine outcome has arrived, store:

`r_j = abs(y_j - mu_j) / s_j`.

Its context `z_j` was recorded at issue time and contains (a) fraction of available history entries in the local spatial group, and (b) `log(1 + mean uncapped observation age)` in that group. Standardize context features using training-generated contexts only. Do not replace issue-time context with hindsight context.

A local spatial group consists of a sensor and up to four nearby sensors reachable along the supplied road graph; choose by shortest-path distance and deterministically break ties by sensor ID. If the graph only supports connectivity, use hop distance. Store groups before evaluating test outcomes. Global pooling uses all sensors within the same dataset, never both cities together.

Maintain separate residual stores for every horizon. Retain residuals by **target timestamp** for the latest seven days. A very old outcome arriving today remains old; arrival time does not refresh its age.

### 7.2 Candidate weights and effective support

At issue time `t`, assign a received residual candidate the weight:

`w_j(t,z) = exp(-(t-target_bin_j)/tau) * exp(-||z-z_j||^2/(2*b^2))`.

Defaults: `tau = 288 bins` and `b = 1.0`. Candidates must have been received by `t`. A missing outcome contributes no residual and no artificial zero score.

Build local and global weighted empirical residual distributions, each normalized separately. To avoid counting simultaneous neighboring outcomes as independent support, sum candidate weights by target bin into block weights `B_u`. Use `ESS = (sum_u B_u)^2 / sum_u B_u^2` as a **support diagnostic**, not a proof of independent samples.

For each horizon and node:

1. Let `ESS_L` and `ESS_G` be local and global support.
2. Set `lambda = ESS_L / (ESS_L + 50)`.
3. If both pools exist, form `F_live = lambda*F_local + (1-lambda)*F_global`. If only one exists, use it. If neither exists, skip to the frozen archive.
4. Let `g` be bins since the freshest received local residual's target time. If local support is absent, derive `g` from the frozen archive's latest local target time; if that is absent, use the global archive and mark fallback.
5. Set `eta = min(1, ESS_G/50) * exp(-g/288)`, or zero if no live global support.
6. Form `F = eta*F_live + (1-eta)*F_frozen`, where `F_frozen` is the node/horizon residual distribution from initial calibration, falling back to its horizon-wide archive if it contains fewer than 50 scores.

The frozen archive remains fixed. The live store is initialized from eligible recent calibration residuals and then updated only on genuine releases. Store frozen empirical distributions compactly if necessary; approximation error must be checked against exact quantiles in tests. Start with exact calculations on smoke data. For full-scale replay, cache horizon-level quantities and vectorize or chunk weighted quantile calculations to bound memory. If candidate subsampling or histogram approximation is required, register its seed and resolution, compare it with exact calculations on a fixed validation subset, and report any material change; do not silently alter the method to meet runtime targets.

### 7.3 Interval construction and explicit low-feedback response

For each desired miscoverage `alpha` in `{0.10, 0.05}`, take the smallest score with cumulative mixture mass at least `1-alpha` as `q_alpha`. Do not apply an exchangeable split-conformal theorem to this weighted mixture.

Let `m` be the unavailable fraction of the current local input history. Define:

`inflation = min(3, 1 + beta_g*min(g/288, 2) + beta_m*m)`.

Output:

`L = max(0, mu - s*q_alpha*inflation)`

`U = max(L, mu + s*q_alpha*inflation)`.

Default `beta_g=0.25`, `beta_m=0.25`. These are transparent heuristic robustness controls. Tune them on the tuning replay only, choosing from `{0, 0.25, 0.5}`. Enforce nested 90%/95% intervals by expanding the 95% interval to contain the 90% interval if numerical issues violate nesting. No arbitrary upper speed cap is applied. Record lower-bound clipping.

The inflation cap is a practical width constraint and may cause undercoverage in severe failures. Report that failure if it occurs. If both live and frozen stores are empty, stop calibration with an actionable error; an uncalibrated prediction can be emitted only with an explicit unavailable-interval status.

### 7.4 Parameter selection and interpretation

Use validation mean interval score as the selection criterion, with a preference for configurations whose aggregate 90% coverage is at least 0.88. This is a tuning rule, not a validity guarantee. If no candidate meets it, choose the lowest interval score and record the miss. Fix the selected configuration before final calibration and test.

Initially hold `tau`, `b`, and support constant at the defaults while tuning the two inflation coefficients. Add a separate registered sensitivity experiment for `tau in {144,288,576}`, `b in {0.5,1,2}`, and support threshold in `{25,50,100}`, one factor at a time. Do not select a new winner from test sensitivity results.

Do not include inverse-propensity weighting in the primary algorithm: availability is not generally identifiable from unobserved outcomes. An optional extension may test known-propensity weighting in an explicitly simulated MAR setting with positivity and clipping; label it an oracle-assisted extension.

## 8. Comparison methods and fair implementation

### Required minimum comparison set

| Category | Methods | Role |
|---|---|---|
| Simple point controls | Last available speed; training historical time-of-week median with sensor-median fallback | Sanity and difficulty checks |
| Point backbones | DCRNN, Graph WaveNet, STID, STAEformer | Accuracy/context comparison; use matching input budgets |
| Fixed uncertainty | Raw quantile head; frozen split-conformal normalized residual intervals | No online adaptation controls |
| Online uncertainty | Rolling received-residual quantiles; horizon-specific ACI; exponentially recency-weighted residual quantiles | Tests whether feedback-aware machinery adds value |
| Relational uncertainty | CoRel using author implementation where reproducible | Direct spatial calibration comparator |
| Proposed | FAR-Cal with the identical frozen FAR-GW point/scale predictions | Main comparison |
| Oracles | Immediate-at-target feedback oracle; complete-feedback oracle under identical impaired inputs | Separates delayed/lost feedback cost; never a deployable method |

For the frozen split-conformal baseline, use the sorted calibration residual at index `ceil((n+1)*(1-alpha))`; if the index exceeds `n`, the mathematical interval is unbounded and must be represented explicitly. Do not silently cap it and call it exact. The exchangeable reference formula does not establish validity under the project's dependent test process.

Implement ACI from its cited algorithm and document any changes needed for missing outcomes and batched spatial arrivals. The received-only variant updates from returned forecast errors, using the originally issued bounds. Batch same-time updates by horizon rather than accidentally taking hundreds of sequential learning-rate steps in an arbitrary sensor order. Any such adaptation must be labeled a project variant, and its guarantees must be re-examined rather than inherited.

For CoRel, distinguish a faithful reproduced setting from a project multihorizon adaptation. Its original protocol is not identical to this project's. Do not change its architecture silently or import its published performance into the main table. Give every calibrator the same point forecasts in the isolated calibration comparison wherever its interface permits; report incompatible joint-learning variants separately.

### Extended comparisons

Add the 2024 missing-data downsampling model and MAS-Mamba if official code or an auditable reproduction is feasible. Add TESTAM if compute permits. These are important extensions, but a partial implementation must not masquerade as the named method. Record exact source commits, dependencies, parameter counts, differences, and unresolved reproduction issues. Use [BasicTS](https://github.com/GestaltCogTeam/BasicTS) where helpful for consistent point forecasting.

No baseline may receive outcomes earlier than the proposed method except explicitly labeled oracles. Calibration methods must share masks, forecast IDs, fault manifests, seeds, and metric denominators. Separate the calibration-only study from retrained-backbone comparisons.

## 9. Real-data fault experiments

Run a small registered core first; avoid an uncontrolled full Cartesian grid.

### 9.1 Primary settings

| ID | Input channel | Feedback channel | Purpose |
|---|---|---|---|
| C0 | Immediate | Immediate at target time | Clean reference |
| C1 | Immediate | Correlated buffered outage | Isolate feedback impairment |
| C2 | Correlated buffered outage | Immediate at target time | Isolate input impairment |
| C3 | Same correlated buffered outage | Same channel as inputs | Primary coupled setting |
| C4 | Same correlated lossy outage | Same channel as inputs | Permanent-label-loss stress |
| C5 | Independent loss | Same channel as inputs | Distinguish independent and spatial failures |

Primary correlated event: choose a connected set containing approximately 10% of sensors; stop releases for six bins (30 minutes); buffer the observations; at restoration, release the backlog over three bins. Each backlog packet gets an extra delay of 0–2 bins according to a saved random draw. Subsequent normal packets may overtake it. Use two nonoverlapping outage events per full test day with a minimum two-hour gap. Seed and save choices before any model test evaluation.

For C4, lose 50% of the packets generated during each outage permanently; release the rest using the C3 schedule. For C5, independent packet loss probability is 0.10. Report realized loss/delay rates; do not imply that C5 and C3 are matched in total severity. Add a matched-rate independent control when interpreting the effect of spatial correlation.

For the C3/C4 independent-versus-spatial matched-rate control, preserve the event windows, number of failed sensor–time entries, and delay/loss draws, but randomly permute affected sensor identities within each window. This holds severity approximately fixed while breaking spatial concentration. If graph-connected sets of the desired size are unavailable, use a documented connected component and report the realized fraction. Call these graph-connected groups rather than verified physical corridors unless geographic metadata confirms a corridor.

### 9.2 Stress tests

One factor at a time around C3/C4:

- Outage durations: 3, 6, 12, 24 bins.
- Affected sensor fractions: 0.05, 0.10, 0.20.
- Additional backlog delays: 0, 3, 12 bins, beyond the outage's inherent wait.
- Permanent loss within outages: 0, 0.25, 0.50, 1.00.
- Cold calibration archive: apply the same reporting process during initial calibration.
- Outcome-dependent failure: increase loss probability during a true low-speed regime, using training-derived thresholds. This is MNAR stress; only the fault generator/evaluator sees the true regime. Never describe it as an identifiable correction experiment.

Use model seeds `{11,22,33}` and fault seeds `{101,202,303}` for the core, crossing them when replay is inexpensive. The same model checkpoint can serve all fault seeds. Use cached point predictions for all calibrators sharing an input stream. Mark one-seed sensitivity results as exploratory unless replicated.

## 10. SUMO experimental specification

### 10.1 What SUMO establishes

SUMO adds controlled physical traffic events and repeatable observation failures. It does not automatically reconstruct Los Angeles or the Bay Area. Speed-only benchmark arrays do not uniquely determine road geometry, routes, or origin–destination demand. Use a **synthetic motorway with real-data-inspired congestion regimes**, and report its differences from the benchmarks.

Keep SUMO demand calibration distinct from statistical interval calibration. SUMO's optional calibrators can modify traffic flows and speeds; the primary experiment does not use them to force predictions or test trajectories to match real data. [Official SUMO calibrator documentation](https://sumo.dlr.de/docs/Simulation/Calibrator.html).

### 10.2 Network and demand to generate

Build a deterministic network generator that writes nodes, edges, connections, routes, detectors, and `.sumocfg` files and invokes `netconvert`.

| Item | Initial specification |
|---|---|
| Main road | One-direction 10 km motorway, twenty approximately 500 m segments |
| Lanes | Two through lanes with explicit merge connections |
| Ramps | One on-ramp near 3 km and one near 7 km; one off-ramp near 8 km |
| Sensor stations | Twenty mainline stations, one per segment, detector on each lane |
| Normal speed limit | 27.78 m/s, approximately 100 km/h |
| Vehicle mix | 90% passenger cars, 10% trucks; serialize explicit type parameters |
| Dynamics | SUMO microscopic simulation, one-second step; record actual car-following/lane-changing models |
| Episode | Eight simulated hours, first hour warm-up |
| Forecast sampling | Five-minute completed intervals |
| Demand | Base mainline 1,800 vehicles/hour total; each on-ramp 150 vehicles/hour; 10% of eligible traffic exits at the off-ramp |
| Peak pattern | Two 60-minute demand surges, 1.4–1.8 times base mainline/ramp demand |
| Randomness | Separate demand, driver, physical-event, and gateway-fault seeds |

These demand values are starting parameters, not validated capacities. Use training-only pilot runs to ensure a mix of free-flow, congestion, and recovery without persistent network-wide gridlock. Adjust demand and merge geometry if needed, log the changes, and freeze the scenario generator before creating tuning/calibration/test splits. Inspect that SUMO has not introduced unintended priority stops or disconnected ramp routes.

Generate route assignments consistently with ramp entry and off-ramp exit constraints. Record attempted departures, successful insertions, insertion delay, completed vehicles, unfinished vehicles, collisions, and teleports. Demand that could not enter the network must not be counted as delivered flow.

### 10.3 Physical event families

Use four equally represented episode families:

- **P0:** Normal demand pattern, no additional physical restriction.
- **P1:** Peak-demand surge with stochastic timing and amplitude.
- **P2:** Local temporary speed restriction: reduce the permitted speed on a selected approximately 500 m downstream section to 8.33 m/s for 20–40 minutes, then restore the exact original settings.
- **P3:** Demand surge overlapping the speed restriction.

Control the restriction through the documented TraCI lane-speed interface, with events recorded to a physical-event log. Call this a speed-restriction bottleneck; it is not an accident or literal lane closure. A separate actual lane-closure extension needs explicit routing and safety behavior and is not required for the core. [SUMO lane-state interface](https://sumo.dlr.de/docs/TraCI/Change_Lane_State.html).

Choose restriction start times at least two hours after simulation start and leave at least two hours after restoration. Event schedules must be unknown to the predictor. Include paired reporting failures that begin before, during, and after the physical event. Keep physical events and reporting failures independently configurable.

### 10.4 Detector target and measurement validity

Use E1 loop interval output with a 300-second period. Define station speed as the lane-speed weighted average using `nVehContrib` as weights; lanes with no contributing vehicles are excluded. If no lane contributes, mark station speed unavailable. E1 speed is a time-mean measurement in m/s; convert to mph using `2.2369362920544`. Preserve counts and occupancy for diagnostics, not extra predictor features in the main speed-only experiment. [Official E1 output definitions](https://sumo.dlr.de/docs/Simulation/Output/Induction_Loops_Detectors_(E1).html).

Zero vehicles is not zero speed. Sparse or stopped traffic may yield missing loop-speed measurements, so report this native detector missingness separately from gateway failures. Optional lane-area measurements are a different target and must not be silently substituted.

The authoritative offline truth stream is the valid detector output **before** reporting faults. This is observation-level ground truth, not every vehicle's exact traffic state. Discard warm-up and incomplete bins. Use one consistent interval-end convention for both XML extraction and online collection.

### 10.5 Gateway and SUMO integration

Implement reporting faults in a Python gateway; SUMO provides traffic dynamics, while the gateway models application-level delay/loss. Do not describe this as a packet-level communications simulator.

For offline experiments, run each traffic episode once and replay its immutable detector stream for all methods/fault settings. Thus a calibration method cannot accidentally benefit from a different traffic realization.

For the live check, connect via TraCI, advance simulation, aggregate only completed detector intervals, pass observations through the gateway, and call the same online engine at five-minute boundaries. Validate the installed API against documentation. Use interval measurements where available; otherwise implement a tested crossing-event accumulator. Do not average last-step speeds across empty timesteps. TraCI supports Python control; libsumo is a compatible performance option after equivalence testing. [Official Python interface](https://sumo.dlr.de/docs/TraCI/Interfacing_TraCI_from_Python.html).

Use a separate evaluator object/process for unrestricted detector truth. The live predictor must not have direct access to TraCI detector values, future event schedules, or the truth store.

### 10.6 SUMO data partitions and runs

Paper-scale initial budget: 120 independently generated eight-hour episodes, allocated as 60 training, 15 tuning, 15 calibration, and 30 test episodes. Balance the four physical families as closely as these counts permit and publish counts. Partition by episode seed, never by randomly mixing windows from the same episode.

Never construct a history or target window across two SUMO episodes. Give calibration episodes a deterministic ordering on a synthetic historical day clock, retaining within-episode time offsets and gaps between episodes. The same frozen archive ordering is used for every held-out episode. Train a separate 20-node backbone on SUMO training episodes. Use the same FAR-Cal algorithm and tuning procedure; do not claim direct zero-shot transfer of node-specific METR-LA/PEMS-BAY weights to the synthetic network. Real-to-SUMO transfer is optional future work.

For each test episode, reset calibration to the same frozen historical SUMO archive, use warm-up observations to establish history, and start scoring only origins with a full post-warm-up 12-bin history and 12 available future target bins. Initial archive ages use a documented synthetic day clock; all archive timestamps precede the test episode, and the latest archive day ends immediately before it. Never carry information from one held-out test episode into another.

Evaluate C0, C3, and C4 across all 30 test episodes. Add C1/C2 on a representative prespecified subset of 12 episodes for mechanism isolation. Use the same three model seeds as the real-data study. Bootstrap uncertainty over episodes, not individual vehicles.

Smoke profile: one short training episode, one calibration episode, and one independent test episode; at least three simulated hours each with 30-minute warm-up. Use two training epochs. This checks execution only and cannot support paper conclusions.

### 10.7 SUMO outputs

Required: reusable XML/network generator, seed manifest, detector truth, released observation streams, forecast archive, uncertainty metrics, outage/recovery timelines, network map, and one demonstrative live run. Optional short GUI video is explanatory only.

Traffic flow, occupancy, and trip summaries validate the scenario. This study does not establish improvements in travel time, emissions, accident prevention, or traffic control because the proposed model does not control vehicles or signals.

## 11. Online engine pseudocode

```python
for now in clock.completed_bins():
    events = gateway.release_due(now)  # only currently released data
    for event in stable_sort(events):
        assert event.arrival_bin <= now
        assert event.observed_bin <= event.arrival_bin
        if event.channel == 'input' and event.original_valid:
            history.ingest(event)
        if event.channel == 'feedback' and event.original_valid:
            for forecast in ledger.match(event.sensor_id, event.observed_bin):
                if not ledger.feedback_seen(forecast.id):
                    calibrator.observe(forecast, event, now)
                    ledger.mark_feedback_seen(forecast.id)

    calibrator.finish_arrival_batch(now)
    calibrator.expire_by_target_time(now)
    if eligible_origin(now):
        x, mask, age = history.causal_window(now, length=12)
        point, scale, raw_quantiles = frozen_model.predict(x, mask, age)
        context = make_observable_context(mask, age)
        intervals, diagnostics = calibrator.predict(now, point, scale, context)
        ledger.append_immutable(now, point, scale, intervals, context, diagnostics)

# Offline evaluator runs separately against originally valid hidden truth.
# Continue gateway draining after the last issue time when needed for logs;
# do not retrospectively change issued forecasts or their scores.
```

The actual implementation must distinguish observation-event deduplication from per-forecast feedback deduplication. Releasing the same observation into both channels is allowed; ingesting its residual twice for the same forecast is not.

## 12. Evaluation parameters and statistical analysis

Compute all predictive metrics on a common valid-target set, irrespective of whether those targets were ever reported to the model. Also report observed-feedback-only metrics as a diagnostic to expose selection bias. Never use them as the only quality estimate when withheld truth exists.

For intervals `[L_j,U_j]`, target `y_j`, point `mu_j`, and nominal miscoverage `alpha`:

| Metric | Definition / interpretation |
|---|---|
| MAE | `mean(abs(y-mu))`, in mph |
| RMSE | `sqrt(mean((y-mu)^2))`, in mph |
| PICP | `mean(1[L <= y <= U])` |
| MPIW | `mean(U-L)`, in mph; interpret jointly with coverage |
| Coverage error | `PICP-(1-alpha)`, signed, plus absolute value |
| Undercoverage | `max(0, (1-alpha)-PICP)` |
| Interval score | `(U-L) + (2/alpha)*(L-y)*1[y<L] + (2/alpha)*(y-U)*1[y>U]` |
| WIS | For two levels, `[0.5*abs(y-mu) + sum_k(alpha_k/2)*IS_k] / (2+0.5)`, averaged over valid cases |
| Feedback availability | Fraction of mature valid outcomes released by a stated wall-clock deadline |
| Feedback delay | Arrival minus target time; summarize median, p95 and unreleased fraction separately |
| Support/staleness | Local/global support diagnostic, freshest residual age, fallback frequency |
| Efficiency | Train time, parameter count, GPU peak memory, online p50/p95 latency and resident memory |

MAPE is secondary only: state a denominator rule such as evaluating valid speeds at least 5 mph and report the excluded fraction. Do not use MAPE to hide poor performance in congestion.

Primary endpoint: mean 90% interval score under C3, averaged over horizons 3/6/12 with equal horizon weights, reported separately for each dataset. Report 90% and 95% coverage and width alongside it. Select the strongest feasible comparator on tuning data and lock its identity before test; still report every comparator.

Stratify by horizon, sensor, predefined group, outage status, outage age, feedback age, and speed regime. Derive a sensor's congestion threshold as its training 20th speed percentile. True future speed may define evaluation subgroups, but cannot be used as an input or online context. Publish group sample counts; flag groups with fewer than 100 valid targets as low support.

### Recovery definition

Define `t_restore` as the end of the gateway outage, not the time when all backlog packets have arrived. For each event and horizon, compute coverage and interval score over affected sensors using a trailing 12-origin window containing only origins at or after `t_restore`.

Coverage recovery occurs at the first window end with at least 100 valid forecast–target pairs and coverage at least `(1-alpha)-0.02`, sustained for three consecutive window ends. Report recovery time from `t_restore`, minimum measurable delay imposed by the window, and censoring if recovery is not observed before episode end or the next overlapping event. Do not assign an arbitrary finite recovery time to nonrecovering events.

Also report interval score and width over the same windows so that trivial widening is visible. Add a joint recovery diagnostic requiring interval score no more than 1.10 times the corresponding immediate-feedback oracle's score over the same cases; when oracle score is numerically zero, use an explicit epsilon and flag it. This is an operational diagnostic, not a coverage theorem.

### Uncertainty of comparisons

Use paired differences with identical traffic truth and faults. For real data, use a moving-block bootstrap over time with 24-hour blocks, retaining all sensors/horizons within blocks; check sensitivity at 12 and 48 hours. For SUMO, bootstrap complete episodes. Use 2,000 bootstrap resamples and report 95% intervals.

Model seeds share test data, so do not present all seed-by-time observations as independent replicates. Report between-training-seed variation separately or use a documented hierarchical procedure. Identify the primary hypothesis before testing and mark secondary subgroup comparisons as exploratory or apply a stated multiple-comparison correction.

## 13. Required ablations

Run each ablation with the same frozen point/scale predictions:

1. Remove spatial pooling: local sensor residuals only, retaining fallback.
2. Remove context matching: set context weight to one.
3. Remove target-age weighting: uniform weights within the seven-day buffer.
4. Remove stale-feedback blend: use the live distribution whenever available.
5. Remove inflation: set both beta coefficients to zero.
6. Use arrival-time freshness instead of target-time freshness as an explicitly flawed diagnostic; test whether backlog bursts expose the problem.
7. Frozen calibration only.
8. Repeat the main calibrator comparison on a frozen STAEformer-based quantile/scale backbone.

The second-backbone experiment is necessary before claiming backbone independence. If unavailable, limit the conclusion to FAR-GW.

## 14. Repository structure and interfaces

```text
ps1-feedback-traffic/
  README.md
  pyproject.toml
  requirements.lock
  configs/
    base.yaml
    metr_la.yaml
    pems_bay.yaml
    sumo.yaml
    smoke.yaml
    experiment_registry.yaml
  src/ps1/
    cli.py
    data/{provenance,prepare,windows,graphs,scaling}.py
    models/{base,graph_wavenet_masked,staeformer_adapter,baselines}.py
    calibration/{base,frozen,rolling,aci_adapter,corel_adapter,far_cal}.py
    online/{events,gateway,history,ledger,replay}.py
    simulation/{network,demand,physical_events,detectors,runner,live}.py
    evaluation/{metrics,recovery,bootstrap,plots,tables}.py
    utils/{seeding,logging,artifacts}.py
  tests/
    test_time_causality.py
    test_feedback_deduplication.py
    test_split_and_masks.py
    test_calibration_math.py
    test_sumo_detector_aggregation.py
    test_smoke_pipeline.py
  data/                     # ignored large/local data; documented paths
  artifacts/                # checkpoints, predictions, configs, manifests
  reports/
  docs/{decisions.md,protocol.md,novelty_matrix.csv,limitations.md}
```

Interfaces to implement:

```python
class Forecaster:
    def fit(self, train_loader, tune_loader, config): ...
    def predict(self, history, mask, age): ...  # point, scale, raw quantiles

class Calibrator:
    def initialize(self, calibration_records): ...
    def observe(self, immutable_forecast, released_outcome, now): ...
    def finish_arrival_batch(self, now): ...
    def expire_by_target_time(self, now): ...
    def predict(self, now, point, scale, context): ...
    def state_dict(self): ...
    def load_state_dict(self, state): ...

class ReportingGateway:
    def ingest_truth(self, observation): ...  # producer-side only
    def release_due(self, now): ...          # consumer-visible interface
```

Persist arrays in a documented efficient format such as Parquet plus NPZ/Zarr. Define dtypes, dimensions, sensor order, units, and timestamps in a schema file. Avoid one JSON file per scalar prediction. Forecast storage can be large; stream by day/episode and retain content hashes.

Use Python 3.11 as the initial compatibility target and a supported PyTorch version compatible with baseline code. Resolve and lock actual versions after environment inspection. Record SUMO binary and TraCI versions and ensure they match. CPU smoke execution is required; GPU training is optional depending on available hardware. Do not hard-code a nonexistent CUDA environment.

## 15. Base configuration contract

The following is a configuration to implement, not a claim that a repository already exists:

```yaml
project: ps1_feedback_traffic
profile: paper
sampling_minutes: 5
history_bins: 12
horizon_bins: 12
report_horizons: [3, 6, 12]
interval_levels: [0.90, 0.95]
units: mph
split:
  train: 0.60
  tune: 0.10
  calibration: 0.10
  test: 0.20
training:
  seeds: [11, 22, 33]
  epochs: 100
  patience: 15
  batch_size: 32
  learning_rate: 0.001
  gradient_clip: 5.0
  quantiles: [0.05, 0.50, 0.95]
calibration:
  method: far_cal
  scale_floor_mph: 1.0
  buffer_days: 7
  age_decay_bins: 288
  context_bandwidth: 1.0
  support_reference: 50
  stale_decay_bins: 288
  beta_gap: 0.25
  beta_missing: 0.25
  inflation_cap: 3.0
faults:
  seeds: [101, 202, 303]
  primary: C3
  connected_fraction: 0.10
  duration_bins: 6
  backlog_release_bins: 3
  events_per_day: 2
sumo:
  mainline_length_m: 10000
  stations: 20
  through_lanes: 2
  step_seconds: 1
  aggregation_seconds: 300
  episode_seconds: 28800
  warmup_seconds: 3600
  train_episodes: 60
  tune_episodes: 15
  calibration_episodes: 15
  test_episodes: 30
  physical_families: [P0, P1, P2, P3]
evaluation:
  bootstrap_resamples: 2000
  real_block_hours: 24
  recovery_window_origins: 12
  recovery_min_pairs: 100
  recovery_tolerance: 0.02
  recovery_consecutive_windows: 3
```

Dataset and profile configuration must override only explicitly named keys. Save the fully resolved configuration with every run. Validate unknown keys as errors rather than silently ignoring misspellings.

## 16. Implementation sequence and acceptance gates

| Stage | Work | Completion gate |
|---|---|---|
| A | Environment, provenance, repository scaffolding | Imports pass; dataset metadata audited; exact versions recorded |
| B | Causal windows, gateway, immutable ledger, evaluator | Leakage, masking, release-order, and deduplication tests pass |
| C | Simple baselines and mask-aware Graph WaveNet | Tiny-batch learning check and independent smoke evaluation pass |
| D | Fixed/rolling calibration and FAR-Cal | Hand-computed quantile tests, interval nesting and replay-resume checks pass |
| E | Real-data core comparisons | Both datasets, registered core conditions, seeds, metrics and uncertainty estimates completed or specifically blocked |
| F | SUMO network and detector truth | Valid network/routes, reproducible traffic, no empty-as-zero error, pilot plausibility report |
| G | SUMO replay and live integration | Paired replay results; live and offline measurements agree within documented numerical tolerance |
| H | Direct competitors, ablations, sensitivity | Required comparisons complete; unavailable reproductions clearly recorded |
| I | Research report and reproducibility bundle | Every numerical table is generated from identified run artifacts; claims match evidence |

Begin SUMO scaffolding after stage B if real data downloads are blocked. A smoke pass never satisfies stages E, H, or I's research-evidence requirements.

### Required automated tests

- A horizon-12 outcome cannot affect its forecast or calibration before its target bin plus reporting delay.
- Changing future truth leaves all earlier emitted forecasts unchanged.
- Changing unreleased truth leaves model-visible state unchanged until release; evaluator scores may change.
- Delayed backlogs update each matching forecast exactly once, even with duplicate events.
- Expiration and freshness use target time, not arrival time.
- Imputed values never become calibration labels.
- Original invalid targets never enter metric denominators.
- Split targets cannot cross boundaries and normalization never uses tuning/test values.
- The fixed-conformal order statistic matches a hand-calculated example, including the small-sample infinite case.
- Weighted ECDF mixture quantiles and absent-pool fallbacks match hand-calculated examples.
- Interval widths are nonnegative and the 95% interval contains the 90% interval.
- Pausing and restoring an online checkpoint reproduces uninterrupted replay.
- E1 lane aggregation weights by contributing vehicles; empty observations remain invalid; unit conversion is correct.
- Gateway failures cannot alter the underlying SUMO traffic trajectory in this open-loop experiment.
- Live collection matches offline interval output on a short deterministic episode.

### CLI commands to implement

```bash
python -m ps1.cli audit-data --dataset metr_la
python -m ps1.cli prepare --dataset metr_la --config configs/metr_la.yaml
python -m ps1.cli smoke --config configs/smoke.yaml
python -m ps1.cli train --dataset metr_la --model far_gw --seed 11
python -m ps1.cli initialize-calibration --dataset metr_la --seed 11
python -m ps1.cli replay --dataset metr_la --scenario C3 --seed 11 --fault-seed 101
python -m ps1.cli sumo-build --config configs/sumo.yaml
python -m ps1.cli sumo-generate --config configs/sumo.yaml --split train
python -m ps1.cli sumo-live --config configs/smoke.yaml
python -m ps1.cli run-registry --config configs/experiment_registry.yaml --resume
python -m ps1.cli evaluate --registry configs/experiment_registry.yaml
python -m ps1.cli report --registry configs/experiment_registry.yaml
```

Make each command idempotent or explicitly resume-aware. Never overwrite a completed run with a different resolved configuration. Cache reuse requires matching data, configuration, source commit, and upstream artifact hashes.

## 17. Required final research deliverables

1. **Executable repository:** installation, data preparation, smoke and paper commands, tests, exact dependency lock, code provenance and licenses.
2. **Experiment registry:** every planned/run/failed/blocked experiment, reason, seed, checkpoint, runtime and artifact path.
3. **SUMO package:** regenerable network, routes, demand parameters, detectors, physical/gateway events and live interface.
4. **Results:** point accuracy, interval quality, subgroup reliability, recovery/censoring, efficiency and ablations, with uncertainty estimates.
5. **Figures:** coverage–width comparison, interval-score versus outage duration, spatial speed/error heatmap, representative outage timeline, recovery curves, and compute–quality comparison. Select representative episodes by a prespecified rule, such as the median baseline interval score, not the most favorable visual.
6. **Manuscript draft:** evidence-backed sections described below, using generated tables and reference links.
7. **Reproduction manifest:** environment, Git commit, dataset and artifact hashes, seeds, configuration, hardware, and a machine-readable summary of all limitations.

Minimum main tables: dataset/protocol; baseline implementations; C0–C5 interval metrics on both datasets; main-horizon point metrics; SUMO physical/fault combinations; recovery with nonrecovery fraction; ablations; computational cost. Do not fill unavailable results with invented numbers or zeros; use `not run` or `blocked` with a reason.

## 18. Manuscript draft framework

### Proposed abstract, before results

Traffic speed prediction intervals require outcome observations for calibration, but sensor reporting failures can simultaneously remove forecasting inputs and delay or suppress calibration feedback. This study investigates this coupled observation problem in multihorizon traffic speed forecasting. We propose FAR-Cal, a feedback-aware residual calibration component that combines target-time freshness, observable missingness context, spatial residual support, and an explicit stale-feedback fallback. The component is evaluated with a mask-aware Graph WaveNet predictor on METR-LA and PEMS-BAY, and with independently generated SUMO motorway episodes containing controlled physical bottlenecks and reporting failures. The evaluation separates input degradation from feedback degradation and measures predictive accuracy, interval coverage, width, interval score, recovery time, and computational cost. The study tests whether feedback-aware calibration provides a better empirical reliability–efficiency trade-off than fixed, online, and relational calibration alternatives. Experimental findings and their uncertainty will be inserted after reproducible evaluation.

Rewrite this into a results-based abstract only after the experiments. Do not retain unverified success language.

### Section plan

1. Introduction: operational need, input-versus-feedback distinction, precise research question, contributions supported by results.
2. Related work: graph forecasting, missing observations, online/conformal calibration, delayed and censored feedback, simulation evidence.
3. Problem formulation: clocks, forecast horizons, masks, arrival process, observable information and evaluation truth.
4. Method: frozen backbone, residuals, weighted pooling, support, fallback, update semantics, complexity and validity boundaries.
5. Experimental protocol: provenance, splits, baselines, training, outages, SUMO, statistical analysis.
6. Results: primary endpoint first, coverage–width trade-offs, mechanism isolation, recovery, SUMO, ablations, efficiency.
7. Discussion: when the method works or fails; effect of label selection; deployment observability; external validity.
8. Conclusion: only the claims demonstrated, followed by narrowly defined future work.

### Claims that require explicit evidence

- “Novel”: completed nearest-work comparison, including unresolved 2026 sources.
- “Better calibrated”: measured coverage error with width and interval score, not coverage alone.
- “Faster recovery”: registered recovery definition and censored-event reporting.
- “Robust”: specific fault families, severity ranges, datasets and uncertainty intervals.
- “Real-time”: measured end-to-end latency on stated hardware; replay speed alone is insufficient.
- “Generalizable”: evidence limited to the tested cities, backbones and synthetic network unless expanded.
- “Guaranteed coverage”: a separate theorem with assumptions and proof; absent from the present proposal.

## 19. Definition of completion for Codex

The implementation is complete when a new user can obtain the documented data, reproduce the CPU smoke pipeline, regenerate the SUMO scenario, run the registered experiments, and trace every result to immutable forecasts and valid targets. The research claim is complete only when the required comparisons, robustness checks, and literature overlap checks support it.

If resources prevent paper-scale execution, deliver the complete runnable system plus the executed smoke evidence and an exact remaining-run manifest. State precisely what has and has not been executed. Preserve the selected PS-1 focus throughout: **reliable uncertainty estimation when the feedback needed for calibration is delayed or missing**.
