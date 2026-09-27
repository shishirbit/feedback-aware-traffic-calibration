# Literature evidence log

Checked through 2026-09-16. The matrix records only features supported by accessible
primary sources. "Not found" means a searched full text did not describe the
feature; it is narrower than a universal absence claim.

## Multi-step ACI

The official PMLR paper was downloaded and inspected. Equation (8) updates one
horizon-specific control input as `epsilon_next = epsilon_current + gamma *
(target_epsilon - miss)`. A horizon remains fixed until its first relevant
outcome arrives. The paper's delay is the ordinary forecast-horizon delay in a
fully observed online sequence. It does not evaluate reporting outages,
permanently lost outcomes, or post-outage recovery.

The project adapter batches all sensor outcomes released at the same replay time
and takes one mean-miss update per horizon and level. Delayed feedback triggers
no update before release; lost feedback triggers none. This spatial batching and
outcome suppression differ from the theorem's protocol, so the implementation
does not inherit the paper's finite-sample guarantee.

Primary source: https://proceedings.mlr.press/v230/hallberg-szabadvary24a.html

## MAS-Mamba

The Wiley full text was inspected. Its predictor uses missingness-aware mask and
age information. Its uncertainty stage uses fixed, held-out split-conformal
calibration scores. Searches of the full text found no reporting-feedback delay,
permanent feedback loss, or outage-recovery evaluation. This supports overlap on
missing-input prediction and fixed calibration, while leaving the project's
feedback-channel benchmark as a distinct evaluation target.

Primary source: https://ietresearch.onlinelibrary.wiley.com/doi/pdfdirect/10.1049/itr2.70304

## Post-Training ACP for Incomplete Time Series

The indexed OpenReview PDF and its algorithms were inspected. The method handles
incomplete predictor inputs through imputation and missing-pattern-conditioned
calibration/reweighting, then adapts its radius after the true test label is
observed. Searches of the reviewed text found no separately modeled delayed or
permanently lost outcome-feedback channel and no outage-recovery protocol. This
is substantial overlap on adaptive calibration under incomplete inputs, but it
does not remove the distinct feedback-channel benchmark contribution.

Primary source: https://openreview.net/forum?id=KMBU4wx79B

## Adaptive CP for spatio-temporal hypergraph neural networks

The indexed publisher article was inspected. CSTH combines a spatiotemporal
hypergraph predictor with topology-adaptive scaling based on hyperedge density
and historical volatility, using blocked-time, one-time offline calibration on
METR-LA and PEMS-BAY. Searches of the reviewed article found no separately
delayed or permanently lost calibration-feedback channel and no outage-recovery
evaluation. It overlaps strongly on relational traffic uncertainty calibration,
while the feedback-arrival and recovery protocol remains distinct.

Primary source: https://www.sciencedirect.com/science/article/pii/S1568494626017047
