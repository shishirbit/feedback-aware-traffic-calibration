# DCRNN backbone reproduction

## Source audit

- Paper: Li et al., *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting*, ICLR 2018.
- Official repository: `liyaguang/DCRNN`.
- Audited commit: `602afd9d767d3aa1c9b3eac51710d6aeee12c227`.
- License: MIT.

The PS-1 PyTorch adapter retains the encoder-decoder recurrent design, dual
random-walk diffusion supports, two diffusion steps, two 64-unit recurrent
layers, and autoregressive multihorizon decoding. It supplies the registered
seven causal input channels and replaces the point head with the common
noncrossing 0.05/0.50/0.95 quantile head used by the calibration audits.

## Registered adaptation

- History and horizon: 12 five-minute bins each.
- Dual forward/reverse row-normalized graph supports and diffusion order 2.
- Two encoder and decoder DCGRU layers with 64 hidden units.
- Adam, learning rate 0.001, weight decay 0.0001.
- Batch size 16 for METR-LA and 8 for PEMS-BAY.
- Maximum 100 epochs and patience 30 on the tuning split.
- Seeds 11, 22, and 33 for both datasets.

GPU feasibility succeeded in the existing Python 3.11 / CUDA PyTorch
environment. The 377,283-parameter model completed one training step in about
0.94 seconds on METR-LA at batch 16 and 0.56 seconds on PEMS-BAY at batch 8.
These timings validate execution only and are not research results.

## Completed audit

All six registered training runs and the 114-stage prediction-cache and
calibration comparison matrix completed. DCRNN met the registered superiority
rule on METR-LA C3/C4 and PEMS-BAY C3/C4. Combined with the other frozen
backbones, the overall matrix passes 12/16 cells. This strengthens transfer
evidence but does not establish universal superiority because FAR-GW and STID
do not pass the PEMS-BAY cells.
