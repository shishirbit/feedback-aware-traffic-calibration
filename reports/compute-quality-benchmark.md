# Matched compute–quality benchmark

This is a descriptive FAR-Cal archive-cap sensitivity study, not a cross-method runtime benchmark. Identical saved forecast/fault inputs are replayed at caps 64/128/256, using C3 duration03 and fixed model/fault seeds 11/101. Three sequential repetitions use rotated cap order. Runtime includes CPU calibration, loading and output serialization, but excludes training and prediction. Filesystem cache is not flushed.

| Dataset | Cap | Median seconds (range) | Interval score | Coverage |
|---|---:|---:|---:|---:|
| metr_la | 64 | 19.28 (19.25–19.89) | 21.2222 | 0.9114 |
| metr_la | 128 | 19.69 (18.71–20.04) | 20.8557 | 0.9133 |
| metr_la | 256 | 19.08 (18.80–19.99) | 20.8365 | 0.9122 |
| pems_bay | 64 | 97.95 (96.88–99.69) | 12.1429 | 0.8980 |
| pems_bay | 128 | 93.76 (91.85–102.98) | 12.0972 | 0.8971 |
| pems_bay | 256 | 96.25 (91.57–105.95) | 12.0705 | 0.8971 |

These quality metrics are equal-horizon averages of pooled valid sensor-target metrics on this single run; the full stress summaries instead average origin-level scores and bootstrap over model/fault seeds. They are distinct estimands and should not be substituted for one another. The cap256 benchmark is checked against the archived same-run output. Timing ranges are observed repetition ranges, not confidence intervals. Other user workloads may share the host; median process CPU seconds are retained in the CSV alongside wall time. Hardware details and the protocol are saved with the benchmark artifacts.

Smaller archives had higher interval scores in both fixed runs. Timing ranges overlap across all caps, so these measurements do not establish a reliable speed advantage for smaller caps. The locked method is not retuned from these test outcomes.
