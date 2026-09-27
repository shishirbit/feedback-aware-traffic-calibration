# FAR-GW core extension and matched independent controls

Positive differences favor FAR-Cal. The rule requires both 95% lower confidence bounds above zero and coverage at least 0.88. Independent-control results test the method within independently assigned outages; they are not correlated-minus-independent effects.

| Stage | Dataset | Scenario | Coverage | Score | Rolling difference (95% CI) | ACI difference (95% CI) | Rule |
|---|---|---|---:|---:|---|---|---|
| core | metr_la | C1 | 0.8985 | 20.3374 | 0.7965 [0.4351, 1.2371] | 0.7757 [0.4179, 1.2074] | Met |
| core | metr_la | C2 | 0.8985 | 20.3374 | 0.7929 [0.4291, 1.2327] | 0.7716 [0.4137, 1.2027] | Met |
| core | metr_la | C5 | 0.8893 | 22.1392 | 0.2279 [0.1420, 0.2896] | 0.2049 [0.1232, 0.2680] | Met |
| independent_control | metr_la | C3 | 0.9025 | 19.3762 | 0.6808 [0.5038, 0.8632] | 0.6927 [0.5183, 0.8722] | Met |
| independent_control | metr_la | C4 | 0.9028 | 19.3799 | 0.6858 [0.5072, 0.8693] | 0.6985 [0.5238, 0.8779] | Met |
| core | pems_bay | C1 | 0.9006 | 10.3272 | -0.0793 [-0.2259, 0.1040] | -0.0940 [-0.2481, 0.0950] | Not met |
| core | pems_bay | C2 | 0.9006 | 10.3272 | -0.0789 [-0.2262, 0.1044] | -0.0937 [-0.2485, 0.0952] | Not met |
| core | pems_bay | C5 | 0.8977 | 11.0428 | -0.0310 [-0.0467, -0.0105] | -0.0362 [-0.0512, -0.0156] | Not met |
| independent_control | pems_bay | C3 | 0.9041 | 10.2345 | 0.0406 [-0.0226, 0.1397] | 0.0402 [-0.0216, 0.1400] | Not met |
| independent_control | pems_bay | C4 | 0.9055 | 10.2372 | 0.0373 [-0.0196, 0.1227] | 0.0373 [-0.0181, 0.1240] | Not met |
