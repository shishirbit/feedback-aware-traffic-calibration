# Completed FAR-GW real-data stress results

All 1,488 stages and 48 summaries are complete. Each cell uses model seeds 11/22/33 and fault seeds 101/202/303. The rule requires both rolling-minus-FAR-Cal and ACI-minus-FAR-Cal 95% lower bounds above zero and coverage at least 0.88.

| Dataset | Scenario | Variant | Coverage | Interval score | Rolling difference (95% CI) | ACI difference (95% CI) | Rule |
|---|---|---|---:|---:|---|---|---|
| metr_la | C3 | duration03 | 0.9042 | 17.8986 | 0.5326 [0.0933, 1.0937] | 0.5260 [0.1170, 1.0408] | Met |
| metr_la | C3 | duration12 | 0.8984 | 21.5178 | 0.6658 [0.3285, 1.0509] | 0.7133 [0.3792, 1.1026] | Met |
| metr_la | C3 | duration24 | 0.8956 | 23.6177 | 0.5900 [0.2385, 1.0053] | 0.6295 [0.2811, 1.0276] | Met |
| metr_la | C3 | fraction05 | 0.8908 | 22.0701 | 0.6792 [0.1733, 1.2938] | 0.7576 [0.2655, 1.3459] | Met |
| metr_la | C3 | fraction20 | 0.9024 | 20.2133 | 0.6631 [0.3132, 1.0534] | 0.7036 [0.3909, 1.0178] | Met |
| metr_la | C3 | extra03 | 0.8985 | 20.3374 | 0.7930 [0.4293, 1.2323] | 0.7728 [0.4154, 1.2046] | Met |
| metr_la | C3 | extra12 | 0.9038 | 20.3522 | 0.7650 [0.3395, 1.2598] | 0.7488 [0.3207, 1.2360] | Met |
| metr_la | C3 | loss025 | 0.8997 | 20.3500 | 0.7855 [0.4065, 1.2445] | 0.7643 [0.3919, 1.2119] | Met |
| metr_la | C3 | loss050 | 0.9009 | 20.3710 | 0.7584 [0.3778, 1.2111] | 0.7398 [0.3696, 1.1763] | Met |
| metr_la | C3 | loss100 | 0.9036 | 20.3636 | 0.7535 [0.3346, 1.2412] | 0.7352 [0.3155, 1.2162] | Met |
| metr_la | C3 | cold_calibration | 0.8985 | 20.3376 | 0.7924 [0.4288, 1.2320] | 0.7713 [0.4132, 1.2021] | Met |
| metr_la | C3 | mnar_low_speed | 0.8985 | 20.3381 | 0.7991 [0.4426, 1.2343] | 0.7785 [0.4254, 1.1977] | Met |
| metr_la | C4 | duration03 | 0.8998 | 21.1438 | 0.4867 [0.1996, 0.7804] | 0.5153 [0.2239, 0.8211] | Met |
| metr_la | C4 | duration12 | 0.8966 | 20.8624 | 0.7316 [0.2705, 1.1539] | 0.8227 [0.4169, 1.2363] | Met |
| metr_la | C4 | duration24 | 0.9017 | 22.4503 | 0.7397 [0.4133, 1.1250] | 0.7565 [0.4103, 1.1715] | Met |
| metr_la | C4 | fraction05 | 0.9068 | 20.5788 | 0.8569 [0.4497, 1.4059] | 0.8905 [0.5077, 1.4077] | Met |
| metr_la | C4 | fraction20 | 0.9072 | 19.6255 | 0.6870 [0.3857, 1.0085] | 0.6969 [0.4110, 0.9740] | Met |
| metr_la | C4 | extra03 | 0.8959 | 23.4334 | 0.9499 [0.5323, 1.4250] | 0.9482 [0.5631, 1.3785] | Met |
| metr_la | C4 | extra12 | 0.8979 | 23.5121 | 0.9263 [0.4852, 1.4197] | 0.9276 [0.5229, 1.3834] | Met |
| metr_la | C4 | loss000 | 0.8939 | 23.4160 | 0.9540 [0.5547, 1.4189] | 0.9548 [0.5803, 1.3619] | Met |
| metr_la | C4 | loss025 | 0.8948 | 23.4088 | 0.9656 [0.5616, 1.4356] | 0.9680 [0.5885, 1.3858] | Met |
| metr_la | C4 | loss100 | 0.8980 | 23.5107 | 0.9282 [0.4892, 1.4230] | 0.9288 [0.5195, 1.3863] | Met |
| metr_la | C4 | cold_calibration | 0.8959 | 23.4334 | 0.9506 [0.5342, 1.4266] | 0.9507 [0.5647, 1.3810] | Met |
| metr_la | C4 | mnar_low_speed | 0.8959 | 23.4295 | 0.9602 [0.5479, 1.4310] | 0.9579 [0.5791, 1.3807] | Met |
| pems_bay | C3 | duration03 | 0.8984 | 10.5658 | -0.0166 [-0.1719, 0.1532] | -0.0257 [-0.1845, 0.1403] | Not met |
| pems_bay | C3 | duration12 | 0.8981 | 11.1271 | -0.1602 [-0.2565, -0.0615] | -0.1865 [-0.2840, -0.0906] | Not met |
| pems_bay | C3 | duration24 | 0.9016 | 12.8819 | -0.3748 [-0.4997, -0.2491] | -0.3979 [-0.5211, -0.2700] | Not met |
| pems_bay | C3 | fraction05 | 0.9018 | 10.8613 | 0.2330 [0.0046, 0.4045] | 0.2256 [-0.0029, 0.4011] | Not met |
| pems_bay | C3 | fraction20 | 0.9003 | 10.2460 | 0.0554 [-0.0572, 0.1827] | 0.0453 [-0.0674, 0.1716] | Not met |
| pems_bay | C3 | extra03 | 0.9094 | 10.4525 | -0.2008 [-0.3438, -0.0539] | -0.2140 [-0.3675, -0.0623] | Not met |
| pems_bay | C3 | extra12 | 0.9101 | 10.5227 | -0.2577 [-0.3986, -0.1201] | -0.2672 [-0.4170, -0.1239] | Not met |
| pems_bay | C3 | loss025 | 0.9062 | 10.4051 | -0.1561 [-0.3088, 0.0052] | -0.1709 [-0.3312, -0.0055] | Not met |
| pems_bay | C3 | loss050 | 0.9087 | 10.4387 | -0.1834 [-0.3257, -0.0338] | -0.1984 [-0.3482, -0.0457] | Not met |
| pems_bay | C3 | loss100 | 0.9101 | 10.5228 | -0.2578 [-0.3997, -0.1202] | -0.2672 [-0.4172, -0.1246] | Not met |
| pems_bay | C3 | cold_calibration | 0.9006 | 10.3273 | -0.0791 [-0.2268, 0.1042] | -0.0940 [-0.2486, 0.0943] | Not met |
| pems_bay | C3 | mnar_low_speed | 0.9013 | 10.3682 | -0.1271 [-0.2732, 0.0189] | -0.1408 [-0.2932, 0.0062] | Not met |
| pems_bay | C4 | duration03 | 0.9067 | 10.5563 | -0.1849 [-0.3520, -0.0357] | -0.2059 [-0.3724, -0.0535] | Not met |
| pems_bay | C4 | duration12 | 0.9018 | 11.0732 | -0.2155 [-0.3308, -0.1132] | -0.2223 [-0.3415, -0.1195] | Not met |
| pems_bay | C4 | duration24 | 0.9017 | 12.8711 | -0.3859 [-0.5208, -0.2718] | -0.4153 [-0.5539, -0.3035] | Not met |
| pems_bay | C4 | fraction05 | 0.9063 | 10.4837 | -0.0826 [-0.2435, 0.0929] | -0.0778 [-0.2401, 0.0993] | Not met |
| pems_bay | C4 | fraction20 | 0.9102 | 10.7717 | -0.2063 [-0.3162, -0.1008] | -0.2024 [-0.3127, -0.0969] | Not met |
| pems_bay | C4 | extra03 | 0.9087 | 10.5993 | -0.2796 [-0.4573, -0.1094] | -0.3048 [-0.5070, -0.1185] | Not met |
| pems_bay | C4 | extra12 | 0.9098 | 10.6075 | -0.2952 [-0.4735, -0.1211] | -0.3188 [-0.5162, -0.1260] | Not met |
| pems_bay | C4 | loss000 | 0.8990 | 10.3254 | -0.0425 [-0.1766, 0.1085] | -0.0704 [-0.2124, 0.0817] | Not met |
| pems_bay | C4 | loss025 | 0.9029 | 10.4518 | -0.1649 [-0.3216, -0.0121] | -0.1919 [-0.3702, -0.0230] | Not met |
| pems_bay | C4 | loss100 | 0.9098 | 10.6076 | -0.2954 [-0.4740, -0.1210] | -0.3192 [-0.5166, -0.1269] | Not met |
| pems_bay | C4 | cold_calibration | 0.9066 | 10.5845 | -0.2849 [-0.4737, -0.1055] | -0.3106 [-0.5213, -0.1195] | Not met |
| pems_bay | C4 | mnar_low_speed | 0.9073 | 10.5579 | -0.2537 [-0.4228, -0.0864] | -0.2797 [-0.4715, -0.0941] | Not met |

METR-LA meets the registered rule in all 24 stress cells; PEMS-BAY meets it in none of 24. This pattern includes duration, affected fraction, extra backlog delay, permanent loss, cold calibration and low-speed MNAR. Coverage remains above the registered floor in all cells. Coverage alone does not establish interval-score superiority. Duration scores use condition-specific focus windows and must not be interpreted as paired causal effects on a fixed set of origins. MNAR truth is available only to the fault generator/evaluator; these tests do not establish identifiable MNAR correction. No universal superiority claim is supported.

The core C1/C2/C5 matrix and matched-rate independent controls show the same dataset-specific rule pattern. The matched control tests superiority within independently assigned outages, not superiority of correlated outages over independent outages.

Provenance audit passed: 48 rule recomputations, 1,296 referenced manifests, and 26784 chunk hashes. Source hashes are in real-stress-provenance-audit.json.
