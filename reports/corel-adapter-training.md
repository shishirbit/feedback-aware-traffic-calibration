# PS-1 CoRel adapter training status

The official CoRel architecture at commit
`4504c4edf76128bfe657762a41ec9eb043038ca2` was trained on the PS-1 METR-LA
frozen-prediction calibration residuals. The adapter aligns each residual
with its target-time release and never exposes it to an earlier input window.

| Seed | Status | Selected / completed epochs | Best validation pinball | CPU seconds |
|---:|---|---:|---:|---:|
| 11 | trained; evaluation complete | 5 / 20 | 0.132512 | 2,643.5 |
| 22 | trained; evaluation complete | 10 / 25 | 0.130697 | 3,274.9 |
| 33 | trained; evaluation complete | 20 / 35 | 0.132488 | 4,589.1 |

Each run uses 3,063 training and 341 validation examples, 207 sensors, 12
output horizons, and 149,989 trainable parameters.

| PEMS-BAY seed | Status | Selected / completed epochs | Best validation pinball | CPU seconds |
|---:|---|---:|---:|---:|
| 11 | trained; evaluation complete | 17 / 32 | 0.226358 | 9,793.2 cumulative |
| 22 | trained; evaluation complete | 19 / 34 | 0.225812 | 10,230.7 |
| 33 | trained; evaluation complete | 18 / 33 | 0.225816 | 9,894.9 |

The PEMS-BAY run uses 4,671 training and 519 validation examples, 325 sensors,
12 output horizons, and 214,653 parameters. It resumed from epoch 30 after the
user-requested system shutdown; the cumulative runtime combines both invocations.

Artifact hashes:

- `model.pt`: `b7a43c15a7d63c7d73053f5036c49d9600daf4c2770b526be47f8950e8c8f815`
- `summary.json`: `6f3249be5d70e1fc49715b3ffe90a6acf49fcd84a201e3bb4271f2275f0c2747`
- `last.pt`: `e24265de6b31cf4050e488ae23d8aa6e0e556cb52819a7ab1d1fd90baeceff03`
- calibration-cache manifest: `ec9f2b5f6c463848ad8f99628cabd22c627821defa22bbf060c1512e7fa18432`

Seed 22:

- `model.pt`: `45807d53f3c8cf72d550835af85e47a40992cdd6088464e86d6a543bb7e7bc8e`
- `summary.json`: `f022b6595ecd9dfba27aca2dad00d6182a3e3d1a1c5522e7a533f00e6df47096`
- `last.pt`: `f97a8509d4685e95fa4f40fc80848a3840c04e87a94d72302a6f89130f39f9f7`
- calibration-cache manifest: `c1f53c8daa21ef8ec7f274edd3fc29deca6d4a5f50efb29bb4a05bc726218e68`

Seed 33:

- `model.pt`: `865310d9b19bb33e3bdccf11dcb1328c13951986013c4d8569747e67c1798af7`
- `summary.json`: `f712fea4bf80abd3935dd67c0ef60c729df729f8d2ed9247922c75b4b56b03fc`
- `last.pt`: `299618b36f7e2a052c75a8857fbc10a19f4f588b479f200ce132a162cfa1133f`
- calibration-cache manifest: `a45df4bc92db90fd6e76b61778023108c632d3dab0b76ba268866e41d991ff49`

PEMS-BAY seed 11:

- `model.pt`: `eebd00dde7276e79001631c39753f66c74d6849133d5ba23e7c90fa5f862d4f5`
- `summary.json`: `530ba7d5c6e2ca997a0ff920f16efd073b8611542902df4b522a6696ceac9c6f`
- `last.pt`: `5d35f850934b8614391c3643ebe77e12229bb22107df950f33f60fc316ed46fd`
- calibration-cache manifest: `7b81230fb29431ac35f46f23fcef473df04561c2f4a33c3f308f55e58115b7b7`

PEMS-BAY seed 22:

- `model.pt`: `2397c4377d468b55f8d8cdf0412dbec4f83c8e52a3e809cee314c85e2e1d5492`
- `summary.json`: `8a2f35a5464af80916dac89611971c7855f4875ef87ee0ca6ba8b6fd8e510881`
- `last.pt`: `21f08ddf7c3433b78c0eedb78f53955d405a840ea71d549754569968283e4ba3`
- calibration-cache manifest: `3291c482406a15b30e40a8a4684488d16b01647be03633de00a1d0f70ef6f4c5`

PEMS-BAY seed 33:

- `model.pt`: `0ed57ee8dbf5252617d52243f1e7b558a10b10cc233155a272d7c59a38d0e903`
- `summary.json`: `2a30feccda375c67203ef91bc442f9df0ef499759aee0b2ecae35041dfe50939`
- `last.pt`: `c17c77c5d1308189ec84afd827153605b8899aed76733eede3e192ac38764c06`
- calibration-cache manifest: `f1fcd7afb21a7fd594c9a4dc4984b4008abe4b10091081a6e1583941b8b569dc`

The checkpoint structure, source commit, seed, 12 scaling vectors, and 39
quantile levels were reloaded successfully. The repository has 70 passing tests.
No test-set CoRel score is claimed from this training result.
