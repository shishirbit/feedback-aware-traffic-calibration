# Provenance starting points

Checked through 2026-09-22:

- https://github.com/liyaguang/DCRNN/tree/602afd9d767d3aa1c9b3eac51710d6aeee12c227 — official MIT-licensed DCRNN source audited for the fourth-backbone adapter. The supplied benchmark split generator is not the PS-1 60/10/10/20 protocol and is not reused.
- https://github.com/nnzhan/Graph-WaveNet — author repository for backbone source
- https://github.com/GestaltCogTeam/STID/tree/e8b313bc591bdd0101a1619962c9b503e75127c0 — official Apache-2.0 STID source audited for the point-backbone adapter
  attribution, pinned at commit `6b162e80c59a1d494809252eca055cff93dc66b1`.
  Its MIT license was inspected. The local adaptation records that commit in every
  resolved architecture; baseline fidelity comparison is still pending.

No downloaded baseline scores have been used as experiment results. Focused
primary-source checks of the nearest identified 2026 calibration works found
substantial overlap on incomplete inputs and relational calibration, but did not
find the project's separately delayed/lost feedback channel or recovery protocol.
The novelty claim remains limited to this search scope.

## Data artifacts acquired 2026-09-09

- METR-LA HDF5: provenance is DCRNN; retrieved from the public GCGRNN GitHub
  mirror because the original Drive folder rejected automated access. SHA-256
  `64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05`.
- PEMS-BAY HDF5: provenance is DCRNN; retrieved through public Google Drive file
  `1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq`. SHA-256
  `65d69fb0a2323dba9867179eb7af47c8b814186bc459ff0a4937d21614153c8f`.
- DCRNN graph artifacts were downloaded byte-for-byte from pinned commit
  `602afd9d767d3aa1c9b3eac51710d6aeee12c227`; ordinary Windows Git checkout
  converted their protocol-0 line endings and was rejected. Correct graph hashes
  are `a35687c6...f61b3` (METR-LA) and `116275f5...97370` (PEMS-BAY).

The DCRNN code is MIT licensed. Neither its README nor the inspected mirror
metadata states a separate dataset redistribution license. The repository keeps
large data ignored and records this unresolved rights point before redistribution.
The official Caltrans PeMS terms describe site information as generally public
domain while reserving third-party rights. The inspected Zenodo DCRNN benchmark
records are openly downloadable but do not display an explicit license for the
exact packaged files. These sources do not establish blanket redistribution
permission for the local METR-LA and PEMS-BAY archives, so reproduction remains
fetch-and-hash rather than bundling raw data.
