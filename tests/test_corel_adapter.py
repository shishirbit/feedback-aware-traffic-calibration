import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "scripts" / "run_corel_ps1.py"
    spec = importlib.util.spec_from_file_location("run_corel_ps1", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_residuals_enter_history_only_at_target_time():
    module = _module()
    q50 = np.zeros((6, 2, 1), dtype=np.float32)
    truth = np.zeros_like(q50)
    truth[:, 0, 0] = np.arange(10, 16)
    truth[:, 1, 0] = np.arange(20, 26)
    arrays = {
        "q50_mph": q50,
        "truth_mph": truth,
        "original_valid": np.ones_like(q50, dtype=bool),
    }

    released, valid = module.causal_residual_stream(arrays)

    assert not valid[0].any()
    assert released[1, 0, 0] == 10  # issue 0, horizon 1, target bin 1
    assert released[1, 0, 1] == 0
    assert released[2, 0, 1] == 20  # issue 0, horizon 2, target bin 2
    assert released[3, 0, 1] == 21  # issue 1, horizon 2, target bin 3
