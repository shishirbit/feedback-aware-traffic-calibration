import importlib.util
from pathlib import Path

import numpy as np


def _module():
    scripts = Path(__file__).parents[1] / "scripts"
    spec = importlib.util.spec_from_file_location("evaluate_corel_ps1", scripts / "evaluate_corel_ps1.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    import sys
    sys.path.insert(0, str(scripts))
    spec.loader.exec_module(module)
    return module


def test_residuals_are_indexed_by_target_not_issue_time():
    module = _module()
    arrays = {
        "issue_bin": np.array([9, 10]),
        "q50_mph": np.zeros((2, 2, 1), dtype=np.float32),
        "truth_mph": np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32),
        "original_valid": np.ones((2, 2, 1), dtype=bool),
    }
    values, valid = module.residuals_by_target(arrays, start=10, length=3)
    assert values[0, 0, 0] == 1.0  # issue 9 + horizon 1
    assert values[1, 0, 0] == 3.0  # issue 10 + horizon 1
    assert values[1, 0, 1] == 2.0  # issue 9 + horizon 2
    assert valid[2, 0, 1]
