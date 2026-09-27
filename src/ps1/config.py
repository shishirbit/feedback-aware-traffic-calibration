"""Strict layered YAML configuration with explicit, recursive overrides."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import yaml


def _merge(base, override, path=""):
    if not isinstance(base, dict) or not isinstance(override, dict):
        raise TypeError("configuration roots must be mappings")
    result = deepcopy(base)
    for key, value in override.items():
        dotted = f"{path}.{key}" if path else key
        if key not in base:
            # Dataset-only metadata is permitted as a complete, named section.
            if path == "" and key in {"dataset", "data"}:
                result[key] = deepcopy(value)
                continue
            raise ValueError(f"unknown configuration key: {dotted}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise TypeError(f"configuration section must remain a mapping: {dotted}")
            result[key] = _merge(base[key], value, dotted)
        else:
            if isinstance(value, dict):
                raise TypeError(f"configuration value cannot become a mapping: {dotted}")
            result[key] = deepcopy(value)
    return result


def _load(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def validate(config):
    fractions = config["split"]
    if abs(sum(fractions.values()) - 1.0) > 1e-9 or any(v <= 0 for v in fractions.values()):
        raise ValueError("split fractions must be positive and sum to one")
    if sorted(config["training"]["quantiles"]) != config["training"]["quantiles"]:
        raise ValueError("training quantiles must be ordered")
    if any(not 0 < value < 1 for value in config["interval_levels"]):
        raise ValueError("interval levels must be in (0,1)")
    if any(h < 1 or h > config["horizon_bins"] for h in config["report_horizons"]):
        raise ValueError("report horizon outside modeled horizons")
    if config["sumo"]["stations"] != 20 or config["sumo"]["through_lanes"] != 2:
        raise ValueError("registered SUMO topology requires 20 stations and two through lanes")
    if config["sumo"]["aggregation_seconds"] % config["sumo"]["step_seconds"]:
        raise ValueError("aggregation must be divisible by simulation step")
    return config


def resolve(base_path, *overrides):
    config = _load(base_path)
    for path in overrides:
        config = _merge(config, _load(path))
    return validate(config)


def canonical_bytes(config):
    return json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def config_hash(config):
    return hashlib.sha256(canonical_bytes(config)).hexdigest()
