"""Streaming, split-safe training for the registered FAR-GW backbone."""
from pathlib import Path
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ps1.data.windows import origins, split_bounds
from ps1.config import config_hash
from ps1.models.graph_wavenet_masked import GraphWaveNetConfig, MaskedGraphWaveNet, masked_pinball_loss
from ps1.utils.artifacts import sha256_file, write_json


def _row_normalize(adjacency):
    adjacency = np.asarray(adjacency, dtype=np.float32)
    return adjacency / np.maximum(adjacency.sum(axis=1, keepdims=True), 1.)


def _causal_state(values, valid, medians, episode_index=None):
    """Return original-stream causal fill and age without future interpolation."""
    filled = np.empty_like(values, dtype=np.float32)
    age = np.empty(values.shape, dtype=np.uint16)
    last_value = np.asarray(medians, dtype=np.float32).copy()
    last_bin = np.full(values.shape[1], -1, dtype=np.int64)
    episode_start = 0
    for now in range(len(values)):
        if episode_index is not None and now and episode_index[now]!=episode_index[now-1]:
            last_value = np.asarray(medians, dtype=np.float32).copy()
            last_bin.fill(-1); episode_start=now
        observed = valid[now] & np.isfinite(values[now])
        last_value[observed] = values[now, observed]
        last_bin[observed] = now
        filled[now] = last_value
        age[now] = np.minimum(np.where(last_bin >= 0, now - last_bin, now-episode_start+1), 65535)
    return filled, age


def _calendar_features(timestamp_ns):
    stamp = np.asarray(timestamp_ns).astype("datetime64[ns]")
    day = stamp.astype("datetime64[D]")
    minute = (stamp - day).astype("timedelta64[m]").astype(np.int64)
    # 1970-01-01 was Thursday; Monday is zero.
    weekday = (day.astype(np.int64) + 3) % 7
    tod = 2 * np.pi * minute / 1440.
    dow = 2 * np.pi * weekday / 7.
    return np.stack((np.sin(tod), np.cos(tod), np.sin(dow), np.cos(dow)), axis=1).astype(np.float32)


def _connected_nodes(adjacency, start, count):
    selected = [int(start)]
    seen = {int(start)}
    cursor = 0
    undirected = (adjacency > 0) | (adjacency.T > 0)
    while cursor < len(selected) and len(selected) < count:
        node = selected[cursor]; cursor += 1
        for neighbor in np.flatnonzero(undirected[node]):
            neighbor = int(neighbor)
            if neighbor not in seen:
                seen.add(neighbor); selected.append(neighbor)
                if len(selected) == count:
                    break
    if len(selected) < count:
        selected.extend([node for node in range(len(adjacency)) if node not in seen][:count-len(selected)])
    return np.asarray(selected, dtype=np.int64)


class TrafficWindowDataset(Dataset):
    """Build windows lazily so real datasets are not expanded into multi-GB tensors."""
    def __init__(self, prepared_path, partition, augment=False, seed=0, history=12, horizon=12):
        data = np.load(prepared_path)
        self.values = data["values_mph"].astype(np.float32)
        self.valid = data["original_valid"].astype(bool) & np.isfinite(self.values)
        self.timestamps = data["timestamp_ns"]
        self.adjacency = data["adjacency"].astype(np.float32)
        self.mean = float(data["train_mean"]); self.std = float(data["train_std"])
        self.medians = data["train_sensor_median"].astype(np.float32)
        self.episode_index = data["episode_index"] if "episode_index" in data else None
        self.filled, self.age = _causal_state(self.values, self.valid, self.medians,self.episode_index)
        self.calendar = _calendar_features(self.timestamps)
        if self.episode_index is not None:
            names={"train":0,"tune":1,"calibration":2,"test":3,"replication":4,"confirmation":4,"asym_validation":4}
            if partition not in names: raise ValueError("unknown partition")
            codes=data["partition_code"]; selected=[]; warmup=int(data["warmup_bins"]) if "warmup_bins" in data else 0
            for episode in np.unique(self.episode_index[codes==names[partition]]):
                episode_rows=np.flatnonzero(self.episode_index==episode)
                start,end=int(episode_rows[0]),int(episode_rows[-1])+1
                selected.extend(range(start+warmup+history-1,end-horizon))
            self.issues=np.asarray(selected,dtype=np.int64)
        else:
            bounds = split_bounds(len(self.values))
            parts = {"train": (bounds[0], bounds[1]), "tune": (bounds[1], bounds[2]),
                     "calibration": (bounds[2], bounds[3]), "test": (bounds[3], bounds[4])}
            if partition not in parts: raise ValueError("unknown partition")
            start, end = parts[partition]
            self.issues = np.asarray(list(origins(start, end, history, horizon)), dtype=np.int64)
        self.history = history; self.horizon = horizon
        self.augment = bool(augment); self.seed = int(seed); self.epoch = 0

    def __len__(self):
        return len(self.issues)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _augmented_history(self, issue, index):
        first = issue - self.history + 1
        mask = self.valid[first:issue+1].copy()
        if not self.augment:
            return self.filled[first:issue+1], mask, self.age[first:issue+1]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(index)]))
        mode = int(rng.integers(3))
        if mode == 1:
            mask &= rng.random(mask.shape) >= rng.uniform(.1, .3)
        elif mode == 2:
            length = int(rng.integers(3, self.history + 1))
            temporal_start = int(rng.integers(0, self.history - length + 1))
            count = max(1, int(np.ceil(rng.uniform(.05, .20) * mask.shape[1])))
            nodes = _connected_nodes(self.adjacency, rng.integers(mask.shape[1]), count)
            mask[temporal_start:temporal_start+length, nodes] = False
        last_value = self.filled[first-1].copy() if first else self.medians.copy()
        last_seen = np.full(mask.shape[1], first - 1, dtype=np.int64)
        if first:
            last_seen = first - 1 - self.age[first-1].astype(np.int64)
        filled = np.empty_like(self.filled[first:issue+1]); age = np.empty_like(mask, dtype=np.uint16)
        for offset, now in enumerate(range(first, issue + 1)):
            observed = mask[offset]
            last_value[observed] = self.values[now, observed]; last_seen[observed] = now
            filled[offset] = last_value
            age[offset] = np.minimum(np.where(last_seen >= 0, now-last_seen, now+1), 65535)
        return filled, mask, age

    def __getitem__(self, index):
        issue = int(self.issues[index]); first = issue-self.history+1
        filled, mask, age = self._augmented_history(issue, index)
        calendar = np.broadcast_to(self.calendar[first:issue+1, None, :], (*mask.shape, 4))
        features = np.concatenate((((filled-self.mean)/self.std)[..., None], mask[..., None].astype(np.float32),
                                   np.minimum(age, 288)[..., None].astype(np.float32)/288., calendar), axis=-1)
        rows = slice(issue+1, issue+self.horizon+1)
        target = (self.values[rows]-self.mean)/self.std
        target_valid = self.valid[rows]
        target = np.where(target_valid, target, 0.).astype(np.float32)
        return torch.from_numpy(features), torch.from_numpy(target), torch.from_numpy(target_valid)


def build_model(dataset):
    support = _row_normalize(dataset.adjacency)
    supports = (torch.from_numpy(support), torch.from_numpy(_row_normalize(dataset.adjacency.T)))
    config = GraphWaveNetConfig(num_nodes=dataset.values.shape[1], input_features=7)
    return MaskedGraphWaveNet(config, supports)


def train_registered(prepared_path, output_dir, config, seed, device="cpu"):
    """Train with early stopping and immutable, hash-addressable evidence files."""
    torch.manual_seed(seed); np.random.seed(seed)
    train = TrafficWindowDataset(prepared_path, "train", augment=True, seed=seed,
                                 history=config["history_bins"], horizon=config["horizon_bins"])
    tune = TrafficWindowDataset(prepared_path, "tune", augment=False, seed=seed,
                                history=config["history_bins"], horizon=config["horizon_bins"])
    model = build_model(train).to(device)
    settings = config["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    tune_loader = DataLoader(tune, batch_size=settings["batch_size"], shuffle=False, num_workers=0)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.pt"; summary_path = output / "training_summary.json"; state_path = output / "run_state.pt"
    prepared_hash = sha256_file(prepared_path); resolved_hash = config_hash(config)
    manifest_path = output / "run_manifest.json"
    manifest = {"status": "running", "seed": seed, "device": str(device),
                "prepared_sha256": prepared_hash, "resolved_config_sha256": resolved_hash,
                "resolved_config": config}
    if summary_path.exists() and checkpoint.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("prepared_sha256") != prepared_hash or existing.get("resolved_config_sha256") != resolved_hash:
            raise FileExistsError(f"existing training artifact has different inputs: {output}")
        return existing
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("prepared_sha256") != prepared_hash or existing.get("resolved_config_sha256") != resolved_hash:
            raise FileExistsError(f"incomplete training artifact has different inputs: {output}")
    else:
        write_json(manifest_path, manifest)
    best = float("inf"); best_state = None; stale = 0; history = []; skipped = 0; first_epoch = 0
    if state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        best, best_state, stale = state["best"], state["best_state"], state["stale"]
        history, skipped, first_epoch = state["history"], state["skipped"], state["epoch"]
    started = time.perf_counter()
    for epoch in range(first_epoch, settings["epochs"]):
        train.set_epoch(epoch); model.train(); losses = []
        epoch_seed = int(np.random.SeedSequence([seed, epoch]).generate_state(1)[0])
        train_loader = DataLoader(train, batch_size=settings["batch_size"], shuffle=True,
                                  generator=torch.Generator().manual_seed(epoch_seed), num_workers=0)
        for inputs, target, valid in train_loader:
            inputs, target, valid = inputs.to(device), target.to(device), valid.to(device)
            optimizer.zero_grad(); loss = masked_pinball_loss(model(inputs), target, valid)
            if loss is None:
                skipped += 1; continue
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        model.eval(); validation = []
        with torch.no_grad():
            for inputs, target, valid in tune_loader:
                loss = masked_pinball_loss(model(inputs.to(device)), target.to(device), valid.to(device))
                if loss is not None: validation.append(float(loss.cpu()))
        tune_loss = float(np.mean(validation))
        history.append({"epoch": epoch+1, "train_pinball": float(np.mean(losses)), "tune_pinball": tune_loss})
        if tune_loss < best:
            best = tune_loss; stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        state = {"epoch": epoch+1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "best": best, "best_state": best_state, "stale": stale, "history": history, "skipped": skipped}
        temporary = state_path.with_suffix(".tmp")
        torch.save(state, temporary); temporary.replace(state_path)
        if stale >= settings["patience"]: break
    torch.save({"model": best_state, "architecture": model.resolved_architecture(), "seed": seed,
                "train_mean": train.mean, "train_std": train.std, "train_sensor_median": train.medians.tolist(),
                "prepared_sha256": prepared_hash, "resolved_config_sha256": resolved_hash}, checkpoint)
    summary = {"status": "trained", "seed": seed, "device": str(device), "epochs_completed": len(history),
               "best_tune_pinball": best, "skipped_empty_batches": skipped,
               "runtime_seconds": time.perf_counter()-started, "history": history,
               "architecture": model.resolved_architecture(), "checkpoint_sha256": sha256_file(checkpoint),
               "prepared_sha256": prepared_hash, "resolved_config_sha256": resolved_hash}
    write_json(summary_path, summary)
    manifest["status"] = "completed"; manifest["checkpoint_sha256"] = summary["checkpoint_sha256"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    state_path.unlink(missing_ok=True)
    return summary


def feasibility_batch(prepared_path, config, seed=11):
    """Measure one CPU optimizer step without claiming a trained model."""
    torch.manual_seed(seed)
    dataset = TrafficWindowDataset(prepared_path, "train", augment=True, seed=seed,
                                   history=config["history_bins"], horizon=config["horizon_bins"])
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=0)
    model = build_model(dataset); optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    started = time.perf_counter(); inputs, target, valid = next(iter(loader))
    optimizer.zero_grad(); loss = masked_pinball_loss(model(inputs), target, valid)
    loss.backward(); optimizer.step()
    return {"status": "feasibility_only", "batch_size": len(inputs), "loss": float(loss.detach()),
            "step_seconds": time.perf_counter()-started, "train_windows": len(dataset),
            "parameters": model.resolved_architecture()["parameters"], "device": "cpu"}
