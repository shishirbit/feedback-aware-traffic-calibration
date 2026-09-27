"""Train the registered DCRNN quantile backbone for PS-1."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ps1.models.dcrnn_quantile import DCRNNQuantile, DCRNNQuantileConfig
from ps1.models.graph_wavenet_masked import masked_pinball_loss
from ps1.training import TrafficWindowDataset, _row_normalize
from ps1.utils.artifacts import sha256_file, write_json


def loader(dataset, batch_size, shuffle, seed):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


def write_progress(path: Path, payload: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--weight-decay", type=float, default=.0001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--feasibility-only", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    train = TrafficWindowDataset(args.prepared, "train", augment=True, seed=args.seed)
    tune = TrafficWindowDataset(args.prepared, "tune", augment=False, seed=args.seed)
    config = DCRNNQuantileConfig(num_nodes=train.values.shape[1])
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    supports = (torch.from_numpy(_row_normalize(train.adjacency)).to(device),
                torch.from_numpy(_row_normalize(train.adjacency.T)).to(device))
    model = DCRNNQuantile(config, supports).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    args.output.mkdir(parents=True, exist_ok=True)
    prepared_hash = sha256_file(args.prepared)
    state_path = args.output / "run_state.pt"
    best, best_state, stale, history, first_epoch = float("inf"), None, 0, [], 0
    if state_path.exists() and not args.feasibility_only:
        state = torch.load(state_path, map_location=device, weights_only=False)
        if state["prepared_sha256"] != prepared_hash or state["architecture"] != model.resolved_architecture():
            raise ValueError("resume state does not match data or architecture")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        best, best_state, stale, history, first_epoch = (state[k] for k in ("best", "best_state", "stale", "history", "epoch"))

    started = time.perf_counter()
    for epoch in range(first_epoch, 1 if args.feasibility_only else args.epochs):
        train.set_epoch(epoch); model.train(); losses = []
        for inputs, target, valid in loader(train, args.batch_size, True, args.seed + epoch):
            optimizer.zero_grad(set_to_none=True)
            inputs, target, valid = inputs.to(device), target.to(device), valid.to(device)
            loss = masked_pinball_loss(model(inputs), target, valid)
            if loss is None: continue
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            losses.append(float(loss.detach()))
            if args.feasibility_only: break
        if args.feasibility_only:
            result = {"status": "feasibility_only", "step_seconds": time.perf_counter() - started,
                      "loss": losses[0], "parameters": model.resolved_architecture()["parameters"],
                      "train_windows": len(train), "batch_size": args.batch_size, "device": str(device)}
            write_json(args.output / "feasibility.json", result); print(json.dumps(result)); return
        model.eval(); validation = []
        with torch.no_grad():
            for inputs, target, valid in loader(tune, args.batch_size, False, args.seed):
                inputs, target, valid = inputs.to(device), target.to(device), valid.to(device)
                loss = masked_pinball_loss(model(inputs), target, valid)
                if loss is not None: validation.append(float(loss))
        tune_loss = float(np.mean(validation)); history.append({"epoch": epoch + 1,
            "train_pinball": float(np.mean(losses)), "tune_pinball": tune_loss})
        if tune_loss < best:
            best, stale = tune_loss, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else: stale += 1
        state = {"epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "best": best, "best_state": best_state, "stale": stale, "history": history,
                 "prepared_sha256": prepared_hash, "architecture": model.resolved_architecture()}
        temporary = state_path.with_suffix(".tmp"); torch.save(state, temporary); temporary.replace(state_path)
        write_progress(args.output / "progress.json", {"status": "running", "epochs_completed": epoch + 1,
                       "best_tune_pinball": best, "last_tune_pinball": tune_loss,
                       "runtime_seconds_this_invocation": time.perf_counter() - started})
        if stale >= args.patience: break

    checkpoint = args.output / "checkpoint.pt"
    torch.save({"model": best_state, "architecture": model.resolved_architecture(), "seed": args.seed,
                "train_mean": train.mean, "train_std": train.std,
                "train_sensor_median": train.medians.tolist(), "prepared_sha256": prepared_hash}, checkpoint)
    summary = {"status": "trained", "seed": args.seed, "device": str(device), "epochs_completed": len(history),
               "best_tune_pinball": best, "runtime_seconds": time.perf_counter() - started,
               "history": history, "architecture": model.resolved_architecture(),
               "checkpoint_sha256": sha256_file(checkpoint), "prepared_sha256": prepared_hash}
    write_json(args.output / "training_summary.json", summary); state_path.unlink(missing_ok=True)
    print(json.dumps({k: summary[k] for k in ("status", "epochs_completed", "best_tune_pinball", "runtime_seconds")}))


if __name__ == "__main__": main()
