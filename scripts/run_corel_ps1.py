"""Train the official CoRel architecture on PS-1 frozen prediction residuals.

Run this with the isolated environment recorded in requirements-corel.lock. The
official repository must be checked out at the pinned commit documented below.
This is a PS-1 multihorizon adaptation, not the paper's source-split experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


COREL_COMMIT = "4504c4edf76128bfe657762a41ec9eb043038ca2"
QUANTILES = np.round(np.arange(0.025, 1.0, 0.025), 3)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cache(cache_dir: Path) -> tuple[dict, dict[str, np.ndarray]]:
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    keys = ("issue_bin", "q50_mph", "truth_mph", "original_valid")
    parts: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    for chunk in manifest["chunks"]:
        path = cache_dir / chunk["file"]
        if sha256(path) != chunk["sha256"]:
            raise ValueError(f"hash mismatch: {path}")
        with np.load(path) as arrays:
            for key in keys:
                parts[key].append(arrays[key].copy())
    result = {key: np.concatenate(value) for key, value in parts.items()}
    if not np.all(np.diff(result["issue_bin"]) == 1):
        raise ValueError("CoRel adapter requires consecutive calibration origins")
    return manifest, result


def causal_residual_stream(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return residuals released at each issue bin and their validity.

    Entry [t, node, h-1] is the error of the horizon-h forecast issued h bins
    earlier, so no future target is exposed to an input window.
    """
    errors = arrays["truth_mph"] - arrays["q50_mph"]
    valid = arrays["original_valid"] & np.isfinite(errors)
    origins, horizons, sensors = errors.shape
    released = np.zeros((origins, sensors, horizons), dtype=np.float32)
    released_valid = np.zeros_like(released, dtype=bool)
    for h_idx in range(horizons):
        lag = h_idx + 1
        released[lag:, :, h_idx] = errors[:-lag, h_idx, :]
        released_valid[lag:, :, h_idx] = valid[:-lag, h_idx, :]
    return released, released_valid


def make_examples(
    arrays: dict[str, np.ndarray], window: int = 12
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    released, released_valid = causal_residual_stream(arrays)
    target = arrays["truth_mph"] - arrays["q50_mph"]
    target_valid = arrays["original_valid"] & np.isfinite(target)

    # Official CoRel has no missing-feedback channel. For clean calibration,
    # rare naturally missing entries are causally carried forward per feature.
    filled = released.copy()
    for t in range(1, len(filled)):
        missing = ~released_valid[t]
        filled[t][missing] = filled[t - 1][missing]

    rows = np.arange(window, len(filled), dtype=np.int64)
    x = np.stack([filled[t - window : t] for t in rows])
    y = np.transpose(target[rows], (0, 2, 1)).astype(np.float32)
    mask = np.transpose(target_valid[rows], (0, 2, 1))
    return x, y, mask, rows


def pinball(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # prediction: [quantile,batch,node,horizon]
    quantiles = torch.as_tensor(QUANTILES, dtype=prediction.dtype, device=prediction.device)
    error = target.unsqueeze(0) - prediction
    loss = torch.maximum(quantiles[:, None, None, None] * error,
                         (quantiles[:, None, None, None] - 1.0) * error)
    expanded_mask = mask.unsqueeze(0).expand_as(loss)
    return loss[expanded_mask].mean()


def build_model(source: Path, sensors: int, horizons: int):
    sys.path.insert(0, str(source))
    from lib.nn.decoder.multiquantile_readout import MultiQuantileDecoder
    from lib.nn.encoder_decoder_model import EncoderDecoderModel
    from lib.nn.encoders.corel_encoder import CoRelEncoder

    return EncoderDecoderModel(
        input_size=horizons,
        output_size=horizons,
        horizon=1,
        encoder_class=CoRelEncoder,
        decoder_class=MultiQuantileDecoder,
        encoder_kwargs=dict(n_instances=sensors, n_neighbors=min(20, sensors), hidden_size=64,
                            emb_size=16, temporal_layers=1, gnn_layers=2, activation="elu",
                            conv_type="aniso", sparsify_gradient=True, at_most_k=False,
                            dropout_emb=0.0),
        decoder_kwargs=dict(quantiles=QUANTILES.tolist(), hidden_size=64),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--corel-source", type=Path, default=Path("third_party/corel"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-batches", type=int, default=50)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    source = args.corel_source.resolve()
    import subprocess
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != COREL_COMMIT:
        raise RuntimeError(f"expected CoRel {COREL_COMMIT}, found {commit}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    manifest, arrays = load_cache(args.cache)
    x, y, mask, rows = make_examples(arrays)
    if args.max_examples:
        x, y, mask, rows = x[: args.max_examples], y[: args.max_examples], mask[: args.max_examples], rows[: args.max_examples]
    split = max(1, int(0.9 * len(x)))
    if split >= len(x):
        split = len(x) - 1

    # Match the official graph-level scaler: one location/scale per horizon.
    train_y = y[:split]
    train_mask = mask[:split]
    means = np.array([train_y[..., h][train_mask[..., h]].mean() for h in range(train_y.shape[-1])], dtype=np.float32)
    stds = np.array([train_y[..., h][train_mask[..., h]].std() for h in range(train_y.shape[-1])], dtype=np.float32)
    stds = np.maximum(stds, 1e-6)
    x = (x - means[None, None, None, :]) / stds[None, None, None, :]
    y = (y - means[None, None, :]) / stds[None, None, :]

    def loader(start: int, stop: int, shuffle: bool, loader_seed: int) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(x[start:stop]), torch.from_numpy(y[start:stop]), torch.from_numpy(mask[start:stop])
        )
        generator = torch.Generator().manual_seed(loader_seed)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, generator=generator)

    val_loader = loader(split, len(x), False, args.seed)
    sensors, horizons = x.shape[2], x.shape[3]
    model = build_model(source, sensors, horizons)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.25)
    args.output.mkdir(parents=True, exist_ok=True)
    last_checkpoint = args.output / "last.pt"
    best, best_state, stale, start_epoch = float("inf"), None, 0, 0
    history: list[dict] = []
    if args.resume:
        if not last_checkpoint.exists():
            raise FileNotFoundError(f"resume checkpoint absent: {last_checkpoint}")
        state = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best, best_state, stale = state["best"], state["best_state"], state["stale"]
        history, start_epoch = state["history"], state["epoch"]
        torch.set_rng_state(state["torch_rng_state"])
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        train_loader = loader(0, split, True, args.seed + epoch)
        model.train()
        train_losses = []
        for batch_idx, (xb, yb, mb) in enumerate(train_loader):
            if batch_idx >= args.train_batches:
                break
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)[:, :, 0]
            loss = pinball(pred, yb, mb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb, mb in val_loader:
                val_losses.append(float(pinball(model(xb)[:, :, 0], yb, mb)))
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch + 1, "train_pinball": float(np.mean(train_losses)), "val_pinball": val_loss})
        if val_loss < best:
            best, stale = val_loss, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        scheduler.step()
        torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "best": best, "best_state": best_state,
                    "stale": stale, "history": history, "epoch": epoch + 1,
                    "torch_rng_state": torch.get_rng_state()}, last_checkpoint)
        (args.output / "progress.json").write_text(json.dumps({
            "status": "running", "epochs_completed": epoch + 1,
            "best_validation_pinball_standardized": best,
            "last_validation_pinball_standardized": val_loss,
            "runtime_seconds_this_invocation": time.perf_counter() - started,
        }, indent=2) + "\n", encoding="utf-8")
        if stale >= args.patience:
            break

    checkpoint = args.output / "model.pt"
    torch.save({"state_dict": best_state, "mean": means, "std": stds, "quantiles": QUANTILES,
                "source_commit": commit, "model_seed": args.seed}, checkpoint)
    summary = {
        "status": "smoke_only" if args.max_examples else "trained",
        "adapter": "PS-1 multihorizon CoRel adaptation",
        "source_commit": commit,
        "cache_manifest": str((args.cache / "manifest.json").resolve()),
        "cache_manifest_sha256": sha256(args.cache / "manifest.json"),
        "dataset": manifest.get("dataset") or args.cache.parent.name.rsplit("-seed", 1)[0],
        "model_seed": args.seed,
        "examples": len(x),
        "train_examples": split,
        "validation_examples": len(x) - split,
        "sensors": sensors,
        "horizons": horizons,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "epochs_completed": len(history),
        "best_validation_pinball_standardized": best,
        "runtime_seconds": time.perf_counter() - started,
        "history": history,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("status", "examples", "parameters", "epochs_completed", "best_validation_pinball_standardized", "runtime_seconds")}))


if __name__ == "__main__":
    main()
