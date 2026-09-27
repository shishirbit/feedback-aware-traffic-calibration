from pathlib import Path
import hashlib
import json
import platform
import subprocess
import sys


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_manifest():
    def command(args):
        result = subprocess.run(args, text=True, capture_output=True, check=True)
        return result.stdout.splitlines()[0]
    manifest = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        manifest["sumo"] = command(["sumo", "--version"])
        manifest["sumo_status"] = "available"
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        manifest["sumo"] = None
        manifest["sumo_status"] = f"unavailable: {error}"
    try:
        manifest["gpu"] = command(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    except (FileNotFoundError, subprocess.CalledProcessError):
        manifest["gpu"] = None
    return manifest


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise FileExistsError(f"refusing to overwrite differing artifact: {path}")
    path.write_text(payload, encoding="utf-8")
