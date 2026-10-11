"""Run provenance and a fail-closed hardware diagnostic, without model weights."""

from datetime import datetime, timezone
import importlib
from importlib.metadata import version
import platform
from pathlib import Path
import re
import shutil
import subprocess

from .data import digest


def hardware() -> dict:
    result = {"checked_at": datetime.now(timezone.utc).isoformat(),
              "platform": platform.platform(), "python": platform.python_version(),
              "nvidia_smi_present": shutil.which("nvidia-smi") is not None,
              "cuda_verified": False, "torch_version": None}
    try:
        torch = importlib.import_module("torch")
        result["torch_version"] = torch.__version__
        if tuple(map(int, re.match(r"(\d+)\.(\d+)", torch.__version__).groups())) < (2, 8):
            raise RuntimeError("Requires PyTorch >=2.8 per hardware mandate")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        if torch.version.cuda is None or tuple(map(int, torch.version.cuda.split(".")[:2])) < (12, 8):
            raise RuntimeError("Requires CUDA >=12.8 wheels per hardware mandate")
        # Diagnostic only: this kernel must really execute, not just import.
        x = torch.ones((16, 16), device="cuda")
        value = float((x @ x).sum().item())
        torch.cuda.synchronize()
        if value != 4096.0:
            raise RuntimeError("CUDA matrix multiplication failed numerical check")
        result.update(cuda_verified=True, cuda_version=torch.version.cuda,
                      device=torch.cuda.get_device_name(0),
                      capability=list(torch.cuda.get_device_capability(0)))
    except Exception as exc:
        result["gpu_blocker"] = f"{type(exc).__name__}: {exc}"
    return result


def provenance(root: Path) -> dict:
    paths = [*sorted((root / "stocksage").glob("*.py")), root / "pyproject.toml", root / "requirements.txt"]
    # Wheel installs lack repository metadata. Still fingerprint actual module
    # bytes, and avoid attributing an unrelated enclosing checkout to this run.
    files = {str(p.relative_to(root)): digest(p.read_bytes()) for p in paths if p.is_file()}
    content = "\n".join(f"{p}:{h}" for p, h in sorted(files.items())).encode()
    try:
        top = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        if Path(top).resolve() != root.resolve():
            raise OSError("Not running in the project checkout")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "git_head_at_run": commit, "working_tree_dirty": dirty,
        "source_sha256": digest(content), "source_files": files,
        "versions": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "threadpoolctl")},
        "threads_per_model": 1,
    }
