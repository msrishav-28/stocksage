"""Exercise complete user paths, fail-closed hardware and artifact provenance."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from stocksage.__main__ import main, run
from stocksage.data import digest
from stocksage.evaluation import Config, predictive_metrics
from stocksage.runtime import hardware, provenance

ROOT = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def invoke(self, args):
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            status = main(args)
        return status, out.getvalue(), err.getvalue()

    def test_module_entrypoint(self):
        result = subprocess.run([sys.executable, "-m", "stocksage", "--help"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("fetch-sample", result.stdout)

    def test_full_offline_run_replay_and_metric_recalculation(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            data, first, second = base / "data", base / "first", base / "second"
            status, _, err = self.invoke(["synthetic", "--out", str(data), "--days", "300", "--symbols", "1"])
            self.assertEqual(status, 0, err)
            inputs = ["--csv", str(data / "prices.csv"), "--manifest", str(data / "manifest.json")]
            self.assertEqual(self.invoke(["validate", *inputs])[0], 0)
            for path in (first, second):
                status, _, err = self.invoke(["evaluate", *inputs, "--out", str(path), "--min-train", "60", "--test-size", "120"])
                self.assertEqual(status, 0, err)
            for filename in ("metrics.json", "predictions.csv", "folds.json", "forecast.json"):
                self.assertEqual((first / filename).read_bytes(), (second / filename).read_bytes(), filename)
            meta = json.loads((first / "run.json").read_text())
            for name, expected in meta["artifacts_sha256"].items():
                self.assertEqual(digest((first / name).read_bytes()), expected)
            self.assertEqual(meta["data"]["kind"], "synthetic")
            self.assertEqual(meta["source_sha256"], provenance(ROOT)["source_sha256"])
            predictions = pd.read_csv(first / "predictions.csv")
            metrics = json.loads((first / "metrics.json").read_text())
            for name, group in predictions.groupby("model"):
                self.assertAlmostEqual(predictive_metrics(group)["mae"], metrics["models"][name]["overall"]["mae"], places=14)
            self.assertEqual(self.invoke(["evaluate", *inputs, "--out", str(first)])[0], 2)
            self.assertEqual(self.invoke(["forecast", *inputs, "--out", str(base / "forecast"), "--min-train", "60"])[0], 0)
            self.assertFalse((base / "forecast" / "metrics.json").exists())

    def test_failed_run_leaves_no_success_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            self.assertEqual(self.invoke(["synthetic", "--out", str(base / "data"), "--days", "300"])[0], 0)
            out = base / "run"
            status, _, error = self.invoke(["evaluate", "--csv", str(base / "data/prices.csv"),
                                            "--manifest", str(base / "data/manifest.json"), "--out", str(out),
                                            "--min-train", "500"])
            self.assertEqual(status, 2)
            self.assertIn("labelled feature sessions", error)
            self.assertFalse(out.exists())
            self.assertFalse(list(base.glob(".stocksage-*")))

    def test_missing_file_nonzero_and_no_fallback(self):
        status, output, err = self.invoke(["validate", "--csv", "/not-a-file.csv", "--manifest", "/not-a-file.json"])
        self.assertEqual(status, 2)
        self.assertEqual(output, "")
        self.assertIn("failed", err)


class RuntimeTests(unittest.TestCase):
    def test_missing_torch_is_blocker(self):
        with patch("stocksage.runtime.importlib.import_module", side_effect=ModuleNotFoundError("torch")):
            result = hardware()
            self.assertFalse(result["cuda_verified"])
            self.assertIn("ModuleNotFoundError", result["gpu_blocker"])
            with redirect_stdout(StringIO()):
                self.assertEqual(main(["doctor", "--require-cuda"]), 2)

    def test_old_torch_and_cpu_only_fail_closed(self):
        for fake in (SimpleNamespace(__version__="2.4.0"),
                     SimpleNamespace(__version__="2.8.0", cuda=SimpleNamespace(is_available=lambda: False))):
            with patch("stocksage.runtime.importlib.import_module", return_value=fake):
                self.assertFalse(hardware()["cuda_verified"])

    def test_cuda_kernel_numerical_gate(self):
        # Simulated hardware checks exercise control flow, not real GPU evidence.
        class Tensor:
            value = 4096.0
            def __matmul__(self, other): return self
            def sum(self): return self
            def item(self): return self.value
        fake = SimpleNamespace(__version__="2.8.0+cu128", version=SimpleNamespace(cuda="12.8"),
                               ones=lambda shape, device: Tensor(), cuda=SimpleNamespace(
                                   is_available=lambda: True, synchronize=lambda: None,
                                   get_device_name=lambda _: "SIMULATED", get_device_capability=lambda _: (12, 0)))
        with patch("stocksage.runtime.importlib.import_module", return_value=fake):
            self.assertTrue(hardware()["cuda_verified"])
            Tensor.value = 1
            self.assertFalse(hardware()["cuda_verified"])
            fake.version.cuda = "12.1"
            self.assertFalse(hardware()["cuda_verified"])

    def test_source_fingerprint_changes_with_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "stocksage").mkdir()
            for name in ("pyproject.toml", "requirements.txt", "stocksage/core.py"):
                (root / name).write_text("original")
            one = provenance(root)
            (root / "stocksage/core.py").write_text("changed")
            two = provenance(root)
            self.assertNotEqual(one["source_sha256"], two["source_sha256"])
            self.assertIsNone(one["git_head_at_run"])


if __name__ == "__main__":
    unittest.main()
