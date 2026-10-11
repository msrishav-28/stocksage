"""Input integrity and causality tests, independent of network/model weights."""

from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from stocksage.data import COLUMNS, digest, fetch_sample, load_dataset, save_dataset, synthetic, validate
from stocksage.features import FEATURES, build_features


class DataTests(unittest.TestCase):
    def setUp(self):
        self.raw = synthetic(300, 2)

    def test_fixture_repeatable(self):
        assert_frame_equal(self.raw, synthetic(300, 2))
        self.assertFalse(self.raw.equals(synthetic(300, 2, seed=1)))

    def test_reject_bad_panels(self):
        for kind in ("duplicate", "missing_session", "unsorted", "missing", "infinite", "negative_volume",
                     "negative_price", "high", "low", "lowercase", "date", "future", "extra_column"):
            with self.subTest(kind=kind):
                bad = self.raw.copy()
                if kind == "duplicate": bad = pd.concat([bad, bad.iloc[[0]]])
                elif kind == "missing_session": bad = bad.drop(index=0)
                elif kind == "unsorted": bad = bad.iloc[::-1]
                elif kind == "missing": bad.loc[0, "close"] = np.nan
                elif kind == "infinite": bad.loc[0, "volume"] = np.inf
                elif kind == "negative_volume": bad.loc[0, "volume"] = -1
                elif kind == "negative_price": bad.loc[0, "open"] = -1
                elif kind == "high": bad.loc[0, "high"] = 1
                elif kind == "low": bad.loc[0, "low"] = 1000
                elif kind == "lowercase": bad.loc[0, "symbol"] = "bad"
                elif kind == "date": bad.loc[0, "date"] = "01/01/2015"
                elif kind == "future": bad.loc[0, "date"] = "2999-01-01"
                else: bad["extra"] = 1
                with self.assertRaises((ValueError, OverflowError)):
                    validate(bad)

    def test_limits(self):
        for args in ((299, 2), (1501, 2), (300, 21), (300, 0)):
            with self.assertRaises(ValueError): synthetic(*args)
        with self.assertRaises(ValueError): validate(self.raw.iloc[:10])
        with patch("stocksage.data.MAX_ROWS", 100):
            with self.assertRaises(ValueError): validate(self.raw)
        too_many = pd.concat([self.raw.iloc[:300].assign(symbol=f"X{i}") for i in range(21)])
        with self.assertRaisesRegex(ValueError, "20 symbols"): validate(too_many)

    def test_snapshot_round_trip_hash_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "sample"
            paths = save_dataset(self.raw, out, {
                "source": "test", "license": "generated", "limitations": "synthetic only", "kind": "synthetic",
                "adjustment": "raw", "point_in_time": False,
            })
            frame, meta = load_dataset(*paths)
            self.assertEqual(len(frame), len(self.raw))
            self.assertEqual(meta["sha256"], digest(paths[0].read_bytes()))
            with self.assertRaises(FileExistsError): save_dataset(self.raw, out, {})
            paths[0].write_text(paths[0].read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "SHA256"): load_dataset(*paths)

    def test_manifest_requires_explicit_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            csv, manifest = save_dataset(self.raw, Path(directory) / "sample", {
                "source": "test", "license": "generated", "limitations": "synthetic", "kind": "synthetic",
                "adjustment": "raw", "point_in_time": False,
            })
            original = json.loads(manifest.read_text())
            cases = {"source": "", "kind": "live", "adjustment": "total_return", "point_in_time": "true",
                     "retrieved_at": "2015-01-01", "license": "", "limitations": ""}
            for key, bad_value in cases.items():
                with self.subTest(key=key):
                    manifest.write_text(json.dumps({**original, key: bad_value}))
                    with self.assertRaises(ValueError): load_dataset(csv, manifest)
            manifest.write_text(json.dumps(original))
            with patch("stocksage.data.MAX_BYTES", 10):
                with self.assertRaisesRegex(ValueError, "budget"): load_dataset(csv, manifest)

    def test_provider_mapping_ignores_external_indicators(self):
        raw = self.raw.iloc[:300].drop(columns="symbol").rename(columns={
            "date": "Date", **{c: "AAPL." + c.title() for c in COLUMNS[2:]}})
        raw["mavg"] = 999999  # Must never enter feature inputs.
        payload = raw.to_csv(index=False).encode()
        with tempfile.TemporaryDirectory() as directory, patch("stocksage.data.urlopen", return_value=BytesIO(payload)):
            frame, meta = load_dataset(*fetch_sample(Path(directory) / "sample"))
            self.assertEqual(frame.columns.tolist(), COLUMNS)
            self.assertEqual(set(frame.symbol), {"AAPL"})
            self.assertEqual(meta["upstream_sha256"], digest(payload))


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.bars = validate(synthetic(300, 2))

    def test_hand_computed_target_is_next_open_to_close(self):
        f = build_features(self.bars)
        for symbol in self.bars.symbol.unique():
            bars = self.bars[self.bars.symbol == symbol].reset_index(drop=True)
            got = f[f.symbol == symbol].reset_index(drop=True)
            self.assertAlmostEqual(got.iloc[0].target, bars.iloc[21].close / bars.iloc[21].open - 1)
            self.assertEqual(got.iloc[0].label_end, bars.iloc[21].date)
            self.assertAlmostEqual(got.iloc[0].return_20, bars.iloc[20].close / bars.iloc[0].close - 1)
            self.assertTrue(pd.isna(got.iloc[-1].target))
            self.assertEqual(got.iloc[-1].date, bars.iloc[-1].date)

    def test_future_perturbation_and_prefix_invariance(self):
        cutoff = self.bars.date.unique()[160]
        original = build_features(self.bars)
        changed = self.bars.copy()
        changed.loc[changed.date > cutoff, ["open", "high", "low", "close", "volume"]] *= 100
        altered = build_features(changed)
        cols = ["date", "symbol", *FEATURES]
        expected = original[original.date <= cutoff][cols].reset_index(drop=True)
        assert_frame_equal(expected, altered[altered.date <= cutoff][cols].reset_index(drop=True))
        assert_frame_equal(expected, build_features(self.bars[self.bars.date <= cutoff])[cols].reset_index(drop=True))

    def test_symbol_isolation(self):
        together = build_features(self.bars)
        for symbol in self.bars.symbol.unique():
            single = build_features(self.bars[self.bars.symbol == symbol])
            assert_frame_equal(together[together.symbol == symbol].reset_index(drop=True), single)

    def test_zero_volume_and_constant_prices_are_finite(self):
        bars = self.bars.copy()
        bars[["open", "high", "low", "close"]] = 100.0
        bars["volume"] = 0.0
        f = build_features(bars)
        self.assertTrue(np.isfinite(f[FEATURES]).all().all())
        self.assertTrue((f[FEATURES] == 0).all().all())


if __name__ == "__main__":
    unittest.main()
