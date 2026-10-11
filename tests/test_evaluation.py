"""Independent chronological boundaries, numerical metrics and model checks."""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from stocksage.data import synthetic, validate
from stocksage.evaluation import Config, economic_metrics, evaluate, forecast_latest, paired_mae_interval, predictive_metrics, walk_forward
from stocksage.features import FEATURES, build_features
from stocksage.models import MODEL_NAMES, predict


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.features = build_features(validate(synthetic(300, 2)))
        self.config = Config(min_train=60, test_size=63)

    def test_boundaries_date_groups_and_final_partial(self):
        folds = list(walk_forward(self.features, self.config))
        seen = []
        dates = sorted(self.features.dropna().date.unique())
        for _, train, test, boundary in folds:
            self.assertLess(train.label_end.max(), boundary)
            self.assertLess(boundary, test.date.min())
            self.assertEqual(dates.index(test.date.min()) - dates.index(boundary), 5)
            self.assertFalse(set(train.date) & set(test.date))
            self.assertTrue((test.groupby("date").symbol.nunique() == 2).all())
            seen.extend(test.date.unique())
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(folds[0][1].date.nunique(), 60)
        self.assertEqual(seen, dates[66:])
        self.assertLess(folds[-1][2].date.nunique(), 63)

    def test_variable_label_duration_is_purged(self):
        _, train, test, _ = next(walk_forward(self.features, self.config))
        f = self.features.copy()
        # Extend every label on the first date into the test period. Purging must
        # remove that entire date, leaving fewer than the required 60 sessions.
        f.loc[f.date == train.date.min(), "label_end"] = test.date.min()
        with self.assertRaisesRegex(ValueError, "Insufficient training history"):
            next(walk_forward(f, self.config))

    def test_no_valid_fold_fails(self):
        with self.assertRaises(ValueError): list(walk_forward(self.features, Config(min_train=500)))
        for kwargs in ({"embargo": 0}, {"test_size": 0}, {"min_train": 59}, {"seed": -1},
                       {"bootstrap_samples": 99}, {"block_size": 0}, {"min_train": 60.5}):
            with self.assertRaises(ValueError): Config(**kwargs)

    def test_latest_respects_label_maturity(self):
        report = forecast_latest(self.features, self.config)
        self.assertLess(report["train_label_end_max"], report["embargo_boundary"])
        self.assertLess(report["embargo_boundary"], report["asof_session_close"])
        self.assertEqual(len(report["forecasts"]), 8)
        self.assertIsNone(report["future_session_date"])


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = build_features(validate(synthetic(300, 2)))
        cls.train, cls.test = list(walk_forward(cls.f, Config(min_train=60, test_size=63)))[0][1:3]

    def test_never_read_test_targets(self):
        changed = self.test.copy()
        changed["target"] = 999.0
        for model in MODEL_NAMES:
            with self.subTest(model=model):
                expected = predict(self.train, self.test, model)
                np.testing.assert_array_equal(expected, predict(self.train, changed, model))
                self.assertTrue((expected[:, 1] <= expected[:, 0]).all())
                self.assertTrue((expected[:, 0] <= expected[:, 2]).all())

    def test_training_mean_and_zero(self):
        mean = predict(self.train, self.test, "train_mean")
        np.testing.assert_allclose(mean[:, 0], self.train.target.mean())
        np.testing.assert_array_equal(predict(self.train, self.test, "zero")[:, 0], 0)

    def test_models_learn_planted_relation(self):
        # A separate deterministic numerical task proves the estimator is fitted.
        rng = np.random.default_rng(2)
        f = pd.DataFrame(rng.normal(size=(500, len(FEATURES))), columns=FEATURES)
        f["target"] = .01 * f[FEATURES[0]]
        for model in ("ridge", "hist_gradient_boosting"):
            got = predict(f.iloc[:400], f.iloc[400:], model)[:, 0]
            self.assertLess(np.mean(abs(got - f.target.iloc[400:])), .003)

    def test_invalid_model_or_training_labels_fail(self):
        with self.assertRaises(ValueError): predict(self.train, self.test, "invented")
        bad = self.train.copy()
        bad["target"] = np.nan
        with self.assertRaises(ValueError): predict(bad, self.test, "ridge")

    def test_future_data_cannot_change_earlier_fold_predictions(self):
        bars = validate(synthetic(300, 1))
        cfg = Config(min_train=60, test_size=100, bootstrap_samples=100)
        first, _, _ = evaluate(build_features(bars), cfg)
        cutoff = pd.Timestamp(first[first.fold == 0].date.max())
        altered = bars.copy()
        altered.loc[altered.date > cutoff, ["open", "high", "low", "close", "volume"]] *= 3
        second, _, _ = evaluate(build_features(altered), cfg)
        cols = ["date", "symbol", "model", "prediction", "q10", "q90"]
        assert_frame_equal(first[first.fold == 0][cols], second[second.fold == 0][cols])


class MetricTests(unittest.TestCase):
    def sample(self):
        return pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                             "symbol": ["A", "A"], "target": [.1, -.1],
                             "prediction": [.05, .02], "q10": [-.1, -.1], "q90": [.1, .1]})

    def test_forecast_metrics_by_hand(self):
        m = predictive_metrics(self.sample())
        self.assertAlmostEqual(m["mae"], .085)
        self.assertAlmostEqual(m["rmse"], np.sqrt((.05**2 + .12**2) / 2))
        self.assertEqual(m["direction_accuracy"], .5)
        self.assertEqual(m["coverage_80"], 1)
        self.assertAlmostEqual(m["mean_interval_width"], .2)
        self.assertAlmostEqual(m["pinball_q10"], .01)

    def test_round_trip_fees_compounding_and_drawdown(self):
        m = economic_metrics(self.sample(), 5)
        self.assertAlmostEqual(m["net_compound_return"], 1.099 * .899 - 1)
        self.assertAlmostEqual(m["max_drawdown"], -.101)
        self.assertEqual(m["active_symbol_sessions"], 2)
        cash = self.sample().assign(prediction=0)
        self.assertEqual(economic_metrics(cash, 5)["net_compound_return"], 0)
        self.assertIsNone(economic_metrics(cash, 5)["annualized_sharpe_zero_rf"])
        self.assertAlmostEqual(economic_metrics(cash, 5, True)["net_compound_return"], 1.099 * .899 - 1)

    def test_initial_loss_counts_as_drawdown(self):
        p = self.sample().iloc[[1]]
        self.assertAlmostEqual(economic_metrics(p, 0)["max_drawdown"], -.1)

    def test_inactive_symbol_keeps_its_cash_weight(self):
        p = pd.concat([self.sample().iloc[[0]], self.sample().iloc[[0]].assign(symbol="B", prediction=0)])
        self.assertAlmostEqual(economic_metrics(p, 5)["net_compound_return"], .099 / 2)

    def test_bootstrap_zero_is_exact_and_reproducible(self):
        p = self.sample().assign(prediction=0)
        ci = paired_mae_interval(p, Config(bootstrap_samples=100))
        self.assertEqual(ci["ci95"], [0, 0])
        self.assertEqual(ci["mean_delta"], 0)
        self.assertEqual(ci, paired_mae_interval(p, Config(bootstrap_samples=100)))

    def test_economic_invalid_data_and_costs(self):
        for cost in (-1, 101, np.nan):
            with self.assertRaises(ValueError): economic_metrics(self.sample(), cost)
        with self.assertRaises(ValueError): economic_metrics(pd.concat([self.sample(), self.sample()]), 5)


if __name__ == "__main__":
    unittest.main()
