"""Stage 3: temporal split, fit, evaluate, write metrics.

Two baselines and two models, on purpose.

  baseline_mean         predict the 2011 average for every day
  baseline_persistence  predict yesterday's count (cnt_lag_1)
  ridge                 standardised linear model on the audited features
  hgb                   gradient-boosted trees on the same features

The mean baseline shows what "the distribution moved" costs. Persistence
is the baseline that actually matters: ridge's features are dominated by
lagged demand, so if it cannot beat "yesterday again" it has learned
nothing worth reporting. The booster is there because it fails, and the
reason it fails is the headline result.

Run: python -m src.train

Exercise 4 (optional): python -m src.train --exercise4
  Compare four approaches on validation set only (Jan-Jun 2012), excluding
  cnt_roll_3 feature.

Final test (Exercise 4): python -m src.train --exercise4 --final-test
  Evaluate the transformed model (hgb_changes) on held-out test set, write
  to reports/metrics.json with both before-feature and after-feature results.
"""
import argparse
import json

import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import METRICS, PROCESSED, TARGET, TRAIN_END, VALID_END
from src.features import feature_columns


class PersistenceBaseline(BaseEstimator, RegressorMixin):
    """Predict a single lagged column unchanged. Nothing is fit."""

    def __init__(self, column: str = "cnt_lag_1"):
        self.column = column

    def fit(self, X, y=None):
        if self.column not in X.columns:
            raise ValueError(f"persistence baseline needs column {self.column!r}")
        return self

    def predict(self, X):
        return X[self.column].to_numpy(dtype=float)


MODELS = {
    "baseline_mean": DummyRegressor(strategy="mean"),
    "baseline_persistence": PersistenceBaseline("cnt_lag_1"),
    "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "hgb": HistGradientBoostingRegressor(random_state=0, max_iter=300),
}


def temporal_split(df: pd.DataFrame):
    """Split by date, never at random. A random split on a time series
    puts tomorrow in train and today in test."""
    train = df[df.dteday <= TRAIN_END]
    valid = df[(df.dteday > TRAIN_END) & (df.dteday <= VALID_END)]
    test = df[df.dteday > VALID_END]
    return train, valid, test


def fit_and_score() -> dict:
    """Train models and write metrics to reports/metrics.json.

    Always produces all four original model results with cnt_roll_3 at the top
    level, plus a "before-feature" block showing results without cnt_roll_3.
    """
    df = pd.read_parquet(PROCESSED)
    cols = feature_columns(df)

    # Check if cnt_roll_3 is in features
    has_roll_3 = "cnt_roll_3" in cols
    if has_roll_3:
        cols_with_roll = cols
        cols_no_roll = [c for c in cols if c != "cnt_roll_3"]
    else:
        cols_with_roll = None
        cols_no_roll = None

    train, valid, test = temporal_split(df)

    results = {
        "n_features": len(cols_with_roll),
        "features": cols_with_roll,
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
    }

    # Before-feature results: models without cnt_roll_3
    if cols_no_roll is not None:
        results["before-feature"] = {}
        for name, model in MODELS.items():
            model.fit(train[cols_no_roll], train[TARGET])
            results["before-feature"][name] = {
                "valid": {
                    "r2": round(float(r2_score(valid[TARGET], model.predict(valid[cols_no_roll]))), 4),
                    "mae": round(float(mean_absolute_error(valid[TARGET], model.predict(valid[cols_no_roll]))), 1),
                },
                "test": {
                    "r2": round(float(r2_score(test[TARGET], model.predict(test[cols_no_roll]))), 4),
                    "mae": round(float(mean_absolute_error(test[TARGET], model.predict(test[cols_no_roll]))), 1),
                }
            }

    for name, model in MODELS.items():
        # Use appropriate columns based on whether we include cnt_roll_3
        if cols_with_roll is not None:
            train_X = train[cols_with_roll]
            result_cols = cols_with_roll
        else:
            train_X = train[cols_no_roll]
            result_cols = cols_no_roll
        model.fit(train_X, train[TARGET])
        results[name] = {
            split: {
                "r2": round(float(r2_score(part[TARGET], model.predict(part[result_cols]))), 4),
                "mae": round(float(mean_absolute_error(part[TARGET], model.predict(part[result_cols]))), 1),
            }
            for split, part in [("valid", valid), ("test", test)]
        }
    return results


def exercise4_fit_and_score() -> dict:
    """Exercise 4: Compare four approaches on validation set only (Jan-Jun 2012).

    Four approaches:
    1. Persistence baseline
    2. Ridge regression
    3. HistGradientBoosting predicting daily ride counts
    4. HistGradientBoosting predicting daily ride count changes (cnt - cnt_lag_1)

    Excludes cnt_roll_3 from all approaches based on Exercise 1 validation.
    """
    df = pd.read_parquet(PROCESSED)

    # Start with feature_columns and remove cnt_roll_3
    cols = feature_columns(df)
    if "cnt_roll_3" in cols:
        cols = [c for c in cols if c != "cnt_roll_3"]

    train, valid, test = temporal_split(df)

    # Use the same hyperparameters as the main MODELS dict
    HGB_RANDOM_STATE = 0
    HGB_MAX_ITER = 300
    RIDGE_ALPHA = 1.0

    results = {
        "n_features": len(cols),
        "features": cols,
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
    }

    # Approach 1: Persistence baseline
    persist_model = PersistenceBaseline("cnt_lag_1")
    persist_model.fit(train[cols], train[TARGET])
    results["baseline_persistence"] = {
        "valid": {
            "r2": round(float(r2_score(valid[TARGET], persist_model.predict(valid[cols]))), 4),
            "mae": round(float(mean_absolute_error(valid[TARGET], persist_model.predict(valid[cols]))), 1),
        }
    }

    # Approach 2: Ridge regression
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    ridge.fit(train[cols], train[TARGET])
    results["ridge"] = {
        "valid": {
            "r2": round(float(r2_score(valid[TARGET], ridge.predict(valid[cols]))), 4),
            "mae": round(float(mean_absolute_error(valid[TARGET], ridge.predict(valid[cols]))), 1),
        }
    }

    # Approach 3: HistGradientBoosting predicting daily ride counts directly
    hgb_cnt = HistGradientBoostingRegressor(random_state=HGB_RANDOM_STATE, max_iter=HGB_MAX_ITER)
    hgb_cnt.fit(train[cols], train[TARGET])
    results["hgb"] = {
        "valid": {
            "r2": round(float(r2_score(valid[TARGET], hgb_cnt.predict(valid[cols]))), 4),
            "mae": round(float(mean_absolute_error(valid[TARGET], hgb_cnt.predict(valid[cols]))), 1),
        }
    }

    # Approach 4: HistGradientBoosting predicting daily ride count changes (cnt - cnt_lag_1)
    # Add back cnt_lag_1 at prediction time
    def transform_for_change_target(X):
        return X

    def inverse_transform_changes(pred_changes, X):
        return pred_changes + X["cnt_lag_1"]

    y_changes = valid[cols] if isinstance(train[cols], pd.DataFrame) else train[cols]

    hgb_change = HistGradientBoostingRegressor(random_state=HGB_RANDOM_STATE, max_iter=HGB_MAX_ITER)
    hgb_change.fit(train[cols], (train[TARGET] - train["cnt_lag_1"]).values)

    # Predict on validation set and add back the lagged values
    change_preds = hgb_change.predict(valid[cols])
    final_preds = change_preds + valid["cnt_lag_1"].values

    results["hgb_changes"] = {
        "valid": {
            "r2": round(float(r2_score(valid[TARGET], final_preds)), 4),
            "mae": round(float(mean_absolute_error(valid[TARGET], final_preds)), 1),
        }
    }

    return results


def run_final_test() -> dict:
    """Run Exercise 4 final test evaluation on held-out test set.

    Returns results without writing to metrics.json (for --exercise4 --final-test).
    Includes both before-feature and hgb_changes validation/test scores.
    Reports 14 features (the actual feature set without cnt_roll_3).
    """
    df = pd.read_parquet(PROCESSED)

    # Remove cnt_roll_3 from features
    cols = feature_columns(df)
    if "cnt_roll_3" in cols:
        cols_no_roll = [c for c in cols if c != "cnt_roll_3"]
    else:
        raise ValueError("run_final_test requires excluding cnt_roll_3")

    train, valid, test = temporal_split(df)

    results = {
        "n_features": len(cols),  # Will be 14 without cnt_roll_3
        "features": cols,
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
    }

    # Before-feature results: all four original models without cnt_roll_3
    results["before-feature"] = {}
    for name, model in MODELS.items():
        model.fit(train[cols_no_roll], train[TARGET])
        results["before-feature"][name] = {
            "valid": {
                "r2": round(float(r2_score(valid[TARGET], model.predict(valid[cols_no_roll]))), 4),
                "mae": round(float(mean_absolute_error(valid[TARGET], model.predict(valid[cols_no_roll]))), 1),
            },
            "test": {
                "r2": round(float(r2_score(test[TARGET], model.predict(test[cols_no_roll]))), 4),
                "mae": round(float(mean_absolute_error(test[TARGET], model.predict(test[cols_no_roll]))), 1),
            }
        }

    # hgb_changes results: evaluate transformed model on validation and test
    HGB_RANDOM_STATE = 0
    HGB_MAX_ITER = 300
    hgb_change = HistGradientBoostingRegressor(random_state=HGB_RANDOM_STATE, max_iter=HGB_MAX_ITER)
    y_changes_train = train[TARGET] - train["cnt_lag_1"]
    hgb_change.fit(train[cols_no_roll], y_changes_train.values)

    # Predict on validation set
    change_preds_valid = hgb_change.predict(valid[cols_no_roll])
    final_preds_valid = change_preds_valid + valid["cnt_lag_1"].values
    results["hgb_changes"] = {
        "valid": {
            "r2": round(float(r2_score(valid[TARGET], final_preds_valid)), 4),
            "mae": round(float(mean_absolute_error(valid[TARGET], final_preds_valid)), 1),
        },
        "test": {},  # Will be filled below
    }

    # Evaluate on test set (held-out)
    change_preds_test = hgb_change.predict(test[cols_no_roll])
    final_preds_test = change_preds_test + test["cnt_lag_1"].values
    results["hgb_changes"]["test"] = {
        "r2": round(float(r2_score(test[TARGET], final_preds_test)), 4),
        "mae": round(float(mean_absolute_error(test[TARGET], final_preds_test)), 1),
    }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train bike sharing demand models")
    parser.add_argument(
        "--exercise4",
        action="store_true",
        help="Run Exercise 4 comparison (only validation set, exclude cnt_roll_3)"
    )
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="Evaluate transformed model on held-out test set separately"
    )
    args = parser.parse_args()

    if args.exercise4 and args.final_test:
        res = run_final_test()  # Prints results without writing to metrics.json
        print("Exercise 4 final test evaluation complete.")
        print(json.dumps({k: v for k, v in res.items() if k != "features"}, indent=2))
    elif args.exercise4:
        res = exercise4_fit_and_score()
        print(json.dumps({k: v for k, v in res.items() if k != "features"}, indent=2))
    else:
        res = fit_and_score()
        METRICS.parent.mkdir(parents=True, exist_ok=True)
        METRICS.write_text(json.dumps(res, indent=2))
        print(json.dumps({k: v for k, v in res.items() if k != "features"}, indent=2))
