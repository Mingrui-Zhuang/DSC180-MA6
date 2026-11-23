"""Guard rails on the ingest contract and the trained model's behaviour."""
import pandas as pd
import pytest
from sklearn.metrics import r2_score

from src.config import LAG_WARMUP, TARGET, FORBIDDEN
from src.features import build_features, feature_columns, reindex_to_calendar
from src.ingest import load_raw
from src.train import MODELS, temporal_split


@pytest.fixture(scope="module")
def raw():
    return load_raw().sort_values("dteday").reset_index(drop=True)


@pytest.fixture(scope="module")
def feats(raw):
    return build_features(raw)


# Explicitly list which feature names are allowed in the model.
# These come from either: (1) raw numeric columns minus FORBIDDEN, or
# (2) features explicitly engineered in src/features.py.

RAW_NUMERIC_COLS = {"dteday", "instant", "season", "month", "weekday",
                    "year", "season_sin", "season_cos", "humidity",
                    "temperature", "atemp", "hum", "windspeed", "temp",
                    "casual", "registered", "cnt"}

# Calendar features engineered in src/features.py line 57-60
CALENDAR_FEATURES = {"dayofweek", "is_weekend", "is_holiday", "is_workday"}

# Cyclical month encoding engineered in src/features.py lines 64-66
MONTH_FEATURES = {"month_sin", "month_cos"}

# Weather encoding from raw.weather column (line 69)
WEATHER_FEATURES = {"weather_severity"}

# Lagged target features (lines 71-78)
LAG_FEATURES = {"cnt_lag_1", "cnt_lag_7", "cnt_roll_7", "cnt_roll_3"}

# Expected feature set: raw numeric (minus forbidden) plus all engineered columns
EXPECTED_FEATURES = ((RAW_NUMERIC_COLS - FORBIDDEN) |
                     CALENDAR_FEATURES | MONTH_FEATURES |
                     WEATHER_FEATURES | LAG_FEATURES)


def test_calendar_is_complete_before_lagging(raw):
    """The reindexed frame must have no gaps, or the lags are wrong."""
    grid = reindex_to_calendar(raw)
    assert set(grid.dteday.diff().dropna().dt.days.unique()) == {1}


def test_only_expected_rows_are_dropped(raw, feats):
    """Rows may only vanish for the lag warm-up or downstream of a genuinely
    absent day -- never because an encoder produced a silent NaN.

    Accounting: the first LAG_WARMUP days have no complete history. Each
    absent calendar day m costs the LAG_WARMUP rows after it (cnt_lag_1 is
    NaN on m+1, cnt_roll_7 is NaN on m+1..m+7, cnt_lag_7 on m+7; their
    union is 7 rows). Day m itself was never in `raw`, so it is not a loss.
    The bound is exact for the current feature set; add a longer lag and
    LAG_WARMUP in src/config.py must grow with it.
    """
    full = pd.date_range(raw.dteday.min(), raw.dteday.max(), freq="D")
    missing_days = len(full) - len(raw)
    allowed_loss = LAG_WARMUP + LAG_WARMUP * missing_days
    lost = len(raw) - len(feats)
    assert lost <= allowed_loss, f"lost {lost} rows; expected at most {allowed_loss}"


def test_no_unexpected_columns(raw):
    """A feature must be explicitly engineered or present in the raw CSV.

    Expected columns: all numeric columns from raw data minus FORBIDDEN, plus
    any features explicitly engineered in src/features.py (calendar, month,
    weather encoding, and lags). Any extra column is unexpected and must be
    removed.
    """
    # Get what we actually produced
    feats = build_features(raw)
    actual_cols = set(feature_columns(feats))

    # Find columns not in our explicit list
    unexpected = actual_cols - EXPECTED_FEATURES
    if unexpected:
        pytest.fail(f"Unexpected numeric columns in features: {sorted(unexpected)}")


def test_beats_the_mean_baseline(feats):
    """A model that cannot beat predicting the training mean is not a result."""
    cols = feature_columns(feats)
    train, _, test = temporal_split(feats)
    ridge = MODELS["ridge"]
    ridge.fit(train[cols], train[TARGET])
    assert r2_score(test[TARGET], ridge.predict(test[cols])) > 0.4


def test_persistence_baseline_is_reported():
    """The baseline that matters for a lag-driven model is 'yesterday again'.
    It must be in the comparison, whatever its score."""
    assert "baseline_persistence" in MODELS
