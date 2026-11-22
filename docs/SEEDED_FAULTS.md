# Seeded Faults

## Fault 1: `feature_columns()` includes `casual`

**Change:** Added `"casual"` to the start of the returned list in `src/features.py`.

```python
def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns that are safe to feed the model."""
    numeric = df.select_dtypes(include="number").columns
    return ["casual"] + [c for c in numeric if c not in FORBIDDEN]
```

**Automatic PostToolUse hook output:**
```
PostToolUse:Edit hook blocking error from command: "${CLAUDE_PROJECT_DIR}/.claude/hooks/check_leakage.sh": [${CLAUDE_PROJECT_DIR}/.claude/hooks/check_leakage.sh]: Leakage tests failed after editing /Users/bobbyzhou/DSC180-MA6-claude/src/features.py:
feats =         dteday  instant  season  ...  cnt_lag_7   cnt_roll_7   cnt_roll_3
0   2011-01-10     10.0  WINTER  ...     134....142857  2183.333333
713 2012-12-31    731.0  WINTER  ...      920.0  1531.428571  2077.333333

[714 rows x 29 columns]

    def test_no_forbidden_features(feats):
        """Guards the gate, not the data: feature_columns() excludes FORBIDDEN by
        construction, so this cannot discover a *new* leaky column. What it
        catches is someone editing feature_columns() or FORBIDDEN -- which is
        exactly the fault the Step 7 hook demonstrates."""
        cols = set(feature_columns(feats))
>       assert not (cols & FORBIDDEN), f"forbidden columns in feature set: {cols & FORBIDDEN}"
E       AssertionError: forbidden columns in feature set: {'casual'}
E       assert not ({'atemp', 'casual', 'cnt_lag_1', 'cnt_lag_7', 'cnt_roll_3', 'cnt_roll_7', ...} & {'casual', 'cnt', 'cnt_2d_bfr', 'days_since_2011', 'instant', 'registered', ...})

tests/test_leakage.py:31: AssertionError
=========================== short test summary ============================
FAILED tests/test_leakage.py::test_no_forbidden_features - AssertionError: fo...
1 failed, 7 passed in 0.46s
```

**Repair:** Removed `"casual"` from the return statement (1 turn).
```
make test: 15 passed in 1.40s
```

**Commit hash (broken):** `7208021`


## Fault 2: `cnt_roll_7` missing `.shift(1)`

**Change:** Removed `.shift(1)` from the calculation of `cnt_roll_7` in `src/features.py`.

```python
out["cnt_roll_7"] = out[TARGET].rolling(LAG_WARMUP).mean()
```

**Automatic PostToolUse hook output:**
```
PostToolUse:Edit hook blocking error from command: "${CLAUDE_PROJECT_DIR}/.claude/hooks/check_leakage.sh": [${CLAUDE_PROJECT_DIR}/.claude/hooks/check_leakage.sh]: Leakage tests failed after editing /Users/bobbyzhou/DSC180-MA6-claude/src/features.py:
[714 rows x 29 columns]

    def test_rolling_mean_excludes_today(feats):
        """cnt_roll_7 must be the mean of the 7 calendar days *before* today."""
        by_date = feats.set_index("dteday")[TARGET]
        date = feats.dteday.iloc[200]
        window = [date - pd.Timedelta(days=k) for k in range(1, 8)]
        if all(d in by_date.index for d in window):
            expected = sum(by_date[d] for d in window) / 7
>           assert feats.set_index("dteday").cnt_roll_7[date] == pytest.approx(expected)
E           assert np.float64(4389.0) == 4414.857142857143 ± 0.00441486
E             
E             comparison failed
E             Obtained: 4389.0
E             Expected: 4414.857142857143 ± 0.00441486

tests/test_leakage.py:57: AssertionError
=========================== short test summary ============================
FAILED tests/test_leakage.py::test_rolling_mean_excludes_today - assert np.fl...
1 failed, 7 passed in 0.42s
```

**Repair:** Added `.shift(1)` back to `cnt_roll_7` (1 turn).
```
make test: 15 passed in 1.13s
```

**Commit hash (broken):** `b4b63fc`


## Fault 3: `encode_weather` silently drops unknown labels

**Change:** Removed the two lines that check for and raise on unknown weather labels:
```python
if unknown:
    raise ValueError(f"unmapped weather labels: {sorted(unknown)}")
```

**Automatic PostToolUse hook output:**
```
PostToolUse:Edit hook reported a problem
[${CLAUDE_PROJECT_DIR}/.claude/hooks/check_leakage.sh]: Leakage tests failed after editing /Users/bobbyzhou/DSC180-MA6-claude/src/features.py:
......F..                                                                [100%]
=================================== FAILURES ===================================
_________________ test_weather_encoder_raises_on_unknown_label _________________

    def test_weather_encoder_raises_on_unknown_label():
        """Fault 3 guard: encode_weather must raise ValueError on unmapped labels."""
        from src.features import encode_weather
    
        # Fault 3: Unknown label "SNOW" (not in WEATHER_ORDER) should raise ValueError
        series_with_snow = pd.Series(["GOOD", "SNOW"])
>       with pytest.raises(ValueError, match=r"\bSNOW\b"):
E       Failed: DID NOT RAISE ValueError

tests/test_leakage.py:86: Failed
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_weather_encoder_raises_on_unknown_label - ...
1 failed, 8 passed in 0.39s
```

**Repair:** Restored the two deleted lines checking for and raising on unknown labels (1 turn).
```
make test: 16 passed in 1.11s
```

**Commit hash (broken):** `81f49af`


## Fault 4: Write to `data/raw/` bypasses PostToolUse hook

**Change:** Copied a marker file into the restricted directory using Bash:
```bash
cp "/Users/bobbyzhou/DSC180-MA6-claude/fault4-fixture.8973Yh/fault4-marker.txt" "/Users/bobbyzhou/DSC180-MA6-claude/fault4-fixture.8973Yh/data/raw/fault4-marker.txt"
```

**Disposable destination:** `fault4-fixture.8973Yh/data/raw/` (a local fixture, not the real `data/raw/`)

**Actual terminal output:**
```
EXIT_CODE: 0
Destination contents:
fault 4 demo
```

**No automatic leakage-hook message appeared.** The copy ran through Bash as a direct shell command. The PostToolUse hook only matches `Edit|Write` tool calls in the harness; it does not intercept arbitrary Bash commands like `cp`. This run used a fixture and did not test the real `data/raw/` permissions or bypass the deny rule that blocks writes to that directory.

**Detection check:** Have `make data` compare the filenames in `data/raw/` with an expected file list and report unexpected additions. Hashing only the expected files would miss an extra marker file.

