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

**Repair:** Removed `"casual"` from the return statement.
