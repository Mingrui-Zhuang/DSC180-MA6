## Audit 1 — Mingrui Zhuang

- Branch: verification
- Commit: 2045fc2ffc89bd7eb8281cd13465e62074d118af

### Auditor output

All five checks pass based on my review:

1. Target leakage. ✅ No feature contains casual, registered, or cnt. Lagged features use shift() on cnt (target), which is forward-looking and non-leaky.
2. Temporal ordering. ✅ .shift(1) is applied before .rolling().mean(). The split uses dteday <= TRAIN_END, TRAIN_END < dteday <= VALID_END, etc. — chronological, not random. Fitting only sees train data via model.fit(train[cols], train[TARGET]).
3. Calendar gaps. ✅ reindex_to_calendar fills missing days with rows of NaN before lags are computed; the test asserts daily continuity.
4. Silent row loss. ✅ .dropna() after lag features has a bounded expectation in test_only_expected_rows_are_dropped, which checks that lost rows ≤ LAG_WARMUP + LAG_WARMUP * missing_days (712 allowed, 728-714=14 lost).
5. Unreported baseline. ✅ Both baseline_mean and baseline_persistence are in MODELS and reported alongside every model score on both valid and test sets.

### Independent verification

- `make test`: 17 passed in 1.10s.
- Inspected `src/config.py` and `src/features.py` to verify forbidden
  columns and lagged feature calculations.
- Inspected `src/train.py` to verify chronological splitting,
  training-set-only fitting, and inclusion of both baselines.
- Inspected `tests/test_contract.py` to verify the row-loss bound.

### Verdict

I agree with the auditor's overall conclusion. The tests passed,
and the code inspection confirmed chronological splitting,
forbidden-column exclusion, and past-only rolling features.
Both required baselines are included.

However, the auditor incorrectly described shift(1) as
forward-looking. It is backward-looking because it uses previous
observations. The auditor's explanation of the row-loss bound
also needs clarification.

No result-changing issue was identified.

## Audit 2 — Bobby Zhou

- Repository: https://github.com/Mingrui-Zhuang/DSC180-MA6
- Branch: feature-and-results
- Commit: 2722bf118621d2fbe95e102f533418e7720dc336

### Auditor output (foreground leakage-auditor)

````
## Audit Findings

### Check 1: Target Leakage ✅ PASS
- `src/config.py` defines `LEAKAGE_COLS = {"casual", "registered", "cnt"}` (line 31)
- `src/features.py` uses `feature_columns(df)` which excludes all forbidden columns via `[c for c in numeric if c not in FORBIDDEN]` (line 86)
- Lagged features use `.shift(1)` on TARGET, not on casual/registered
- **No leakage detected**

### Check 2: Temporal Ordering ✅ PASS  
- `src/features.py`: `.shift(1)` is applied before `.rolling().mean()` (lines 74-78)
- `src/train.py`: Split is by date using `dteday <= TRAIN_END`, `TRAIN_END < dteday <= VALID_END`, etc. (lines 61-67)
- Model fitting: `model.fit(train[cols], train[TARGET])` only sees training rows (line 83)
- **No temporal violations**

### Check 3: Calendar Gaps ✅ PASS
- `src/features.py`: `reindex_to_calendar()` fills missing calendar days with NaN rows before lag features are computed (lines 34-50)
- This ensures `.shift(1)` aligns to actual calendar days, not row index
- **No calendar gap contamination**

### Check 4: Silent Row Loss ✅ PASS
- `src/features.py`: `.dropna()` after lag features at line 80
- Bounded by test contract: `test_only_expected_rows_are_dropped` verifies lost rows ≤ expected warmup + missing days
- **Row loss is bounded**

### Check 5: Unreported Baseline ❌ FINDING IN EXERCISE 4
**File**: `src/train.py`, lines 94-181 (`exercise4_fit_and_score()`)

The `exercise4_fit_and_score()` function reports only `baseline_persistence` and **omits `baseline_mean`**:

```python
results["baseline_persistence"] = {
    "valid": {
        "r2": round(float(r2_score(valid[TARGET], persist_model.predict(valid[cols]))), 4),
        "mae": round(float(mean_absolute_error(valid[TARGET], persist_model.predict(valid[cols]))), 1),
    }
}
```

This violates the CLAUDE.md rule: *"Report the mean baseline **and** the persistence baseline (yesterday's count) alongside every model."*

The main `fit_and_score()` function correctly reports both baselines on valid and test sets, but Exercise 4 does not.

### Minimal Fix
Add `baseline_mean` to `exercise4_fit_and_score()`:

```python
# Add after Approach 1 (persistence baseline):
results["baseline_mean"] = {
    "valid": {
        "r2": round(float(r2_score(valid[TARGET], np.full(len(valid[TARGET]), train[TARGET].mean()))), 4),
        "mae": round(float(mean_absolute_error(valid[TARGET], np.full(len(valid[TARGET]), train[TARGET].mean()))), 1),
    }
}
```
````

### Verdict

I agree with the auditor that Exercise 4 omits the training-mean baseline. I ran python -B -m src.train --exercise4 with scikit-learn 1.9.1 and confirmed that baseline_persistence was the only baseline returned. I reproduced the reported validation R² gain of 0.3894 and MAE reduction of 462.3 rides for transformed HGB. Adding the mean baseline would complete the comparison. The reported improvement over original HGB still holds.