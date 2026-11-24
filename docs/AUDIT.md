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