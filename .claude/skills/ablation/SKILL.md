---
name: ablation
description: Measure the effect of removing a specified feature on validation performance.
---

# Feature ablation

Feature to evaluate: $ARGUMENTS

1. Read src/features.py and src/train.py. Verify that the specified
   feature exists in the permitted feature set.
2. Construct two feature sets, one with the specified feature and
   one without it.
3. Use identical training and validation rows, model hyperparameters,
   and random seeds.
4. Train on 2011 and evaluate on January–June 2012 only.
5. Report validation R² and MAE for ridge, gradient boosting,
   and both required baselines.
6. Calculate metric changes as with-feature minus without-feature.
7. Explain whether retaining the feature is justified by the results.

Preserve the raw data, temporal split, and default metrics report.
Do not evaluate the held-out test set.
