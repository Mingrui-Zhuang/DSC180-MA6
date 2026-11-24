# Decisions

One line per methodological decision: what you chose, why, and what it changed.
Agents append here via the `/new-feature` skill; you append here by hand.

- Exclude `casual`, `registered`: they sum to `cnt` exactly. Leaky model R^2 1.000 -> honest ridge 0.583.
- Exclude `cnt_2d_bfr`: a shipped 2-day lag we cannot test the alignment of. Build our own lags instead.
- Exclude `instant`, `yr`, `days_since_2011`: time indexes let a tree predict a flat constant past 2011 and a linear model extrapolate a trend we have no grounds for.
- Split 2011 / H1 2012 / H2 2012 by date. Test set touched once, after all feature work.
- Report mean and persistence baselines with every score. Persistence (test R^2 0.510, MAE 840) is within 0.07 R^2 of ridge and beats it on MAE; that caveat goes in the report.

- 2026-11-23 — Tested predicting daily ridership changes with HGB
  to address the growth trend. Validation R² improved from 0.2185
  to 0.6079 and MAE decreased from 1296.6 to 834.3. We consider
  this transformation defensible for one-day-ahead forecasting.
  Ridge remains our preferred model based on validation.

- 2026-11-24 — Added `cnt_roll_3`, the mean ridership over the previous three days. Shifting by one day ensures that all observations are available at forecast time. Ridge validation R² decreased from 0.6475 to 0.6439, and HGB validation R² decreased from 0.2185 to 0.1983.

- 2026-11-24 — Retained `cnt_roll_3` in the submitted pipeline for reproducible before-and-after comparison, despite its negative validation results. Excluded it from Exercise 4 to maintain a consistent 14-feature comparison.

- 2026-11-24 — Prefer Ridge when minimizing squared prediction errors based on validation performance, while acknowledging that persistence achieves lower test MAE.

- 2026-11-24 — Updated `CLAUDE.md` to enforce strictly historical inputs for `cnt_roll_3` and document the `/ablation` skill, which restricts feature comparisons to validation data.