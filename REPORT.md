# Bikeshare Demand Forecasting

## Results

We evaluated four models using a chronological split: training
on 2011, validation on the first half of 2012, and testing on
the second half of 2012. The datasets contain 348, 182, and
184 observations, respectively.

| Feature Set | Model | Validation R² | Validation MAE | Test R² | Test MAE |
|---|---|---:|---:|---:|---:|
| Before | Mean baseline | -0.9318 | 2002.2 | -2.1441 | 2847.7 |
| Before | Persistence | 0.4507 | 900.6 | 0.5099 | 840.1 |
| Before | Ridge | 0.6475 | 832.9 | 0.5833 | 947.1 |
| Before | HGB | 0.2185 | 1296.6 | -0.4975 | 1972.6 |
| After | Mean baseline | -0.9318 | 2002.2 | -2.1441 | 2847.7 |
| After | Persistence | 0.4507 | 900.6 | 0.5099 | 840.1 |
| After | Ridge | 0.6439 | 836.5 | 0.5800 | 946.2 |
| After | HGB | 0.1983 | 1311.6 | -0.5338 | 1998.7 |

The added feature, `cnt_roll_3`, represents the average
ridership over the previous three days. We shift the target
by one day before calculating the rolling mean to ensure
that the current day's ridership is excluded.

The feature reduced Ridge's validation R² by 0.0036 and
increased its validation MAE by 3.6 rides. HGB's validation
R² also decreased by 0.0202. Therefore, the validation
results do not support using this feature to improve
predictive performance.

## Headline finding

The original leaky model achieved an R² of approximately
1.000 because casual and registered ridership directly
sum to total ridership. After eliminating leakage and
using a chronological split, Ridge achieved a test R²
of 0.5833, compared with 0.5099 for persistence and
-2.1441 for the mean baseline.

HGB achieved a negative test R² of -0.4975 but still
outperformed the mean baseline. Its weak performance
reflects the difficulty of tree-based models in
extrapolating beyond the demand levels observed during
training.

We retained `cnt_roll_3` in the submitted pipeline to
reproduce the before-and-after comparison, but its
validation performance does not justify recommending
it for deployment.

## Exercise 4: target transformation

We transformed the prediction target from daily ridership
to the difference between today's and yesterday's ridership:

$$
\Delta y_t = y_t-y_{t-1}
$$

Predictions were converted back to daily ride counts by
adding yesterday's observed count:

$$
\hat{y}_t = \widehat{\Delta y_t}+y_{t-1}
$$

The experiment excluded `cnt_roll_3` to maintain a
consistent feature set across the model comparisons.

| Model | Validation R² | Validation MAE | Test R² | Test MAE |
|---|---:|---:|---:|---:|
| Persistence | 0.4507 | 900.6 | 0.5099 | 840.1 |
| Ridge | 0.6475 | 832.9 | 0.5833 | 947.1 |
| Original HGB | 0.2185 | 1296.6 | -0.4975 | 1972.6 |
| Transformed HGB | 0.6079 | 834.3 | 0.5440 | 921.7 |
| Mean baseline | -0.9318 | 2002.2 | -2.1441 | 2847.7 |

The transformation increased HGB's validation R² by
0.3894 and reduced its validation MAE by 462.3 rides.
Although Ridge still performed better on validation,
the transformed HGB became competitive.

We selected the transformation using validation results
and evaluated it on the held-out test set at the end.
Test MAE decreased by approximately 53.3% relative to
the original HGB.

We consider the transformation methodologically defensible
because yesterday's observed ridership is available at
forecast time for one-day-ahead predictions. Predicting
daily changes helps the tree model adapt to growing
ridership without directly extrapolating absolute counts.
This assumption would need reconsideration for multi-day
forecasting, where the preceding day's actual count may
not be available.

## Exercise 5: ridge vs persistence

We would prefer Ridge when the forecasting objective
prioritizes reducing squared prediction errors. Before
feature engineering, Ridge achieved a validation R² of
0.6475 and MAE of 832.9, outperforming persistence on
both validation metrics.

On the test set, Ridge achieved an R² of 0.5833 compared
with persistence's 0.5099. However, its MAE of 947.1 was
higher than persistence's 840.1. Therefore, Ridge is
preferable under a squared-error objective, but it is
not universally superior. If operational costs were
proportional to absolute forecasting errors, persistence
would be a reasonable deployment choice.

## Attribution

The primary coding agent was Claude Code using
`qwen3.5:9b` via Ollama (`bikeshare-coder`), with a
65,536-token context window.

Agent-generated or agent-assisted work:

- Feature implementation and verification tests.
- Target transformation and model evaluation code.
- Reusable feature-ablation skill.
- Assistance with repository audits and documentation.
- ChatGPT assistance with organizing the final report.

Human methodological decisions:

- Maintaining the leakage restrictions and chronological
  data split.
- Adding a three-day rolling mean using strictly
  historical observations.
- Evaluating the new feature using validation metrics.
- Retaining the feature for reproducible comparison
  despite its negative validation delta.
- Predicting daily ridership changes to address the
  growth trend.
- Selecting Ridge based on validation performance while
  acknowledging persistence's advantage in test MAE.