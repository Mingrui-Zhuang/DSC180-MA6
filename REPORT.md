## Exercise 4: target transformation

We transformed the prediction target from daily ridership to the
difference between today's and yesterday's ridership. Predictions
were converted back to daily ride counts by adding yesterday's
observed count.

| Model | Validation R² | Validation MAE |
|---|---:|---:|
| Persistence | 0.4507 | 900.6 |
| Ridge | 0.6475 | 832.9 |
| Original HGB | 0.2185 | 1296.6 |
| Transformed HGB | 0.6079 | 834.3 |

The transformation increased HGB's validation R² by 0.3894 and
reduced its MAE by 462.3 rides. Although ridge still performed
better, the transformed HGB became competitive.

We consider the transformation methodologically defensible
because yesterday's observed ridership is available at forecast
time for one-day-ahead predictions. Predicting daily changes
allows the tree model to adapt to the growing ridership level
without requiring direct extrapolation of absolute counts.