# ML design notes

## Problem
Predict `P(call-out)` for a person entering a specific shift. Binary classification, imbalanced (about 15% positive).

## Features (11)
`is_weekend`, `is_evening`, `is_night`, `is_nurse`, `part_time`, `consecutive_days`, `weekly_hours_before`, `hours_ratio_after`, `rest_hours`, `nights_this_week`, `over_contract`.

All are computed from the schedule itself (`features_for`), so the same code builds training rows, scores the live roster, and answers what-if questions. Features describe the person's state *entering* the shift, using only earlier assignments, so there is no look-ahead leakage.

## Data
`generate_training_data` simulates attendance with assumed drivers (streaks, quick turnarounds under 11h, night and weekend work, overtime) plus Gaussian noise. The label is drawn from a logistic probability, so perfect prediction is impossible by design.

**This is synthetic.** Replace it with real attendance records to get meaningful predictions.

## Method
1. Stratified 75/25 train/test split.
2. Compare logistic regression, random forest, gradient boosting using 3-fold CV ROC-AUC on the training split only.
3. Choose the decision threshold that maximises F1 on cross-validated training predictions.
4. Refit the winner on the full training split, evaluate once on the test split.
5. Permutation importance on the test split (ROC-AUC drop when a feature is shuffled).

Class weights are deliberately not used: probabilities are used directly as risk percentages, and re-weighting would inflate them.

## Typical result (seed 42)
ROC-AUC 0.79, recall 62%, precision 38%, accuracy 78%. Precision is modest because call-outs are rare; the model is used to rank risk and prompt standby cover, not to accuse anyone.

## Risk levels
Low under 12%, medium 12% to 25%, high 25% or more (`LOW_BELOW`, `HIGH_FROM` in `ml_model.py`).

## Ethics and use
Risk scores are for planning cover, not for judging employees. Never use them for discipline or hiring decisions. Check for bias before using real data (for example, part-time or night staff may be flagged more because of their schedules, not their behaviour).
