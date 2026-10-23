# Delhivery Last-Mile Delay Intelligence Pipeline

**Program:** Artificial Intelligence and Machine Learning Certification Program · E&ICT Academy, IIT Roorkee  
**Purpose:** Use synthetic parcel data to estimate delivery time, identify likely late deliveries, segment hubs by observed operating behaviour, and produce an action list for new shipments.

> All shipment data in this project is synthetic. It is not real Delhivery customer or shipment data. Amounts, where shown, are in Indian Rupees (INR).

## How to run

Run these commands from the repository root in a terminal with Python installed. They are listed in execution order:

```bash
python -m venv .venv
```

Activate the environment (Windows PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

On macOS/Linux, activate it with `source .venv/bin/activate` instead. Then run:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

`generate_data.py` creates the three synthetic input files in `data/`. Part A creates the modelling table and data-quality outputs. Parts B and C train and compare models; Part D segments the hubs. `package_model.py` **regenerates `models/delay_model.pkl`** and the companion model metadata files. `predict_delay.py` loads those artefacts and writes `outputs/new_shipment_actions.csv`.

The exact versions used for this repository are pinned in `requirements.txt`. A pickle is version-sensitive: `models/delay_model.pkl` is expected to load reliably with the scikit-learn version pinned there. If a grading environment has different library versions, rerun `package_model.py` in that environment rather than relying on the committed pickle. The scripts are deterministic with the specified random seeds.

## Data findings

- **Data quality:** The raw file has 1,224 rows. Exact duplicate removal removes 24, leaving 1,200 unique shipments. The data contains 36 missing distances, 24 missing weights, 12 missing declared values, 18 missing payment modes, and 5 scanner-fault weight sentinels. Service-type spellings fall from 12 distinct forms to 3 after normalization. These counts are in `outputs/part_a_checks.csv`.
- **Overall delay:** 278 of 1,200 shipments are delayed, a delay rate of **0.2317** (`outputs/part_a_checks.csv`). The modelling table has 1,200 rows and zero missing cells across the required 17 features and the target columns.
- **Service type:** Express is delayed most often: **119 of 243**, or **0.4897**. Standard is **132 of 642 (0.2056)**, while Economy is **27 of 315 (0.0857)** (`outputs/part_a_delay_by_group.csv`).
- **Hub tier:** The delay rate rises from **0.0400 for Tier 1** to **0.2150 for Tier 2** and **0.4400 for Tier 3** (`outputs/part_a_delay_by_group.csv`). The tier label is therefore useful as a broad signal, but Part D finds that four hubs do not behave like their assigned tier.
- **Distance:** The delay rate rises with distance: **0.0219** for 0–200 km, **0.1368** for 201–400 km, **0.4364** for 401–600 km, and **0.8523** for 601+ km (`outputs/part_a_delay_by_group.csv`).
- **Strongest correlations with delivery hours:** Ranking the numeric correlation matrix by absolute correlation and excluding `delivery_hours` itself, the three strongest are `promised_hours` (**0.8372**), `distance_km` (**0.4013**), and `n_stops` (**0.3247**). Signed values are copied from `outputs/part_a_correlations.csv`.
- **Declared value transform:** The median-filled `declared_value_inr` has mean **INR 1,533.0542** and median **INR 1,121.5000**, while its natural-log transform has mean **7.0277** and median **7.0224** (`outputs/part_a_checks.csv`). The log transform compresses the strongly right-skewed value range so a small number of high-value parcels have less leverage on a linear model.

## Model results (regression)

- The mean-only validation baseline has **R² = -0.0212** (`outputs/part_b_metrics.csv`). A negative R² means this constant prediction is worse than using the validation set's own mean as a reference; it does not capture the meaningful differences between parcels.
- **Linear regression** wins on validation with **RMSE 4.8912 hours** and **R² 0.9313**. Gradient boosting is close at RMSE **4.9165 hours** and R² **0.9306** (`outputs/part_b_metrics.csv`). The final test evaluation for linear regression is RMSE **4.5235 hours** and R² **0.9493**.
- The depth-1 tree has high bias: train RMSE **11.6334** and validation RMSE **12.1324**. Depths 4–6 improve the fit; depth 6 has the lowest validation RMSE (**6.2796**). The unlimited tree has train RMSE **0.0000** but validation RMSE **7.8368**, a large generalization gap and a clear high-variance warning (`outputs/part_b_tree_depth.csv`).
- Learning rate **0.1** is best among the tested gradient-boosting rates (validation R² **0.9306**). Rate **0.01** is too slow for 100 estimators (R² **0.7127**); rate **1.0** overshoots and generalizes worse (R² **0.8440**) (`outputs/part_b_gb_learning_rate.csv`).
- Five-fold cross-validation on the training-plus-validation rows gives linear regression mean R² **0.9261** (standard deviation **0.0048**) and gradient boosting **0.9353** (standard deviation **0.0045**) (`outputs/part_b_cv.csv`). This slightly favours gradient boosting across folds, but linear regression is the selected model because the stated selection rule is lowest validation RMSE; its validation score is almost identical and its model is simpler to explain. Test data is not used to choose the model.

## Model results (classification)

- The blind all-on-time baseline has accuracy **0.7667** but recall and F1 of **0.0000**: it catches none of the delayed parcels, so its accuracy is misleading in an operational setting (`outputs/part_c_models.csv`).
- The 17-feature forest assigns a combined importance of **0.2245** to the five bookkeeping/noise columns. Four of those noise columns have F-ranks outside the top seven, but `label_print_batch` ranks **8**, so **all five are outside the top seven**. The top seven F-score features are distance, hub tier, hub daily capacity, stops, promised hours, rain and sale-event flag (`outputs/part_c_feature_scores.csv`).
- The validation F1 scores for k = 5, 7, 10, 12, 17 are **0.6988, 0.8333, 0.7857, 0.8049, 0.7848**. Seven features work best on this split, and adding weaker features does not improve the result (`outputs/part_c_selectk.csv`).
- On the same seven selected features, the untuned forest reaches validation F1 **0.8333**, higher than grid search (**0.7750**) and randomized search (**0.7848**) (`outputs/part_c_tuning.csv`). This illustrates that tuning can overfit cross-validation or fail to improve a small validation sample of 180 parcels; tuning should not be assumed to help automatically.
- The selected alert threshold is **0.5**. At this threshold validation precision is **0.7955** and recall is **0.8333**: it captures at least 80% of delayed validation parcels while limiting unnecessary alerts. The rule is to use the highest threshold whose validation recall is at least 0.80 (`outputs/part_c_thresholds.csv`).
- On the held-out test set, precision is **0.6889**, recall **0.7561**, and F1 **0.7209** (`outputs/part_c_final_test.csv`). Test recall is lower than validation recall; this is a real difference, not something to conceal. There are only 41 delayed parcels in the test set, so each missed parcel changes recall by about 1/41, or 2.44 percentage points.

## Operations recommendation

- Use the **Escalate** segment for manager attention. Its nine hubs have mean hub-level delay rate **0.4600** (`outputs/segment_profile.csv`). The hub IDs in this segment are listed in `outputs/hub_segments.csv`; operationally, the highest-delay hubs should receive capacity, route and dispatch-process reviews first.
- The tier/segment mismatches are **HUB07 Kolkata (Tier 1, Watch)**, **HUB13 Nagpur (Tier 2, Escalate)**, **HUB15 Kochi (Tier 2, Escalate)**, and **HUB22 Dehradun (Tier 3, Watch)** (`outputs/tier_vs_segment_mismatch.csv`). Their current tier labels should remain administrative descriptors, not be treated as a reliable substitute for observed operating performance. In particular, HUB13 and HUB15 deserve escalation playbooks despite Tier 2 labels; HUB22 should be managed like Watch rather than automatically treated as Escalate.
- Use alert threshold **0.5**. Test precision **0.6889** means roughly **6.9 out of every 10 flagged parcels** are truly delayed; test recall **0.7561** means roughly **7.6 out of every 10 delayed parcels** are caught (`outputs/part_c_final_test.csv`). These are estimates from a small synthetic test set, not guarantees for live operations.
- Of the 48 new parcels, **8 are flagged** and **6 of those 8** are assigned “Escalate to hub manager”; the other two flagged parcels receive a proactive customer call. The row-by-row decisions are in `outputs/new_shipment_actions.csv`.
- Keep **K = 3** because operations requires exactly three playbooks: Standard, Watch and Escalate. The separation ratio rises from **2.894 for K = 2** to **5.3928 for K = 3**, then to **6.3047 for K = 4** and **6.8866 for K = 5** (`outputs/part_d_separation.csv`). The larger ratios for K = 4 and K = 5 come with clusters smaller than six hubs, too small to support a stable standalone playbook. The names are assigned by ascending cluster delay rate, not by arbitrary K-Means cluster number.

## Production plan

### Readiness mapped to this repository

- **Accuracy:** regression MAE/RMSE/R², classifier precision/recall/F1 and the one-time test outputs are saved as CSV files under `outputs/`.
- **Reliability:** the final model is tested on a held-out set; `outputs/part_d_reload_check.csv` confirms that the saved and reloaded model probabilities match exactly in this run.
- **Reproducibility:** fixed random seeds are specified in the scripts, generated data is committed, package versions are pinned, and the README gives the run order.
- **Performance:** the final scoring script processes all 48 sample parcels in one batch and writes one CSV action list.
- **Maintainability:** cleaning and feature engineering are centralized in `src/shipment_prep.py`, instead of duplicated between training and scoring.
- **Consistency:** training and prediction use the same `clean_shipments` and `build_features` functions, stored imputation medians, and saved feature order.
- **Monitorability:** outputs are explicit CSV artefacts; a production deployment should retain predicted probabilities, flags, segment, eventual delivery outcome and decision timestamps for monitoring.

### Suggested monitoring triggers

These are proposed initial operational guardrails, not statistically established limits. Review them with the operations owner before live use.

1. **Weekly delayed share among delivered parcels:** investigate if it exceeds **30%** for two consecutive weeks, comparing equivalent service mix and hub mix.
2. **Share of dispatched parcels flagged:** investigate if it rises above **25%** or changes by more than **10 percentage points** week over week; first check for changes in volume, service mix and sale activity.
3. **Recall on parcels whose outcome is known:** investigate and recalibrate if rolling four-week recall falls below **0.70**. Track precision as well so a recall improvement does not create an unmanageable call volume.
4. **Feature drift:** investigate if the weekly mean distance or stop count shifts by more than **20%** from the training reference, or if any feature has more than **5%** missing values before imputation.

### Retraining and sale events

Re-evaluate performance monthly and retrain quarterly, or sooner if a monitoring trigger is breached and investigation confirms meaningful drift. Keep a time-based holdout for evaluation before deploying a replacement model. Treat the 15–28 September sale window as a special-event period: compare sale weeks with the same type of sale period in another cycle, not only with ordinary weeks. Drift here could appear as a changed distance/stops distribution, different service mix, altered delay rate for Express during sale periods, or a change in calibration between predicted probabilities and observed delay outcomes.

## Limitations

1. **Synthetic data:** the relationships are generated by a known simulation and do not establish real-world Delhivery performance or prove that the same features are available or reliable in production.
2. **Small test set:** only 180 parcels and 41 delayed outcomes are in the classification test split. Precision and recall can move materially when only a few predictions change.
3. **Single split and threshold:** the alert threshold was selected using one 180-row validation set. Temporal validation and repeated backtesting would provide stronger evidence before operational use.
4. **Noise features:** bookkeeping fields have non-zero random-forest importance despite being intended as noise. This demonstrates why feature importance should not be interpreted as causal evidence.
5. **Hub segmentation is descriptive:** K-Means uses only four aggregate profile values and does not account for uncertainty, seasonality or changing volumes. Segments should be reviewed periodically with hub managers.
6. **Operational costs are simplified:** the threshold rule prioritizes recall but does not explicitly optimize the monetary costs of missed delays, customer calls or manager escalations.
