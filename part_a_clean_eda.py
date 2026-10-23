from pathlib import Path
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features, FEATURES, PROMISED_HOURS

ROOT = Path(__file__).resolve().parent
for folder in ["outputs", "models"]: (ROOT / folder).mkdir(exist_ok=True)
raw = pd.read_csv(ROOT / "data/raw_shipments.csv")
hubs = pd.read_csv(ROOT / "data/hubs.csv")
raw_rows = len(raw)
dedup = raw.drop_duplicates().copy()
spell_before = int(dedup["service_type"].nunique())
missing_distance = int(dedup["distance_km"].isna().sum())
missing_weight = int(dedup["parcel_weight_kg"].isna().sum())
missing_value = int(dedup["declared_value"].isna().sum())
missing_pay = int(dedup["payment_mode"].isna().sum())
sentinel = int((dedup["parcel_weight_kg"] > 100).sum())
clean, medians = clean_shipments(raw)
features = build_features(clean, hubs)
spell_after = int(features["service_type"].nunique())
features["is_delayed"] = (features["delivery_hours"] > features["promised_hours"]).astype(int)
features.to_csv(ROOT / "data/modelling_table.csv", index=False)

v = features["declared_value_inr"]
logv = np.log(v)
q1, q3 = features["delivery_hours"].quantile([0.25, 0.75])
iqr = q3-q1
lower, upper = q1-1.5*iqr, q3+1.5*iqr
metrics = [
("raw_rows", raw_rows), ("duplicate_rows_removed", raw_rows-len(dedup)), ("rows_after_dedup", len(dedup)),
("missing_distance_km_before", missing_distance), ("missing_parcel_weight_before", missing_weight),
("missing_declared_value_before", missing_value), ("missing_payment_mode_before", missing_pay),
("sentinel_weight_rows", sentinel), ("service_type_spellings_before", spell_before),
("service_type_spellings_after", spell_after), ("median_distance_km", medians["distance_km"]),
("median_parcel_weight_kg", medians["parcel_weight_kg"]), ("median_declared_value_inr", medians["declared_value_inr"]),
("rows_after_hub_merge", len(features)), ("nulls_in_features_and_target", int(features[FEATURES+["delivery_hours","is_delayed"]].isna().sum().sum())),
("delayed_count", int(features["is_delayed"].sum())), ("delay_rate", features["is_delayed"].mean()),
("hours_q1", q1), ("hours_q3", q3), ("hours_lower_bound", lower), ("hours_upper_bound", upper),
("hours_outlier_count", int(((features["delivery_hours"]<lower)|(features["delivery_hours"]>upper)).sum())),
("declared_value_mean_inr", v.mean()), ("declared_value_median_inr", v.median()),
("log_declared_value_mean", logv.mean()), ("log_declared_value_median", logv.median()),
]
pd.DataFrame(metrics, columns=["metric","value"]).round(4).to_csv(ROOT/"outputs/part_a_checks.csv",index=False)
rows=[]
for dim, col in [("service_type","service_type"),("hub_tier","hub_tier")]:
    for group, g in features.groupby(col, sort=True):
        rows.append((dim, group, len(g), int(g.is_delayed.sum()), g.is_delayed.mean(), g.delivery_hours.mean()))
features["distance_band"] = pd.cut(features.distance_km, bins=[0,200,400,600,2000], labels=["0-200","201-400","401-600","601+"], right=True, include_lowest=True)
for group, g in features.groupby("distance_band", observed=False, sort=True):
    rows.append(("distance_band", group, len(g), int(g.is_delayed.sum()), g.is_delayed.mean(), g.delivery_hours.mean()))
pd.DataFrame(rows, columns=["dimension","group","shipments","delayed","delay_rate","mean_hours"]).round(4).to_csv(ROOT/"outputs/part_a_delay_by_group.csv",index=False)
features.select_dtypes(include=[np.number]).corr().round(4).rename_axis("column").to_csv(ROOT/"outputs/part_a_correlations.csv",index=True)
pd.DataFrame(medians.items(),columns=["column","median"]).round(4).to_csv(ROOT/"models/impute_values.csv",index=False)
print(f"Part A complete: {len(features)} shipments; delayed={features.is_delayed.sum()}; rate={features.is_delayed.mean():.4f}")
