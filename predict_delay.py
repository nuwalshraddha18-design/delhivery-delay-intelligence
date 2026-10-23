from pathlib import Path
import joblib, pandas as pd
ROOT=Path(__file__).resolve().parent
from src.shipment_prep import clean_shipments, build_features
raw=pd.read_csv(ROOT/'data/new_shipments.csv'); hubs=pd.read_csv(ROOT/'data/hubs.csv')
meddf=pd.read_csv(ROOT/'models/impute_values.csv'); medians=dict(zip(meddf["column"],meddf["median"]))
clean, _=clean_shipments(raw,medians=medians); feat=build_features(clean,hubs)
order=pd.read_csv(ROOT/'models/feature_order.csv').feature.tolist(); model=joblib.load(ROOT/'models/delay_model.pkl')
threshold=float(pd.read_csv(ROOT/'models/delay_threshold.csv').threshold.iloc[0]); seg=pd.read_csv(ROOT/'outputs/hub_segments.csv')[['hub_id','segment']]
feat['delay_probability']=model.predict_proba(feat[order])[:,1].round(4); feat['flagged']=(feat.delay_probability>=threshold).astype(int)
feat=feat.merge(seg,on='hub_id',how='left',validate='many_to_one')
feat['action']=feat.apply(lambda r:'No action' if r.flagged==0 else ('Escalate to hub manager' if r.segment=='Escalate' else 'Proactive customer call'),axis=1)
feat[['shipment_id','hub_id','segment','delay_probability','flagged','action']].to_csv(ROOT/'outputs/new_shipment_actions.csv',index=False,float_format='%.4f')
print(f"Scored {len(feat)} new shipments; flagged {int(feat.flagged.sum())}.")
