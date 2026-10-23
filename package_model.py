from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
ROOT=Path(__file__).resolve().parent; (ROOT/'models').mkdir(exist_ok=True); OUT=ROOT/'outputs'
from src.shipment_prep import FEATURES
cfg=pd.read_csv(OUT/'part_c_final_config.csv').iloc[0]; df=pd.read_csv(ROOT/'data/modelling_table.csv'); X=df[FEATURES]; y=df.is_delayed
Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=.3,random_state=42,stratify=y)
Xv,Xte,yv,yte=train_test_split(Xtmp,ytmp,test_size=.5,random_state=42,stratify=ytmp)
md=None if str(cfg.max_depth)=='unlimited' or pd.isna(cfg.max_depth) else int(cfg.max_depth)
model=Pipeline([('selector',SelectKBest(f_classif,k=int(cfg.selector_k))),('model',RandomForestClassifier(n_estimators=int(cfg.n_estimators),max_depth=md,min_samples_leaf=int(cfg.min_samples_leaf),random_state=42))]).fit(Xtr,ytr)
joblib.dump(model,ROOT/'models/delay_model.pkl')
pd.DataFrame({'threshold':[float(cfg.threshold)]}).to_csv(ROOT/'models/delay_threshold.csv',index=False)
pd.DataFrame({'feature':FEATURES}).to_csv(ROOT/'models/feature_order.csv',index=False)
raw=pd.read_csv(ROOT/'data/raw_shipments.csv'); from src.shipment_prep import clean_shipments
_, medians=clean_shipments(raw); pd.DataFrame(medians.items(),columns=['column','median']).to_csv(ROOT/'models/impute_values.csv',index=False,float_format='%.4f')
loaded=joblib.load(ROOT/'models/delay_model.pkl'); a=model.predict_proba(Xte)[:,1]; b=loaded.predict_proba(Xte)[:,1]
pd.DataFrame([('test_rows_scored',len(Xte)),('reload_max_abs_probability_diff',float(np.max(np.abs(a-b)))),('threshold',float(cfg.threshold))],columns=['metric','value']).round(4).to_csv(OUT/'part_d_reload_check.csv',index=False)
print('Model packaged and reload checked')
