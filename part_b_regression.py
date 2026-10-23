from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
from src.shipment_prep import FEATURES
df=pd.read_csv(ROOT/'data/modelling_table.csv'); X=df[FEATURES]; y=df.delivery_hours
Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=.3,random_state=42)
Xv,Xte,yv,yte=train_test_split(Xtmp,ytmp,test_size=.5,random_state=42)
sc=StandardScaler().fit(Xtr); ztr=sc.transform(Xtr); zte=sc.transform(Xte)
pd.DataFrame([('train_rows',len(Xtr)),('val_rows',len(Xv)),('test_rows',len(Xte)),('train_scaled_max_abs_mean',np.abs(ztr.mean(axis=0)).max()),('test_scaled_max_abs_mean',np.abs(zte.mean(axis=0)).max())],columns=['metric','value']).round(4).to_csv(OUT/'part_b_checks.csv',index=False)
depths=[1,2,3,4,6,8,12,None]; depthrows=[]
for d in depths:
 m=DecisionTreeRegressor(max_depth=d,random_state=42).fit(Xtr,ytr)
 depthrows.append(('unlimited' if d is None else d,mean_squared_error(ytr,m.predict(Xtr))**.5,mean_squared_error(yv,m.predict(Xv))**.5))
pd.DataFrame(depthrows,columns=['max_depth','train_rmse','val_rmse']).round(4).to_csv(OUT/'part_b_tree_depth.csv',index=False)
best_depth=min(depths,key=lambda d:(mean_squared_error(yv,DecisionTreeRegressor(max_depth=d,random_state=42).fit(Xtr,ytr).predict(Xv)),999 if d is None else d))
models={'linear_regression':Pipeline([('scaler',StandardScaler()),('model',LinearRegression())]),'decision_tree':DecisionTreeRegressor(max_depth=best_depth,random_state=42),'random_forest':RandomForestRegressor(n_estimators=100,random_state=42),'gradient_boosting':GradientBoostingRegressor(n_estimators=100,learning_rate=.1,random_state=42)}
def metricrow(name,split,truth,pred):
 mse=mean_squared_error(truth,pred); return (name,split,mean_absolute_error(truth,pred),mse,mse**.5,r2_score(truth,pred))
rows=[metricrow('mean_baseline','val',yv,np.repeat(ytr.mean(),len(yv)))]
for name,m in models.items():
 m.fit(Xtr,ytr); rows.append(metricrow(name,'val',yv,m.predict(Xv)))
pd.DataFrame(rows,columns=['model','split','MAE','MSE','RMSE','R2']).round(4).to_csv(OUT/'part_b_metrics.csv',index=False)
lrrows=[]
for lr in [.01,.1,1.0]:
 m=GradientBoostingRegressor(n_estimators=100,learning_rate=lr,random_state=42).fit(Xtr,ytr); p=m.predict(Xv)
 lrrows.append((lr,mean_squared_error(yv,p)**.5,r2_score(yv,p)))
pd.DataFrame(lrrows,columns=['learning_rate','val_rmse','val_r2']).round(4).to_csv(OUT/'part_b_gb_learning_rate.csv',index=False)
rank=sorted(models,key=lambda n:mean_squared_error(yv,models[n].predict(Xv)))[:2]
Xtv=pd.concat([Xtr,Xv]); ytv=pd.concat([ytr,yv]); cv=KFold(n_splits=5,shuffle=True,random_state=42); cvrows=[]
for name in rank:
 m=models[name]; scores=cross_val_score(m,Xtv,ytv,cv=cv)
 cvrows.append((name,scores.mean(),scores.std()))
pd.DataFrame(cvrows,columns=['model','mean_r2','std_r2']).round(4).to_csv(OUT/'part_b_cv.csv',index=False)
win=min(models,key=lambda n:mean_squared_error(yv,models[n].predict(Xv)))
rows.append(metricrow(win,'test',yte,models[win].predict(Xte)))
pd.DataFrame(rows,columns=['model','split','MAE','MSE','RMSE','R2']).round(4).to_csv(OUT/'part_b_metrics.csv',index=False)
print('Part B complete. Validation winner:',win,'depth:',best_depth)
