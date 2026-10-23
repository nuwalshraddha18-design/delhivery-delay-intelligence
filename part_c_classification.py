from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
from src.shipment_prep import FEATURES, NOISE_COLUMNS
df=pd.read_csv(ROOT/'data/modelling_table.csv'); X=df[FEATURES]; y=df.is_delayed
Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=.3,random_state=42,stratify=y)
Xv,Xte,yv,yte=train_test_split(Xtmp,ytmp,test_size=.5,random_state=42,stratify=ytmp)
def metrics(name,truth,pred):
 tn,fp,fn,tp=confusion_matrix(truth,pred,labels=[0,1]).ravel()
 return (name,tn,fp,fn,tp,accuracy_score(truth,pred),precision_score(truth,pred,zero_division=0),recall_score(truth,pred,zero_division=0),f1_score(truth,pred,zero_division=0))
rows=[metrics('blind_all_on_time',yv,np.zeros(len(yv),dtype=int))]
log=Pipeline([('scaler',StandardScaler()),('model',LogisticRegression(random_state=42,max_iter=2000))]).fit(Xtr,ytr); rows.append(metrics('logistic_regression',yv,log.predict(Xv)))
depths=[2,3,4,5,6,8,10]; tree_models=[]
for d in depths:
 m=DecisionTreeClassifier(criterion='gini',max_depth=d,random_state=42).fit(Xtr,ytr); tree_models.append((d,m,f1_score(yv,m.predict(Xv))))
bestd=sorted(tree_models,key=lambda z:(-z[2],z[0]))[0][0]; tree=dict((d,m) for d,m,_ in tree_models)[bestd]; rows.append(metrics(f'decision_tree_depth_{bestd}',yv,tree.predict(Xv)))
rf17=RandomForestClassifier(n_estimators=100,random_state=42).fit(Xtr,ytr); rows.append(metrics('random_forest_default',yv,rf17.predict(Xv)))
pd.DataFrame(rows,columns=['model','TN','FP','FN','TP','accuracy','precision','recall','f1']).round(4).to_csv(OUT/'part_c_models.csv',index=False)
selector=SelectKBest(f_classif,k='all').fit(Xtr,ytr); scores=selector.scores_; ranks=pd.Series(scores,index=FEATURES).rank(method='min',ascending=False).astype(int)
score_df=pd.DataFrame({'feature':FEATURES,'f_score':scores,'f_rank':[ranks[f] for f in FEATURES],'rf_importance':rf17.feature_importances_,'is_noise':[f in NOISE_COLUMNS for f in FEATURES]})
score_df.round(4).to_csv(OUT/'part_c_feature_scores.csv',index=False)
selectrows=[]
for k in [5,7,10,12,17]:
 m=Pipeline([('selector',SelectKBest(f_classif,k=k)),('model',RandomForestClassifier(n_estimators=100,random_state=42))]).fit(Xtr,ytr)
 selectrows.append((k,f1_score(yv,m.predict(Xv))))
pd.DataFrame(selectrows,columns=['k','val_f1']).round(4).to_csv(OUT/'part_c_selectk.csv',index=False)
sel=SelectKBest(f_classif,k=7).fit(Xtr,ytr); A=sel.transform(Xtr); B=sel.transform(Xv)
param_grid={'n_estimators':[50,100,150],'max_depth':[4,6,8],'min_samples_leaf':[1,5,10]}
grid=GridSearchCV(RandomForestClassifier(random_state=42),param_grid,cv=5).fit(A,ytr)
rand=RandomizedSearchCV(RandomForestClassifier(random_state=42),param_grid,n_iter=9,cv=5,random_state=42).fit(A,ytr)
def valf(est): return f1_score(yv,est.predict(B))
def paramsstr(p): return '; '.join(f'{k}={v}' for k,v in sorted(p.items()))
rows_t=[]
rows_t.append(('grid_search',27,5,135,paramsstr(grid.best_params_),grid.best_score_,valf(grid.best_estimator_)))
rows_t.append(('randomized_search',9,5,45,paramsstr(rand.best_params_),rand.best_score_,valf(rand.best_estimator_)))
deforest=RandomForestClassifier(n_estimators=100,random_state=42).fit(A,ytr)
rows_t.append(('default_forest_same_7_features',1,0,1,'max_depth=None; min_samples_leaf=1; n_estimators=100',np.nan,valf(deforest)))
pd.DataFrame(rows_t,columns=['search','candidates','cv_folds','total_fits','best_params','best_cv_accuracy','val_f1']).round(4).to_csv(OUT/'part_c_tuning.csv',index=False)
tune=pd.DataFrame(rows_t,columns=['search','candidates','cv_folds','total_fits','best_params','best_cv_accuracy','val_f1'])
# Tie priority: default forest, then grid, then randomized.
priority={'default_forest_same_7_features':0,'grid_search':1,'randomized_search':2}
chosen=sorted(rows_t,key=lambda r:(-r[6],priority[r[0]]))[0][0]
paramdict={'default_forest_same_7_features':{},'grid_search':grid.best_params_,'randomized_search':rand.best_params_}[chosen]
final=Pipeline([('selector',SelectKBest(f_classif,k=7)),('model',RandomForestClassifier(random_state=42,**paramdict))]).fit(Xtr,ytr)
probs=final.predict_proba(Xv)[:,1]; thresholds=[.1,.2,.3,.4,.5,.6,.7,.8,.9]; thrrows=[]
for t in thresholds: thrrows.append((t,*metrics('x',yv,(probs>=t).astype(int))[1:]))
thdf=pd.DataFrame(thrrows,columns=['threshold','TN','FP','FN','TP','accuracy','precision','recall','f1']); thdf.round(4).to_csv(OUT/'part_c_thresholds.csv',index=False)
eligible=thdf[thdf.recall>=.80]; threshold=float(eligible.threshold.max()) if len(eligible) else .1
pt=final.predict_proba(Xte)[:,1]; testrow=metrics('test',yte,(pt>=threshold).astype(int))
pd.DataFrame([(threshold,*testrow[1:])],columns=['threshold','TN','FP','FN','TP','accuracy','precision','recall','f1']).round(4).to_csv(OUT/'part_c_final_test.csv',index=False)
mp=paramdict.get('n_estimators',100); md=paramdict.get('max_depth',None); ml=paramdict.get('min_samples_leaf',1)
pd.DataFrame([{'selector_k':7,'n_estimators':mp,'max_depth':'unlimited' if md is None else md,'min_samples_leaf':ml,'threshold':threshold,'chosen_from':chosen}]).to_csv(OUT/'part_c_final_config.csv',index=False)
print('Part C complete. chosen:',chosen,'threshold:',threshold)
