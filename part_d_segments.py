from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
df=pd.read_csv(ROOT/'data/modelling_table.csv'); hubs=pd.read_csv(ROOT/'data/hubs.csv')
profile=df.groupby('hub_id',sort=True).agg(shipments=('shipment_id','count'),mean_delivery_hours=('delivery_hours','mean'),delay_rate=('is_delayed','mean'),mean_distance_km=('distance_km','mean'),mean_n_stops=('n_stops','mean')).reset_index()
profile=profile.round(4)
profile.to_csv(OUT/'hub_profile.csv',index=False,float_format='%.4f')
cols=['mean_delivery_hours','delay_rate','mean_distance_km','mean_n_stops']; z=StandardScaler().fit_transform(profile[cols])
sep=[]
for k in [2,3,4,5]:
 labels=KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(z)
 same=[]; across=[]
 for i in range(len(z)):
  for j in range(i+1,len(z)):
   dist=float(np.linalg.norm(z[i]-z[j]))
   (same if labels[i]==labels[j] else across).append(dist)
 sizes=sorted(np.bincount(labels).tolist(),reverse=True)
 sep.append((k,np.mean(across)/np.mean(same),'-'.join(map(str,sizes))))
pd.DataFrame(sep,columns=['K','across_over_within','cluster_sizes']).round(4).to_csv(OUT/'part_d_separation.csv',index=False)
labels=KMeans(n_clusters=3,random_state=42,n_init=10).fit_predict(z)
prof=profile.copy(); prof['cluster']=labels
order=prof.groupby('cluster').delay_rate.mean().sort_values().index.tolist(); names={order[0]:'Standard',order[1]:'Watch',order[2]:'Escalate'}
prof['segment']=prof.cluster.map(names)
seg=prof.merge(hubs[['hub_id','hub_city','hub_tier']],on='hub_id',how='left')
seg[['hub_id','segment','hub_city','hub_tier']].sort_values('hub_id').to_csv(OUT/'hub_segments.csv',index=False)
sp=prof.groupby('segment').agg(hubs=('hub_id','count'),mean_delivery_hours=('mean_delivery_hours','mean'),delay_rate=('delay_rate','mean'),mean_distance_km=('mean_distance_km','mean'),mean_n_stops=('mean_n_stops','mean')).reindex(['Standard','Watch','Escalate']).reset_index()
sp.round(4).to_csv(OUT/'segment_profile.csv',index=False)
expected={1:'Standard',2:'Watch',3:'Escalate'}
mis=seg[seg.apply(lambda r:r.segment!=expected[r.hub_tier],axis=1)][['hub_id','segment','hub_city','hub_tier']].sort_values('hub_id')
mis.to_csv(OUT/'tier_vs_segment_mismatch.csv',index=False)
print('Part D segmentation complete')
