
import os
import pickle
import importlib

import pandas as pd
import numpy as np

import cufflinks as cf
import matplotlib.ticker as tkr
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-white')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 100)
cf.set_config_file(theme='white',dimensions=(650,450))
cf.go_offline()


sys.path.insert(1, '../../Datasets/')

# String
import StringUtils 
importlib.reload(StringUtils)

import PlotUtils 
importlib.reload(PlotUtils)

import SupervisedUtils 
importlib.reload(SupervisedUtils)

import UnsupervisedUtils 
importlib.reload(SupervisedUtils)

def save_object(obj,name):
  with open(name + '.pkl', 'wb') as file:
    pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)

def load_object(name):
    with open(name, 'rb') as file:
        return(pickle.load(file))

TMDb = pd.read_feather('../M1/DBM1')
TMDb.head(2)

ratings = pd.read_feather('../../Datasets/MovieLens/ratings')
ratings.head(2)
ratings.drop(columns=['timestamp'],inplace=True)
ratings.columns = ['userId','id','rating']

print(f'Hay un total de {len(ratings["userId"].value_counts())} usuarios')
print(f'Hay un total de {len(ratings["id"].value_counts())} peliculas calificadas')
PlotUtils.hist_box(ratings,'rating',ctitle = 'Distribución de las calificaciones de los usuarios')

movies = TMDb.copy()
movies.drop(columns=['overview','month','day','budget','revenue','status','tagline','vote_count','keywords','poster_path','backdrop_path'],inplace=True)
movies = movies.loc[movies['genres'].isna()==False].reset_index(drop=True)
movies = movies.loc[movies['runtime'].isna()==False].reset_index(drop=True)
movies = movies.loc[movies['year'].isna()==False].reset_index(drop=True)
movies = movies.loc[movies['vote_average']>0].reset_index(drop=True)
print(movies.shape)
movies.head(2)

movies.isna().sum().to_frame().T
movies = movies.fillna(0)

counts = movies['genres'].str.split('-').explode().value_counts(True)
mask = counts>0.05
genres_mask = counts[mask].index
genresaux = movies['genres'].str.split('-').explode().reset_index()
genresaux.loc[~genresaux['genres'].isin(genres_mask),'genres'] = 'OTROS'
genresaux = genresaux.reset_index().pivot_table(index='index',columns='genres',values='level_0',aggfunc='count',fill_value=0)
genresaux.columns = ['genre_' + col for col in genresaux.columns]
genresaux = genresaux.fillna(0)
movies = pd.concat([movies,genresaux],axis=1).drop(columns = ['genres'])
movies.head(2)

counts = movies['original_language'].value_counts(True)
mask = counts>0.05
language_mask = counts[mask].index
movies.loc[~movies['original_language'].isin(language_mask),'original_language'] = 'OTROS'
movies = pd.get_dummies(movies,columns = ['original_language'])
movies.head(2)

ratings['rating'] = ratings['rating']/10
ratings['rating'] = ratings['rating'].apply(lambda x: 1 if x>0.7 else 0)
ratings.head(2)

TAD = ratings.merge(movies,on='id')
TAD.shape

genres_cols = [col for col in TAD.columns if 'genre_' in col]
language_cols = [col for col in TAD.columns if 'original_language_' in col]

for v in genres_cols:
    TAD[v] = np.multiply(TAD['rating'],TAD[v])

for v in language_cols:
    TAD[v] = np.multiply(TAD['rating'],TAD[v])

TAD.head()

feat_op= {'id':['count'],
         'popularity': ['median'],
         'year': ['median'],
         'runtime':['median'],
         'vote_average' : ['median']
         }

TAD = TAD.groupby('userId').agg(feat_op)

TAD_columns = [key + '_' + str(j) for key in feat_op.keys() for j in feat_op.get(key)]
TAD_columns = list(map(lambda x: x.replace('id_count','n'),TAD_columns))
TAD.columns=TAD_columns
TAD.head(2)

TAD.shape

X = TAD.copy()
X

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
X=scaler.transform(X)
pd.DataFrame(X)

n_comp_wb=3
pca_f = UnsupervisedUtils.PCA(n_components=n_comp_wb,random_state=1610)
pca_f.fit(X)
X_PCA = pca_f.transform(X)
X_PCA =  pd.DataFrame(X_PCA)
X_PCA.columns = ['C' + str(x +1)  for x in range(n_comp_wb)]

mds = UnsupervisedUtils.MDS(n_components=3,random_state=1610)
X_MDS = mds.fit_transform(X)
X_MDS =  pd.DataFrame(X_MDS)
X_MDS.columns = ['V1','V2','V3']

plt.figure(figsize=(12,7))
sns.scatterplot(x=X_PCA['C1'],y=X_PCA['C2'])
sns.despine(top = True, right = True)
plt.suptitle('PCA',size=30,fontweight='bold')
plt.show()

# %%
plt.figure(figsize=(12,7))
sns.scatterplot(x=X_MDS['V1'],y=X_MDS['V2'])
sns.despine(top = True, right = True)
plt.suptitle('MDS',size=30,fontweight='bold')
plt.show()

n_clust=3

Ward = UnsupervisedUtils.AgglomerativeClustering(n_clusters=n_clust)
Ward.fit(X)
clusters_Ward= Ward.labels_
TAD['Ward'] = clusters_Ward

# Kmeans = UnsupervisedUtils.KMeans(n_clusters=n_clust)
# Kmeans.fit(X)
Kmeans = load_object('kmeans_model.pkl')
clusters_Kmeans= Kmeans.labels_
TAD['KMeans'] = clusters_Kmeans

# GM = UnsupervisedUtils.GaussianMixture(n_components=n_clust, random_state=1610)
# GM = GM.fit(X)
GM = load_object('gm_model.pkl')
clusters_GM= GM.predict(X)
TAD['GM'] = clusters_GM

TAD

save_object(Kmeans,'kmeans_model')
save_object(GM,'gm_model')

plt.figure(figsize=(20,8),dpi=150)
method = 'Ward'
PlotUtils.pie(TAD,method,title='Cantidad de elementos en cada cluster\n Método: '+  method,legend=False)
plt.savefig('n_' + method + '_final.png')

plt.figure(figsize=(20,8),dpi=150)
method = 'KMeans'
PlotUtils.pie(TAD,method,title='Cantidad de elementos en cada cluster\n Método: '+  method,legend = False, labels=True)
plt.savefig('n_' + method + '_final.png')

plt.figure(figsize=(20,8),dpi=150)
method = 'GM'
PlotUtils.pie(TAD,method,title='Cantidad de elementos en cada cluster\n Método: '+  method,legend=False)
plt.savefig('n_' + method + '_final.png')


method = 'Ward'

import plotly.express as px

fig = px.scatter(x=X_PCA['C1'],y=X_PCA['C2'],color = TAD[method].astype(str))
fig.update_traces(marker=dict(size=5,
                                  line=dict(width=0.2,
                                            color='white')),
                      selector=dict(mode='markers'))
fig.update_layout(height=500, width=750,title='PCA')

fig = px.scatter(x=X_MDS['V1'],y=X_MDS['V2'],color = TAD[method].astype(str))
fig.update_traces(marker=dict(size=5,
                                  line=dict(width=0.2,
                                            color='white')),
                      selector=dict(mode='markers'))
fig.update_layout(height=500, width=750,title='PCA')

X_f = pd.DataFrame(X)
X_f_cols = X_f.columns
X_f['Ward'] = clusters_Ward
X_f['KMeans'] = clusters_Kmeans
X_f['GM'] = clusters_GM
X_f

print(UnsupervisedUtils.silhouette_score(X_f[X_f_cols],X_f['Ward']))
print(UnsupervisedUtils.silhouette_score(X_f[X_f_cols],X_f['KMeans']))
print(UnsupervisedUtils.silhouette_score(X_f[X_f_cols],X_f['GM']))

clusters = []

for i in range(n_clust):
    clusters.append(TAD.loc[TAD['KMeans']==i])

feat_cols = clusters[0].columns[0:5]
feat_cols

VarsKruskal = pd.DataFrame([UnsupervisedUtils.kruskal(*[x[y] for x in clusters]) for y in feat_cols]).applymap(lambda x : '{:.10f}'.format(x))
VarsKruskal.insert(0,'Variable',feat_cols)
VarsKruskal

def highlight(s, props=''):
    return np.where(s == 'Distribución parecida', props, '')

for i in range(n_clust):
    print('TAD VS ' + 'Cluster ' + str(i))
    display(UnsupervisedUtils.pruebas_hipotesis(TAD, clusters[i], feat_cols).style.apply(highlight, props='color:white;background-color:darkblue', axis=1))
    

tuckeycol = []
for col in feat_cols:
    tuckeycol.append(UnsupervisedUtils.pairwise_tukeyhsd(endog=TAD[col], groups=TAD['KMeans'], alpha=0.05))
import io
tuckeyresults = []
for i,col in enumerate(feat_cols): 
    print(col)
    dbtemp = pd.read_csv(io.StringIO(tuckeycol[i].summary().as_csv()), sep=",",header=1)
    dbtemp.columns = ['group1', 'group2', 'meandiff', 'p-adj', 'lower', 'upper','reject']
    dbtemp = dbtemp[['group1','group2','p-adj','reject']]
    dbtemp = dbtemp.applymap(lambda x: str(x).strip())
    dbtemp['reject'] = np.where(dbtemp['reject']=='False','Distribución parecida','Distribución diferente')
    tuckeyresults.append(dbtemp['reject'].value_counts(dropna=False).T)
    display(dbtemp.style.apply(highlight, props='color:white;background-color:darkblue', axis=1))
    

tuckeydecision = pd.concat(tuckeyresults,axis=1).T
tuckeydecision.index = feat_cols
tuckeydecision = tuckeydecision.fillna(0)
tuckeydecision['%'] = tuckeydecision['Distribución diferente']/3
tuckeydecision['tuckey_decision'] = np.where(tuckeydecision['%'] > 0.6, 'Conservar la variable' ,'Quitar variable')
tuckeydecision

notsignificant = []
TAD_final = TAD.drop(columns = notsignificant)
TAD_final

from sklearn.preprocessing import MinMaxScaler
TMDb = pd.read_feather('../M1/DBM1')
ratings = pd.read_feather('../../Datasets/MovieLens/ratings')
ratings.drop(columns=['timestamp'],inplace=True)
ratings.columns = ['userId','id','rating']
movies = TMDb.copy()
movies.drop(columns=['overview','month','day','budget','revenue','status','tagline','vote_count','keywords','poster_path','backdrop_path'],inplace=True)
movies = movies.loc[movies['genres'].isna()==False].reset_index(drop=True)
movies = movies.loc[movies['runtime'].isna()==False].reset_index(drop=True)
movies = movies.loc[movies['year'].isna()==False].reset_index(drop=True)
movies = movies.loc[movies['vote_average']>0].reset_index(drop=True)
movies = movies.fillna(0)
counts = movies['genres'].str.split('-').explode().value_counts(True)
mask = counts>0.05
genres_mask = counts[mask].index
genresaux = movies['genres'].str.split('-').explode().reset_index()
genresaux.loc[~genresaux['genres'].isin(genres_mask),'genres'] = 'OTROS'
genresaux = genresaux.reset_index().pivot_table(index='index',columns='genres',values='level_0',aggfunc='count',fill_value=0)
genresaux.columns = ['genre_' + col for col in genresaux.columns]
genresaux = genresaux.fillna(0)
movies = pd.concat([movies,genresaux],axis=1).drop(columns = ['genres'])
counts = movies['original_language'].value_counts(True)
mask = counts>0.05
language_mask = counts[mask].index
movies.loc[~movies['original_language'].isin(language_mask),'original_language'] = 'OTROS'
movies = pd.get_dummies(movies,columns = ['original_language'])
ratings['rating'] = ratings['rating']/10
ratings['rating'] = ratings['rating'].apply(lambda x: 1 if x>0.7 else 0)
TAD = ratings.merge(movies,on='id')
genres_cols = [col for col in TAD.columns if 'genre_' in col]
language_cols = [col for col in TAD.columns if 'original_language_' in col]
for v in genres_cols:
    TAD[v] = np.multiply(TAD['rating'],TAD[v])

for v in language_cols:
    TAD[v] = np.multiply(TAD['rating'],TAD[v])

feat_op= {'id':['count'],
         'popularity': ['median'],
         'year': ['median'],
         'runtime':['median'],
         'vote_average' : ['median']
         }
genre_op = dict(zip(genres_cols,np.tile(['sum'],(len(genres_cols),1)).tolist()))
language_op = dict(zip(language_cols,np.tile(['sum'],(len(language_cols),1)).tolist()))
feat_op.update(genre_op)
feat_op.update(language_op)
TAD = TAD.groupby('userId').agg(feat_op)
TAD_columns = [key + '_' + str(j) for key in feat_op.keys() for j in feat_op.get(key)]
TAD_columns = list(map(lambda x: x.replace('id_count','n'),TAD_columns))
TAD.columns=TAD_columns
TAD

KMeans = load_object('kmeans_model.pkl')
TAD['KMeans'] = KMeans.labels_
TAD.head(2)

clusters = []

for i in range(3):
    clusters.append(TAD.loc[TAD['KMeans']==i])

cluster_resume = clusters[0][TAD_columns].describe()
print(f'En este cluster, tenemos un total de  {cluster_resume["n"][0]} usuarios')
cluster_resume  

cluster_resume = clusters[1][TAD_columns].describe()
print(f'En este cluster, tenemos un total de  {cluster_resume["n"][0]} usuarios')
cluster_resume  

cluster_resume = clusters[2][TAD_columns].describe()
print(f'En este cluster, tenemos un total de  {cluster_resume["n"][0]} usuarios')
cluster_resume  

TMDb = pd.read_feather('../M1/DBM1')
ratings = pd.read_feather('../../Datasets/MovieLens/ratings')
ratings.drop(columns=['timestamp'],inplace=True)
ratings.columns = ['userId','id','rating']
movies = TMDb.copy()
movies.drop(columns=['overview','month','day','budget','revenue','status','tagline','vote_count','keywords','poster_path','backdrop_path'],inplace=True)
movies = movies.loc[movies['genres'].isna()==False].reset_index(drop=True)
movies = movies.loc[movies['runtime'].isna()==False].reset_index(drop=True)
movies = movies.loc[movies['year'].isna()==False].reset_index(drop=True)
movies = movies.loc[movies['vote_average']>0].reset_index(drop=True)
movies = movies.fillna(0)
counts = movies['genres'].str.split('-').explode().value_counts(True)
mask = counts>0.05
genres_mask = counts[mask].index
genresaux = movies['genres'].str.split('-').explode().reset_index()
genresaux.loc[~genresaux['genres'].isin(genres_mask),'genres'] = 'OTROS'
genresaux = genresaux.reset_index().pivot_table(index='index',columns='genres',values='level_0',aggfunc='count',fill_value=0)
genresaux.columns = ['genre_' + col for col in genresaux.columns]
genresaux = genresaux.fillna(0)
movies = pd.concat([movies,genresaux],axis=1).drop(columns = ['genres'])
counts = movies['original_language'].value_counts(True)
mask = counts>0.05
language_mask = counts[mask].index
movies.loc[~movies['original_language'].isin(language_mask),'original_language'] = 'OTROS'
movies = pd.get_dummies(movies,columns = ['original_language'])
ratings['rating'] = ratings['rating']/10
ratings['rating'] = ratings['rating'].apply(lambda x: 1 if x>0.7 else 0)
TAD = ratings.merge(movies,on='id')
genres_cols = [col for col in TAD.columns if 'genre_' in col]
language_cols = [col for col in TAD.columns if 'original_language_' in col]
for v in genres_cols:
    TAD[v] = np.multiply(TAD['rating'],TAD[v])

for v in language_cols:
    TAD[v] = np.multiply(TAD['rating'],TAD[v])

feat_op= {'id':['count'],
         'popularity': ['median'],
         'year': ['median'],
         'runtime':['median'],
         'vote_average' : ['median']
         }
TAD = TAD.groupby('userId').agg(feat_op)
TAD_columns = [key + '_' + str(j) for key in feat_op.keys() for j in feat_op.get(key)]
TAD_columns = list(map(lambda x: x.replace('id_count','n'),TAD_columns))
TAD.columns=TAD_columns
X = TAD.copy()
scaler = MinMaxScaler()
scaler.fit(X)
X=scaler.transform(X)
pd.DataFrame(X)

n_predict = 110
popularity_predict = 7
year_predict = 2005
runtime_predict = 100
vote_average_predict = 7

X_predict = pd.DataFrame([[n_predict,popularity_predict,year_predict,runtime_predict,vote_average_predict]],columns = ['n', 'popularity_median', 'year_median', 'runtime_median','vote_average_median'])
X_predict
X_predict = scaler.transform(X_predict)
X_predict

cluster_predict  = KMeans.predict(X_predict)
cluster_predict

n_predict = 10
popularity_predict = 15
year_predict = 2010
runtime_predict = 120
vote_average_predict = 7

X_predict = pd.DataFrame([[n_predict,popularity_predict,year_predict,runtime_predict,vote_average_predict]],columns = ['n', 'popularity_median', 'year_median', 'runtime_median','vote_average_median'])
X_predict
X_predict = scaler.transform(X_predict)
X_predict

cluster_predict  = KMeans.predict(X_predict)
cluster_predict
