import os
import pickle
import importlib
import warnings
from tqdm import tqdm

import pandas as pd
import numpy as np

import cufflinks as cf
import matplotlib.ticker as tkr
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
plt.style.use('seaborn-white')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 100)
cf.set_config_file(theme='white',dimensions=(650,450))
cf.go_offline()

sys.path.insert(1, '../../Datasets/')

import StringUtils 
importlib.reload(StringUtils)

import PlotUtils 
importlib.reload(PlotUtils)

import SupervisedUtils 
importlib.reload(SupervisedUtils)

def save_object(obj,name):
  with open(name + '.pkl', 'wb') as file:
    pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)

def load_object(name):
    with open(name, 'rb') as file:
        return(pickle.load(file))

TMDb = pd.read_feather('../M1/DBM1')
TMDb.head(2)

movies = TMDb.copy()
movies = movies.loc[movies['overview'].isna()==False].reset_index(drop=True)
movies = movies.loc[movies['budget']>0].reset_index(drop=True)
movies = movies.loc[movies['vote_average']>0].reset_index(drop=True)
movies = movies.loc[movies['genres'].isna()==False].reset_index(drop=True)
movies = movies.drop(columns= ['poster_path','backdrop_path','day','revenue','status'])
print(movies.shape)

movies.isna().sum().to_frame().T

movies = movies.dropna(subset=['year','month','runtime']).reset_index(drop=True)

PlotUtils.hist(movies,'vote_average')

movies['y']=pd.cut(movies['vote_average'],[0,6.5,10],labels=['Malo','Buena'],include_lowest=True)
movies['y'].value_counts()
le = SupervisedUtils.LabelEncoder()
le.fit(movies['y'])
y = le.transform(movies['y'])

features = ['genres','original_language','overview','year','month','budget','runtime','tagline','keywords']
X = movies[features]
X.head(2)

X.isna().sum().to_frame().T

X_train, X_test, y_train, y_test = SupervisedUtils.train_test_split(X, y, test_size=0.3, random_state=12345)

counts = X_train['genres'].str.split('-').explode().value_counts(True)
mask = counts>0.05
genres_mask = counts[mask].index
genresaux = X_train['genres'].str.split('-')
genresaux = genresaux.apply(lambda x: '-'.join(list(map(lambda y: 'OTROS' if y not in genres_mask else y,x))))
genre_vc = StringUtils.CountVectorizer(tokenizer= lambda x: x.split('-'))
genre_vc.fit(genresaux)
genresaux = pd.DataFrame(genre_vc.transform(genresaux).toarray()).set_index(X_train.index)
genresaux.columns = ['genre_' + str(col) for col in genre_vc.get_feature_names_out()]
X_train = pd.concat([X_train,genresaux],axis=1).drop(columns = ['genres'])
X_train.head(2)

genresaux = X_test['genres'].str.split('-')
genresaux = genresaux.apply(lambda x: '-'.join(list(map(lambda y: 'OTROS' if y not in genres_mask else y,x))))
genresaux = pd.DataFrame(genre_vc.transform(genresaux).toarray()).set_index(X_test.index)
genresaux.columns = ['genre_' + str(col) for col in genre_vc.get_feature_names_out()]
X_test = pd.concat([X_test,genresaux],axis=1).drop(columns = ['genres'])
X_test.head(2)

counts = X_train['original_language'].value_counts(True)
mask = counts>0.10
language_mask = counts[mask].index
X_train.loc[~X_train['original_language'].isin(language_mask),'original_language'] = 'OTROS'
X_train = pd.get_dummies(X_train,columns = ['original_language'])
X_train.head(2)

X_test.loc[~X_test['original_language'].isin(language_mask),'original_language'] = 'OTROS'
X_test = pd.get_dummies(X_test,columns = ['original_language'])
X_test.head(2)

engstopwords = StringUtils.stopwords.words('english')
customsw  = ['one','film','movie','man','two','story']
engstopwords = engstopwords + customsw

X_train['overview_clean']=StringUtils.clean_re(X_train['overview'])
X_train['overview_clean']=StringUtils.remove_stopwords(X_train['overview_clean'],engstopwords)
X_train['overview_clean']=StringUtils.remove_accents(X_train['overview_clean'])
X_test['overview_clean']=StringUtils.clean_re(X_test['overview'])
X_test['overview_clean']=StringUtils.remove_stopwords(X_test['overview_clean'],engstopwords)
X_test['overview_clean']=StringUtils.remove_accents(X_test['overview_clean'])

overview_vc = StringUtils.TfidfVectorizer(max_features=100)
overview_vc.fit(X_train['overview_clean'])
X_overview_train = overview_vc.transform(X_train['overview_clean']).toarray()
X_overview_train = pd.DataFrame(X_overview_train).set_index(X_train.index)
X_overview_train.columns  = ['overview_' + str(j) for j in range(X_overview_train.shape[1])]
X_overview_test  = overview_vc.transform(X_test['overview_clean']).toarray()
X_overview_test = pd.DataFrame(X_overview_test).set_index(X_test.index)
X_overview_test.columns  = ['overview_' + str(j) for j in range(X_overview_test.shape[1])]

keywords_vc = StringUtils.TfidfVectorizer(tokenizer= lambda x: x.split('-'),max_features=100)
keywords_vc.fit(X_train['keywords'].astype(str))
X_keywords_train = keywords_vc.transform(X_train['keywords'].astype(str)).toarray()
X_keywords_train = pd.DataFrame(X_keywords_train).set_index(X_train.index)
X_keywords_train.columns  = ['keywords_' + str(j) for j in range(X_keywords_train.shape[1])]
X_keywords_test  = keywords_vc.transform(X_test['keywords'].astype(str)).toarray()
X_keywords_test = pd.DataFrame(X_keywords_test).set_index(X_test.index)
X_keywords_test.columns  = ['keywords_' + str(j) for j in range(X_keywords_test.shape[1])]

X_string_train = pd.concat([X_overview_train,X_keywords_train],axis=1)
X_string_test = pd.concat([X_overview_test,X_keywords_test],axis=1)

stringvars = ['overview','keywords','tagline','overview_clean']

X_train = pd.concat([X_train.drop(columns=stringvars),X_string_train],axis=1)
X_test = pd.concat([X_test.drop(columns=stringvars),X_string_test],axis=1)

X_train.head(2)

X_test.head(2)

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

LogModel = SupervisedUtils.LogisticRegression(random_state=12345, n_jobs=-1)
LogModel.fit(X_train,y_train)
y_LogModel=LogModel.predict(X_test)
LogModel_train_score = LogModel.score(X_train,y_train)
print(f'Score en train: {LogModel_train_score}')
LogModel_test_score = LogModel.score(X_test,y_test)
print(f'Score en test: {LogModel_test_score}')
acc_LogModel, prec_LogModel, rec_LogModel, roc_LogModel = SupervisedUtils.model_cf('Regresión\nlogística\n(Default)',y_test,y_LogModel)
SupervisedUtils.grafica_curva_roc(y_test,y_LogModel)

clist = np.logspace(-2,4,20,base=2)
LogitTemp=SupervisedUtils.LogisticRegression(random_state=12345, n_jobs=-1)
train_scores, test_scores = SupervisedUtils.validation_curve(
    LogitTemp,
    X_train,
    y_train,
    param_name='C',
    param_range=clist,
    cv=3,
    n_jobs=-1,
    scoring="accuracy")
cscores=list(map(np.mean,test_scores))
ctunning = pd.DataFrame(cscores,index=clist,columns=['Score en test'])
cbest=clist[np.argmax(cscores)]
ctunning.plot(logx=True,figsize=(20,8))

plt.axvline(x=cbest,label='Valor óptimo',linestyle=':',color='green')
plt.plot([], [],' ',label=f'$C={round(cbest,4)}$')
plt.legend(fontsize = 'large')
for ind,i in enumerate(clist):
    plt.text(i,cscores[ind]-0.15*np.std(cscores),round(i,3)) 
plt.xticks([])
plt.title('Efectividad media de una regresión logística con parámetro C',size=20,fontweight='bold')

BestLogModel = SupervisedUtils.LogisticRegression(C=cbest,random_state=12345, n_jobs=-1)
BestLogModel.fit(X_train,y_train)
y_BestLogModel=BestLogModel.predict(X_test)
BestLogModel_train_score = BestLogModel.score(X_train,y_train)
print(f'Score en train: {BestLogModel_train_score}')
BestLogModel_test_score = BestLogModel.score(X_test,y_test)
print(f'Score en test: {BestLogModel_test_score}')
acc_BestLogModel, prec_BestLogModel, rec_BestLogModel, roc_BestLogModel = SupervisedUtils.model_cf('\nRegresión\nlogística\n\n(C=3.4568)',y_test,y_BestLogModel)
SupervisedUtils.grafica_curva_roc(y_test,y_BestLogModel)

RFModel = SupervisedUtils.RandomForestClassifier(random_state=12345, n_jobs=-1)
RFModel.fit(X_train,y_train)
y_RFModel=RFModel.predict(X_test)
RFModel_train_score = RFModel.score(X_train,y_train)
print(f'Score en train: {RFModel_train_score}')
RFModel_test_score = RFModel.score(X_test,y_test)
print(f'Score en test: {RFModel_test_score}')
acc_RFModel, prec_RFmodel, rec_RFModel, roc_RFModel = SupervisedUtils.model_cf('Random Forest\n(Default)',y_test,y_RFModel)
SupervisedUtils.grafica_curva_roc(y_test,y_RFModel)

param_grid={'n_estimators' : [10,50,100,200],
            'criterion' : ['gini','entropy'],
            'max_depth' : [None, 2,5,10,20,50],
            'min_samples_split' : [2,4,6,10],
            'min_samples_leaf' : [1,2,3],
           }
RF = SupervisedUtils.RandomForestClassifier(random_state=12345, n_jobs=-1)
search_RF = SupervisedUtils.HalvingGridSearchCV(RF, cv=3, param_grid=param_grid,factor = 2,random_state=12345,verbose=np.inf,scoring="accuracy").fit(X_train, y_train)
rftunning=pd.DataFrame(search_RF.cv_results_)
indrfmax=np.argmax(rftunning['mean_test_score'])
rfbest=rftunning.iloc[indrfmax,:]

plt.figure(figsize=(8, 6), dpi=80)
rftunning.plot(use_index=True, y='mean_test_score',figsize=(20,8))
plt.gca().set_xticks([])
plt.axvline(x=indrfmax,label='Modelo óptimo',linestyle=':',color='green')
plt.plot([],[],' ',label=f"Score: {round(rftunning['mean_test_score'][indrfmax],4)}")
plt.legend()

plt.suptitle('Efectividad media usando diversos parámetros en un modelo Random Forest\n(3K Fold CV)',size=20,fontweight='bold')

rftunning[[x for x in rftunning if 'param_' in x] + ['mean_train_score','mean_test_score']].tail(20)

BestRFModel = SupervisedUtils.RandomForestClassifier(max_depth=50,min_samples_leaf=1,min_samples_split=4,n_estimators=200,random_state=12345, n_jobs=-1)
BestRFModel.fit(X_train,y_train)
y_BestRFModel=BestRFModel.predict(X_test)
BestRFModel_train_score = BestRFModel.score(X_train,y_train)
print(f'Score en train: {BestRFModel_train_score}')
BestRFModel_test_score = BestRFModel.score(X_test,y_test)
print(f'Score en test: {BestRFModel_test_score}')
acc_BestRFModel, prec_BestRFModel, rec_BestRFModel, roc_BestRFModel = SupervisedUtils.model_cf('\nRandom Forest\n (Mejor modelo)',y_test,y_BestRFModel)
SupervisedUtils.grafica_curva_roc(y_test,y_BestRFModel)

KNNModel = SupervisedUtils.KNeighborsClassifier(n_jobs=-1)
KNNModel.fit(X_train,y_train)
y_KNNModel=KNNModel.predict(X_test)
KNNModel_train_score = KNNModel.score(X_train,y_train)
print(f'Score en train: {KNNModel_train_score}')
KNNModel_test_score = KNNModel.score(X_test,y_test)
print(f'Score en test: {KNNModel_test_score}')
acc_KNNModel, prec_KNNModel, rec_KNNModel, roc_KNNModel = SupervisedUtils.model_cf('K Nearest Neighbors\n(Default K=5)',y_test,y_KNNModel)
SupervisedUtils.grafica_curva_roc(y_test,y_KNNModel)

param_grid={'n_neighbors':[2,5,7,10,50],
            'weights' : ['uniform', 'distance'],
            'leaf_size':[10,30,50,100],
            'p':[1,2]
           }

KNN = SupervisedUtils.KNeighborsClassifier(n_jobs=-1)
search_KNN = SupervisedUtils.HalvingGridSearchCV(KNN, cv=3, param_grid=param_grid,factor = 2,random_state=12345,verbose=np.inf,scoring="accuracy").fit(X_train, y_train)
knntunning=pd.DataFrame(search_KNN.cv_results_)
indknnmax=np.argmax(knntunning['mean_test_score'])
knnbest=knntunning.iloc[indknnmax,:]

plt.figure(figsize=(8, 6), dpi=80)
knntunning.plot(use_index=True, y='mean_test_score',figsize=(20,8))
plt.gca().set_xticks([])
plt.axvline(x=indknnmax,label='Modelo óptimo',linestyle=':',color='green')
plt.plot([],[],' ',label=f"Score: {round(rftunning['mean_test_score'][indrfmax],4)}")
plt.legend()
plt.suptitle('Efectividad media usando diversos parámetros en un modelo Random Forest\n(3K Fold CV)',size=20,fontweight='bold')

knntunning[[x for x in knntunning if 'param_' in x] + ['mean_train_score','mean_test_score']].tail(20)

BestKNNModel = SupervisedUtils.KNeighborsClassifier(leaf_size=50, n_jobs=-1, n_neighbors=50, weights='distance')
BestKNNModel.fit(X_train,y_train)
y_BestKNNModel=BestKNNModel.predict(X_test)
BestKNNModel_train_score = BestKNNModel.score(X_train,y_train)
print(f'Score en train: {BestKNNModel_train_score}')
BestKNNModel_test_score = BestKNNModel.score(X_test,y_test)
print(f'Score en test: {BestKNNModel_test_score}')
acc_BestKNNModel, prec_BestKNNModel, rec_BestKNNModel, roc_BestKNNModel = SupervisedUtils.model_cf('\nKNN\n (Mejor modelo)',y_test,y_BestKNNModel)
SupervisedUtils.grafica_curva_roc(y_test,y_BestKNNModel)

y_preds_def= pd.DataFrame(LogModel.predict(X_train), columns=['Regresión logística'])
y_preds_def['Random Forest']=RFModel.predict(X_train)
y_preds_def['KNN']=KNNModel.predict(X_train)
y_preds_def['Emsable modelos default']=y_preds_def.mode(axis=1)
y_ensdefault=y_preds_def['Emsable modelos default']
y_preds_def.head(10)

acc_ensdefault, prec_ensdefault, rec_ensdefault, roc_ensdefault = SupervisedUtils.model_cf('Emsable:\n modelos default',y_train,y_ensdefault)

y_preds_def= pd.DataFrame(y_LogModel, columns=['Regresión logística'])
y_preds_def['Random Forest']=y_RFModel
y_preds_def['KNN']=y_KNNModel
y_preds_def['Emsable modelos default']=y_preds_def.mode(axis=1)
y_ensdefault=y_preds_def['Emsable modelos default']
y_preds_def.head(10)

acc_ensdefault, prec_ensdefault, rec_ensdefault, roc_ensdefault = SupervisedUtils.model_cf('Emsable:\n modelos default',y_test,y_ensdefault)

y_preds_best= pd.DataFrame(BestLogModel.predict(X_train), columns=['Regresión logística'])
y_preds_best['Random Forest']=BestRFModel.predict(X_train)
y_preds_best['KNN']=BestKNNModel.predict(X_train)
y_preds_best['Emsable mejores modelos']=y_preds_best.mode(axis=1)
y_ensbest=y_preds_best['Emsable mejores modelos']
y_preds_best.head(10)

acc_ensbest, prec_ensbest, rec_ensbest, roc_ensbest = SupervisedUtils.model_cf('Emsable:\n mejores modelos',y_train,y_ensbest)

y_preds_best= pd.DataFrame(y_BestLogModel, columns=['Regresión logística'])
y_preds_best['Random Forest']=y_BestRFModel
y_preds_best['KNN']=y_BestKNNModel
y_preds_best['Emsable mejores modelos']=y_preds_best.mode(axis=1)
y_ensbest=y_preds_best['Emsable mejores modelos']
y_preds_best.head(10)

acc_ensbest, prec_ensbest, rec_ensbest, roc_ensbest = SupervisedUtils.model_cf('Emsable:\n mejores modelos',y_test,y_ensbest)

ModelMetricsDef= pd.DataFrame([acc_LogModel,prec_LogModel,rec_LogModel,roc_LogModel],index=['Efectividad','Precision','Recall','ROC AUC - Score'],columns=['Regresión logística'])
ModelMetricsDef['Random Forest']=[acc_RFModel,prec_RFmodel,rec_RFModel,roc_RFModel]
ModelMetricsDef['KNN']=[acc_KNNModel,prec_KNNModel,rec_KNNModel,roc_KNNModel]
ModelMetricsDef['Ensamble: Default']=[acc_ensdefault,prec_ensdefault,rec_ensdefault,roc_ensdefault]
ModelMetricsDef

ModelMetricsDef_TFID=ModelMetricsDef
ModelMetricsDef_TFID.to_excel('ModelMetricsDef_TFID.xlsx')

ModelMetrics= pd.DataFrame([acc_BestLogModel,prec_BestLogModel,rec_BestLogModel,roc_BestLogModel],index=['Efectividad','Precision','Recall','ROC AUC - Score'],columns=['Regresión logística'])
ModelMetrics['Random Forest']=[acc_BestRFModel,prec_BestRFModel,rec_BestRFModel,roc_BestRFModel]
ModelMetrics['KNN']=[acc_BestKNNModel,prec_BestKNNModel,rec_BestKNNModel,roc_BestKNNModel]
ModelMetrics['Ensamble: Mejores modelos']=[acc_ensbest,prec_ensbest,rec_ensbest,roc_ensbest]
ModelMetrics

ModelMetrics_TFID=ModelMetrics
ModelMetrics_TFID.to_excel('ModelMetrics_TFID.xlsx')

fig = plt.figure(figsize=(15,8))
ax = fig.add_axes([0,0,1,1])
xticks = np.arange(4)
colors = ['b','r','g','y']
ax.plot([],[],' ',label="$\\bf{Modelos\ seleccionados}$")
for ind,i in enumerate(ModelMetrics.columns):
    ax.bar(xticks + ind/5, ModelMetrics[i], color = colors[ind], width = 1/7,alpha=0.5,label=i)   
plt.plot([],[],' ',label='Modelos default')

for ind,i in enumerate(ModelMetricsDef.columns):
    ax.bar(xticks + ind/5, ModelMetricsDef[i],edgecolor =colors[ind], width = 1/9,label=i,fill=False) 
handles, labels = plt.gca().get_legend_handles_labels()
order = [0,2,3,4,5,1,6,7,8,9]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],ncol=2) 
ax.set_xticks(xticks+0.3, ('Efectividad', 'Precision', 'Recall', 'ROC AUC'))
fig.suptitle('Métricas de modelos entrenados', size=25,fontstyle='italic',fontweight='bold')

iterables=[['Modelos seleccionados','Modelos default'],ModelMetrics.columns]
Metrics = pd.concat([ModelMetrics,ModelMetricsDef],axis=1)
Metrics.columns=pd.MultiIndex.from_product(iterables, names=["", ""])
Metrics

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

TMDb = pd.read_feather('../M1/DBM1')
movies = TMDb.copy()
movies = movies.loc[movies['overview'].isna()==False].reset_index(drop=True)
movies = movies.loc[movies['budget']>0].reset_index(drop=True)
movies = movies.loc[movies['vote_average']>0].reset_index(drop=True)
movies = movies.loc[movies['genres'].isna()==False].reset_index(drop=True)
movies = movies.drop(columns= ['poster_path','backdrop_path','day','revenue','status'])
movies = movies.dropna(subset=['year','month','runtime']).reset_index(drop=True)
movies['y']=pd.cut(movies['vote_average'],[0,6.5,10],labels=['Malo','Buena'],include_lowest=True)
le = SupervisedUtils.LabelEncoder()
le.fit(movies['y'])
y = le.transform(movies['y'])
features = ['genres','original_language','overview','year','month','budget','runtime','tagline','keywords']
X = movies[features]
X_train, X_test, y_train, y_test = SupervisedUtils.train_test_split(X, y, test_size=0.3, random_state=12345)
counts = X_train['genres'].str.split('-').explode().value_counts(True)
mask = counts>0.05
genres_mask = counts[mask].index
genresaux = X_train['genres'].str.split('-')
genresaux = genresaux.apply(lambda x: '-'.join(list(map(lambda y: 'OTROS' if y not in genres_mask else y,x))))
genre_vc = StringUtils.CountVectorizer(tokenizer= lambda x: x.split('-'))
genre_vc.fit(genresaux)
genresaux = pd.DataFrame(genre_vc.transform(genresaux).toarray()).set_index(X_train.index)
genresaux.columns = ['genre_' + str(col) for col in genre_vc.get_feature_names_out()]
X_train = pd.concat([X_train,genresaux],axis=1).drop(columns = ['genres'])
genresaux = X_test['genres'].str.split('-')
genresaux = genresaux.apply(lambda x: '-'.join(list(map(lambda y: 'OTROS' if y not in genres_mask else y,x))))
genresaux = pd.DataFrame(genre_vc.transform(genresaux).toarray()).set_index(X_test.index)
genresaux.columns = ['genre_' + str(col) for col in genre_vc.get_feature_names_out()]
X_test = pd.concat([X_test,genresaux],axis=1).drop(columns = ['genres'])
counts = X_train['original_language'].value_counts(True)
mask = counts>0.10
language_mask = counts[mask].index
X_train.loc[~X_train['original_language'].isin(language_mask),'original_language'] = 'OTROS'
X_train = pd.get_dummies(X_train,columns = ['original_language'])
X_test.loc[~X_test['original_language'].isin(language_mask),'original_language'] = 'OTROS'
X_test = pd.get_dummies(X_test,columns = ['original_language'])
engstopwords = StringUtils.stopwords.words('english')
customsw  = ['one','film','movie','man','two','story']
engstopwords = engstopwords + customsw

X_train['overview_clean']=StringUtils.clean_re(X_train['overview'])
X_test['overview_clean']=StringUtils.clean_re(X_test['overview'])
overview_tags = [TaggedDocument(words = x.split(), tags = [y]) for x,y in zip(X_train['overview_clean'],y_train)]
print('Entrenando modelo d2v_overview')
d2v_overview = Doc2Vec(vector_size=100, min_count=1, epochs=10)
d2v_overview.build_vocab(overview_tags)
d2v_overview.train(overview_tags, total_examples=d2v_overview.corpus_count, epochs=d2v_overview.epochs)
print('Creando matriz de entrenamiento')
auxlist=list()
for t in tqdm(overview_tags):
    v = d2v_overview.infer_vector(t[0])
    auxlist.append(v)
X_overview_train=pd.DataFrame(auxlist)
X_overview_train.columns  = ['overview_' + str(j) for j in range(X_overview_train.shape[1])]
X_overview_train.index = X_train.index
print('Creando matriz de validacion')
auxlist=list()
for t in tqdm(X_test['overview_clean']):
    v = d2v_overview.infer_vector(t.split())
    auxlist.append(v)
X_overview_test=pd.DataFrame(auxlist)
X_overview_test.columns  = ['overview_' + str(j) for j in range(X_overview_test.shape[1])]
X_overview_test.index = X_test.index

keywords_tags = [TaggedDocument(words = x.split('-'), tags = [y]) for x,y in zip(X_train['keywords'].astype(str),y_train)]
print('Entrenando modelo d2v_keywords')
d2v_keywords = Doc2Vec(vector_size=100, min_count=1, epochs=10)
d2v_keywords.build_vocab(keywords_tags)
d2v_keywords.train(keywords_tags, total_examples=d2v_keywords.corpus_count, epochs=d2v_keywords.epochs)
print('Creando matriz de entrenamiento')
auxlist=list()
for t in tqdm(keywords_tags):
    v = d2v_keywords.infer_vector(t[0])
    auxlist.append(v)
X_keywords_train=pd.DataFrame(auxlist)
X_keywords_train.columns  = ['keywords_' + str(j) for j in range(X_keywords_train.shape[1])]
X_keywords_train.index = X_train.index
print('Creando matriz de validacion')
auxlist=list()
for t in tqdm(X_test['keywords'].astype(str)):
    v = d2v_keywords.infer_vector(t.split('-'))
    auxlist.append(v)
X_keywords_test=pd.DataFrame(auxlist)
X_keywords_test.columns  = ['keywords_' + str(j) for j in range(X_keywords_test.shape[1])]
X_keywords_test.index = X_test.index

X_string_train = pd.concat([X_overview_train,X_keywords_train],axis=1)
X_string_test = pd.concat([X_overview_test,X_keywords_test],axis=1)

stringvars = ['overview','keywords','tagline','overview_clean']
X_train = pd.concat([X_train.drop(columns=stringvars),X_string_train],axis=1)
X_test = pd.concat([X_test.drop(columns=stringvars),X_string_test],axis=1)

X_train
X_test

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

LogModel = SupervisedUtils.LogisticRegression(random_state=12345, n_jobs=-1)
LogModel.fit(X_train,y_train)
y_LogModel=LogModel.predict(X_test)
LogModel_train_score = LogModel.score(X_train,y_train)
print(f'Score en train: {LogModel_train_score}')
LogModel_test_score = LogModel.score(X_test,y_test)
print(f'Score en test: {LogModel_test_score}')
acc_LogModel, prec_LogModel, rec_LogModel, roc_LogModel = SupervisedUtils.model_cf('Regresión\nlogística\n(Default)',y_test,y_LogModel)
SupervisedUtils.grafica_curva_roc(y_test,y_LogModel)

clist = np.logspace(-2,4,20,base=2)
LogitTemp=SupervisedUtils.LogisticRegression(random_state=12345, n_jobs=-1)
train_scores, test_scores = SupervisedUtils.validation_curve(
    LogitTemp,
    X_train,
    y_train,
    param_name='C',
    param_range=clist,
    cv=3,
    n_jobs=-1,
    scoring="accuracy")
cscores=list(map(np.mean,test_scores))
ctunning = pd.DataFrame(cscores,index=clist,columns=['Score en test'])
cbest=clist[np.argmax(cscores)]
ctunning.plot(logx=True,figsize=(20,8))
plt.axvline(x=cbest,label='Valor óptimo',linestyle=':',color='green')
plt.plot([], [],' ',label=f'$C={round(cbest,4)}$')
plt.legend(fontsize = 'large')
for ind,i in enumerate(clist):
    plt.text(i,cscores[ind]-0.15*np.std(cscores),round(i,3))
plt.xticks([])
plt.title('Efectividad media de una regresión logística con parámetro C',size=20,fontweight='bold')

BestLogModel = SupervisedUtils.LogisticRegression(C=8.297240355569535,random_state=12345, n_jobs=-1)
BestLogModel.fit(X_train,y_train)
y_BestLogModel=BestLogModel.predict(X_test)
BestLogModel_train_score = BestLogModel.score(X_train,y_train)
print(f'Score en train: {BestLogModel_train_score}')
BestLogModel_test_score = BestLogModel.score(X_test,y_test)
print(f'Score en test: {BestLogModel_test_score}')
acc_BestLogModel, prec_BestLogModel, rec_BestLogModel, roc_BestLogModel = SupervisedUtils.model_cf('\nRegresión\nlogística\n\n(C=8.2972)',y_test,y_BestLogModel)
SupervisedUtils.grafica_curva_roc(y_test,y_BestLogModel)

RFModel = SupervisedUtils.RandomForestClassifier(random_state=12345, n_jobs=-1)
RFModel.fit(X_train,y_train)
y_RFModel=RFModel.predict(X_test)
RFModel_train_score = RFModel.score(X_train,y_train)
print(f'Score en train: {RFModel_train_score}')
RFModel_test_score = RFModel.score(X_test,y_test)
print(f'Score en test: {RFModel_test_score}')
acc_RFModel, prec_RFmodel, rec_RFModel, roc_RFModel = SupervisedUtils.model_cf('Random Forest\n(Default)',y_test,y_RFModel)
SupervisedUtils.grafica_curva_roc(y_test,y_RFModel)

param_grid={'n_estimators' : [10,50,100,200],
            'criterion' : ['gini','entropy'],
            'max_depth' : [None, 2,5,10,20,50],
            'min_samples_split' : [2,4,6,10],
            'min_samples_leaf' : [1,2,3],
           }
RF = SupervisedUtils.RandomForestClassifier(random_state=12345, n_jobs=-1)
search_RF = SupervisedUtils.HalvingGridSearchCV(RF, cv=3, param_grid=param_grid,factor = 2,random_state=12345,verbose=np.inf,scoring="accuracy").fit(X_train, y_train)
rftunning=pd.DataFrame(search_RF.cv_results_)
indrfmax=np.argmax(rftunning['mean_test_score'])
rfbest=rftunning.iloc[indrfmax,:]
plt.figure(figsize=(8, 6), dpi=80)
rftunning.plot(use_index=True, y='mean_test_score',figsize=(20,8))
plt.gca().set_xticks([])
plt.axvline(x=indrfmax,label='Modelo óptimo',linestyle=':',color='green')
plt.plot([],[],' ',label=f"Score: {round(rftunning['mean_test_score'][indrfmax],4)}")
plt.legend()
plt.suptitle('Efectividad media usando diversos parámetros en un modelo Random Forest\n(3K Fold CV)',size=20,fontweight='bold')
rfbest
rftunning[[x for x in rftunning if 'param_' in x] + ['mean_train_score','mean_test_score']].tail(20)

BestRFModel = SupervisedUtils.RandomForestClassifier(**rfbest['params'],random_state=12345, n_jobs=-1)
BestRFModel.fit(X_train,y_train)
y_BestRFModel=BestRFModel.predict(X_test)
BestRFModel_train_score = BestRFModel.score(X_train,y_train)
print(f'Score en train: {BestRFModel_train_score}')
BestRFModel_test_score = BestRFModel.score(X_test,y_test)
print(f'Score en test: {BestRFModel_test_score}')
acc_BestRFModel, prec_BestRFModel, rec_BestRFModel, roc_BestRFModel = SupervisedUtils.model_cf('\nRandom Forest\n (Mejor modelo)',y_test,y_BestRFModel)
SupervisedUtils.grafica_curva_roc(y_test,y_BestRFModel)

KNNModel = SupervisedUtils.KNeighborsClassifier(n_jobs=-1)
KNNModel.fit(X_train,y_train)
y_KNNModel=KNNModel.predict(X_test)
KNNModel_train_score = KNNModel.score(X_train,y_train)
print(f'Score en train: {KNNModel_train_score}')
KNNModel_test_score = KNNModel.score(X_test,y_test)
print(f'Score en test: {KNNModel_test_score}')
acc_KNNModel, prec_KNNModel, rec_KNNModel, roc_KNNModel = SupervisedUtils.model_cf('K Nearest Neighbors\n(Default K=5)',y_test,y_KNNModel)
SupervisedUtils.grafica_curva_roc(y_test,y_KNNModel)

param_grid={'n_neighbors':[2,5,7,10,50],
            'weights' : ['uniform', 'distance'],
            'leaf_size':[10,30,50,100],
            'p':[1,2]
           }
KNN = SupervisedUtils.KNeighborsClassifier(n_jobs=-1)
search_KNN = SupervisedUtils.HalvingGridSearchCV(KNN, cv=3, param_grid=param_grid,factor = 2,random_state=12345,verbose=np.inf,scoring="accuracy").fit(X_train, y_train)
knntunning=pd.DataFrame(search_KNN.cv_results_)
indknnmax=np.argmax(knntunning['mean_test_score'])
knnbest=knntunning.iloc[indknnmax,:]
plt.figure(figsize=(8, 6), dpi=80)
knntunning.plot(use_index=True, y='mean_test_score',figsize=(20,8))
plt.gca().set_xticks([])
plt.axvline(x=indknnmax,label='Modelo óptimo',linestyle=':',color='green')
plt.plot([],[],' ',label=f"Score: {round(rftunning['mean_test_score'][indrfmax],4)}")
plt.legend()
plt.suptitle('Efectividad media usando diversos parámetros en un modelo Random Forest\n(3K Fold CV)',size=20,fontweight='bold')

knnbest
knntunning[[x for x in knntunning if 'param_' in x] + ['mean_train_score','mean_test_score']].tail(20)

BestKNNModel = SupervisedUtils.KNeighborsClassifier(**knnbest['params'],n_jobs=-1)
BestKNNModel.fit(X_train,y_train)
y_BestKNNModel=BestKNNModel.predict(X_test)
BestKNNModel_train_score = BestKNNModel.score(X_train,y_train)
print(f'Score en train: {BestKNNModel_train_score}')
BestKNNModel_test_score = BestKNNModel.score(X_test,y_test)
print(f'Score en test: {BestKNNModel_test_score}')
acc_BestKNNModel, prec_BestKNNModel, rec_BestKNNModel, roc_BestKNNModel = SupervisedUtils.model_cf('\nKNN\n (Mejor modelo)',y_test,y_BestKNNModel)
SupervisedUtils.grafica_curva_roc(y_test,y_BestKNNModel)

y_preds_def= pd.DataFrame(LogModel.predict(X_train), columns=['Regresión logística'])
y_preds_def['Random Forest']=RFModel.predict(X_train)
y_preds_def['KNN']=KNNModel.predict(X_train)
y_preds_def['Emsable modelos default']=y_preds_def.mode(axis=1)
y_ensdefault=y_preds_def['Emsable modelos default']
y_preds_def.head(10)
acc_ensdefault, prec_ensdefault, rec_ensdefault, roc_ensdefault = SupervisedUtils.model_cf('Emsable:\n modelos default',y_train,y_ensdefault)

y_preds_def= pd.DataFrame(y_LogModel, columns=['Regresión logística'])
y_preds_def['Random Forest']=y_RFModel
y_preds_def['KNN']=y_KNNModel
y_preds_def['Emsable modelos default']=y_preds_def.mode(axis=1)
y_ensdefault=y_preds_def['Emsable modelos default']
y_preds_def.head(10)
acc_ensdefault, prec_ensdefault, rec_ensdefault, roc_ensdefault = SupervisedUtils.model_cf('Emsable:\n modelos default',y_test,y_ensdefault)

y_preds_best= pd.DataFrame(BestLogModel.predict(X_train), columns=['Regresión logística'])
y_preds_best['Random Forest']=BestRFModel.predict(X_train)
y_preds_best['KNN']=BestKNNModel.predict(X_train)
y_preds_best['Emsable mejores modelos']=y_preds_best.mode(axis=1)
y_ensbest=y_preds_best['Emsable mejores modelos']
y_preds_best.head(10)
acc_ensbest, prec_ensbest, rec_ensbest, roc_ensbest = SupervisedUtils.model_cf('Emsable:\n mejores modelos',y_train,y_ensbest)

y_preds_best= pd.DataFrame(y_BestLogModel, columns=['Regresión logística'])
y_preds_best['Random Forest']=y_BestRFModel
y_preds_best['KNN']=y_BestKNNModel
y_preds_best['Emsable mejores modelos']=y_preds_best.mode(axis=1)
y_ensbest=y_preds_best['Emsable mejores modelos']
y_preds_best.head(10)
acc_ensbest, prec_ensbest, rec_ensbest, roc_ensbest = SupervisedUtils.model_cf('Emsable:\n mejores modelos',y_test,y_ensbest)

ModelMetricsDef= pd.DataFrame([acc_LogModel,prec_LogModel,rec_LogModel,roc_LogModel],index=['Efectividad','Precision','Recall','ROC AUC - Score'],columns=['Regresión logística'])
ModelMetricsDef['Random Forest']=[acc_RFModel,prec_RFmodel,rec_RFModel,roc_RFModel]
ModelMetricsDef['KNN']=[acc_KNNModel,prec_KNNModel,rec_KNNModel,roc_KNNModel]
ModelMetricsDef['Ensamble: Default']=[acc_ensdefault,prec_ensdefault,rec_ensdefault,roc_ensdefault]
ModelMetricsDef
ModelMetricsDef_d2v=ModelMetricsDef
ModelMetricsDef_d2v.to_excel('ModelMetricsDef_d2v.xlsx')

ModelMetrics= pd.DataFrame([acc_BestLogModel,prec_BestLogModel,rec_BestLogModel,roc_BestLogModel],index=['Efectividad','Precision','Recall','ROC AUC - Score'],columns=['Regresión logística'])
ModelMetrics['Random Forest']=[acc_BestRFModel,prec_BestRFModel,rec_BestRFModel,roc_BestRFModel]
ModelMetrics['KNN']=[acc_BestKNNModel,prec_BestKNNModel,rec_BestKNNModel,roc_BestKNNModel]
ModelMetrics['Ensamble: Mejores modelos']=[acc_ensbest,prec_ensbest,rec_ensbest,roc_ensbest]
ModelMetrics
ModelMetrics_d2v=ModelMetrics
ModelMetrics_d2v.to_excel('ModelMetrics_d2v.xlsx')

fig = plt.figure(figsize=(15,8))
ax = fig.add_axes([0,0,1,1])
xticks = np.arange(4)
colors = ['b','r','g','y']
ax.plot([],[],' ',label="$\\bf{Modelos\ seleccionados}$")
for ind,i in enumerate(ModelMetrics.columns):
    ax.bar(xticks + ind/5, ModelMetrics[i], color = colors[ind], width = 1/7,alpha=0.5,label=i)  
plt.plot([],[],' ',label='Modelos default')
for ind,i in enumerate(ModelMetricsDef.columns):
    ax.bar(xticks + ind/5, ModelMetricsDef[i],edgecolor =colors[ind], width = 1/9,label=i,fill=False)
handles, labels = plt.gca().get_legend_handles_labels()
order = [0,2,3,4,5,1,6,7,8,9]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],ncol=2) 
ax.set_xticks(xticks+0.3, ('Efectividad', 'Precision', 'Recall', 'ROC AUC'))
fig.suptitle('Métricas de modelos entrenados', size=25,fontstyle='italic',fontweight='bold')

iterables=[['Modelos seleccionados','Modelos default'],ModelMetrics.columns]
Metrics = pd.concat([ModelMetrics,ModelMetricsDef],axis=1)
Metrics.columns=pd.MultiIndex.from_product(iterables, names=["", ""])
Metrics

ModelMetricsDef_TFID = pd.read_excel('ModelMetricsDef_TFID.xlsx',index_col=0)
ModelMetricsDef_d2v = pd.read_excel('ModelMetricsDef_d2v.xlsx',index_col=0)
ModelMetrics_TFID = pd.read_excel('ModelMetrics_TFID.xlsx',index_col=0)
ModelMetrics_d2v = pd.read_excel('ModelMetrics_d2v.xlsx',index_col=0)

fig = plt.figure(figsize=(15,8))
ax = fig.add_axes([0,0,1,1])
xticks = np.arange(4)
colors = ['b','r','g','y']
ax.plot([],[],' ',label="Modelos entrenados con TF-IDF")
for ind,i in enumerate(ModelMetricsDef_TFID.columns):
    ax.bar(xticks + ind/5, ModelMetricsDef_TFID[i], color = colors[ind], width = 1/7,alpha=0.5,label=i) 
plt.plot([],[],' ',label='Modelos entrenados con Doc2Vec')
for ind,i in enumerate(ModelMetricsDef_d2v.columns):
    ax.bar(xticks + ind/5, ModelMetricsDef_d2v[i],edgecolor =colors[ind], width = 1/9,label=i,fill=False)
handles, labels = plt.gca().get_legend_handles_labels()
order = [0,2,3,4,5,1,6,7,8,9]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],ncol=2) 
ax.set_xticks(xticks+0.3, ('Efectividad', 'Precision', 'Recall', 'ROC AUC'))
fig.suptitle('TF-IDF vs Doc2Vec\n (Modelos default)', y=1.115,size=25,fontstyle='italic',fontweight='bold')

fig = plt.figure(figsize=(15,8))
ax = fig.add_axes([0,0,1,1])
xticks = np.arange(4)
colors = ['b','r','g','y']
ax.plot([],[],' ',label="Modelos entrenados con TF-IDF")
for ind,i in enumerate(ModelMetrics_TFID.columns):
    ax.bar(xticks + ind/5, ModelMetrics_TFID[i], color = colors[ind], width = 1/7,alpha=0.5,label=i)
plt.plot([],[],' ',label='Modelos entrenados con Doc2Vec')
for ind,i in enumerate(ModelMetrics_d2v.columns):
    ax.bar(xticks + ind/5, ModelMetrics_d2v[i],edgecolor =colors[ind], width = 1/9,label=i,fill=False)
handles, labels = plt.gca().get_legend_handles_labels()
order = [0,2,3,4,5,1,6,7,8,9]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],ncol=2) 
ax.set_xticks(xticks+0.3, ('Efectividad', 'Precision', 'Recall', 'ROC AUC'))
fig.suptitle('TF-IDF vs Doc2Vec\n (Mejores modelos)', y=1.115,size=25,fontstyle='italic',fontweight='bold')

BestLogModel = SupervisedUtils.LogisticRegression(C=3.456887573126422,random_state=12345, n_jobs=-1)
BestRFModel = SupervisedUtils.RandomForestClassifier(max_depth=50,min_samples_leaf=1,min_samples_split=4,n_estimators=200,random_state=12345, n_jobs=-1)
BestKNNModel = SupervisedUtils.KNeighborsClassifier(leaf_size=50, n_jobs=-1, n_neighbors=50, weights='distance')

TMDb = pd.read_feather('../M1/DBM1')
movies = TMDb.copy()
movies = movies.loc[movies['overview'].isna()==False].reset_index(drop=True)
movies = movies.loc[movies['budget']>0].reset_index(drop=True)
movies = movies.loc[movies['vote_average']>0].reset_index(drop=True)
movies = movies.loc[movies['genres'].isna()==False].reset_index(drop=True)
movies = movies.drop(columns= ['poster_path','backdrop_path','day','revenue','status'])
movies = movies.dropna(subset=['year','month','runtime']).reset_index(drop=True)
movies['y']=pd.cut(movies['vote_average'],[0,6.5,10],labels=['Malo','Buena'],include_lowest=True)
le = SupervisedUtils.LabelEncoder()
le.fit(movies['y'])
y = le.transform(movies['y'])
y = pd.Series(y)
features = ['genres','original_language','overview','year','month','budget','runtime','tagline','keywords']
X = movies[features]
engstopwords = StringUtils.stopwords.words('english')
customsw  = ['one','film','movie','man','two','story']
engstopwords = engstopwords + customsw
stringvars = ['overview','keywords','tagline','overview_clean']
X['overview_clean']=StringUtils.clean_re(X['overview'])
X['overview_clean']=StringUtils.remove_stopwords(X['overview_clean'],engstopwords)
X['overview_clean']=StringUtils.remove_accents(X['overview_clean'])

num_records = X.shape[0]
bootstrap_errors_logit = []
bootstrap_errors_rf = []
bootstrap_errors_knn = []
bootstrap_errors_ens = []
np.random.seed(0)

for _ in tqdm(range(1250)):
    train_indices = np.random.choice(range(num_records), num_records, replace=True)
    test_indices = np.setdiff1d(range(num_records), train_indices)
    X_train_b, y_train_b = X.iloc[train_indices,:], y[train_indices]
    X_test_b, y_test_b = X.iloc[test_indices,:], y[test_indices]
    counts = X_train_b['genres'].str.split('-').explode().value_counts(True)
    mask = counts>0.05
    genres_mask = counts[mask].index
    genresaux = X_train_b['genres'].str.split('-')
    genresaux = genresaux.apply(lambda x: '-'.join(list(map(lambda y: 'OTROS' if y not in genres_mask else y,x))))
    genre_vc = StringUtils.CountVectorizer(tokenizer= lambda x: x.split('-'))
    genre_vc.fit(genresaux)
    genresaux = pd.DataFrame(genre_vc.transform(genresaux).toarray()).set_index(X_train_b.index)
    genresaux.columns = ['genre_' + str(col) for col in genre_vc.get_feature_names_out()]
    X_train_b = pd.concat([X_train_b,genresaux],axis=1).drop(columns = ['genres'])
    genresaux = X_test_b['genres'].str.split('-')
    genresaux = genresaux.apply(lambda x: '-'.join(list(map(lambda y: 'OTROS' if y not in genres_mask else y,x))))
    genresaux = pd.DataFrame(genre_vc.transform(genresaux).toarray()).set_index(X_test_b.index)
    genresaux.columns = ['genre_' + str(col) for col in genre_vc.get_feature_names_out()]
    X_test_b = pd.concat([X_test_b,genresaux],axis=1).drop(columns = ['genres'])
    counts = X_train_b['original_language'].value_counts(True)
    mask = counts>0.10
    language_mask = counts[mask].index
    X_train_b.loc[~X_train_b['original_language'].isin(language_mask),'original_language'] = 'OTROS'
    X_train_b = pd.get_dummies(X_train_b,columns = ['original_language'])
    X_test_b.loc[~X_test_b['original_language'].isin(language_mask),'original_language'] = 'OTROS'
    X_test_b = pd.get_dummies(X_test_b,columns = ['original_language'])
    overview_vc = StringUtils.TfidfVectorizer(max_features=100)
    overview_vc.fit(X_train_b['overview_clean'])
    X_overview_train = overview_vc.transform(X_train_b['overview_clean']).toarray()
    X_overview_train = pd.DataFrame(X_overview_train).set_index(X_train_b.index)
    X_overview_train.columns  = ['overview_' + str(j) for j in range(X_overview_train.shape[1])]
    X_overview_test  = overview_vc.transform(X_test_b['overview_clean']).toarray()
    X_overview_test = pd.DataFrame(X_overview_test).set_index(X_test_b.index)
    X_overview_test.columns  = ['overview_' + str(j) for j in range(X_overview_test.shape[1])]
    keywords_vc = StringUtils.TfidfVectorizer(tokenizer= lambda x: x.split('-'),max_features=100)
    keywords_vc.fit(X_train_b['keywords'].astype(str))
    X_keywords_train = keywords_vc.transform(X_train_b['keywords'].astype(str)).toarray()
    X_keywords_train = pd.DataFrame(X_keywords_train).set_index(X_train_b.index)
    X_keywords_train.columns  = ['keywords_' + str(j) for j in range(X_keywords_train.shape[1])]
    X_keywords_test  = keywords_vc.transform(X_test_b['keywords'].astype(str)).toarray()
    X_keywords_test = pd.DataFrame(X_keywords_test).set_index(X_test_b.index)
    X_keywords_test.columns  = ['keywords_' + str(j) for j in range(X_keywords_test.shape[1])]
    X_string_train = pd.concat([X_overview_train,X_keywords_train],axis=1)
    X_string_test = pd.concat([X_overview_test,X_keywords_test],axis=1)
    X_train_b = pd.concat([X_train_b.drop(columns=stringvars),X_string_train],axis=1)
    X_test_b = pd.concat([X_test_b.drop(columns=stringvars),X_string_test],axis=1)
    sc = MinMaxScaler()
    sc.fit(X_train_b)
    X_train_b = sc.transform(X_train_b)
    X_test_b = sc.transform(X_test_b)

    BestLogModel.fit(X_train_b, y_train_b)
    y_Log=BestLogModel.predict(X_test_b)
    bootstrap_errors_logit.append(SupervisedUtils.roc_auc_score(y_test_b, y_Log))

    BestRFModel.fit(X_train_b, y_train_b)
    y_RF=BestRFModel.predict(X_test_b)
    bootstrap_errors_rf.append(SupervisedUtils.roc_auc_score(y_test_b, y_RF))
    BestKNNModel.fit(X_train_b, y_train_b)
    y_KNN = BestKNNModel.predict(X_test_b)
    bootstrap_errors_knn.append(SupervisedUtils.roc_auc_score(y_test_b, y_KNN))

    y_preds_best= pd.DataFrame(y_Log, columns=['Regresión logística'])
    y_preds_best['Random Forest']=y_RF
    y_preds_best['KNN']=y_KNN
    y_preds_best['Ensable mejores modelos']=y_preds_best.mode(axis=1)
    y_ens = y_preds_best['Ensable mejores modelos']
    bootstrap_errors_ens.append(SupervisedUtils.roc_auc_score(y_test_b, y_ens))

bootstrapdb=pd.DataFrame(np.array([bootstrap_errors_logit,bootstrap_errors_rf,bootstrap_errors_knn,bootstrap_errors_ens]).transpose(),columns=['Regresión logistica','Random Forest','KNN','Ensamble: Mejores modelos'])
bootstrapdb.to_excel('BootstrapMetrics.xlsx')

bootstrap_metrics=bootstrapdb.describe(percentiles=[0.025,0.975])
bootstrap_metrics

plt.figure(figsize=(15,8))
sns.boxplot(data=bootstrapdb,palette='pastel',whis=(0.025,0.975),orient = 'h',fliersize=2,linewidth=2.5)
plt.xlabel('ROC AUC')
plt.title('Distribución de valores ROC AUC de nuestros modelos\n (Usando bootstrap)',size=20,fontweight='bold')

plt.figure(figsize=(15,8))
g = sns.barplot(x=bootstrap_metrics.T.index,y=bootstrap_metrics.T['mean'],palette="Blues_d")
g.bar_label(g.containers[0])
plt.title('Valor medio de los valores ROC AUC de nuestros modelos\n(usando bootstrap)',size=20,fontweight='bold')

fig, ax = plt.subplots(1,4,figsize=(20,7))
sns.histplot(bootstrapdb,x='Regresión logistica',ax=ax[0],element="step",alpha=0.5,color='red')
sns.histplot(bootstrapdb,x='Random Forest',ax=ax[1],element="step",alpha=0.5,color='blue')
sns.histplot(bootstrapdb,x='KNN',ax=ax[2],element="step",alpha=0.5,color='green')
sns.histplot(bootstrapdb,x='Ensamble: Mejores modelos',ax=ax[3],element="step",alpha=0.5,color='yellow')
fig.suptitle('Distribución de valores ROC AUC de nuestros modelos\n (Usando bootstrap)',size=20,fontweight='bold')

from keras import Model
from keras.models import Sequential
from keras.layers import Input
from keras.layers import BatchNormalization,Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Embedding, GRU, LSTM, MaxPooling1D, Conv1D
from keras.layers import concatenate
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
import tensorflow as tf
tf.config.list_physical_devices('GPU') 

from sklearn.preprocessing import MinMaxScaler

TMDb = pd.read_feather('../M1/DBM1')
movies = TMDb.copy()
movies = movies.loc[movies['overview'].isna()==False].reset_index(drop=True)
movies = movies.loc[movies['budget']>0].reset_index(drop=True)
movies = movies.loc[movies['vote_average']>0].reset_index(drop=True)
movies = movies.loc[movies['genres'].isna()==False].reset_index(drop=True)
movies = movies.drop(columns= ['poster_path','backdrop_path','day','revenue','status'])
movies = movies.dropna(subset=['year','month','runtime']).reset_index(drop=True)
movies['y']=pd.cut(movies['vote_average'],[0,6.5,10],labels=['Malo','Buena'],include_lowest=True)
le = SupervisedUtils.LabelEncoder()
le.fit(movies['y'])
y = le.transform(movies['y'])
features = ['genres','original_language','overview','year','month','budget','runtime','tagline','keywords']
X = movies[features]
engstopwords = StringUtils.stopwords.words('english')
customsw  = ['one','film','movie','man','two','story']
engstopwords = engstopwords + customsw
stringvars = ['overview','keywords','tagline','overview_clean']
X['overview_clean']=StringUtils.clean_re(X['overview'])
X['overview_clean']=StringUtils.remove_stopwords(X['overview_clean'],engstopwords)
X['overview_clean']=StringUtils.remove_accents(X['overview_clean'])
X_train, X_test, y_train, y_test = SupervisedUtils.train_test_split(X, y, test_size=0.3, random_state=12345)
counts = X_train['genres'].str.split('-').explode().value_counts(True)
mask = counts>0.05
genres_mask = counts[mask].index
genresaux = X_train['genres'].str.split('-')
genresaux = genresaux.apply(lambda x: '-'.join(list(map(lambda y: 'OTROS' if y not in genres_mask else y,x))))
genre_vc = StringUtils.CountVectorizer(tokenizer= lambda x: x.split('-'))
genre_vc.fit(genresaux)
genresaux = pd.DataFrame(genre_vc.transform(genresaux).toarray()).set_index(X_train.index)
genresaux.columns = ['genre_' + str(col) for col in genre_vc.get_feature_names_out()]
X_train = pd.concat([X_train,genresaux],axis=1).drop(columns = ['genres'])
genresaux = X_test['genres'].str.split('-')
genresaux = genresaux.apply(lambda x: '-'.join(list(map(lambda y: 'OTROS' if y not in genres_mask else y,x))))
genresaux = pd.DataFrame(genre_vc.transform(genresaux).toarray()).set_index(X_test.index)
genresaux.columns = ['genre_' + str(col) for col in genre_vc.get_feature_names_out()]
X_test = pd.concat([X_test,genresaux],axis=1).drop(columns = ['genres'])
counts = X_train['original_language'].value_counts(True)
mask = counts>0.10
language_mask = counts[mask].index
X_train.loc[~X_train['original_language'].isin(language_mask),'original_language'] = 'OTROS'
X_train = pd.get_dummies(X_train,columns = ['original_language'])
X_test.loc[~X_test['original_language'].isin(language_mask),'original_language'] = 'OTROS'
X_test = pd.get_dummies(X_test,columns = ['original_language'])
overview_vc = StringUtils.TfidfVectorizer(max_features=100)
overview_vc.fit(X_train['overview_clean'])
X_overview_train = overview_vc.transform(X_train['overview_clean']).toarray()
X_overview_train = pd.DataFrame(X_overview_train).set_index(X_train.index)
X_overview_train.columns  = ['overview_' + str(j) for j in range(X_overview_train.shape[1])]
X_overview_test  = overview_vc.transform(X_test['overview_clean']).toarray()
X_overview_test = pd.DataFrame(X_overview_test).set_index(X_test.index)
X_overview_test.columns  = ['overview_' + str(j) for j in range(X_overview_test.shape[1])]
keywords_vc = StringUtils.TfidfVectorizer(tokenizer= lambda x: x.split('-'),max_features=100)
keywords_vc.fit(X_train['keywords'].astype(str))
X_keywords_train = keywords_vc.transform(X_train['keywords'].astype(str)).toarray()
X_keywords_train = pd.DataFrame(X_keywords_train).set_index(X_train.index)
X_keywords_train.columns  = ['keywords_' + str(j) for j in range(X_keywords_train.shape[1])]
X_keywords_test  = keywords_vc.transform(X_test['keywords'].astype(str)).toarray()
X_keywords_test = pd.DataFrame(X_keywords_test).set_index(X_test.index)
X_keywords_test.columns  = ['keywords_' + str(j) for j in range(X_keywords_test.shape[1])]
X_string_train = pd.concat([X_overview_train,X_keywords_train],axis=1)
X_string_test = pd.concat([X_overview_test,X_keywords_test],axis=1)
X_train = pd.concat([X_train.drop(columns=stringvars),X_string_train],axis=1)
X_test = pd.concat([X_test.drop(columns=stringvars),X_string_test],axis=1)
sc = MinMaxScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

x_input = Input(shape = (X_train.shape[1],))
x = Dropout(0.2) (x_input)
x = Dense(100,activation = 'relu',kernel_regularizer='l2') (x_input)
pred = Dense(1,activation = 'sigmoid') (x)

model = Model(inputs=x_input, outputs=pred)
model.summary()

plot_model(model)

model.compile(optimizer=Adam(), loss="binary_crossentropy",metrics='accuracy')

early_stopping = EarlyStopping(monitor='val_accuracy', patience=50, min_delta=0.001,verbose=1,restore_best_weights=True)

history = model.fit(x = X_train,
                  y = y_train,
                  batch_size = 50,
                  epochs = 1000,
                  callbacks = early_stopping,
                  validation_data=(X_test,y_test)
                )

y_DL1 = model.predict(X_test)
y_DL1 = y_DL1.round(0)

acc_DL1, prec_DL1, rec_DL1, roc_DL1 = SupervisedUtils.model_cf('Redes neuronales\n(Estructura 1)',y_test,y_DL1)
SupervisedUtils.grafica_curva_roc(y_test,y_DL1)

TMDb = pd.read_feather('../M1/DBM1')
movies = TMDb.copy()
movies = movies.loc[movies['overview'].isna()==False].reset_index(drop=True)
movies = movies.loc[movies['budget']>0].reset_index(drop=True)
movies = movies.loc[movies['vote_average']>0].reset_index(drop=True)
movies = movies.loc[movies['genres'].isna()==False].reset_index(drop=True)
movies = movies.drop(columns= ['poster_path','backdrop_path','day','revenue','status'])
movies = movies.dropna(subset=['year','month','runtime']).reset_index(drop=True)
movies['y']=pd.cut(movies['vote_average'],[0,6.5,10],labels=['Malo','Buena'],include_lowest=True)
le = SupervisedUtils.LabelEncoder()
le.fit(movies['y'])
features = ['genres','original_language','overview','year','month','budget','runtime','tagline','keywords']
X = movies[features]
engstopwords = StringUtils.stopwords.words('english')
customsw  = ['one','film','movie','man','two','story']
engstopwords = engstopwords + customsw
stringvars = ['overview','keywords','tagline','overview_clean']
X['overview_clean']=StringUtils.clean_re(X['overview'])
X['overview_clean']=StringUtils.remove_stopwords(X['overview_clean'],engstopwords)
X['overview_clean']=StringUtils.remove_accents(X['overview_clean'])
X_train, X_test, y_train, y_test = SupervisedUtils.train_test_split(X, y, test_size=0.3, random_state=12345)
counts = X_train['genres'].str.split('-').explode().value_counts(True)
mask = counts>0.05
genres_mask = counts[mask].index
genresaux = X_train['genres'].str.split('-')
genresaux = genresaux.apply(lambda x: '-'.join(list(map(lambda y: 'OTROS' if y not in genres_mask else y,x))))
genre_vc = StringUtils.CountVectorizer(tokenizer= lambda x: x.split('-'))
genre_vc.fit(genresaux)
X_train_genre = pd.DataFrame(genre_vc.transform(genresaux).toarray()).set_index(X_train.index)
X_train_genre.columns = ['genre_' + str(col) for col in genre_vc.get_feature_names_out()]
genresaux = X_test['genres'].str.split('-')
genresaux = genresaux.apply(lambda x: '-'.join(list(map(lambda y: 'OTROS' if y not in genres_mask else y,x))))
X_test_genre = pd.DataFrame(genre_vc.transform(genresaux).toarray()).set_index(X_test.index)
X_test_genre.columns = ['genre_' + str(col) for col in genre_vc.get_feature_names_out()]
counts = X_train['original_language'].value_counts(True)
mask = counts>0.10
language_mask = counts[mask].index
X_train.loc[~X_train['original_language'].isin(language_mask),'original_language'] = 'OTROS'
X_train_language = pd.get_dummies(X_train['original_language'])
X_test.loc[~X_test['original_language'].isin(language_mask),'original_language'] = 'OTROS'
X_test_language = pd.get_dummies(X_test['original_language'])
overview_vc = StringUtils.TfidfVectorizer(max_features=100)
overview_vc.fit(X_train['overview_clean'])
X_overview_train = overview_vc.transform(X_train['overview_clean']).toarray()
X_overview_train = pd.DataFrame(X_overview_train).set_index(X_train.index)
X_overview_train.columns  = ['overview_' + str(j) for j in range(X_overview_train.shape[1])]
X_overview_test  = overview_vc.transform(X_test['overview_clean']).toarray()
X_overview_test = pd.DataFrame(X_overview_test).set_index(X_test.index)
X_overview_test.columns  = ['overview_' + str(j) for j in range(X_overview_test.shape[1])]
keywords_vc = StringUtils.TfidfVectorizer(tokenizer= lambda x: x.split('-'),max_features=100)
keywords_vc.fit(X_train['keywords'].astype(str))
X_keywords_train = keywords_vc.transform(X_train['keywords'].astype(str)).toarray()
X_keywords_train = pd.DataFrame(X_keywords_train).set_index(X_train.index)
X_keywords_train.columns  = ['keywords_' + str(j) for j in range(X_keywords_train.shape[1])]
X_keywords_test  = keywords_vc.transform(X_test['keywords'].astype(str)).toarray()
X_keywords_test = pd.DataFrame(X_keywords_test).set_index(X_test.index)
X_keywords_test.columns  = ['keywords_' + str(j) for j in range(X_keywords_test.shape[1])]
X_train_feat = X_train[['year','month','budget','runtime']]
X_test_feat = X_test[['year','month','budget','runtime']]
sc_feat = MinMaxScaler()
sc_feat.fit(X_train_feat)
X_train_feat = sc_feat.transform(X_train_feat)
X_test_feat = sc_feat.transform(X_test_feat)

x_genre_input = Input(shape = (X_train_genre.shape[1],), name = 'genre')
x_language_input = Input(shape = (X_train_language.shape[1],), name = 'language')
x_overview_input = Input(shape = (X_overview_train.shape[1],), name = 'overview')
x_keywords_input = Input(shape = (X_keywords_train.shape[1],), name = 'keywords')
x_feat_input = Input(shape= (X_train_feat.shape[1],), name = 'features')
x_cat = concatenate([x_genre_input,x_language_input])
x_cat = Dense(5,activation = 'tanh')(x_cat)
x_str = concatenate([x_overview_input,x_keywords_input])
x_str = Dense(100,activation = 'relu') (x_str)
x = concatenate([x_cat,x_str,x_feat_input])
x = Dense(5,activation = 'tanh',kernel_regularizer='l2')(x)
pred = Dense(1,activation = 'sigmoid') (x)

model = Model(inputs=[x_genre_input,x_language_input,x_overview_input,x_keywords_input,x_feat_input], outputs=pred)
model.summary()

plot_model(model)

model.compile(optimizer=Adam(), loss="binary_crossentropy",metrics='accuracy')

early_stopping = EarlyStopping(monitor='val_accuracy', patience=50, min_delta=0.001,verbose=1,restore_best_weights=True)

history = model.fit(x = [X_train_genre,X_train_language,X_overview_train,X_keywords_train,X_train_feat],
                  y = y_train,
                  batch_size = 50,
                  epochs = 1000,
                  callbacks = early_stopping,
                  validation_data=([X_test_genre,X_test_language,X_overview_test,X_keywords_test,X_test_feat],y_test)
                )

y_DL2 = model.predict([X_test_genre,X_test_language,X_overview_test,X_keywords_test,X_test_feat])
y_DL2 = y_DL2.round(0)

acc_DL2, prec_DL2, rec_DL2, roc_DL2 = SupervisedUtils.model_cf('Redes neuronales\n(Estructura 2)',y_test,y_DL2)
SupervisedUtils.grafica_curva_roc(y_test,y_DL2)

TMDb = pd.read_feather('../M1/DBM1')

movies = TMDb.copy()
movies = movies.loc[movies['overview'].isna()==False].reset_index(drop=True)
movies = movies.loc[movies['budget']>0].reset_index(drop=True)
movies = movies.loc[movies['vote_average']>0].reset_index(drop=True)
movies = movies.loc[movies['genres'].isna()==False].reset_index(drop=True)
movies = movies.drop(columns= ['poster_path','backdrop_path','day','revenue','status'])
movies = movies.dropna(subset=['year','month','runtime']).reset_index(drop=True)
movies['y']=pd.cut(movies['vote_average'],[0,6.5,10],labels=['Malo','Buena'],include_lowest=True)
le = SupervisedUtils.LabelEncoder()
le.fit(movies['y'])
y = le.transform(movies['y'])
features = ['genres','original_language','overview','year','month','budget','runtime','tagline','keywords']
X = movies[features]
engstopwords = StringUtils.stopwords.words('english')
customsw  = ['one','film','movie','man','two','story']
engstopwords = engstopwords + customsw
stringvars = ['overview','keywords','tagline','overview_clean']
X['overview_clean']=StringUtils.clean_re(X['overview'])
X['overview_clean']=StringUtils.remove_stopwords(X['overview_clean'],engstopwords)
X['overview_clean']=StringUtils.remove_accents(X['overview_clean'])
X['keywords'] = X['keywords'].astype(str)
X_train, X_test, y_train, y_test = SupervisedUtils.train_test_split(X, y, test_size=0.3, random_state=12345)
counts = X_train['genres'].str.split('-').explode().value_counts(True)
mask = counts>0.05
genres_mask = counts[mask].index
genresaux = X_train['genres'].str.split('-')
genresaux = genresaux.apply(lambda x: '-'.join(list(map(lambda y: 'OTROS' if y not in genres_mask else y,x))))
genre_vc = StringUtils.CountVectorizer(tokenizer= lambda x: x.split('-'))
genre_vc.fit(genresaux)
X_train_genre = pd.DataFrame(genre_vc.transform(genresaux).toarray()).set_index(X_train.index)
X_train_genre.columns = ['genre_' + str(col) for col in genre_vc.get_feature_names_out()]
genresaux = X_test['genres'].str.split('-')
genresaux = genresaux.apply(lambda x: '-'.join(list(map(lambda y: 'OTROS' if y not in genres_mask else y,x))))
X_test_genre = pd.DataFrame(genre_vc.transform(genresaux).toarray()).set_index(X_test.index)
X_test_genre.columns = ['genre_' + str(col) for col in genre_vc.get_feature_names_out()]
counts = X_train['original_language'].value_counts(True)
mask = counts>0.10
language_mask = counts[mask].index
X_train.loc[~X_train['original_language'].isin(language_mask),'original_language'] = 'OTROS'
X_train_language = pd.get_dummies(X_train['original_language'])
X_test.loc[~X_test['original_language'].isin(language_mask),'original_language'] = 'OTROS'
X_test_language = pd.get_dummies(X_test['original_language'])

VocabOverview = StringUtils.Vocabulary(X_train,'overview_clean')
TokenOverview = StringUtils.VectorizeVariable(StringUtils.Tokenizer(num_words = 10000),X_train['overview_clean'],100)
X_overview_train = TokenOverview.X_pad
X_overview_test = TokenOverview.transform(X_test['overview_clean'])
VocabKeywords = StringUtils.Vocabulary(X_train,'keywords','-')
TokenKeywords = StringUtils.VectorizeVariable(StringUtils.Tokenizer(num_words = 7000),X_train['keywords'],100)
X_keywords_train = TokenKeywords.X_pad
X_keywords_test = TokenKeywords.transform(X_test['keywords'])
X_train_feat = X_train[['year','month','budget','runtime']]
X_test_feat = X_test[['year','month','budget','runtime']]
sc_feat = MinMaxScaler()
sc_feat.fit(X_train_feat)
X_train_feat = sc_feat.transform(X_train_feat)
X_test_feat = sc_feat.transform(X_test_feat)

x_genre_input = Input(shape = (X_train_genre.shape[1],), name = 'genre')
x_language_input = Input(shape = (X_train_language.shape[1],),name = 'language')
x_feat_input = Input(shape= (X_train_feat.shape[1],), name = 'features')
x_overview_input = Input(shape = (100,), name = 'overview')
overview_embedding = Embedding(100000, 100)(x_overview_input)
overview_layer = Conv1D(64,5) (overview_embedding)
overview_layer = MaxPooling1D(3) (overview_layer)
overview_layer = LSTM(15) (overview_layer)
x_keywords_input = Input(shape = (100,), name = 'keywords')
keywords_embedding = Embedding(100000, 100)(x_keywords_input)
keywords_layer = Conv1D(32,3) (keywords_embedding)
keywords_layer = MaxPooling1D(2) (keywords_layer)
keywords_layer = LSTM(5) (keywords_layer)
x_cat = concatenate([x_genre_input,x_language_input])
x_cat = Dense(15,activation = 'relu')(x_cat)
x_str = concatenate([overview_layer,keywords_layer])
x_str = Dense(50) (x_str)
x = concatenate([x_cat,x_str,x_feat_input])
x = Dropout(0.2) (x)
x = Dense(100,activation = 'relu')(x)
pred = Dense(1,activation = 'sigmoid') (x)

model = Model(inputs=[x_genre_input,x_language_input,x_overview_input,x_keywords_input,x_feat_input], outputs=pred)
model.summary()

plot_model(model, show_shapes=False,dpi=200)

model.compile(optimizer=Adam(), loss="binary_crossentropy",metrics='accuracy')

early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, min_delta=0.001,verbose=1,restore_best_weights=True)

history = model.fit(x = [X_train_genre,X_train_language,X_overview_train,X_keywords_train,X_train_feat],
                  y = y_train,
                  batch_size = 50,
                  epochs = 1000,
                  callbacks = early_stopping,
                  validation_data=([X_test_genre,X_test_language,X_overview_test,X_keywords_test,X_test_feat],y_test)
                )

y_DL3 = model.predict([X_test_genre,X_test_language,X_overview_test,X_keywords_test,X_test_feat])
y_DL3 = y_DL3.round(0)

acc_DL3, prec_DL3, rec_DL3, roc_DL3 = SupervisedUtils.model_cf('Redes neuronales\n(Estructura 3)',y_test,y_DL3)
SupervisedUtils.grafica_curva_roc(y_test,y_DL3)

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

TMDb = pd.read_feather('../M1/DBM1')
movies = TMDb.copy()
movies = movies.loc[movies['overview'].isna()==False].reset_index(drop=True)
movies = movies.loc[movies['budget']>0].reset_index(drop=True)
movies = movies.loc[movies['vote_average']>0].reset_index(drop=True)
movies = movies.loc[movies['genres'].isna()==False].reset_index(drop=True)
movies = movies.drop(columns= ['poster_path','backdrop_path','day','revenue','status'])
movies = movies.dropna(subset=['year','month','runtime']).reset_index(drop=True)
movies['y']=pd.cut(movies['vote_average'],[0,6.5,10],labels=['Malo','Buena'],include_lowest=True)
le = SupervisedUtils.LabelEncoder()
le.fit(movies['y'])
y = le.transform(movies['y'])
features = ['genres','original_language','overview','year','month','budget','runtime','tagline','keywords']
X = movies[features]
engstopwords = StringUtils.stopwords.words('english')
customsw  = ['one','film','movie','man','two','story']
engstopwords = engstopwords + customsw
stringvars = ['overview','keywords','tagline','overview_clean']
X['overview_clean']=StringUtils.clean_re(X['overview'])
X['overview_clean']=StringUtils.remove_stopwords(X['overview_clean'],engstopwords)
X['overview_clean']=StringUtils.remove_accents(X['overview_clean'])
X_train, X_test, y_train, y_test = SupervisedUtils.train_test_split(X, y, test_size=0.3, random_state=12345)
counts = X_train['genres'].str.split('-').explode().value_counts(True)
mask = counts>0.05
genres_mask = counts[mask].index
genresaux = X_train['genres'].str.split('-')
genresaux = genresaux.apply(lambda x: '-'.join(list(map(lambda y: 'OTROS' if y not in genres_mask else y,x))))
genre_vc = StringUtils.CountVectorizer(tokenizer= lambda x: x.split('-'))
genre_vc.fit(genresaux)
genresaux = pd.DataFrame(genre_vc.transform(genresaux).toarray()).set_index(X_train.index)
genresaux.columns = ['genre_' + str(col) for col in genre_vc.get_feature_names_out()]
X_train = pd.concat([X_train,genresaux],axis=1).drop(columns = ['genres'])
genresaux = X_test['genres'].str.split('-')
genresaux = genresaux.apply(lambda x: '-'.join(list(map(lambda y: 'OTROS' if y not in genres_mask else y,x))))
genresaux = pd.DataFrame(genre_vc.transform(genresaux).toarray()).set_index(X_test.index)
genresaux.columns = ['genre_' + str(col) for col in genre_vc.get_feature_names_out()]
X_test = pd.concat([X_test,genresaux],axis=1).drop(columns = ['genres'])
counts = X_train['original_language'].value_counts(True)
mask = counts>0.10
language_mask = counts[mask].index
X_train.loc[~X_train['original_language'].isin(language_mask),'original_language'] = 'OTROS'
language_encoder = OneHotEncoder()
language_encoder.fit(X_train['original_language'].to_numpy().reshape(-1,1))
languagesaux = pd.DataFrame(language_encoder.transform(X_train['original_language'].to_numpy().reshape(-1,1)).toarray()).set_index(X_train.index)
languagesaux.columns = ['original_language_' + str(x.replace('x0_','')) for x in language_encoder.get_feature_names_out()]
X_train = pd.concat([X_train,languagesaux],axis=1).drop(columns = ['original_language'])
X_test.loc[~X_test['original_language'].isin(language_mask),'original_language'] = 'OTROS'
languagesaux = pd.DataFrame(language_encoder.transform(X_test['original_language'].to_numpy().reshape(-1,1)).toarray()).set_index(X_test.index)
languagesaux.columns = ['original_language_' + str(x.replace('x0_','')) for x in language_encoder.get_feature_names_out()]
X_test = pd.concat([X_test,languagesaux],axis=1).drop(columns = ['original_language'])
overview_vc = StringUtils.TfidfVectorizer(max_features=100)
overview_vc.fit(X_train['overview_clean'])
X_overview_train = overview_vc.transform(X_train['overview_clean']).toarray()
X_overview_train = pd.DataFrame(X_overview_train).set_index(X_train.index)
X_overview_train.columns  = ['overview_' + str(j) for j in range(X_overview_train.shape[1])]
X_overview_test  = overview_vc.transform(X_test['overview_clean']).toarray()
X_overview_test = pd.DataFrame(X_overview_test).set_index(X_test.index)
X_overview_test.columns  = ['overview_' + str(j) for j in range(X_overview_test.shape[1])]
keywords_vc = StringUtils.TfidfVectorizer(tokenizer= lambda x: x.split('-'),max_features=100)
keywords_vc.fit(X_train['keywords'].astype(str))
X_keywords_train = keywords_vc.transform(X_train['keywords'].astype(str)).toarray()
X_keywords_train = pd.DataFrame(X_keywords_train).set_index(X_train.index)
X_keywords_train.columns  = ['keywords_' + str(j) for j in range(X_keywords_train.shape[1])]
X_keywords_test  = keywords_vc.transform(X_test['keywords'].astype(str)).toarray()
X_keywords_test = pd.DataFrame(X_keywords_test).set_index(X_test.index)
X_keywords_test.columns  = ['keywords_' + str(j) for j in range(X_keywords_test.shape[1])]
X_string_train = pd.concat([X_overview_train,X_keywords_train],axis=1)
X_string_test = pd.concat([X_overview_test,X_keywords_test],axis=1)
X_train = pd.concat([X_train.drop(columns=stringvars),X_string_train],axis=1)
X_test = pd.concat([X_test.drop(columns=stringvars),X_string_test],axis=1)
sc = MinMaxScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

BestRFModel = SupervisedUtils.RandomForestClassifier(max_depth=50,min_samples_leaf=1,min_samples_split=4,n_estimators=200,random_state=12345, n_jobs=-1)
BestRFModel.fit(X_train,y_train)
y_BestRFModel=BestRFModel.predict(X_test)
BestRFModel_train_score = BestRFModel.score(X_train,y_train)
print(f'Score en train: {BestRFModel_train_score}')
BestRFModel_test_score = BestRFModel.score(X_test,y_test)
print(f'Score en test: {BestRFModel_test_score}')
acc_BestRFModel, prec_BestRFModel, rec_BestRFModel, roc_BestRFModel = SupervisedUtils.model_cf('\nRandom Forest\n (Mejor modelo)',y_test,y_BestRFModel)

genre_predict = 'Drama-Thriller-Animation'
original_language_predict = 'en'
overview_predict = 'end world with zombies and cowboys'
year_predict = 2023
month_predict = 2
budget_predict = 10000000
runtime_predict = 120
keywords_predict = 'love'

X_predict = pd.DataFrame([[genre_predict,original_language_predict,overview_predict,year_predict,month_predict,budget_predict,runtime_predict,keywords_predict]],columns = ['genres','original_language','overview','year','month','budget','runtime','keywords'])
X_predict

X_predict['overview_clean']=StringUtils.clean_re(X_predict['overview'])
X_predict['overview_clean']=StringUtils.remove_stopwords(X_predict['overview_clean'],engstopwords)
X_predict['overview_clean']=StringUtils.remove_accents(X_predict['overview_clean'])

genresaux = X_predict['genres'].str.split('-')
genresaux = genresaux.apply(lambda x: '-'.join(list(map(lambda y: 'OTROS' if y not in genres_mask else y,x))))
genresaux = pd.DataFrame(genre_vc.transform(genresaux).toarray()).set_index(X_predict.index)
genresaux.columns = ['genre_' + str(col) for col in genre_vc.get_feature_names_out()]
X_predict = pd.concat([X_predict,genresaux],axis=1).drop(columns = ['genres'])

X_predict.loc[~X_predict['original_language'].isin(language_mask),'original_language'] = 'OTROS'
languagesaux = pd.DataFrame(language_encoder.transform(X_predict['original_language'].to_numpy().reshape(-1,1)).toarray()).set_index(X_predict.index)
languagesaux.columns = ['original_language_' + str(x.replace('x0_','')) for x in language_encoder.get_feature_names_out()]
X_predict = pd.concat([X_predict,languagesaux],axis=1).drop(columns = ['original_language'])

X_overview_predict  = overview_vc.transform(X_predict['overview_clean']).toarray()
X_overview_predict = pd.DataFrame(X_overview_predict).set_index(X_predict.index)
X_overview_predict.columns  = ['overview_' + str(j) for j in range(X_overview_predict.shape[1])]

X_keywords_predict  = keywords_vc.transform(X_predict['keywords'].astype(str)).toarray()
X_keywords_predict = pd.DataFrame(X_keywords_predict).set_index(X_predict.index)
X_keywords_predict.columns  = ['keywords_' + str(j) for j in range(X_keywords_predict.shape[1])]

X_string_predict = pd.concat([X_overview_predict,X_keywords_predict],axis=1)
X_predict = pd.concat([X_predict.drop(columns=['overview','keywords','overview_clean']),X_string_predict],axis=1)
X_predict
X_predict = sc.transform(X_predict)

y_predict = BestRFModel.predict(X_predict)
le.inverse_transform(y_predict)
y_predict = BestRFModel.predict_proba(X_predict)
y_predict

genre_predict = 'Drama-Thriller-Animation'
original_language_predict = 'en'
overview_predict = 'end world with zombies and cowboys'
year_predict = 2023
month_predict = 2
budget_predict = 10000000
runtime_predict = 120
keywords_predict = 'love-frienship'

X_predict = pd.DataFrame([[genre_predict,original_language_predict,overview_predict,year_predict,month_predict,budget_predict,runtime_predict,keywords_predict]],columns = ['genres','original_language','overview','year','month','budget','runtime','keywords'])
X_predict

X_predict['overview_clean']=StringUtils.clean_re(X_predict['overview'])
X_predict['overview_clean']=StringUtils.remove_stopwords(X_predict['overview_clean'],engstopwords)
X_predict['overview_clean']=StringUtils.remove_accents(X_predict['overview_clean'])
genresaux = X_predict['genres'].str.split('-')
genresaux = genresaux.apply(lambda x: '-'.join(list(map(lambda y: 'OTROS' if y not in genres_mask else y,x))))
genresaux = pd.DataFrame(genre_vc.transform(genresaux).toarray()).set_index(X_predict.index)
genresaux.columns = ['genre_' + str(col) for col in genre_vc.get_feature_names_out()]
X_predict = pd.concat([X_predict,genresaux],axis=1).drop(columns = ['genres'])

X_predict.loc[~X_predict['original_language'].isin(language_mask),'original_language'] = 'OTROS'
languagesaux = pd.DataFrame(language_encoder.transform(X_predict['original_language'].to_numpy().reshape(-1,1)).toarray()).set_index(X_predict.index)
languagesaux.columns = ['original_language_' + str(x.replace('x0_','')) for x in language_encoder.get_feature_names_out()]
X_predict = pd.concat([X_predict,languagesaux],axis=1).drop(columns = ['original_language'])
X_overview_predict  = overview_vc.transform(X_predict['overview_clean']).toarray()
X_overview_predict = pd.DataFrame(X_overview_predict).set_index(X_predict.index)
X_overview_predict.columns  = ['overview_' + str(j) for j in range(X_overview_predict.shape[1])]
X_keywords_predict  = keywords_vc.transform(X_predict['keywords'].astype(str)).toarray()
X_keywords_predict = pd.DataFrame(X_keywords_predict).set_index(X_predict.index)
X_keywords_predict.columns  = ['keywords_' + str(j) for j in range(X_keywords_predict.shape[1])]
X_string_predict = pd.concat([X_overview_predict,X_keywords_predict],axis=1)
X_predict = pd.concat([X_predict.drop(columns=['overview','keywords','overview_clean']),X_string_predict],axis=1)
X_predict
X_predict = sc.transform(X_predict)

y_predict = BestRFModel.predict(X_predict)
le.inverse_transform(y_predict)
y_predict = BestRFModel.predict_proba(X_predict)
y_predict
