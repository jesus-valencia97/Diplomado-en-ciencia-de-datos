import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from scipy.stats import kruskal
from sklearn.metrics import silhouette_score
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def get_random_string(length):

    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    result_str = 'trybase_' + result_str
    return result_str

def plot_method(df,var1,var2,tad,method,title=None):
    fig = px.scatter(x=df[var1],y=df[var2],color=tad[method].astype(str), symbol=tad[method].astype(str))
    fig.update_traces(marker=dict(size=5,
                                  line=dict(width=0.2,
                                            color='white')),
                      selector=dict(mode='markers'))
    if title:
        fig.update_layout(height=500, width=750,title=title)
        fig.show()
    else:
        fig.update_layout(height=500, width=750)
        fig.show()

    
def inertia(df, min_, max_):
    
    x = [i for i in range(min_, max_)]
    inertia_ = []
    
    for ki in x:
        km = KMeans(n_clusters=ki)
        km.fit(df)
        inertia_.append(km.inertia_)
        
    sns.lineplot(
        x=x, y=inertia_,
        marker="o"
    )
    
def pruebas_hipotesis(df1, df2, col_list):
    p_values = []
    dec = []
    for col in col_list:
        stat, p_value = kruskal(df1[col], df2[col])
        p_values.append(p_value)
        if p_value <= 0.05:
            decision = 'Distribución diferente'
        else:
            decision = 'Distribución parecida'
        dec.append(decision)
    return pd.DataFrame(data = {'variables': col_list, 'p_value': p_values, 'decision': dec})