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
    # choose from all lowercase letter
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
    
    

def plot_method_all(method,tad,title=None):
    n_colors = len(set(tad[method]))

    fig1 = px.scatter(x=X_PCA['C1'],y=X_PCA['C2'],color=tad[method].astype(str), symbol=tad[method].astype(str))
    fig2 = px.scatter(x=X_PCA['C1'],y=X_PCA['C3'],color=tad[method].astype(str), symbol=tad[method].astype(str))
    fig3 = px.scatter(x=X_PCA['C1'],y=X_PCA['C4'],color=tad[method].astype(str), symbol=tad[method].astype(str))
    fig4 = px.scatter(x=X_PCA['C2'],y=X_PCA['C3'],color=tad[method].astype(str), symbol=tad[method].astype(str))
    fig5 = px.scatter(x=X_PCA['C2'],y=X_PCA['C4'],color=tad[method].astype(str), symbol=tad[method].astype(str))
    fig6 = px.scatter(x=X_PCA['C3'],y=X_PCA['C4'],color=tad[method].astype(str), symbol=tad[method].astype(str))


    fig11 = px.scatter(x=X_MDS['V1'],y=X_MDS['V2'],color=tad[method].astype(str), symbol=tad[method].astype(str))
    fig22 = px.scatter(x=X_MDS['V1'],y=X_MDS['V3'],color=tad[method].astype(str), symbol=tad[method].astype(str))
    fig33 = px.scatter(x=X_MDS['V1'],y=X_MDS['V4'],color=tad[method].astype(str), symbol=tad[method].astype(str))
    fig44 = px.scatter(x=X_MDS['V2'],y=X_MDS['V3'],color=tad[method].astype(str), symbol=tad[method].astype(str))
    fig55 = px.scatter(x=X_MDS['V2'],y=X_MDS['V4'],color=tad[method].astype(str), symbol=tad[method].astype(str))
    fig66 = px.scatter(x=X_MDS['V3'],y=X_MDS['V4'],color=tad[method].astype(str), symbol=tad[method].astype(str))


    fig = make_subplots(rows=3, cols=4,subplot_titles=("PCA", "MDS","PCA", "MDS"))

    for i in range(n_colors):
        fig.add_trace(fig1['data'][i], row=1, col=1)
        fig.add_trace(fig11['data'][i], row=1, col=2)

        fig.add_trace(fig2['data'][i], row=2, col=1)
        fig.add_trace(fig22['data'][i], row=2, col=2)

        fig.add_trace(fig3['data'][i], row=3, col=1)
        fig.add_trace(fig33['data'][i], row=3, col=2)

        fig.add_trace(fig4['data'][i], row=1, col=3)
        fig.add_trace(fig44['data'][i], row=1, col=4)

        fig.add_trace(fig5['data'][i], row=2, col=3)
        fig.add_trace(fig55['data'][i], row=2, col=4)

        fig.add_trace(fig6['data'][i], row=3, col=3)
        fig.add_trace(fig66['data'][i], row=3, col=4)


    names = set()
    fig.for_each_trace(
        lambda trace:
            trace.update(showlegend=False)
            if (trace.name in names) else names.add(trace.name))

    fig['layout']['xaxis']['title']="C1"
    fig['layout']['yaxis']['title']='C2'
    fig['layout']['xaxis2']['title']="V1"
    fig['layout']['yaxis2']['title']='V2'

    fig['layout']['xaxis3']['title']="C2"
    fig['layout']['yaxis3']['title']='C3'
    fig['layout']['xaxis4']['title']="V2"
    fig['layout']['yaxis4']['title']='V3'

    fig['layout']['xaxis5']['title']="C1"
    fig['layout']['yaxis5']['title']='C3'
    fig['layout']['xaxis6']['title']="V1"
    fig['layout']['yaxis6']['title']='V3'

    fig['layout']['xaxis7']['title']="C2"
    fig['layout']['yaxis7']['title']='C4'
    fig['layout']['xaxis8']['title']="V2"
    fig['layout']['yaxis8']['title']='V4'

    fig['layout']['xaxis9']['title']="C1"
    fig['layout']['yaxis9']['title']='C4'
    fig['layout']['xaxis10']['title']="V1"
    fig['layout']['yaxis10']['title']='V4'

    fig['layout']['xaxis11']['title']="C3"
    fig['layout']['yaxis11']['title']='C4'
    fig['layout']['xaxis12']['title']="V3"
    fig['layout']['yaxis12']['title']='V4'





    fig.update_traces(marker=dict(size=5,
                                  line=dict(width=0.2,
                                            color='white')),
                      selector=dict(mode='markers'))
    
    if title:
        fig.update_layout(height=1000, width=1250, title_text=title)
    else:
        fig.update_layout(height=1000, width=1250, title_text= method+ ": " + str(n_colors) + " Clusters")
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