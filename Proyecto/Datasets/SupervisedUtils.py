
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error, roc_curve, confusion_matrix, roc_auc_score
from sklearn.model_selection import validation_curve

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors               import KNeighborsClassifier
from sklearn.ensemble                import RandomForestClassifier
from sklearn.model_selection         import StratifiedKFold

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV


def plot_cf(cf_matrix,wtitle=True,ax=None):
    
    off_diag_mask = np.eye(*cf_matrix.shape, dtype=bool)

    
    vmine = np.min(cf_matrix[~off_diag_mask])
    vmaxe = np.max(cf_matrix[~off_diag_mask])+4*np.mean(cf_matrix[~off_diag_mask])
    vmin = np.min(cf_matrix[off_diag_mask])-np.mean(cf_matrix[off_diag_mask])
    vmax = np.max(cf_matrix)
    sns.heatmap(cf_matrix, annot=True,fmt='g', mask=~off_diag_mask, cmap='Blues', vmin=vmin, vmax=vmax,cbar=False,square=True,linewidths=2,annot_kws={'fontsize':'x-large','weight':'bold'},ax=ax)
    sns.heatmap(cf_matrix, annot=True,fmt='g', mask=off_diag_mask, cmap='Blues', vmin=vmine, vmax=vmaxe, cbar=False,square=True,linewidths=2,ax=ax)
    if wtitle and ax is None:
        plt.title("Matriz de confusión", fontsize=20,fontfamily='fantasy')
    if wtitle and ax is not None:
        ax.set_title("Matriz de confusión", fontsize='xx-large',fontfamily='fantasy')

def grafica_curva_roc(y, y_pred, title='Curva ROC'):
    """
    Función para graficar la curva ROC
    Los parámetros fpr y tpr son el output de ejecutar la función roc_curve de sklearn
    
    Args:
        fpr        Tasa de falsos positivos 
        tpr        Tasa de verdaderos positivos
        title      sting para definir el título de la gráfica
        note       Nota para mostrar en la gráfica
    """
    fpr, tpr, _ = roc_curve(y, y_pred)
    roc_score=roc_auc_score(y, y_pred)
    
    plt.figure(figsize=(12,6))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('Tasa de falsos positivos ')
    plt.ylabel('Tasa de verdaderos positivos')
    plt.title(title)
    plt.text(0.6, 0.2, f'ROC AUC: {round(roc_score,4)}',size=15)
    plt.show()
    

def model_cf(label,y,y_pred):

    cf_matrix = confusion_matrix(y, y_pred)
 
    
    n_i = list(map(sum,cf_matrix))
    perrors=1-(cf_matrix.diagonal()/list(map(sum,cf_matrix)))
    errors = n_i*perrors
    x=[1]*len(errors)
    

    recall = round(cf_matrix[1][1]/(cf_matrix[1][1] + cf_matrix[1][0]),5)
    precision=round(cf_matrix[1][1]/(cf_matrix[1][1] + cf_matrix[0][1]),5)
    roc_score=roc_auc_score(y, y_pred)
    efectividad = round((1-np.sum(errors)/len(y)),5)
    
    
    plt.figure(figsize=(15, 6), dpi=80)
    
    ax1 = plt.subplot2grid((1, 20), (0, 0),colspan=6)
    ax2 = plt.subplot2grid((1, 20), (0, 8),colspan=8)
    ax3 = plt.subplot2grid((1, 20), (0, 16))
    
    ax1.axis('off')
    ax1.text(0.5,0.55,label,fontstyle='italic',fontsize=40,ha='center',fontweight='bold')
    
    ax1.text(0.20,0.30,'Efectividad:',fontstyle='italic',fontsize=15,ha='center',fontweight='bold')
    ax1.text(0.20,0.20,f'{format(efectividad*100, ".15g")}%',fontstyle='italic',fontsize=30,ha='center')
    
    ax1.text(0.80,0.30,'Precision:',fontstyle='italic',fontsize=15,ha='center',fontweight='bold')
    ax1.text(0.80,0.20,f'{format(precision*100, ".15g")}%',fontstyle='italic',fontsize=30,ha='center')
    
    ax1.text(0.20,0.10,'Recall:',fontstyle='italic',fontsize=15,ha='center',fontweight='bold')
    ax1.text(0.20,0,f'{format(recall*100, ".15g")}%',fontstyle='italic',fontsize=30,ha='center')
    
    ax1.text(0.80,0.1,'ROC AUC:',fontstyle='italic',fontsize=15,ha='center',fontweight='black')
    ax1.text(0.80,0,round(roc_score,5),fontstyle='italic',fontsize=30,ha='center')
    
    plot_cf(cf_matrix,True,ax2)
    
    ax3.sharey(ax2)
    ax3.axis('off')
    #ax3.set_xlim(0.1,0.3)
    #ax3.set_xlim(0,2)
    #plt.tight_layout()
    
    
    for ind,i in enumerate(errors):
        
        ax3.text(0,ind+0.55,f'{int(round(i,0))} de {int(n_i[ind])} ({round(perrors[ind]*100,2)}%)',ha='left',fontsize=10)
        
    ax3.text(0.4,0.15,'Errores',fontsize=12,fontweight='bold')
    ax3.text(1,1,'+',fontsize=12,ha='center')
    ax3.text(1,1.85,'=',fontsize=12,ha='center')
    ax3.text(1.2,2.2,f'{int(round(np.sum(errors),0))} errores',fontsize=20,fontweight='bold',ha='center')
    
    return efectividad, precision, recall, roc_score

def text_transformer(X_train, X_test, vectorizer):
    """ Función para crear word embeddings (TF-IDF y BoW)
      
        Args:  
        X_train pandas dataframe
        X_test  pandas dataframe
        varToVector string que contiene nombre de variable de texto
        vectorizer objeto para crear un word embedding de tipo TF-IDF y BoW
    """
    # Entrena método para convertir de texto a matrix numérica
    vectorizer_ = vectorizer
    vectorizer_.fit(X_train)

    X_train = vectorizer_.transform(X_train)
    X_test = vectorizer_.transform(X_test)
    return X_train, X_test

def try_model(model,n_features,X_train,y_train,X_test,y_test):
    
    X_train_TFID, X_test_TFID = text_transformer(X_train, X_test, StringUtils.TfidfVectorizer(max_features=n_features))
    X_train_TFID = pd.DataFrame(X_train_TFID.toarray())
    X_test_TFID = pd.DataFrame(X_test_TFID.toarray())
    
    model.fit(X_train_TFID,y_train)
    return(model.score(X_test_TFID,y_test))