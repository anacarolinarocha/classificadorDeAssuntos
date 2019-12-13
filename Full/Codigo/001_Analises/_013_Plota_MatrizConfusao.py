import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

#------------------------------------------------------------------------------
def imprime_matriz_de_confusao(cm, title,pathfigura):
    sns.set(font_scale=3.5)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(65,40))
    sns.heatmap(cmn, ax=ax,annot=True, fmt='.2f', linewidth=2,linecolor='lightgray',xticklabels=target_names, yticklabels=target_names,cmap="BuPu",annot_kws={"size": 28,"weight": "bold"})#,annot_kws={"fontsize":18}
    # ax.figure.axes[-1].yaxis.label.set_size(20)
    ax.set_title(title, fontsize =58)
    plt.ylabel('Assunto Principal', fontsize = 45 )
    plt.xlabel('Assunto Predito', fontsize = 45)
    # plt.show(block=True)
    fig.tight_layout()
    plt.savefig("{0}{1}.png".format(pathfigura, 'ConfusionMatrix_' + title), bbox_inches = 'tight',
    pad_inches = 0)

#------------------------------------------------------------------------------
#MLP
#------------------------------------------------------------------------------

mlp = '/media/DATA/classificadorDeAssuntos/Dados/Resultados/EXP26_MelhoresModelos_TextsoReduzidos_LSI/predicao_LSI250_Multi-Layer Perceptron.csv'
mlp = pd.read_csv(mlp)
mlp_y_true = mlp.y_true
mlp_y_pred = mlp.y_pred

cm =confusion_matrix(mlp_y_true, mlp_y_pred)
pathfigura = '/media/DATA/classificadorDeAssuntos/Dados/Resultados/EXP26_MelhoresModelos_TextsoReduzidos_LSI/'
imprime_matriz_de_confusao(cm, 'Multilayer Perceptron', pathfigura)

#------------------------------------------------------------------------------
#RANDOM FOREST
#------------------------------------------------------------------------------

mlp = '/media/DATA/classificadorDeAssuntos/Dados/Resultados/EXP25_MelhoresModelos_TextsoReduzidos_BM25/predicao_BM25_Random Forest.csv'
mlp = pd.read_csv(mlp)
mlp_y_true = mlp.y_true
mlp_y_pred = mlp.y_pred

cm =confusion_matrix(mlp_y_true, mlp_y_pred)
pathfigura = '/media/DATA/classificadorDeAssuntos/Dados/Resultados/EXP25_MelhoresModelos_TextsoReduzidos_BM25/'
imprime_matriz_de_confusao(cm, 'Random Forest', pathfigura)

#------------------------------------------------------------------------------
#SVM
#------------------------------------------------------------------------------

mlp = '/media/DATA/classificadorDeAssuntos/Dados/Resultados/EXP24_MelhoresModelos_TextsoReduzidos_TFIDF/predicao_SVM.csv'
mlp = pd.read_csv(mlp)
mlp_y_true = mlp.y_true
mlp_y_pred = mlp.y_pred

cm =confusion_matrix(mlp_y_true, mlp_y_pred)
pathfigura = '/media/DATA/classificadorDeAssuntos/Dados/Resultados/EXP24_MelhoresModelos_TextsoReduzidos_TFIDF/'
imprime_matriz_de_confusao(cm, 'SVM', pathfigura)

#------------------------------------------------------------------------------
#NAIVE BAYES
#------------------------------------------------------------------------------

mlp = '/media/DATA/classificadorDeAssuntos/Dados/Resultados/EXP25_MelhoresModelos_TextsoReduzidos_BM25/predicao_BM25_Multinomial Naive Bayes.csv'
mlp = pd.read_csv(mlp)
mlp_y_true = mlp.y_true
mlp_y_pred = mlp.y_pred

cm =confusion_matrix(mlp_y_true, mlp_y_pred)
pathfigura = '/media/DATA/classificadorDeAssuntos/Dados/Resultados/EXP25_MelhoresModelos_TextsoReduzidos_BM25/'
imprime_matriz_de_confusao(cm, 'Naïve Bayes', pathfigura)