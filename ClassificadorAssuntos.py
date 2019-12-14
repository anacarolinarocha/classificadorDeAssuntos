#!/usr/bin/env python
# coding: utf-8

# # Classificador de Assuntos
# 
# Por Ana Carolina Pereira Rocha

from docutils.nodes import header
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC,LinearSVC
from datetime import timedelta
import time
import sys
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import uuid
import os
from sklearn.calibration import CalibratedClassifierCV
import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd

# Verificando o ambiente de execução do conda

import os
print(os.environ['CONDA_DEFAULT_ENV'])

import funcoes as func
from modelo import *

n_cores = mp.cpu_count()
n_cores_grande = round(n_cores * 0.8)
n_cores_pequeno = round(n_cores * 0.35)

# #### ATENÇÃO:
#
# A célula abaixo deve ser editada para conter o caminho correto para a pasta onde
# os dados serão buscados, e a pasta onde serão gravadas as saídas do processamento
# deste código. O caminho de cada pasta deve ser terminado com a '/' no final.

path_fonte_de_dados = '/home//DocumentosClassificadorAssuntos/'
path_resultados = '/home/DocumentosClassificadorAssuntos/DocsProcessados/'

if not os.path.exists(path_resultados):
    os.makedirs(path_resultados)

float_formatter = lambda x: "%.4f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

columnsResultados=['id_execucao', 'data', 'nome','feature_type','tempo_processamento',
                   'tamanho_conjunto_treinamento', 'accuracy','balanced_accuracy',
                   'micro_precision','micro_recall','micro_fscore','macro_precision',
                   'macro_recall','macro_fscore','best_params_','best_estimator_',
                   'grid_scores_','grid_cv_results','confusion_matrix',
                   'classification_report','num_estimators','max_samples']
df_resultados = pd.DataFrame(columns = columnsResultados)
nome_arquivo_destino = path_resultados + "Metricas.csv"
if  not (os.path.isfile(nome_arquivo_destino)):
    with open(nome_arquivo_destino, 'a') as f:
        df_resultados.to_csv(f, header=True)
nome_classification_reports = path_resultados + 'ClassificationReport'

id_execucao = str(uuid.uuid1())[:7]
data = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

modelos = []

listaAssuntos=[2546,2086,1855,2594,2458,2704,2656,2140,2435,2029,2583,2554,8808,
               2117,2021,5280,1904,1844,2055,1907, 1806,55220,2506, 4437,10570,
               1783,1888,2478,5356,1773,1663,5272,2215,1767,1661,1690]

# Definindo modelos que serão usados

classificadorNB = MultinomialNB()
classificadorRF = RandomForestClassifier(random_state=42)
classificadorSVM = CalibratedClassifierCV(LinearSVC(class_weight='balanced',
                                            max_iter=10000,random_state=42),
                                          method='sigmoid', cv=5)
classificadorMLP = MLPClassifier(early_stopping= True,random_state=42)

nomeAlgoritmoNB='Multinomial Naive Bayes'
nomeAlgoritmoRF='Random Forest'
nomeAlgoritmoSVM='SVM'
nomeAlgoritmoMLP="Multi-Layer Perceptron"

# Pré-processamento dos documentos

path_destino_de_dados = path_fonte_de_dados + 'DocumentosProcessados/'
if not os.path.exists(path_destino_de_dados):
        os.makedirs(path_destino_de_dados)
        
#func.processaDocumentos(path_fonte_de_dados,path_destino_de_dados)
print("Todos os documentos disponíveis foram processados")


# Recuperando textos

qtdElementosPorAssunto=1000000
df_amostra = func.recupera_amostras_de_todos_regionais(listaAssuntos,
                                qtdElementosPorAssunto, path_destino_de_dados)


# Juntando os assuntos 55220 e 1855, ambos Indenização por Dano Moral

df_amostra.loc[df_amostra['cd_assunto_nivel_3'] == 55220, 'cd_assunto_nivel_3'] = 1855
df_amostra.loc[df_amostra['cd_assunto_nivel_2'] == 55218, 'cd_assunto_nivel_3'] = 2567

print('Total de textos recuperados: ' + str(len(df_amostra)))
df_amostra = df_amostra.dropna(subset=['texto_stemizado'])
print('Total de textos recuperados com conteúdo: ' + str(len(df_amostra)))


# Analisando tamanho dos textos

df_amostra['quantidade_de_palavras'] = \
    [len(x.split()) for x in df_amostra['texto_processado'].tolist()]
sns.boxplot(df_amostra['quantidade_de_palavras'])
plt.savefig("{0}{1}.png".format(path_resultados, "Distribuicao_Tamanho_Textos_Original"))

df_amostra_f = df_amostra[((df_amostra.quantidade_de_palavras < 400) &
                           (df_amostra.quantidade_de_palavras > 0))]
print('Quantidade de textos entre 0 e 400 palavras: ' + str(len(df_amostra_f)))
df_amostra_f = df_amostra[(df_amostra.quantidade_de_palavras > 10000)]
print('Quantidade de textos com mais de 10.000 palavras: ' + str(len(df_amostra_f)))
df_amostra.shape
df_amostra_f = df_amostra[((df_amostra.quantidade_de_palavras < 10000) &
                           (df_amostra.quantidade_de_palavras > 400))]
df_amostra_f= df_amostra_f.sort_values(by='quantidade_de_palavras', ascending=True)
df_amostra_f.shape
df_amostra = df_amostra_f
plt.clf()
plt.cla()
plt.close()
sns.boxplot(df_amostra['quantidade_de_palavras'])
plt.savefig("{0}{1}.png".format(path_resultados, "Distribuicao_Tamanho_Textos Final"))

print('Total de textos utilizados: ' + str(len(df_amostra)))
X_train, X_test, y_train, y_test = func.splitTrainTest(df_amostra)
print("Amostra de teste de " + str(X_test.shape[0]) + " elementos")
print("Amostra de treinamento de " + str(X_train.shape[0]) + " elementos")

title = "Balanceamento de assuntos na amostra de "  + str(X_train.shape[0])
func.mostra_balanceamento_assunto(y_train.value_counts(), title,
                                  "Quantidade Elementos", "Código Assunto",
                                  path_resultados, y_train.shape[0])


# ## Criando matrizes

# #### TF-IDF

start_time = time.time()
tfidf_transformer,x_tfidf_train, x_tfidf_test = \
            func.extraiFeaturesTFIDF_train_test(df_amostra,
            X_train['texto_stemizado'], X_test['texto_stemizado'], path_resultados)
total_time = time.time() - start_time
print("Tempo para montar matrizes TF-IDF (features:  "
      + str(x_tfidf_train.shape[1]) + ") :"
      + str(timedelta(seconds=total_time)))

# #### BM25

bm25_transformer,x_bm25_train, x_bm25_test = func.extraiFeaturesBM25(df_amostra,
                        tfidf_transformer, x_tfidf_train, x_tfidf_test, path_resultados)

# #### LSI

lsi100_transformer,x_lsi100_train, x_lsi100_test = func.extraiFeaturesLSI(df_amostra,
                              X_train['texto_stemizado'], X_test['texto_stemizado'],
                              100, path_resultados)
lsi250_transformer,x_lsi250_train, x_lsi250_test = func.extraiFeaturesLSI(df_amostra,
                              X_train['texto_stemizado'], X_test['texto_stemizado'],
                               250, path_resultados)

# ## Grid Search
# #### Com TF-IDF
# 
# Coloque aqui a quantidade de configurações diferentes a serem testadas
# no GridSearch para cada modelo.

numero_de_configuracoes_por_modelo=2

# #### Multinomial Naïve-Bayes (NB)

param_grid_NB = {
    'estimator__n_estimators': [3,5],
    'estimator__max_samples': [0.8,0.5],
    'estimator__base_estimator__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1]
}
modeloNB = func.chama_treinamento_modelo(x_tfidf_train, y_train, x_tfidf_test,
                 y_test, classificadorNB,
                 nomeAlgoritmoNB , 'TFIDF',param_grid_NB,
                 numero_de_configuracoes_por_modelo,
                 n_cores_grande,id_execucao ,data,path_resultados,df_resultados,
                 nome_arquivo_destino,X_test)
modelos.append([modeloNB.getNome(),modeloNB.getFeatureType(),
                modeloNB.getMicroPrecision(),modeloNB])

# #### SVM

param_grid_SVM = {
    'estimator__n_estimators': [3, 5],
    'estimator__max_samples': [0.8, 0.5],
    'estimator__base_estimator__base_estimator__C': [0.01, 0.1, 1, 10]
}
modeloSVM = func.chama_treinamento_modelo(x_tfidf_train,y_train, x_tfidf_test,y_test,
                  classificadorSVM, nomeAlgoritmoSVM,'TFIDF',
                  param_grid_SVM, numero_de_configuracoes_por_modelo,
                  n_cores_grande,id_execucao ,data,path_resultados,df_resultados,
                  nome_arquivo_destino,X_test)
modelos.append([modeloSVM.getNome(), modeloSVM.getFeatureType(),
                modeloSVM.getMicroPrecision(),modeloSVM])

# #### Random Forest (RF)

param_grid_RF = {
    'estimator__n_estimators': [3,5],
    'estimator__max_samples': [0.8,0.5],
    'estimator__base_estimator__max_depth': [30,50,100],
    'estimator__base_estimator__n_estimators': [100,200,300],
    'estimator__base_estimator__min_samples_leaf': [0.05, 0.1, 0.5],
    'estimator__base_estimator__min_samples_split': [0.05, 0.1, 0.5],
    'estimator__base_estimator__max_features': [0.3, 0.5, 0.8]
}
modeloRF = func.chama_treinamento_modelo(x_tfidf_train,y_train,
                 x_tfidf_test,y_test, classificadorRF,
                 nomeAlgoritmoRF,'TFIDF',param_grid_RF,
                 numero_de_configuracoes_por_modelo,
                 n_cores_grande,id_execucao ,data,path_resultados,df_resultados,
                 nome_arquivo_destino,X_test)
modelos.append([modeloRF.getNome(),modeloRF.getFeatureType(),
                modeloRF.getMicroPrecision(),modeloRF])

# #### Multi-layer Perceptron

param_grid_MLP = {
    'estimator__n_estimators': [3,5],
    'estimator__max_samples': [0.8,0.5],
    'estimator__base_estimator__hidden_layer_sizes': [(10,10),(10,5,10)],
    'estimator__base_estimator__activation': ['identity', 'logistic', 'tanh', 'relu'],
    'estimator__base_estimator__solver': ['sgd', 'adam','lbfgs'],
    'estimator__base_estimator__alpha': [0.001, 0.01, 0.05, 0.1],
    'estimator__base_estimator__learning_rate': ['constant','adaptive','invscaling'],
    'estimator__base_estimator__max_iter': [200,300,400]
}
modeloMLP = func.chama_treinamento_modelo(x_tfidf_train,y_train, x_tfidf_test,y_test,
                  classificadorMLP, nomeAlgoritmoMLP, 'TFIDF',
                  param_grid_MLP, numero_de_configuracoes_por_modelo,
                  n_cores_pequeno,id_execucao ,data,path_resultados,
                  df_resultados, nome_arquivo_destino,X_test)
modelos.append([modeloMLP.getNome(),modeloMLP.getFeatureType(),
                modeloMLP.getMicroPrecision(),modeloMLP])

# #### Criando dicionarios com a melhor configuração de cada modelo

#MNB
param_grid_melhor_NB = {
    'estimator__n_estimators':
        [modeloNB.getBestParams().get('estimator__n_estimators')],
    'estimator__max_samples':
        [modeloNB.getBestParams().get('estimator__max_samples')],
    'estimator__base_estimator__alpha':
        [modeloNB.getBestParams().get('estimator__base_estimator__alpha')]
}

# SVM
param_grid_melhor_SVM = {
    'estimator__n_estimators':
        [modeloSVM.getBestParams().get('estimator__n_estimators')],
    'estimator__max_samples':
        [modeloSVM.getBestParams().get('estimator__max_samples')],
    'estimator__base_estimator__base_estimator__C':
        [modeloSVM.getBestParams().
             get('estimator__base_estimator__base_estimator__C')]
}

# RF
param_grid_melhor_RF = {
    'estimator__n_estimators':
        [modeloRF.getBestParams().get('estimator__n_estimators')],
    'estimator__max_samples':
        [modeloRF.getBestParams().get('estimator__max_samples')],
    'estimator__base_estimator__max_depth':
        [modeloRF.getBestParams().get('estimator__base_estimator__max_depth')],
    'estimator__base_estimator__n_estimators':
        [modeloRF.getBestParams().get('estimator__base_estimator__n_estimators')],
    'estimator__base_estimator__min_samples_leaf':
        [modeloRF.getBestParams().get('estimator__base_estimator__min_samples_leaf')],
    'estimator__base_estimator__min_samples_split':
        [modeloRF.getBestParams().get('estimator__base_estimator__min_samples_split')],
    'estimator__base_estimator__max_features':
        [modeloRF.getBestParams().get('estimator__base_estimator__max_features')]
}

# MLP
param_grid_melhor_MLP = {
    'estimator__n_estimators':
        [modeloMLP.getBestParams().get('estimator__n_estimators')],
    'estimator__max_samples':
        [modeloMLP.getBestParams().get('estimator__max_samples')],
    'estimator__base_estimator__hidden_layer_sizes':
        [modeloMLP.getBestParams().get('estimator__base_estimator__hidden_layer_sizes')],
    'estimator__base_estimator__activation':
        [modeloMLP.getBestParams().get('estimator__base_estimator__activation')],
    'estimator__base_estimator__solver':
        [modeloMLP.getBestParams().get('estimator__base_estimator__solver')],
    'estimator__base_estimator__alpha':
        [modeloMLP.getBestParams().get('estimator__base_estimator__alpha')],
    'estimator__base_estimator__learning_rate':
        [modeloMLP.getBestParams().get('estimator__base_estimator__learning_rate')],
    'estimator__base_estimator__max_iter':
        [modeloMLP.getBestParams().get('estimator__base_estimator__max_iter')]
}

# #### BM25

modeloNB_BM25 = func.chama_treinamento_modelo(x_bm25_train,y_train, x_bm25_test,
                  y_test, classificadorNB,
                  nomeAlgoritmoNB,'BM25',param_grid_melhor_NB, 1,n_cores_grande,
                  id_execucao ,data,path_resultados,df_resultados,nome_arquivo_destino,
                  X_test)
modelos.append([modeloNB_BM25.getNome(),modeloNB_BM25.getFeatureType(),
                modeloNB_BM25.getMicroPrecision(),modeloNB_BM25])

modeloSVM_BM25 = func.chama_treinamento_modelo(x_bm25_train,y_train, x_bm25_test,
                   y_test, classificadorSVM,
                   nomeAlgoritmoSVM,'BM25',param_grid_melhor_SVM, 1,n_cores_grande,
                   id_execucao ,data,path_resultados,df_resultados,nome_arquivo_destino,
                   X_test)
modelos.append([modeloSVM_BM25.getNome(),modeloSVM_BM25.getFeatureType(),
                modeloSVM_BM25.getMicroPrecision(),modeloSVM_BM25])

modeloRF_BM25 = func.chama_treinamento_modelo(x_bm25_train,y_train, x_bm25_test,
                  y_test, classificadorRF,nomeAlgoritmoRF,
                  'BM25',param_grid_melhor_RF, 1,n_cores_grande,id_execucao,
                  data,path_resultados,df_resultados,nome_arquivo_destino,X_test)
modelos.append([modeloRF_BM25.getNome(),modeloRF_BM25.getFeatureType(),
                modeloRF_BM25.getMicroPrecision(),modeloRF_BM25])

modeloMLP_BM25 = func.chama_treinamento_modelo(x_bm25_train,y_train, x_bm25_test,
                   y_test, classificadorMLP,
                   nomeAlgoritmoMLP,'BM25',param_grid_melhor_MLP, 1,n_cores_pequeno,
                   id_execucao ,data,path_resultados,df_resultados,nome_arquivo_destino,
                   X_test)
modelos.append([modeloMLP_BM25.getNome(),modeloMLP_BM25.getFeatureType(),
                modeloMLP_BM25.getMicroPrecision(), modeloMLP_BM25])

# #### LSI 100

modeloSVM_LSI100 = func.chama_treinamento_modelo(x_lsi100_train,y_train, x_lsi100_test ,
                 y_test, classificadorSVM,
                 nomeAlgoritmoSVM, 'LSI100',param_grid_melhor_SVM, 1,n_cores_grande,
                 id_execucao ,data,path_resultados,df_resultados,nome_arquivo_destino,
                 X_test)
modelos.append([modeloSVM_LSI100.getNome(),modeloSVM_LSI100.getFeatureType(),
                modeloSVM_LSI100.getMicroPrecision(),
                modeloSVM_LSI100])

modeloRF_LSI100 = func.chama_treinamento_modelo(x_lsi100_train, y_train,x_lsi100_test ,
                y_test, classificadorRF,
                nomeAlgoritmoRF, 'LSI100',param_grid_melhor_RF, 1,n_cores_grande,
                id_execucao ,data,path_resultados,df_resultados,nome_arquivo_destino,
                X_test)
modelos.append([modeloRF_LSI100.getNome(),modeloRF_LSI100.getFeatureType(),
                modeloRF_LSI100.getMicroPrecision(), modeloRF_LSI100])

modeloMLP_LSI100 = func.chama_treinamento_modelo(x_lsi100_train,y_train, x_lsi100_test ,
                 y_test, classificadorMLP,
                 nomeAlgoritmoMLP,'LSI100',param_grid_melhor_MLP, 1,n_cores_pequeno,
                 id_execucao ,data,path_resultados,df_resultados,nome_arquivo_destino,
                 X_test)
modelos.append([modeloMLP_LSI100.getNome(),modeloMLP_LSI100.getFeatureType(),
                modeloMLP_LSI100.getMicroPrecision(), modeloMLP_LSI100])

# #### LSI 250

modeloSVM_LSI250 = func.chama_treinamento_modelo(x_lsi250_train,y_train, x_lsi250_test ,
                 y_test, classificadorSVM,
                 nomeAlgoritmoSVM, 'LSI250',param_grid_melhor_SVM, 1,n_cores_grande,
                 id_execucao ,data,path_resultados,df_resultados,nome_arquivo_destino,
                 X_test)
modelos.append([modeloSVM_LSI250.getNome(),modeloSVM_LSI250.getFeatureType(),
                modeloSVM_LSI250.getMicroPrecision(), modeloSVM_LSI250])

modeloRF_LSI250 = func.chama_treinamento_modelo(x_lsi250_train, y_train,x_lsi250_test ,
                y_test, classificadorRF,
                nomeAlgoritmoRF, 'LSI250',param_grid_melhor_RF, 1,n_cores_grande,
                id_execucao ,data,path_resultados,df_resultados,nome_arquivo_destino,
                X_test)
modelos.append([modeloRF_LSI250.getNome(),modeloRF_LSI250.getFeatureType(),
                modeloRF_LSI250.getMicroPrecision(),  modeloRF_LSI250])

modeloMLP_LSI250 = func.chama_treinamento_modelo(x_lsi250_train,y_train, x_lsi250_test ,
                 y_test, classificadorMLP,
                 nomeAlgoritmoMLP,'LSI250',param_grid_melhor_MLP, 1,n_cores_pequeno,
                 id_execucao ,data,path_resultados,df_resultados,nome_arquivo_destino,
                 X_test)
modelos.append([modeloMLP_LSI250.getNome(),modeloMLP_LSI250.getFeatureType(),
                modeloMLP_LSI250.getMicroPrecision(), modeloMLP_LSI250])

# Encontrando o modelo vencedor

modelos_df = pd.DataFrame(modelos,
                  columns=['Nome Modelo','Feature Type','micro_precision','Modelo'])
modelos_df= modelos_df.sort_values(by='micro_precision', ascending=False)
print("O modelo vencedor foi o " + modelos_df.iloc[0]['Nome Modelo'] + ", com " +
      (str("%.2f" % modelos_df.iloc[0]['micro_precision'])) + " de micro precisão")

modelo_vencedor = modelos_df.iloc[0]['Modelo']

arquivoPickle = open(path_resultados + "MelhorModelo.p", 'wb')
pickle.dump(modelo_vencedor.getBestEstimator(), arquivoPickle)
arquivoPickle.close()

if modelo_vencedor.getFeatureType() == 'LSI100':
    feature_vencedora = open(path_resultados + "MelhorModeloFeature.p", 'wb')
    pickle.dump(lsi100_transformer, feature_vencedora)
    feature_vencedora.close()
elif modelo_vencedor.getFeatureType() == 'LSI250':
    feature_vencedora = open(path_resultados + "MelhorModeloFeature.p", 'wb')
    pickle.dump(lsi250_transformer, feature_vencedora)
    feature_vencedora.close()
elif modelo_vencedor.getFeatureType() == 'TFIDF':
    feature_vencedora = open(path_resultados + "MelhorModeloFeature.p", 'wb')
    pickle.dump(tfidf_transformer, feature_vencedora)
    feature_vencedora.close()
elif modelo_vencedor.getFeatureType() == 'BM25':
    feature_vencedora = open(path_resultados + "MelhorModeloFeature.p", 'wb')
    pickle.dump(bm25_transformer, feature_vencedora)
    feature_vencedora.close()
    
print("O modelo para transformação dos textos pré-processados se encontra no arquivo "
      + path_resultados +
      "MelhorModeloFeature.p" + " e o modelo de classificação no arquivo "
      + path_resultados + "MelhorModelo.p")

