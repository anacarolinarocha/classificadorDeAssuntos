#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 23:36:36 2019

@author: anarocha
"""
from select_documentos import *
import pandas as pd
from sqlalchemy import create_engine
import time
from datetime import timedelta
from funcoes import *
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import os
import lxml
import csv
import multiprocessing as mp
from sklearn.metrics import multilabel_confusion_matrix

import os

# =============================================================================
# Recuperando dados
# =============================================================================
start_time = time.time()
engine = create_engine('postgresql://c054997:elefante@10.0.17.3:5438/pje_2grau')
df = pd.read_sql_query(select_teste, engine)
end_time = time.time() - start_time
print('Tempo para recuperar documentos:' + str(timedelta(seconds=end_time)))

df.to_csv('./listaProcessos.csv')
#df_100 = df.head(1000)
df = pd.read_csv('./listaProcessos.csv')     
df = df.rename( columns ={'identificador_documento': 'ds_orgao_julgador_colegiado'})
df = df.rename( columns ={'right': 'codigo_documento'})
df_metadados= df[['ds_orgao_julgador', 'identificador_documento','nr_processo','id_processo_documento','right','dt_juntada']]





# =============================================================================
# Geração de arquivos
# =============================================================================
is_2594 = df['in_2594']==1
df_2594 = df[is_2594]
df_2594=df_2594.reset_index(drop=True)
df_2594 = df_2594.sort_values(by=['ds_orgao_julgador_colegiado'],ascending=True)

os.makedirs("./RecursosOrdinarios/2594/")
ojcs = df_2594['ds_orgao_julgador_colegiado'].drop_duplicates()
for ojc in ojcs:
    os.makedirs("./RecursosOrdinarios/2594/" + ojc + "/")

for index, row in df_2594.iterrows():
    nome_arquivo = "./RecursosOrdinarios/2594/" + row['ds_orgao_julgador_colegiado'] + "/"+ row['nr_processo'] + ".html"
    conteudo_documento = row['ds_modelo_documento']
    arquivo= open(nome_arquivo, "w")
    arquivo.write(conteudo_documento)
    arquivo.close()
    

    
#y_2594 = df['in_25.46']
#y_2594 = df['in_20.86']
# =============================================================================
# Pre-processamento do texto
# =============================================================================
start_time = time.time()
listaProcessada=[]
for index, documento in df.iterrows():
    if(documento['ds_modelo_documento'] != None):
        listaProcessada.append(processa_texto(documento['ds_modelo_documento']))
print('Tempo para processar o texto:' + str(timedelta(seconds=end_time)))
# TODO: melhorar o pre processamento fazendo uso dos metodos já existentes no TFIDFVectorizer
#       e fazer proocesasmento paralelo aqui se for preciso.        
dados_X = pd.DataFrame(df_metadados)
dados_X = dados_X.join(pd.DataFrame(listaProcessada,columns=['texto_processado']))

vectorizer = TfidfVectorizer()
X = vectorizer.fit(dados_X['texto_processado'])
# TODO: Testar com BM25 e com Word Embeddings

# =============================================================================
# Criando conjuntos de treinamento, validação e teste
# =============================================================================

X_2594_train, X_2594_test, y_2594_train, y_2594_test = train_test_split( dados_X, y_2594, test_size=0.33, random_state=42, stratify=y_2594)

X_2594_train_tfidf = X.transform(X_2594_train['texto_processado'])
X_2594_test_tfidf = X.transform(X_2594_test['texto_processado'])
X_2594_train_tfidf.shape
X_2594_test_tfidf.shape


# =============================================================================
# Verificando distribuição e fazendo balanceamento
# =============================================================================

ros =RandomOverSampler(random_state=42)
X_2594_train_tfidf_resampled, y_2594_train_resampled = ros.fit_resample(X_2594_train_tfidf, y_2594_train)
pd.DataFrame(y_2594_test).hist()

pd.DataFrame(y_2594_test).hist()


# =============================================================================
# Criando modelo
# =============================================================================

#clf = SVC(class_weight='balanced')
#clf=MultinomialNB()
clf=SVC(kernel='linear', probability=True)
start_time = time.time()
clf.fit(X_2594_train_tfidf_resampled, y_2594_train_resampled) 
end_time = time.time() - start_time
print('Tempo para a criação do modelo:' + str(timedelta(seconds=end_time)))

y_pred = clf.predict(X_2594_test_tfidf)
# =============================================================================
# Verificando resultado
# =============================================================================

clf.score(X_2594_test_tfidf,y_2594_test)

conf_mat = confusion_matrix(y_true=y_2594_test, y_pred=y_pred)
print('Confusion matrix:\n', conf_mat)
#
#labels = ['Class 0', 'Class 1']
#fig = plt.figure()
#ax = fig.add_subplot(111)
#cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
#fig.colorbar(cax)
#ax.set_xticklabels([''] + labels)
#ax.set_yticklabels([''] + labels)
#plt.xlabel('Predicted')
#plt.ylabel('Expected')
#plt.show()



# =============================================================================
# Fazendo o Predict
# =============================================================================
y_pred_proba = clf.predict_proba(X_2594_test_tfidf)
#https://stackoverflow.com/questions/20113206/scikit-learn-svc-decision-function-and-predict
#https://prateekvjoshi.com/2015/12/15/how-to-compute-confidence-measure-for-svm-classifiers/
df_to_csv = None
df_to_csv = pd.DataFrame(X_2594_test, columns=['ds_orgao_julgador', 'identificador_documento','nr_processo','id_processo_documento','right','dt_juntada'])
df_to_csv = df_to_csv.rename( columns ={'identificador_documento': 'ds_orgao_julgador_colegiado'})
df_to_csv = df_to_csv.rename( columns ={'right': 'codigo_documento'})
df_to_csv = df_to_csv.reset_index(drop=True)
df_to_csv = df_to_csv.join(pd.DataFrame(y_pred_proba, columns=['p0','p1']).reset_index(drop=True))
df_to_csv = df_to_csv.join(pd.DataFrame(y_pred, columns=['predict']).reset_index(drop=True))
df_to_csv = df_to_csv.join(pd.DataFrame(pd.DataFrame(y_2594_test)['in_2594']).reset_index(drop=True))
df_to_csv = df_to_csv.sort_values(by=['predict','p1','p0'], ascending=False)

df_to_csv.to_csv('./Processos_Classificados_SVC_2594.csv')
