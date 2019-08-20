#!/usr/bin/env python
# coding: utf-8

# # Classificação de Processos Trabalhistas segundo Assuntos da TPU-CNJ

# ## Abordagem **Multi Classe (Single Label)**

# **Por Ana Carolina Pereira Rocha**

# In[3]:

import csv
import multiprocessing as mp
import sys
import time
import nltk
import os
from imblearn.over_sampling import RandomOverSampler
from matplotlib import path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.svm import SVC
from sqlalchemy import create_engine
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.metrics import precision_recall_fscore_support as score
from funcoes_novas import *
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier

sys.path.insert(1, '/home/anarocha/Documents/credentials')
from credentials import *

# -----------------------------------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------------------------------
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

path_base='/home/anarocha/myGit/classificadorDeAssuntos/Resultados/'
nome_experimento='EXP1_TRT21_TodoAssuntosSelecionados_MultiClasse_V1'
path_base_exeperimento = path_base + nome_experimento + '/'
if not os.path.exists(path_base_exeperimento):
    os.makedirs(path_base_exeperimento)
if not os.path.exists(path_base_exeperimento + 'Modelos/'):
    os.makedirs(path_base_exeperimento + 'Modelos/')

assuntos_sheet = pd.ExcelFile('/home/anarocha/myGit/classificadorDeAssuntos/Dados/SelecaoAssuntos.xlsx')
assuntos_nivel_3_df_completo = assuntos_sheet.parse('Assuntos De Nivel 3')[
    ['codigo_assunto_nivel_3', 'assunto_nivel_3', 'selecionado', 'Selecionado pelo GNN']]
is_assunto_nivel_3_selecionado = assuntos_nivel_3_df_completo[
    assuntos_nivel_3_df_completo['selecionado'] == 'Selecionado Previamente'].reset_index().sort_values(
    'codigo_assunto_nivel_3')
assuntos_nivel_3_df_selecionado = is_assunto_nivel_3_selecionado[['codigo_assunto_nivel_3', 'assunto_nivel_3']]
del(is_assunto_nivel_3_selecionado)
del(assuntos_nivel_3_df_completo)
del(assuntos_sheet)

# ### ====================================================================================
# # Classificação Multi Classe
#
# #### Classifica quanto ao assunto principal, dentre vários assuntos possíveis
# ### ====================================================================================

# -----------------------------------------------------------------------------------------------------
# Fazendo nova recuperacao dos dados para que eles venham com uma única coluna com o assunto principal
# -----------------------------------------------------------------------------------------------------
with open('/home/anarocha/myGit/classificadorDeAssuntos/Scripts/001-Consultas/002-SelectDocumentosMultiClasse.sql', 'r') as file:
    sql = file.read().replace('\n', ' ')
engine = create_engine(credentials_trt19_2g) #bugfix trt 21
start_time = time.time()
teste = """

CREATE SERVER foreign_server
        FOREIGN DATA WRAPPER postgres_fdw
        OPTIONS (host '10.0.3.150', port '5192', dbname 'pje_2grau_bugfix_log');
        
CREATE USER MAPPING FOR bugfix
        SERVER foreign_server
        OPTIONS (user 'bugfix', password 'bii6f1x');
       

CREATE FOREIGN TABLE tb_log_fdw (
        ds_entidade text,
        ds_id_entidade text,
        dt_log timestamp
)
        SERVER foreign_server
        OPTIONS (schema_name 'pje', table_name 'tb_log');
"""

df_mc = pd.read_sql_query(sql, engine)
total_time = time.time() - start_time
print('Tempo para recuperar documentos:' + str(timedelta(seconds=total_time)))
df_mc.to_csv('./Dados/naoPublicavel/listaProcessos_MultiLabel_TRT19_2G_2010-2019.csv', sep='#', quoting=csv.QUOTE_NONNUMERIC)

df_mc.to_csv('./Dados/naoPublicavel/listaProcessos_MultiLabel_TRT24_2G_2010-2019.csv', sep='#', quoting=csv.QUOTE_NONNUMERIC)
df_mc_07 = pd.read_csv('./Dados/naoPublicavel/listaProcessos_MultiLabel_TRT07_2G_2010-2019.csv', sep='#', quoting=csv.QUOTE_NONNUMERIC)
df_mc_21 = pd.read_csv('./Dados/naoPublicavel/listaProcessos_MultiLabel_TRT21_2G_2010-2019.csv', sep='#', quoting=csv.QUOTE_NONNUMERIC)
df_mc_07.head(2)
df_mc_21.head(2)
del(df_mc)
pretty_size(getsize(df_mc))
del(df_mc)


#Processando texto
pool = mp.Pool(mp.cpu_count())
start_time = time.time()
df_mc.shape
df_mc = df_mc.dropna()
df_mc['texto_processado'] = pool.map(removeHTML, [row for row in df_mc['ds_modelo_documento']])
pool.close()
total_time = time.time() - start_time
print('Tempo para processamento do texto:' + str(timedelta(seconds=total_time)))
df_mc.head()

# -----------------------------------------------------------------------------------------------------
# ### Criacao de features
# -----------------------------------------------------------------------------------------------------

#TFIDF
stopwords = nltk.corpus.stopwords.words('portuguese')
tfidf_vectorizer = TfidfVectorizer(strip_accents='ascii', stop_words=stopwords, token_pattern=r'(?u)\b[A-Za-z]+\b', max_df=0.8)
#TODO: STEMMIZAR
%time matriz_tfidf = tfidf_vectorizer.fit(df_mc['texto_processado'])
len(matriz_tfidf.vocabulary_)
pretty_size(getsize(matriz_tfidf))


# -----------------------------------------------------------------------------------------------------
# ### Manipulando dataset
# -----------------------------------------------------------------------------------------------------
# Separando conjunto de treinamento e teste:
Xs_mc = df_mc.drop(['cd_assunto_nivel_3'], axis=1)
Ys_mc = df_mc[['cd_assunto_nivel_3']]
Counter(Ys_mc['cd_assunto_nivel_3'])


plot_ys_mc(Ys_mc['cd_assunto_nivel_3'],'Distribuição inicial',path_base_exeperimento)

X_train_mc, X_test_mc, y_train_mc, y_test_mc = train_test_split( Xs_mc, Ys_mc, test_size=0.3, random_state=42, stratify=Ys_mc)

#Filtra os  elementos que tiverem menos que a quantidade minima de assuntos permitida
qtdMinimaAssuntos = 50

y_train_mc_que_serao_removidos = y_train_mc.groupby("cd_assunto_nivel_3").filter(lambda x: len(x) <= qtdMinimaAssuntos)
assuntos_removidos_por_falta_de_exemplos = y_train_mc_que_serao_removidos.cd_assunto_nivel_3.unique()

#Removendo exemplos de treinamento
qtd_anterior_treinamento = X_train_mc.shape[0]
y_train_mc = y_train_mc.groupby("cd_assunto_nivel_3").filter(lambda x: len(x) > qtdMinimaAssuntos)
X_train_mc = X_train_mc.loc[y_train_mc.index]
qtd_posteior_treinamento = X_train_mc.shape[0]
qtd_removida_treinamento = qtd_anterior_treinamento-qtd_posteior_treinamento
print('Foram excluidos %d registros de treinamento' % qtd_removida_treinamento)

#Removendo exemplos de teste
qtd_anterior_teste = X_test_mc.shape[0]
y_test_mc=y_test_mc[~y_test_mc['cd_assunto_nivel_3'].isin(assuntos_removidos_por_falta_de_exemplos)]
X_test_mc = X_test_mc.loc[y_test_mc.index]
qtd_posteior_teste = X_test_mc.shape[0]
qtd_removida_teste = qtd_anterior_teste-qtd_posteior_teste
print('Foram excluidos %d registros de teste' % qtd_removida_teste)


plot_ys_mc(y_train_mc['cd_assunto_nivel_3'],'Treinamento',path_base_exeperimento)
plot_ys_mc(y_test_mc['cd_assunto_nivel_3'],'Teste',path_base_exeperimento)


ros =RandomOverSampler()
X_train_mc_resampled, y_train_mc_resampled = ros.fit_resample(X_train_mc, y_train_mc['cd_assunto_nivel_3'])
#print(ros.sample_indices_) pega os primeiros em ordem sequencial e depois qnd acaba vai pro bootstrap
Counter(y_train_mc_resampled)
plot_ys_mc(pd.Series(y_train_mc_resampled),'Treinamento balanceado',path_base_exeperimento)
X_train_mc_resampled_tfidf = tfidf_vectorizer.transform(X_train_mc_resampled[:,-1])
pretty_size(getsize(X_train_mc_resampled_tfidf))
X_test_mc_tfidf = tfidf_vectorizer.transform(X_test_mc['texto_processado'])
pretty_size(getsize(X_test_mc_tfidf))
# -----------------------------------------------------------------------------------------------------
# ### Treinamento
# -----------------------------------------------------------------------------------------------------



# ---------------------
# Com bagging
# ---------------------
#A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction. 

from sklearn.naive_bayes import MultinomialNB

nome_classificador="Multinomial_Naive_Bayes"

n_estimators = 5
max_samples=round(X_train_mc_resampled_tfidf.shape[0] * 0.7)
start_time = time.time()
#clf_ovr_bagged = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', probability=True, class_weight='balanced'), max_samples=max_samples, n_estimators=n_estimators, n_jobs=-1))
clf_nb_bagged = OneVsRestClassifier(BaggingClassifier(MultinomialNB(), max_samples=max_samples, n_estimators=n_estimators, n_jobs=-1))
clf_nb_bagged.fit(X_train_mc_resampled_tfidf, y_train_mc_resampled)
total_time = time.time() - start_time
print("Tempo para a criação do modelo Bagging Naive Bayes Multinomial", str(timedelta(seconds=total_time)))

filename = path_base_exeperimento + 'Modelos/clf_nb_bagged_trt21_todosassuntosfiltrados.sav'
import pickle
pickle.dump(clf_nb_bagged, open(filename, 'wb'))

# clf_ovr_bagged = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test_mc_tfidf, y_test_mc)
# print(result)

y_pred_bagged = clf_nb_bagged.predict(X_test_mc_tfidf)
clf_nb_bagged.score(X_test_mc_tfidf,y_test_mc)
accuracy = clf_nb_bagged.score(X_test_mc_tfidf,y_test_mc)
print('macro_precision %s \nmacro_recall    %s \nmacro_fscore    %s' % score(y_test_mc,y_pred_bagged,average='macro')[:3])
print('micro_precision %s \nmicro_recall    %s \nmicro_fscore    %s' % score(y_test_mc,y_pred_bagged,average='weighted')[:3])
conf_mat = multilabel_confusion_matrix(y_true=y_test_mc, y_pred=y_pred_bagged)
print('Confusion matrix:\n', conf_mat)

macro_score = str('macro_precision %s \nmacro_recall    %s \nmacro_fscore    %s' % score(y_test_mc,y_pred_bagged,average='macro')[:3])
micro_score = str('micro_precision %s \nmicro_recall    %s \nmicro_fscore    %s' % score(y_test_mc,y_pred_bagged,average='weighted')[:3])
imprime_resultado_classificador(path_base_exeperimento, nome_experimento, nome_classificador, timedelta(seconds=total_time), accuracy, macro_score, micro_score)

y_pred_bagged_proba = clf_nb_bagged.predict_proba(X_test_mc_tfidf)

#%%

y_pred_bagged_proba[0:5]

#%% md

### Abrindo um documento específico para validção

#%%

# f = open("./teste.html", "w")
# f.write(X_test_mc.iloc[0]['ds_modelo_documento'])
# f.close()
# import webbrowser
# webbrowser.get('firefox').open_new_tab('./teste.html')
#
# os.remove('./teste.html')

# ----------------------------------------------------------------------------------------------------------------------
# Cruva de aprendizagem
# ----------------------------------------------------------------------------------------------------------------------
tamanho_amostra = X_train_mc_resampled_tfidf.shape[0]
acuracia_amostra = accuracy
tupla=(tamanho_amostra, acuracia_amostra)
lista_tuplas = []
# lista_tuplas.append(tupla)
lista_tuplas.append((90000, 0.80))
lista_tuplas.append((100000, 0.81))
lista_tuplas.append((120000, 0.82))
lista_tuplas.append((150000, 0.83))
lista_tuplas.append((170000, 0.84))
lista_tuplas.append((180000, 0.85))
lista_tuplas.append((200000, 0.86))
lista_tuplas.append((210000, 0.865))
lista_tuplas.append((230000, 0.869))
lista_tuplas.append((250000, 0.87))
pd.DataFrame(lista_tuplas, columns=['Quantidade elementos na amostra','Acurácia']).plot(kind='line', x=0, y=1)
plt.title("Análise da quantidade de amostras")
plt.show()


# ---------------------
# Sem bagging
# ---------------------

clf_ovr = OneVsRestClassifier(SVC(kernel='linear', probability=True))
start_time = time.time()
clf_ovr.fit(X_train_mc_resampled_tfidf, y_train_mc_resampled)
total_time = time.time() - start_time
print('Tempo para a criação do modelo SVC:' + str(timedelta(seconds=total_time)))


y_pred = clf_ovr.predict(X_test_mc_tfidf)
clf_ovr.score(X_test_mc_tfidf,y_test_mc)
print('macro_precision %s \nmacro_recall    %s \nmacro_fscore    %s' % score(y_test_mc,y_pred,average='macro')[:3])
print('micro_precision %s \nmicro_recall    %s \nmicro_fscore    %s' % score(y_test_mc,y_pred,average='weighted')[:3])

resultado_svc = """
# ------------------------------------------
# TRT 15 - 3 ASSUNTOS - SVC (com bagging)
#0:22:23.284952
#0.8060234813680449 accuracy
# macro_precision 0.7320162691006681
# macro_recall    0.7387239963113159
# macro_fscore    0.7352905332368164
# micro_precision 0.8073105826285133
# micro_recall    0.8060234813680449
# micro_fscore    0.8066059495676242
# ------------------------------------------
"""
f = open(path_base_exeperimento + "./svc.txt", "w")
f.write(resultado_svc)
f.close()


conf_mat = multilabel_confusion_matrix(y_true=y_test_mc, y_pred=y_pred)
print('Confusion matrix:\n', conf_mat)

filename = './Modelos/svc_trt15_3assuntos_2086_2546_2549.sav'
import pickle
pickle.dump(clf_ovr, open(filename, 'wb'))



# keep: https://www.anoreg.org.br/site/2019/06/10/cnj-inteligencia-artificial-sera-usada-para-verificar-qualidade-de-dados-processuais/

