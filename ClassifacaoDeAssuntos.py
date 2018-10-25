#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 17:09:43 2018

@author: anarocha
"""

import nltk
from unicodedata import normalize
import re

from SolrClient import SolrClient
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import multiprocessing as mp
#from multiprocessing import Pool

import matplotlib.pyplot as plt

from gensim import corpora
from gensim.models import LsiModel, TfidfModel
from gensim.corpora import MmCorpus

from sklearn.model_selection import train_test_split

from joblib import Parallel, delayed

import csv
import time

from gensim import  matutils
import gc
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import sys

from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

from sklearn import grid_search
from numpy.random import random, random_integers
from sklearn.model_selection import train_test_split

import json
##############################################################################
# DEFINE FUNÇÕES
##############################################################################
stopwords = nltk.corpus.stopwords.words('portuguese')
stopwords.extend(['microsoftinternetexplorer','false','none','trabalho','juiz',
                  'reclamado','reclamada','autos','autor','excelentissimo',
                  'senhor','normal'])
stopwords = [normalize('NFKD', palavra).encode('ASCII','ignore').decode('ASCII') for palavra in stopwords]


nltk.download('rslp')
stemmer = nltk.stem.RSLPStemmer()

def processa_texto(texto):
        textoProcessado = BeautifulSoup(texto, 'html.parser').string
        #TODO: ainda é preciso remover as tags XML e word....
        textoProcessado = normalize('NFKD', textoProcessado).encode('ASCII','ignore').decode('ASCII')
        textoProcessado = re.sub('[^a-zA-Z]',' ',textoProcessado)
        textoProcessado = textoProcessado.lower()
        textoProcessado = textoProcessado.split()
        textoProcessado = [palavra for palavra in textoProcessado if not palavra in stopwords]
        textoProcessado = [palavra for palavra in textoProcessado if len(palavra)>3]
        textoProcessado =  [stemmer.stem(palavra) for palavra in textoProcessado]
        return textoProcessado

# =============================================================================
# Cria matrix de confusão
# =============================================================================

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure(figsize=(25,25))    
    plt.rcParams.update({'font.size': 10})
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout(pad=1.4)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# =============================================================================
# Cria grafico de barras
# =============================================================================

def cria_grafico_barra(data, title, ylabel, xlabel, fontSize, figSize_x, figSize_y, nomeImagem):
    path = '/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/imagens/'
    fig = plt.figure(figsize=(figSize_x,figSize_y))
    plt.rcParams.update({'font.size': fontSize})
    data.plot.bar(ylim=0)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()
    fig.savefig(path + nomeImagem + '.png')  
    

# =============================================================================
# Multinomial Nayve Bayes
# =============================================================================
def naive_bayes(training_corpus,training_classes,test_corpus, test_classes, classNumber,classes,featureType):
    classesCM = []
    classesCM = classes
    
    param_grid = {
        'fit_prior':[True, False],
        'alpha':[0,1]
    }
    
    clf_NB = MultinomialNB(random_state=0, class_weight='balanced')
    clf_NB_grid = grid_search.GridSearchCV(estimator=clf_NB, param_grid=param_grid,
                                         scoring='f1_weighted',n_jobs=7,cv=5)
    
    clf_NB_grid.fit(training_corpus, training_classes)
    
    clf_NB = clf_NB_grid.best_estimator_
    predicted_NB = clf_NB.predict(test_corpus)
    np.mean(predicted_NB == test_classes)
    
    confusion_matrix_NB = confusion_matrix(test_classes,predicted_NB)
    
    matrixHeaderString = 'Naive Bayes \nMacro Class ' + str(classNumber) +' - ' +featureType + '\nAccuracy: {0:.3f}'.format(accuracy_score(test_classes,predicted_NB))
    plot_confusion_matrix(confusion_matrix_NB, classesCM, title=matrixHeaderString)
    figureFile = '/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/imagens/confusion_matrix_NB_mc'+ str(classNumber) +'_tfidf_testeFuncao.png'
    plt.savefig(figureFile) 
    
    macro_precision,macro_recall,macro_fscore,macro_support=score(test_classes,predicted_NB,average='macro')
    micro_precision,micro_recall,micro_fscore,micro_support=score(test_classes,predicted_NB,average='weighted')
    
    global avaliacaoFinal
    avaliacaoFinal = avaliacaoFinal.append({'Macro Class':classNumber,
                                            'Model':'Naive Bayes',
                                            'Features':featureType,
                                            'Macro Precision':macro_precision,
                                            'Macro Recall':macro_recall,
                                            'Macro F1-Measure':macro_fscore,
                                            'Micro Precision':micro_precision,
                                            'Micro Recall':micro_recall,
                                            'Micro F1-Measure':micro_fscore}, ignore_index=True)   
    
# =============================================================================
# SVM
# =============================================================================
def svm(training_corpus,training_classes,test_corpus, test_classes, classNumber,classes,featureType):
    param_grid = {
        'kernel':['linear'],#, 'poly', 'rbf', 'sigmoid'],
        'C': [0.1],#,10,100],
        'gamma':[0.1],#,1,10,30]
        #'l1_ratio': [0, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1],
    }
    clf_SVM = SVC(random_state=0, class_weight='balanced')
    clf_SVM_grid = grid_search.GridSearchCV(estimator=clf_SVM, param_grid=param_grid,
                                         scoring='f1_weighted',n_jobs=7,cv=5)
    clf_SVM_grid.fit(training_corpus, training_classes)
    
    clf_SVM = clf_SVM_grid.best_estimator_
    clf_SVM.fit(training_corpus, training_classes)
    predicted_SVM =  clf_SVM.predict(test_corpus)
    np.mean(predicted_SVM == test_classes)
    
    confusion_matrix_SVM = confusion_matrix(test_classes,predicted_SVM)
    
    classesCM = []
    classesCM = classes
    matrixHeaderString = 'SVM \nMacro Class ' + str(classNumber) +' - ' +featureType + '\nAccuracy: {0:.3f}'.format(accuracy_score(test_classes,predicted_SVM))
    plot_confusion_matrix(confusion_matrix_SVM, classesCM, title=matrixHeaderString)
    figureFile = '/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/imagens/confusion_matrix_SVM_mc'+ str(classNumber) +'_tfidf_testeFuncao.png'
    plt.savefig(figureFile) 
    
    macro_precision,macro_recall,macro_fscore,macro_support=score(test_classes,predicted_SVM,average='macro')
    micro_precision,micro_recall,micro_fscore,micro_support=score(test_classes,predicted_SVM,average='weighted')
    
    global avaliacaoFinal
    avaliacaoFinal = avaliacaoFinal.append({'Macro Class':classNumber,
                                            'Model':'SVM',
                                            'Features':featureType,
                                            'Macro Precision':macro_precision,
                                            'Macro Recall':macro_recall,
                                            'Macro F1-Measure':macro_fscore,
                                            'Micro Precision':micro_precision,
                                            'Micro Recall':micro_recall,
                                            'Micro F1-Measure':micro_fscore}, ignore_index=True)   
    
    
# =============================================================================
# Random Forest
# =============================================================================
def random_forest(training_corpus,training_classes,test_corpus, test_classes, classNumber,classes,featureType):
    
    param_grid = {
       'max_depth': [50,75,100],
       'n_estimators':[100,300,600],
       'min_samples_split':[5,10,30,50],
       'min_samples_leaf':[5,10,30,50],
       'criterion':['gini', 'entropy'],
       'max_features':[0.2,0.5,1],     
       'class_weight':['balanced','balanced_subsample']        
    }
    clf_RF = RandomForestClassifier(random_state=1986,n_jobs=7,bootstrap=False)
    clf_RF_grid = grid_search.GridSearchCV(estimator=clf_RF, param_grid=param_grid,
                                         scoring='f1_weighted', verbose=2,cv=5)
    clf_RF_grid.fit(training_corpus, training_classes)

    clf_RF = clf_RF_grid.best_estimator_
    clf_RF.fit(training_corpus, training_classes)
    predicted_RF =  clf_RF.predict(test_corpus)
    np.mean(predicted_RF == test_classes)
    
    confusion_matrix_RF = confusion_matrix(test_classes,predicted_RF)
    
    classesCM = []
    classesCM = classes
    matrixHeaderString = 'Random Forest \nMacro Class ' + str(classNumber) +' - ' +featureType + '\nAccuracy: {0:.3f}'.format(accuracy_score(test_classes,predicted_RF))
    plot_confusion_matrix(confusion_matrix_RF, classesCM, title=matrixHeaderString)
    figureFile = '/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/imagens/confusion_matrix_RF_mc'+ str(classNumber) +'_tfidf_testeFuncao.png'
    plt.savefig(figureFile) 
    
    macro_precision,macro_recall,macro_fscore,macro_support=score(test_classes,predicted_RF,average='macro')
    micro_precision,micro_recall,micro_fscore,micro_support=score(test_classes,predicted_RF,average='weighted')
    
    global avaliacaoFinal
    avaliacaoFinal = avaliacaoFinal.append({'Macro Class':classNumber,
                                            'Model':'Random Forest',
                                            'Features':featureType,
                                            'Macro Precision':macro_precision,
                                            'Macro Recall':macro_recall,
                                            'Macro F1-Measure':macro_fscore,
                                            'Micro Precision':micro_precision,
                                            'Micro Recall':micro_recall,
                                            'Micro F1-Measure':micro_fscore}, ignore_index=True)   
    
   
# =============================================================================
# Multilayer Perceptron
# =============================================================================
def mlp(training_corpus,training_classes,test_corpus, test_classes, classNumber,classes,featureType):
    classesCM = []
    classesCM = classes
    
    param_grid = {
        'hidden_layer_sizes':[(5,5), (5)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver':['lbfgs', 'sgd', 'adam'],
        'learning_rate':['constant', 'invscaling', 'adaptive'],
        'learning_rate_init':[0.01,0.001,0.0001],
        'momentum':[0.3,0.6,0.9]
     }
    
    clf_MLP = MLPClassifier( batch_size='auto',
           beta_1=0.9, beta_2=0.999, early_stopping=False,
            momentum=0.9,   random_state=1, 
           solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
           warm_start=False)
    
    clf_MLP_grid = GridSearchCV(estimator = clf_MLP, param_grid = param_grid, 
                             verbose = 2,cv=5)
    
    clf_MLP_grid.fit(training_corpus, training_classes)    
    
   # result_grid_MLP = pd.DataFrame(clf_MLP_grid.grid_scores_)
    
    clf_MLP = clf_MLP_grid.best_estimator_
    clf_MLP.fit(training_corpus, training_classes)
    predicted_MLP =  clf_MLP.predict(test_corpus)
    np.mean(predicted_MLP == test_corpus)
    
    
    confusion_matrix_MLP = confusion_matrix(test_classes,predicted_MLP)
    
    matrixHeaderString = 'Multilayer Perceptron \nMacro Class ' + str(classNumber) +' - ' +featureType + '\nAccuracy: {0:.3f}'.format(accuracy_score(test_classes,predicted_MLP))
    plot_confusion_matrix(confusion_matrix_MLP, classesCM, title=matrixHeaderString)
    figureFile = '/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/imagens/confusion_matrix_MLP_mc'+ str(classNumber) +'_tfidf_testeFuncao.png'
    plt.savefig(figureFile) 
    
    macro_precision,macro_recall,macro_fscore,macro_support=score(test_classes,predicted_MLP,average='macro')
    micro_precision,micro_recall,micro_fscore,micro_support=score(test_classes,predicted_MLP,average='weighted')
    
    global avaliacaoFinal
    avaliacaoFinal = avaliacaoFinal.append({'Macro Class':classNumber,
                                            'Model':'Multilayer Perceptron',
                                            'Features':featureType,
                                            'Macro Precision':macro_precision,
                                            'Macro Recall':macro_recall,
                                            'Macro F1-Measure':macro_fscore,
                                            'Micro Precision':micro_precision,
                                            'Micro Recall':micro_recall,
                                            'Micro F1-Measure':micro_fscore}, ignore_index=True)   
    
##############################################################################    
##############################################################################
# BUSCA OS DADOS 
##############################################################################
##############################################################################
solr = SolrClient('http://localhost:8983/solr')
solrDataAnalise = solr.query('classificacaoDeDocumentos_hierarquiaCompleta',{
'q':'tx_conteudo_documento:[* TO *]','fl':'id_processo_documento,cd_assunto_nivel_1,cd_assunto_nivel_2,cd_assunto_nivel_3,cd_assunto_nivel_4,cd_assunto_nivel_5', 'rows':'300000'
})
dfGeral = pd.DataFrame(solrDataAnalise.docs)    
del(solrDataAnalise)


#USANDO FACETS.....
solrDataAnalise = solr.query('classificacaoDeDocumentos_hierarquiaCompleta',{
'q':'tx_conteudo_documento:[* TO *]',
'rows':'10',
'facet':True,
'facet.field':'cd_assunto_nivel_2',
'facet.mincount':'10'
})
columns = ['cd_assunto_nivel_2','count']
facets = pd.DataFrame(columns=columns)
facets['cd_assunto_nivel_2']= solrDataAnalise.get_facet_keys_as_list('cd_assunto_nivel_2')
facets['count']= solrDataAnalise.get_facet_values_as_list('cd_assunto_nivel_2')
dfFacet = pd.DataFrame(solrDataAnalise.facet_pivot)    

#-----------------------------------------------------------------------------
# Verifica a distribuição dos dados para assunto de nível 2
#-----------------------------------------------------------------------------
fig = plt.figure(figsize=(20,5))
plt.rcParams.update({'font.size': 10})
dfGeral.groupby('cd_assunto_nivel_2').id_processo_documento.count().plot.bar(ylim=0)
plt.title('Distribuição geral dos dados')
plt.ylabel('Quantidade de Documentos')
plt.xlabel('Código do assunto')
plt.show()
fig.savefig('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/imagens/TRT15_2GRAU_DistribuicaoClasses_2Nivel_Geral.png')  


plt.subplot(212)
dfGeral.groupby('cd_assunto_nivel_2').id_processo_documento.count().plot.bar(ylim=0)
plt.gca().axes.get_xaxis().set_visible(False)


plt.figure(1,figsize=(8,3))
plt.subplot(511)
dfGeral.groupby('cd_assunto_nivel_1').id_processo_documento.count().plot.bar(ylim=0)
plt.gca().axes.get_xaxis().set_visible(False)

plt.subplot(512)
dfGeral.groupby('cd_assunto_nivel_2').id_processo_documento.count().plot.bar(ylim=0)
plt.gca().axes.get_xaxis().set_visible(False)

plt.subplot(513)
dfGeral.groupby('cd_assunto_nivel_3').id_processo_documento.count().plot.bar(ylim=0)
plt.gca().axes.get_xaxis().set_visible(False)

plt.subplot(514)
dfGeral.groupby('cd_assunto_nivel_4').id_processo_documento.count().plot.bar(ylim=0)
plt.gca().axes.get_xaxis().set_visible(False)

plt.subplot(515)
dfGeral.groupby('cd_assunto_nivel_5').id_processo_documento.count().plot.bar(ylim=0)
plt.gca().axes.get_xaxis().set_visible(False)
plt.show()
#TODO:tirar print manual
fig.savefig('todo/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/imagens/TRT15_2GRAU_Distribuicao_De_Processos_Por_Nivel_Assunto.png')  

# =============================================================================
# Criando conjuntos de treinamento e teste estratificados
# =============================================================================

query = 'tx_conteudo_documento:[* TO *]'

solrDataAnalise = solr.query('classificacaoDeDocumentos_hierarquiaCompleta',{
'q':query,'fl':'id,id_processo_documento,cd_assunto_nivel_1,cd_assunto_nivel_2,cd_assunto_nivel_3,cd_assunto_nivel_4,cd_assunto_nivel_5', 'rows':'300000'
})
df_idProcessos = pd.DataFrame(solrDataAnalise.docs)    
df_idProcessos = df_idProcessos.groupby('cd_assunto_nivel_2').filter(lambda x: len(x) > 10)

df_idProcessos_treinamento, df_idProcessos_teste, df_codigoAssunto_treinamento, df_codigoAssunto_teste = train_test_split(df_idProcessos[['id','id_processo_documento']], df_idProcessos['cd_assunto_nivel_2'],
                                                    stratify=df_idProcessos['cd_assunto_nivel_2'], 
                                                    test_size=0.25)


# =============================================================================
# Analisando a distribuição de dados entre treinamento e teste - devem ser similares
# =============================================================================
df_codigoAssunto_treinamento= pd.DataFrame(df_codigoAssunto_treinamento)
quantidadesPorAssunto_Treinamento =  pd.DataFrame(df_codigoAssunto_treinamento.groupby('cd_assunto_nivel_2').cd_assunto_nivel_2.count().nlargest(70))
quantidadesPorAssunto_Treinamento.index.names = ['Códigos de Assunto - Nível 2']
quantidadesPorAssunto_Treinamento.reset_index(inplace=True)
quantidadesPorAssunto_Treinamento.columns=['Códigos de Assunto - Nível 2','quantidadeDocumentos']

df_codigoAssunto_teste= pd.DataFrame(df_codigoAssunto_teste)
quantidadesPorAssunto_Teste =  pd.DataFrame(df_codigoAssunto_teste.groupby('cd_assunto_nivel_2').cd_assunto_nivel_2.count().nlargest(70))
quantidadesPorAssunto_Teste.index.names = ['Códigos de Assunto - Nível 2']
quantidadesPorAssunto_Teste.reset_index(inplace=True)
quantidadesPorAssunto_Teste.columns=['Códigos de Assunto - Nível 2','quantidadeDocumentos']


fig = plt.figure()
fig.suptitle("Análise dos conjuntos de treinamento e teste")

ax1 = fig.add_subplot(2,1,1)
ax1.title.set_text('Treinamento')
quantidadesPorAssunto_Treinamento.groupby('Códigos de Assunto - Nível 2').quantidadeDocumentos.sum().plot.bar(ylim=0)

ax2 = fig.add_subplot(2,1,2)
ax2.title.set_text('Teste')
quantidadesPorAssunto_Teste.groupby('Códigos de Assunto - Nível 2').quantidadeDocumentos.sum().plot.bar(ylim=0)

fig.set_figheight(8)
fig.set_figwidth(12)
fig.tight_layout(pad=3)

# =============================================================================
# Marca os elementos que serão usados para teste no Solr
# =============================================================================

idsTeste = df_idProcessos_teste['id']
documentosDeTeste = []
for ids in idsTeste:
    doc = {'id': ids, 'isTeste':{'set': 'true'}}
    documentosDeTeste.append(doc)
solr.index_json('classificacaoDeDocumentos_hierarquiaCompleta',json.dumps(documentosDeTeste))
solr.commit(openSearcher=True, collection='classificacaoDeDocumentos_hierarquiaCompleta')

################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################

# =============================================================================
# Processa conjunto de treinamento e de teste
# =============================================================================

queryTreinamento = 'tx_conteudo_documento:[* TO *] AND NOT isTeste:true'
queryTeste  = 'tx_conteudo_documento:[* TO *] AND isTeste:true'
################################################################################################################################
# BUSCA DADOS E CRIA O DICIONÁRIO
################################################################################################################################
dicionarioFinal = corpora.Dictionary('')

start_time = time.time()
listaProcessada = []
for resCursor in solr.cursor_query('classificacaoDeDocumentos_hierarquiaCompleta',{'q':queryTreinamento,'rows':'100','fl':'tx_conteudo_documento','sort':'id asc'}):  
    listaProcessada = Parallel(n_jobs = 7)(delayed(processa_texto)(documento.get('tx_conteudo_documento')) for documento in resCursor.docs)
    dicionarioParcial = corpora.Dictionary(listaProcessada)
    dicionarioFinal.merge_with(dicionarioParcial)
    with open("/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/listaProcessadaFinal_Treinamento.csv", "a") as fp:
        wr = csv.writer(fp, dialect='excel')
        for row in listaProcessada:
            wr.writerow(row)           
print(time.time() - start_time)

start_time = time.time()
listaProcessada = []
for resCursor in solr.cursor_query('classificacaoDeDocumentos_hierarquiaCompleta',{'q':queryTeste,'rows':'1000','fl':'tx_conteudo_documento','sort':'id asc'}):  
    listaProcessada = Parallel(n_jobs = 7)(delayed(processa_texto)(documento.get('tx_conteudo_documento')) for documento in resCursor.docs)
    dicionarioParcial = corpora.Dictionary(listaProcessada)
    dicionarioFinal.merge_with(dicionarioParcial)
    with open("/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/listaProcessadaFinal_Teste.csv", "a") as fp:
        wr = csv.writer(fp, dialect='excel')
        for row in listaProcessada:
            wr.writerow(row)           
print(time.time() - start_time)

#------------------------------------------------------------------------------
# Salva dicionario
#------------------------------------------------------------------------------

dicionarioFinal.save('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/dicionarioFinal.dict')    

#carrega dicionaria
#dicionarioFinal=corpora.Dictionary.load('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/dicionarioFinal.dict', mmap='r')
print(dicionarioFinal)
tamanho_dicionario = 181842
del(dicionarioParcial,listaProcessada,row,stopwords)
gc.collect()

###############################################################################################################################
# CRIA VETORES DE TEXTO
###############################################################################################################################

#******************************************************************************************************************************
# Dados de Treinamento
#******************************************************************************************************************************

#------------------------------------------------------------------------------
# Cria o corpus de Bag of Words
#------------------------------------------------------------------------------

start_time = time.time()        
class MyCorpus_Treinamento_Doc2Bow(object):
    def __iter__(self):
        for line in open('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/listaProcessadaFinal_Treinamento.csv'):
            yield dicionarioFinal.doc2bow(line.split(','))
corpora.MmCorpus.serialize('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/corpusTreinamento_BOW.mm', MyCorpus_Treinamento_Doc2Bow())
print(time.time() - start_time)


corpus_treinamento_bow_sparse = matutils.corpus2csc(corpora.MmCorpus('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/corpusTreinamento_BOW.mm'), tamanho_dicionario).transpose()
corpus_treinamento_bow_sparse.shape

#------------------------------------------------------------------------------
# Cria o corpus TF-IDF
#------------------------------------------------------------------------------
start_time = time.time()
modeloTfidfTreinamento = TfidfModel(corpora.MmCorpus('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/corpusTreinamento_BOW.mm') , id2word=dicionarioFinal, normalize=True)
modeloTfidfTreinamento.save('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/corpusTreinamento_TFIDF.tfidf_model')
MmCorpus.serialize('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/corpusTreinamento_TFIDF.mm', modeloTfidfTreinamento[corpora.MmCorpus('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/corpusTreinamento_BOW.mm')], progress_cnt=10000)
del(modeloTfidfTreinamento)
print(time.time() - start_time)


corpus_treinamento_tfidf_sparse = matutils.corpus2csc(corpora.MmCorpus('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/corpusTreinamento_TFIDF.mm'), tamanho_dicionario).transpose()
corpus_treinamento_tfidf_sparse.shape

#------------------------------------------------------------------------------
# Cria o corpus LSI
#------------------------------------------------------------------------------
num_topics=300
start_time = time.time()
modeloLSITreinamento = LsiModel(corpora.MmCorpus('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/corpusTreinamento_TFIDF.mm'), id2word=dicionarioFinal, num_topics=num_topics)
modeloLSITreinamento.save('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/corpusTreinamento_LSI.lsi_model')
MmCorpus.serialize('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/corpusTreinamento_LSI.mm', modeloLSITreinamento[corpora.MmCorpus('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/corpusTreinamento_TFIDF.mm')], progress_cnt=10000)
del(modeloLSITreinamento)
print(time.time() - start_time)

corpus_treinamento_lsi_sparse = matutils.corpus2csc(corpora.MmCorpus('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/corpusTreinamento_LSI.mm'), num_topics).transpose()
corpus_treinamento_lsi_sparse.shape

#------------------------------------------------------------------------------
# Busca o target do conjunto de treinamento: assunto de nível 2
#------------------------------------------------------------------------------
 
assuntos_Treinamento = solr.query('classificacaoDeDocumentos_hierarquiaCompleta',{'q':queryTreinamento,'rows':'1000000','fl':'cd_assunto_nivel_2','sort':'id asc'})
assuntos_Treinamento = pd.DataFrame(assuntos_Treinamento.docs)    
assuntos_Treinamento.shape


#verifica se o numero de documentos bate com o numero de assuntos
#assuntosMacroClasse1_Treinamento.shape
#row_count = sum(1 for line in open("/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/listaProcessadaFinal_mc1_Treinamento.csv"))


gc.collect()
#******************************************************************************************************************************
# TESTE
#******************************************************************************************************************************

#------------------------------------------------------------------------------
# Cria o corpus de Bag of Words
#------------------------------------------------------------------------------
start_time = time.time()
class MyCorpus_Teste_Doc2Bow(object):
    def __iter__(self):
        for line in open('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/listaProcessadaFinal_Teste.csv'):
            yield dicionarioFinal_mc1.doc2bow(line.split(','))
corpora.MmCorpus.serialize('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/corpusTeste_BOW.mm', MyCorpus_Teste_Doc2Bow())
print(time.time() - start_time)       

corpus_teste_bow_sparse = matutils.corpus2csc(corpora.MmCorpus('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/corpusTeste_BOW.mm'), tamanho_dicionario).transpose()
corpus_teste_bow_sparse.shape
  
#------------------------------------------------------------------------------
# Cria o corpus TF-IDF
#------------------------------------------------------------------------------
start_time = time.time()
modeloTfidfTeste = TfidfModel(corpora.MmCorpus('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/corpusTeste_BOW.mm'), id2word=dicionarioFinal, normalize=True)
modeloTfidfTeste.save('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/corpusTeste_TFIDF.tfidf_model')
MmCorpus.serialize('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/corpusTeste_TFIDF.mm', modeloTfidfTeste[corpora.MmCorpus('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/corpusTeste_BOW.mm')], progress_cnt=10000)
print(time.time() - start_time)

corpus_teste_tfidf_sparse = matutils.corpus2csc(corpora.MmCorpus('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/corpusTeste_TFIDF.mm'), tamanho_dicionario).transpose()
corpus_teste_tfidf_sparse.shape

#------------------------------------------------------------------------------
# Cria o corpus LSI
#------------------------------------------------------------------------------
num_topics_mc1=300
start_time = time.time()
modeloLSITeste = LsiModel(corpora.MmCorpus('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/corpusTeste_TFIDF.mm'), id2word=dicionarioFinal, num_topics=num_topics)
modeloLSITeste.save('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/corpusTeste_LSI.lsi_model')
MmCorpus.serialize('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/corpusTeste_LSI.mm', modeloLSITeste[corpora.MmCorpus('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/corpusTeste_TFIDF.mm')], progress_cnt=10000)
del(modeloLSITeste)
print(time.time() - start_time)

corpus_teste_lsi_sparse = matutils.corpus2csc(corpora.MmCorpus('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/corpusTeste_TFIDF.mm'), num_topics).transpose()
corpus_teste_lsi_sparse.shape

#------------------------------------------------------------------------------
# Busca o target do conjunto de teste: assunto de nível 2
#------------------------------------------------------------------------------
assuntos_Teste = solr.query('classificacaoDeDocumentos_hierarquiaCompleta',{'q':queryTeste,'rows':'1000000','fl':'cd_assunto_nivel_2','sort':'id asc'})
assuntos_Teste = pd.DataFrame(assuntos_Teste.docs)    

#verifica se o numero de documentos bate com o numero de assuntos
#assuntosMacroClasse1_Teste.shape
#row_count = sum(1 for line in open("/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/listaProcessadaFinal_mc1_Teste.csv"))


################################################################################################################################
# INDUÇÃO DE MODELOS 
################################################################################################################################

# =============================================================================
# TF-IDF
# =============================================================================

avaliacaoFinal = pd.DataFrame(columns=['Model','Features','Macro Precisão', 'Macro Revocação', 'Macro F1-Measure','Micro Precisão', 'Micro Revocação', 'Micro F1-Measure'])
avaliacaoFinal.columns=['Model','Features','Macro Precision', 'Macro Recall', 'Macro F1-Measure','Micro Precision', 'Micro Recall', 'Micro F1-Measure']

classes = pd.DataFrame(assuntos_Teste['cd_assunto_nivel_2'].astype('category').values.describe())
classes.reset_index(inplace=True)
classes = classes.categories.tolist()

#//TODO: fazer validacao cruzada
#------------------------------------------------------------------------------
# Modelos
#------------------------------------------------------------------------------

naive_bayes(corpus_treinamento_tfidf_sparse,assuntos_Treinamento['cd_assunto_nivel_2'].astype('category').values,corpus_teste_tfidf_sparse, assuntos_Teste['cd_assunto_nivel_2'].astype('category').values, 1,classes,'TFIDF')
svm(corpus_treinamento_tfidf_sparse,assuntos_Treinamento['cd_assunto_nivel_2'],corpus_teste_tfidf_sparse, assuntos_Teste['cd_assunto_nivel_2'], 1,classes,'TFIDF')
random_forest(corpus_treinamento_tfidf_sparse,assuntos_Treinamento['cd_assunto_nivel_2'],corpus_teste_tfidf_sparse,  assuntos_Teste['cd_assunto_nivel_2'], 1,classes,'TFIDF')
mlp(corpus_treinamento_tfidf_sparse,assuntos_Treinamento['cd_assunto_nivel_2'],corpus_teste_tfidf_sparse,  assuntos_Teste['cd_assunto_nivel_2'], 1,classes,'TFIDF')

    
naive_bayes(corpus_treinamento_bow_sparse,assuntos_Treinamento['cd_assunto_nivel_2'].astype('category').values,corpus_teste_bow_sparse, assuntos_Teste['cd_assunto_nivel_2'].astype('category').values, 1,classes,'BOW')
svm(corpus_treinamento_bow_sparse,assuntos_Treinamento['cd_assunto_nivel_2'],corpus_teste_bow_sparse, assuntos_Teste['cd_assunto_nivel_2'], 1,classes,'BOW')
random_forest(corpus_treinamento_bow_sparse,assuntos_Treinamento['cd_assunto_nivel_2'],corpus_teste_bow_sparse,  assuntos_Teste['cd_assunto_nivel_2'], 1,classes,'BOW')
mlp(corpus_treinamento_bow_sparse,assuntos_Treinamento['cd_assunto_nivel_2'],corpus_teste_bow_sparse,  assuntos_Teste['cd_assunto_nivel_2'], 1,classes,'BOW')




#TODO: fazer a curva de aprendizagem do ganho do algoritmo com a quantidade de elementos para verificar se precisa rodar com tudo etc. 
    #https://www.kaggle.com/residentmario/learning-curves-with-zillow-economics-data/

# =============================================================================
# LSI
# =============================================================================

#------------------------------------------------------------------------------
# SVM MC1
#------------------------------------------------------------------------------


param_grid = {
    'loss': [ 'modified_huber', 'squared_hinge'],
    'penalty': ['elasticnet','l2'],
    'alpha': [1e-4,1e-3]
    #'l1_ratio': [0, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1],
}
clf_SVM = SGDClassifier(random_state=0, class_weight='balanced',n_jobs=7)
clf_SVM_grid = grid_search.GridSearchCV(estimator=clf_SVM, param_grid=param_grid,
                                     scoring='f1_weighted',cv = 3)
clf_SVM_grid.fit(corpus_treinamento_mc1_lsi_sparse, assuntosMacroClasse1_Treinamento.cd_assunto_nivel_2)


print(clf_SVM_grid.best_score_)
print(clf_SVM_grid.best_params_)
teste_svm_c1_segunda_execucao = pd.DataFrame(clf_SVM_grid.grid_scores_)

clf_SVM = clf_SVM_grid.best_estimator_
clf_SVM.fit(corpus_treinamento_mc1_lsi_sparse, assuntosMacroClasse1_Treinamento.cd_assunto_nivel_2)
predicted_SVM_mc1_lsi =  clf_SVM.predict(corpus_teste_mc1_lsi_sparse)
np.mean(predicted_SVM_mc1_lsi == assuntosMacroClasse1_Teste.cd_assunto_nivel_2)


codigos= pd.DataFrame(assuntosMacroClasse1_Treinamento['cd_assunto_nivel_2'].astype('category').values.describe())
codigos.reset_index(inplace=True)
codigos = codigos.categories.tolist()

confusion_matrix_SVM_mc1_lsi = confusion_matrix(assuntosMacroClasse1_Teste['cd_assunto_nivel_2'].astype('category').values,predicted_SVM_mc1_lsi)
fig = plt.figure(figsize=(10,10))

plot_confusion_matrix(confusion_matrix_SVM_mc1_lsi, codigos,
                      title='SVM \nMacro Class 1 - TF-IDF\nAccuracy: {0:.3f}'.format(accuracy_score(assuntosMacroClasse1_Teste['cd_assunto_nivel_2'].astype('category').values,predicted_SVM_mc1_lsi)))

plt.savefig('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/imagens/confusion_matrix_SVM_mc1_lsi.png') 

macro_precision,macro_recall,macro_fscore,macro_support=score(assuntosMacroClasse1_Teste['cd_assunto_nivel_2'].astype('category').values,predicted_SVM_mc1_lsi,average='macro')
micro_precision,micro_recall,micro_fscore,micro_support=score(assuntosMacroClasse1_Teste['cd_assunto_nivel_2'].astype('category').values,predicted_SVM_mc1_lsi,average='weighted')

avaliacaoFinal_MC1_TFIDF.loc[4]= ['SVM','LSI',macro_precision,macro_recall,macro_fscore,micro_precision,micro_recall,micro_fscore]

print(classification_report(assuntosMacroClasse1_Teste['cd_assunto_nivel_2'].astype('category').values,predicted_SVM_mc1_lsi))
print('Micro average precision = {:.2f} (dâ o mesmo peso para cada instância)'.format(precision_score(assuntosMacroClasse1_Teste['cd_assunto_nivel_2'].astype('category').values,predicted_SVM_mc1_lsi,average = 'weighted')))
print('Macro average precision = {:.2f} (dâ o mesmo peso para cada classe)'.format(precision_score(assuntosMacroClasse1_Teste['cd_assunto_nivel_2'].astype('category').values,predicted_SVM_mc1_lsi,average = 'macro')))




# =============================================================================
# Ensemble
# =============================================================================
from sklearn.ensemble import VotingClassifier
estimators = []

model1 = MultinomialNB(class_prior=None,fit_prior=True)
model2 = clf_SVM_mc1_tfidf = SGDClassifier(loss='modified_huber', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)
model3 = clf_RF_mc1_tfidf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=50, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=50, min_samples_split=50,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=7,
            oob_score=False, random_state=0, verbose=0, warm_start=False)
model4 = clf_MLP_mc1_tfidf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(10, 5), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9, 
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)

estimators.append(('Naive Bayes', model1))
estimators.append(('SVM', model2))
estimators.append(('Random Forest', model3))
#estimators.append(('Naive Bayes', model4))
# create the ensemble model
ensemble = VotingClassifier(estimators)

from sklearn import model_selection
kfold = model_selection.KFold(n_splits=5, random_state=12)
results = model_selection.cross_val_score(ensemble, corpus_treinamento_mc2_tfidf_sparse, assuntosMacroClasse2_Treinamento, cv=kfold)
print(results.mean())


type(corpus_treinamento_mc3_tfidf_sparse)
teste = corpus_treinamento_mc1_tfidf_sparse.todense()




##############################################################################
# VERIFICANDO A DISTRIBUIÇÃO DOS DADOS
##############################################################################

#-----------------------------------------------------------------------------
# Quantidade média de palavras por classe
#-----------------------------------------------------------------------------
    
contador =  pd.DataFrame(columns=['qtd_palavras','cd_assunto_nivel_2'])
count = 0;
for y in range(0,len(df)):
    #normalizando o texto
        documento =  df['tx_conteudo_documento'][y].split()
        for palavra in documento :
            count += 1    
        contador.loc[y] = [count, df['cd_assunto_nivel_2'][y]]
        count = 0
        
for y in range(0,len(contador)):
    contador['qtd_palavras'][y] = float(contador['qtd_palavras'][y])

contador.groupby(['cd_assunto_nivel_2']).mean()  

grouped2=pd.to_numeric(contador['qtd_palavras']).groupby(contador['cd_assunto_nivel_2'])
grouped2.mean()
  
    


#Teste para ver se o parallel manteria a ordem        
df = pd.DataFrame(columns=['inicial','final'])
for i in range(0,10000):
    df.loc[i] = [i,i]

def quadrado(f):
    return f*f
df.inicial[15]

resultado = Parallel(n_jobs = 7)(delayed(quadrado)(item) for item in df.inicial)