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
from sklearn.linear_model import SGDClassifier
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
##############################################################################
# DEFINE FUNÇÕES
##############################################################################
stopwords = nltk.corpus.stopwords.words('portuguese')
stopwords.extend(['microsoftinternetexplorer','false','none','trabalho','juiz',
                  'reclamado','reclamada','autos','autor','excelentissimo',
                  'senhor','normal'])

nltk.download('rslp')
stemmer = nltk.stem.RSLPStemmer()

def processa_texto(texto):
        textoProcessado = BeautifulSoup(texto, 'html.parser').string
        textoProcessado = normalize('NFKD', textoProcessado).encode('ASCII','ignore').decode('ASCII')
        textoProcessado = re.sub('[^a-zA-Z]',' ',textoProcessado)
        textoProcessado = textoProcessado.lower()
        textoProcessado = textoProcessado.split()
        textoProcessado = [palavra for palavra in textoProcessado if not palavra in stopwords]
        textoProcessado = [palavra for palavra in textoProcessado if len(palavra)>3]
        textoProcessado =  [stemmer.stem(palavra) for palavra in textoProcessado]
        return textoProcessado


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
    plt.figure(figsize=(15,15))    
    plt.rcParams.update({'font.size': 25})
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

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# =============================================================================
# Multinomial Nayve Bayes
# =============================================================================


def naive_bayes(training_corpus,training_classes,test_corpus, test_classes, classNumber,classes,featureType):
    classesCM = []
    classesCM = classes
    clf_NB = MultinomialNB(class_prior=None,fit_prior=True).fit(training_corpus, training_classes)

    predicted_NB = clf_NB.predict(test_corpus)
    np.mean(predicted_NB == test_classes)
    
    confusion_matrix_NB = confusion_matrix(test_classes,predicted_NB)
    
    matrixHeaderString = 'Naive Bayes \nMacro Class ' + str(classNumber) +' - ' +featureType + '\nAccuracy: {0:.3f}'.format(accuracy_score(test_classes,predicted_NB))
    plot_confusion_matrix(confusion_matrix_NB, classesCM, title=matrixHeaderString)
    figureFile = '/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/imagens/confusion_matrix_NB_mc'+ str(classNumber) +'_tfidf_testeFuncao.png'
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
        'loss': [ 'modified_huber', 'squared_hinge'],
        'penalty': ['elasticnet','l2'],
        'alpha': [1e-4,1e-3]
        #'l1_ratio': [0, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1],
    }
    clf_SVM = SGDClassifier(random_state=0, class_weight='balanced',n_jobs=7)
    clf_SVM_grid = grid_search.GridSearchCV(estimator=clf_SVM, param_grid=param_grid,
                                         scoring='f1_weighted',cv = 3)
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
    figureFile = '/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/imagens/confusion_matrix_SVM_mc'+ str(classNumber) +'_tfidf_testeFuncao.png'
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
        'class_weight':['balanced','balanced_subsample'],
        'n_estimators':[300,400,600]
        
    }
    #param_grid = {
    #    'max_depth': [25,50],'class_weight':['balanced','balanced_subsample'],
    #    'criterion': ['gini','entropy'],'min_weight_fraction_leaf':[0.0,0.5],
    #    'bootstrap': [True,False] , 'max_features':['auto','log2'], 'n_estimators':[50,100]}
    clf_RF = RandomForestClassifier(random_state=1986,n_jobs=7,max_leaf_nodes=None,max_features='auto',
                                    min_samples_split=15, min_samples_leaf=50, bootstrap=False, criterion='gini')
    clf_RF_grid = grid_search.GridSearchCV(estimator=clf_RF, param_grid=param_grid,
                                         scoring='f1_weighted',cv = 3, verbose=2)
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
    figureFile = '/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/imagens/confusion_matrix_RF_mc'+ str(classNumber) +'_tfidf_testeFuncao.png'
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
        #TODO: tocar o 5,10,15 para 2,5
        #'hidden_layer_sizes': [(2, 5), (5,10)],
        'hidden_layer_sizes': [(5,10),(5,5,5)],
        'max_iter': [100,200],
        #'activation' : ['identity', 'logistic', 'tanh', 'relu'],
        'activation' : ['identity', 'logistic', 'tanh']
     }

    
    
    clf_MLP = MLPClassifier( batch_size='auto',
           beta_1=0.9, beta_2=0.999, early_stopping=False,
            momentum=0.9,   random_state=1, 
           solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
           warm_start=False)
    
    
    clf_MLP_grid = GridSearchCV(estimator = clf_MLP_mc3_tfidf, param_grid = param_grid, 
                              cv = 3,  verbose = 2)
    
    clf_MLP_grid.fit(training_corpus, training_classes)    

    
    result_grid_MLP = pd.DataFrame(clf_MLP_grid.grid_scores_)
    
    clf_MLP = clf_MLP_grid.best_estimator_
    clf_MLP.fit(training_corpus, training_classes)
    predicted_MLP =  clf_MLP.predict(test_corpus)
    np.mean(predicted_MLP == test_corpus)
    
    
    confusion_matrix_MLP = confusion_matrix(test_classes,predicted_MLP)
    
    matrixHeaderString = 'Multilayer Perceptron \nMacro Class ' + str(classNumber) +' - ' +featureType + '\nAccuracy: {0:.3f}'.format(accuracy_score(test_classes,predicted_MLP))
    plot_confusion_matrix(confusion_matrix_MLP, classesCM, title=matrixHeaderString)
    figureFile = '/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/imagens/confusion_matrix_MLP_mc'+ str(classNumber) +'_tfidf_testeFuncao.png'
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
    
   
    
mlp(corpus_treinamento_mc3_tfidf_sparse,assuntosMacroClasse3_Treinamento.cd_assunto_nivel_2,corpus_teste_mc3_tfidf_sparse, assuntosMacroClasse3_Teste.cd_assunto_nivel_2, 3,classes_mc3,'TFIDF')
mlp(corpus_treinamento_mc2_tfidf_sparse,assuntosMacroClasse2_Treinamento.cd_assunto_nivel_2,corpus_teste_mc2_tfidf_sparse, assuntosMacroClasse2_Teste.cd_assunto_nivel_2, 2,classes_mc2,'TFIDF')
mlp(corpus_treinamento_mc1_tfidf_sparse,assuntosMacroClasse1_Treinamento.cd_assunto_nivel_2,corpus_teste_mc1_tfidf_sparse, assuntosMacroClasse1_Teste.cd_assunto_nivel_2, 1,classes_mc1,'TFIDF')



    
    
##############################################################################
# BUSCA OS DADOS 
##############################################################################
solr = SolrClient('http://localhost:8983/solr')
solrDataAnalise = solr.query('classificacaoDeDocumentos_hierarquiaCompleta',{
'q':'tx_conteudo_documento:[* TO *]','fl':'id_processo_documento,cd_assunto_nivel_1,cd_assunto_nivel_2,cd_assunto_nivel_3,cd_assunto_nivel_4,cd_assunto_nivel_5', 'rows':'300000'
})
dfGeral = pd.DataFrame(solrDataAnalise.docs)    
del(solrDataAnalise)
#-----------------------------------------------------------------------------
# Verifica a distribuição dos dados
#-----------------------------------------------------------------------------
fig = plt.figure(figsize=(20,5))
plt.rcParams.update({'font.size': 10})
dfGeral.groupby('cd_assunto_nivel_2').id_processo_documento.count().plot.bar(ylim=0)
plt.title('Distribuição geral dos dados')
plt.ylabel('Quantidade de Documentos')
plt.xlabel('Código do assunto')
plt.show()
fig.savefig('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/imagens/TRT15_2GRAU_DistribuicaoClasses_2Nivel_Geral.png')  


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
fig.savefig('todo/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/imagens/TRT15_2GRAU_DistribuicaoClasses_2Nivel_Processo_por_nivel.png')  

#------------------------------------------------------------------------------
# O problema se trata de uma uma classificação multi-classe com dasdos bastante 
# desbalanceados. Portanto, será criado 3 classificadores diferentes, conforme abaixo:
#  1 - MACRO CLASSE 1: Entre 15 mil e 80 mil dados
#  2 - MACRO CLASSE 2: Entre mil e 15 mil dados
#  3 - MACRO CLASSE 3: Menos que mil dados
#------------------------------------------------------------------------------
quantidadesPorAssunto =  pd.DataFrame(dfGeral.groupby('cd_assunto_nivel_2').id_processo_documento.count().nlargest(70))
quantidadesPorAssunto.reset_index(inplace=True)
quantidadesPorAssunto.columns=['cd_assunto_nivel_2','quantidadeDocumentos']
codigosMacroClasse1 = quantidadesPorAssunto[(quantidadesPorAssunto['quantidadeDocumentos'] > 15000)]
codigosMacroClasse2 = quantidadesPorAssunto[(quantidadesPorAssunto['quantidadeDocumentos'] > 1000) & (quantidadesPorAssunto['quantidadeDocumentos'] <15000)]
codigosMacroClasse3 = quantidadesPorAssunto[(quantidadesPorAssunto['quantidadeDocumentos'] < 1000) & (quantidadesPorAssunto['quantidadeDocumentos'] >50)]
codigosMacroClasse_excluidos = quantidadesPorAssunto[(quantidadesPorAssunto['quantidadeDocumentos'] < 51)]

codigosMacroClasse1.groupby('cd_assunto_nivel_2').quantidadeDocumentos.sum()
codigosMacroClasse1.describe()
codigosMacroClasse1.quantidadeDocumentos.sum()

codigosMacroClasse2.groupby('cd_assunto_nivel_2').quantidadeDocumentos.sum()
codigosMacroClasse2.describe()
codigosMacroClasse2.quantidadeDocumentos.sum()

codigosMacroClasse3.groupby('cd_assunto_nivel_2').quantidadeDocumentos.sum()
codigosMacroClasse3.describe()
codigosMacroClasse3.quantidadeDocumentos.sum()

codigosMacroClasse_excluidos.groupby('cd_assunto_nivel_2').quantidadeDocumentos.sum()
codigosMacroClasse_excluidos.describe()
codigosMacroClasse_excluidos.quantidadeDocumentos.sum()
#-----------------------------------------------------------------------------
# Verifica a distribuição dos dados por macroclasse
#-----------------------------------------------------------------------------
def cria_grafico_barra(data, title, ylabel, xlabel, fontSize, figSize_x, figSize_y, nomeImagem):
    path = '/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/imagens/'
    fig = plt.figure(figsize=(figSize_x,figSize_y))
    plt.rcParams.update({'font.size': fontSize})
    data.plot.bar(ylim=0)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()
    fig.savefig(path + nomeImagem + '.png')  
    

cria_grafico_barra(codigosMacroClasse1.groupby('cd_assunto_nivel_2').quantidadeDocumentos.sum(),
                   'Quantidade de documentos na Macro Classe 1','Quantidade de Documentos','Código do assunto',10,10,4,'TRT15_2GRAU_DistribuicaoClasses_2Nivel_MC1')

cria_grafico_barra(codigosMacroClasse2.groupby('cd_assunto_nivel_2').quantidadeDocumentos.sum(),
                   'Quantidade de documentos na Macro Classe 1','Quantidade de Documentos','Código do assunto',10,10,4,'TRT15_2GRAU_DistribuicaoClasses_2Nivel_MC1')

cria_grafico_barra(codigosMacroClasse3.groupby('cd_assunto_nivel_2').quantidadeDocumentos.sum(),
                   'Quantidade de documentos na Macro Classe 1','Quantidade de Documentos','Código do assunto',10,10,4,'TRT15_2GRAU_DistribuicaoClasses_2Nivel_MC1')
        
#TODO:parei aqui na refatoracao....

fig = plt.figure(figsize=(10,4))
plt.rcParams.update({'font.size': 10})
codigosMacroClasse1.groupby('cd_assunto_nivel_2').quantidadeDocumentos.sum().plot.bar(ylim=0)
plt.title('Quantidade de documentos na Macro Classe 1')
plt.ylabel('Quantidade de Documentos')
plt.xlabel('Código do assunto')
plt.show()
fig.savefig('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/imagens/TRT15_2GRAU_DistribuicaoClasses_2Nivel_MC1.png')  

fig = plt.figure(figsize=(10,4))
plt.rcParams.update({'font.size': 10})
codigosMacroClasse2.groupby('cd_assunto_nivel_2').quantidadeDocumentos.sum().plot.bar(ylim=0)
plt.title('Quantidade de documentos na Macro Classe 2')
plt.ylabel('Quantidade de Documentos')
plt.xlabel('Código do assunto')
plt.show()
fig.savefig('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/imagens/TRT15_2GRAU_DistribuicaoClasses_2Nivel_MC2.png')  

fig = plt.figure(figsize=(10,4))
plt.rcParams.update({'font.size': 10})
codigosMacroClasse3.groupby('cd_assunto_nivel_2').quantidadeDocumentos.sum().plot.bar(ylim=0)
plt.title('Quantidade de documentos na Macro Classe 3')
plt.ylabel('Quantidade de Documentos')
plt.xlabel('Código do assunto')
plt.show()
fig.savefig('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/imagens/TRT15_2GRAU_DistribuicaoClasses_2Nivel_MC3.png')  


# Data for plotting

# Create two subplots sharing y axis



plt.figure(1,figsize=(8,3))
plt.subplot(311)
codigosMacroClasse1.groupby('cd_assunto_nivel_2').quantidadeDocumentos.sum().plot.bar(ylim=0)
plt.gca().axes.get_xaxis().set_visible(False)

plt.subplot(312)
codigosMacroClasse2.groupby('cd_assunto_nivel_2').quantidadeDocumentos.sum().plot.bar(ylim=0)
plt.gca().axes.get_xaxis().set_visible(False)

plt.subplot(313)
codigosMacroClasse3.groupby('cd_assunto_nivel_2').quantidadeDocumentos.sum().plot.bar(ylim=0)
plt.gca().axes.get_xaxis().set_visible(False)
plt.show()
#TODO:tirar print manual
fig.savefig('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/imagens/TRT15_2GRAU_DistribuicaoClasses_2Nivel_MacroClasses.png')  




#-----------------------------------------------------------------------------
# Separa e analisa conjunto de treinamento
#----------------------------------------------------------------------------

queryTreinamento = 'tx_conteudo_documento:[* TO *] AND dt_juntada:[* TO 2017-07-01T00:00:00Z ]'

solrDataAnalise = solr.query('classificacaoDeDocumentos_hierarquiaCompleta',{
'q':queryTreinamento,'fl':'id_processo_documento,cd_assunto_nivel_1,cd_assunto_nivel_2,cd_assunto_nivel_3,cd_assunto_nivel_4,cd_assunto_nivel_5', 'rows':'300000'
})
dfTreinamento = pd.DataFrame(solrDataAnalise.docs)    

fig = plt.figure(figsize=(20,5))
dfTreinamento.groupby('cd_assunto_nivel_2').id_processo_documento.count().plot.bar(ylim=0)
plt.title('Distribuição dos dados de treinamento')
plt.ylabel('Quantidade de Documentos')
plt.xlabel('Código do assunto')
plt.show()
fig.savefig('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/imagens/TRT15_2GRAU_DistribuicaoClasses_2Nivel_Treinamento_Geral.png')  

#-----------------------------------------------------------------------------
# Separa e analisa conjunto de teste
#-----------------------------------------------------------------------------

queryTeste = 'tx_conteudo_documento:[* TO *] AND dt_juntada:[2017-07-01T00:00:00Z TO 2018-06-30T23:59:59Z]'

solrDataAnalise = solr.query('classificacaoDeDocumentos_hierarquiaCompleta',{
'q':queryTeste,'fl':'id_processo_documento,cd_assunto_nivel_1,cd_assunto_nivel_2,cd_assunto_nivel_3,cd_assunto_nivel_4,cd_assunto_nivel_5', 'rows':'300000'
})
dfTeste = pd.DataFrame(solrDataAnalise.docs)    


fig = plt.figure(figsize=(20,5))
dfTeste.groupby('cd_assunto_nivel_2').id_processo_documento.count().plot.bar(ylim=0)
plt.title('Distribuição dos dados de teste')
plt.ylabel('Quantidade de Documentos')
plt.xlabel('Código do assunto')
plt.show()
fig.savefig('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/imagens/TRT15_2GRAU_DistribuicaoClasses_2Nivel_Teste_Geral.png')  

del(fig)


del(dfTeste,dfTreinamento)
del(dfGeral)
gc.collect()

############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
#
# MACRO CLASSE 1: Mais que 15 mil dados
#
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################

codigosMacroClasse1 = " OR ".join([str(codigo) for codigo in codigosMacroClasse1.cd_assunto_nivel_2])
queryMacroClasse1 = 'cd_assunto_nivel_2:('  + codigosMacroClasse1  + ')' 
#******************************************************************************************************************************
# Analisa o conjunto de Treinamento e Teste para a Macro Classe 1
#******************************************************************************************************************************
#-----------------------------------------------------------------------------
# Treinamento
#----------------------------------------------------------------------------

queryTreinamento_mc1 = 'tx_conteudo_documento:[* TO *] AND dt_juntada:[* TO 2017-07-01T00:00:00Z ] AND ' + queryMacroClasse1

#busca os dados da MC1 com seus respectivos códigos
solrDataAnalise = solr.query('classificacaoDeDocumentos_hierarquiaCompleta',{
'q':queryTreinamento_mc1,'fl':'id_processo_documento,cd_assunto_nivel_2', 'rows':'300000'
})
dfTreinamento_mc1 = pd.DataFrame(solrDataAnalise.docs)    


#busca os dados que não são da MC1 e traz todo mundo com um código -1
queryTreinamento_mc1_outros = 'tx_conteudo_documento:[* TO *] AND dt_juntada:[* TO 2017-07-01T00:00:00Z ] AND NOT ' + queryMacroClasse1
solrDataAnalise = solr.query('classificacaoDeDocumentos_hierarquiaCompleta',{
'q':queryTreinamento_mc1_outros,'fl':'id_processo_documento,cd_assunto_nivel_2', 'rows':'300000'
})
dfTreinamento_mc1_outros = pd.DataFrame(solrDataAnalise.docs)    
dfTreinamento_mc1_outros['cd_assunto_nivel_2'] = -1 

dfTreinamento_mc1 = dfTreinamento_mc1.append(dfTreinamento_mc1_outros)

fig = plt.figure(figsize=(18,13))
dfTreinamento_mc1.groupby('cd_assunto_nivel_2').id_processo_documento.count().plot.bar(ylim=0)
plt.title('Distribuição dos dados de treinamento')
plt.ylabel('Quantidade de Documentos')
plt.xlabel('Código do assunto')
plt.show()
fig.savefig('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/imagens/TRT15_2GRAU_DistribuicaoClasses_2Nivel_Treinamento_MC1.png')  

#-----------------------------------------------------------------------------
# Teste
#-----------------------------------------------------------------------------

queryTeste_mc1 = 'tx_conteudo_documento:[* TO *] AND dt_juntada:[2017-07-01T00:00:00Z TO 2018-06-30T23:59:59Z] AND ' + queryMacroClasse1

#busca os dados da MC1 com seus respectivos códigos
solrDataAnalise = solr.query('classificacaoDeDocumentos_hierarquiaCompleta',{
'q':queryTeste_mc1,'fl':'id_processo_documento,cd_assunto_nivel_1,cd_assunto_nivel_2,cd_assunto_nivel_3,cd_assunto_nivel_4,cd_assunto_nivel_5', 'rows':'300000'
})
dfTeste_mc1 = pd.DataFrame(solrDataAnalise.docs)    

#busca os dados que não são da MC1 e traz todo mundo com um código -1
queryTeste_mc1_outros = 'tx_conteudo_documento:[* TO *] AND dt_juntada:[* TO 2017-07-01T00:00:00Z ] AND NOT ' + queryMacroClasse1
solrDataAnalise = solr.query('classificacaoDeDocumentos_hierarquiaCompleta',{
'q':queryTeste_mc1_outros,'fl':'id_processo_documento,cd_assunto_nivel_2', 'rows':'300000'
})
dfTeste_mc1_outros = pd.DataFrame(solrDataAnalise.docs)    
dfTeste_mc1_outros['cd_assunto_nivel_2'] = -1 

dfTeste_mc1 = dfTeste_mc1.append(dfTeste_mc1_outros)


fig = plt.figure(figsize=(18,13))
dfTeste_mc1.groupby('cd_assunto_nivel_2').id_processo_documento.count().plot.bar(ylim=0)
plt.title('Distribuição dos dados de teste')
plt.ylabel('Quantidade de Documentos')
plt.xlabel('Código do assunto')
plt.show()
fig.savefig('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/imagens/TRT15_2GRAU_DistribuicaoClasses_2Nivel_Teste_MC1.png')  


del(dfTreinamento_mc1,dfTreinamento_mc1_outros,dfTeste_mc1,dfTeste_mc1_outros,fig)

################################################################################################################################
# BUSCA DADOS E CRIA O DICIONÁRIO
################################################################################################################################
dicionarioFinal_mc1 = corpora.Dictionary('')
#------------------------------------------------------------------------------
# Busca dados de Treinamento
#------------------------------------------------------------------------------
queryMacroClasse1_Treinamento = queryMacroClasse1 + ' AND ' + queryTreinamento

#busca todos os documentos que de fato são da classe 1
start_time = time.time()
listaProcessada = []
for resCursor in solr.cursor_query('classificacaoDeDocumentos_hierarquiaCompleta',{'q':queryMacroClasse1_Treinamento,'rows':'10000','fl':'tx_conteudo_documento','sort':'id asc'}):  
    listaProcessada = Parallel(n_jobs = 7)(delayed(processa_texto)(documento.get('tx_conteudo_documento')) for documento in resCursor.docs)
    dicionarioParcial = corpora.Dictionary(listaProcessada)
    dicionarioFinal_mc1.merge_with(dicionarioParcial)
    print(time.time() - start_time)
    with open("/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/listaProcessadaFinal_mc1_Treinamento.csv", "a") as fp:
        wr = csv.writer(fp, dialect='excel')
        for row in listaProcessada:
            wr.writerow(row)       
    with open("/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/listaProcessadaFinal_mc1_Treinamento backup.csv", "a") as fp:
        wr = csv.writer(fp, dialect='excel')
        for row in listaProcessada:
            wr.writerow(row)       
print(time.time() - start_time)

#busca todos os documentos que de fato NÃO SÃO da classe 1, para serem tratados como 'outros'
queryMacroClasse1_Treinamento_outros = 'NOT ' + queryMacroClasse1_Treinamento
start_time = time.time()
listaProcessada = []
for resCursor in solr.cursor_query('classificacaoDeDocumentos_hierarquiaCompleta',{'q':queryMacroClasse1_Treinamento_outros,'rows':'10000','fl':'tx_conteudo_documento','sort':'id asc'}):  
    listaProcessada = Parallel(n_jobs = 7)(delayed(processa_texto)(documento.get('tx_conteudo_documento')) for documento in resCursor.docs)
    dicionarioParcial = corpora.Dictionary(listaProcessada)
    dicionarioFinal_mc1.merge_with(dicionarioParcial)
    print(time.time() - start_time)
    with open("/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/listaProcessadaFinal_mc1_Treinamento.csv", "a") as fp:
        wr = csv.writer(fp, dialect='excel')
        for row in listaProcessada:
            wr.writerow(row)       
print(time.time() - start_time)

#------------------------------------------------------------------------------
# Busca dados de Teste
#------------------------------------------------------------------------------
#busca todos os documentos que de fato são da classe 1
queryMacroClasse1_Teste = queryMacroClasse1 + ' AND ' + queryTeste
start_time = time.time()
listaProcessada = []
for resCursor in solr.cursor_query('classificacaoDeDocumentos_hierarquiaCompleta',{'q':queryMacroClasse1_Teste,'rows':'10000','fl':'tx_conteudo_documento','sort':'id asc'}):  
    listaProcessada = Parallel(n_jobs = 7)(delayed(processa_texto)(documento.get('tx_conteudo_documento')) for documento in resCursor.docs)
    dicionarioParcial = corpora.Dictionary(listaProcessada)
    dicionarioFinal_mc1.merge_with(dicionarioParcial)
    with open("/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/listaProcessadaFinal_mc1_Teste.csv", "a") as fp:
        wr = csv.writer(fp, dialect='excel')
        for row in listaProcessada:
            wr.writerow(row)
    
print(time.time() - start_time)


#busca todos os documentos que de fato NÃO SÃO da classe 1, para serem tratados como 'outros'
queryMacroClasse1_Teste_outros = 'NOT ' + queryMacroClasse1_Teste
start_time = time.time()
listaProcessada = []
for resCursor in solr.cursor_query('classificacaoDeDocumentos_hierarquiaCompleta',{'q':queryMacroClasse1_Teste_outros,'rows':'10000','fl':'tx_conteudo_documento','sort':'id asc'}):  
    listaProcessada = Parallel(n_jobs = 7)(delayed(processa_texto)(documento.get('tx_conteudo_documento')) for documento in resCursor.docs)
    dicionarioParcial = corpora.Dictionary(listaProcessada)
    dicionarioFinal_mc1.merge_with(dicionarioParcial)
    with open("/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/listaProcessadaFinal_mc1_Teste.csv", "a") as fp:
        wr = csv.writer(fp, dialect='excel')
        for row in listaProcessada:
            wr.writerow(row)
print(time.time() - start_time)
#------------------------------------------------------------------------------
# Salva dicionario
#------------------------------------------------------------------------------

dicionarioFinal_mc1.save('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/dicionarioFinal_mc1.dict')    
#dicionarioFinal_mc1=corpora.Dictionary.load('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/dicionarioFinal_mc1.dict', mmap='r')
print(dicionarioFinal_mc1)
tamanho_dicionario = 181846
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
class MyCorpus_Treinamento_MacroClasse1(object):
    def __iter__(self):
        for line in open('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/listaProcessadaFinal_mc1_Treinamento.csv'):
            yield dicionarioFinal_mc1.doc2bow(line.split(','))
corpora.MmCorpus.serialize('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc1_BOW.mm', MyCorpus_Treinamento_MacroClasse1())
print(time.time() - start_time)


corpus_treinamento_mc1_bow_sparse = matutils.corpus2csc(corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc1_BOW.mm'), tamanho_dicionario).transpose()
corpus_treinamento_mc1_bow_sparse.shape

#------------------------------------------------------------------------------
# Cria o corpus TF-IDF
#------------------------------------------------------------------------------
start_time = time.time()
modeloTfidfTreinamento_mc1 = TfidfModel(corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc1_BOW.mm') , id2word=dicionarioFinal_mc1, normalize=True)
modeloTfidfTreinamento_mc1.save('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc1_TFIDF.tfidf_model')
MmCorpus.serialize('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc1_TFIDF.mm', modeloTfidfTreinamento_mc1[corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc1_BOW.mm')], progress_cnt=10000)
del(modeloTfidfTreinamento_mc1)
print(time.time() - start_time)


corpus_treinamento_mc1_tfidf_sparse = matutils.corpus2csc(corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc1_TFIDF.mm'), tamanho_dicionario).transpose()
corpus_treinamento_mc1_tfidf_sparse.shape

#------------------------------------------------------------------------------
# Cria o corpus LSI
#------------------------------------------------------------------------------
num_topics_mc1=300
start_time = time.time()
modeloLSITreinamento_mc1 = LsiModel(corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc1_TFIDF.mm'), id2word=dicionarioFinal_mc1, num_topics=num_topics_mc1)
modeloLSITreinamento_mc1.save('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc1_LSI.lsi_model')
MmCorpus.serialize('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc1_LSI.mm', modeloLSITreinamento_mc1[corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc1_TFIDF.mm')], progress_cnt=10000)
del(modeloLSITreinamento_mc1)
print(time.time() - start_time)

corpus_treinamento_mc1_lsi_sparse = matutils.corpus2csc(corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc1_LSI.mm'), num_topics_mc1).transpose()
corpus_treinamento_mc1_lsi_sparse.shape

#------------------------------------------------------------------------------
# Busca o target do conjunto de treinamento: assunto de nível 2
#------------------------------------------------------------------------------
 
assuntosMacroClasse1_Treinamento = solr.query('classificacaoDeDocumentos_hierarquiaCompleta',{'q':queryMacroClasse1_Treinamento,'rows':'1000000','fl':'cd_assunto_nivel_2','sort':'id asc'})
assuntosMacroClasse1_Treinamento = pd.DataFrame(assuntosMacroClasse1_Treinamento.docs)    
assuntosMacroClasse1_Treinamento.shape

assuntosMacroClasse1_Treinamento_outros = solr.query('classificacaoDeDocumentos_hierarquiaCompleta',{'q':queryMacroClasse1_Treinamento_outros,'rows':'1000000','fl':'cd_assunto_nivel_2','sort':'id asc'})
assuntosMacroClasse1_Treinamento_outros = pd.DataFrame(assuntosMacroClasse1_Treinamento_outros.docs)    
assuntosMacroClasse1_Treinamento_outros['cd_assunto_nivel_2'] = 0 

assuntosMacroClasse1_Treinamento = assuntosMacroClasse1_Treinamento.append(assuntosMacroClasse1_Treinamento_outros)

#verifica se o numero de documentos bate com o numero de assuntos
#assuntosMacroClasse1_Treinamento.shape
#row_count = sum(1 for line in open("/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/listaProcessadaFinal_mc1_Treinamento.csv"))

assuntosMacroClasse1_Treinamento.reset_index(inplace=True)
assuntosMacroClasse1_Treinamento.cd_assunto_nivel_2

assuntosMacroClasse1_Treinamento = assuntosMacroClasse1_Treinamento['cd_assunto_nivel_2'].astype('category').values



gc.collect()
#******************************************************************************************************************************
# TESTE
#******************************************************************************************************************************

#------------------------------------------------------------------------------
# Cria o corpus de Bag of Words
#------------------------------------------------------------------------------
start_time = time.time()
class MyCorpus_Teste(object):
    def __iter__(self):
        for line in open('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/listaProcessadaFinal_mc1_Teste.csv'):
            yield dicionarioFinal_mc1.doc2bow(line.split(','))
corpora.MmCorpus.serialize('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc1_BOW.mm', MyCorpus_Teste())
print(time.time() - start_time)       

corpus_teste_mc1_bow_sparse = matutils.corpus2csc(corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc1_BOW.mm'), tamanho_dicionario).transpose()
corpus_teste_mc1_bow_sparse.shape
  
#------------------------------------------------------------------------------
# Cria o corpus TF-IDF
#------------------------------------------------------------------------------
start_time = time.time()
modeloTfidfTeste_mc1 = TfidfModel(corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc1_BOW.mm'), id2word=dicionarioFinal_mc1, normalize=True)
modeloTfidfTeste_mc1.save('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc1_TFIDF.tfidf_model')
MmCorpus.serialize('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc1_TFIDF.mm', modeloTfidfTeste_mc1[corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc1_BOW.mm')], progress_cnt=10000)
print(time.time() - start_time)

corpus_teste_mc1_tfidf_sparse = matutils.corpus2csc(corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc1_TFIDF.mm'), tamanho_dicionario).transpose()
corpus_teste_mc1_tfidf_sparse.shape
     


#------------------------------------------------------------------------------
# Cria o corpus LSI
#------------------------------------------------------------------------------
num_topics_mc1=300
start_time = time.time()
modeloLSITeste_mc1 = LsiModel(corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc1_TFIDF.mm'), id2word=dicionarioFinal_mc1, num_topics=num_topics_mc1)
modeloLSITeste_mc1.save('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc1_LSI.lsi_model')
MmCorpus.serialize('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc1_LSI.mm', modeloLSITeste_mc1[corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc1_TFIDF.mm')], progress_cnt=10000)
del(modeloLSITeste_mc1)
print(time.time() - start_time)

corpus_teste_mc1_lsi_sparse = matutils.corpus2csc(corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc1_TFIDF.mm'), num_topics_mc1).transpose()
corpus_teste_mc1_lsi_sparse.shape

#------------------------------------------------------------------------------
# Busca o target do conjunto de teste: assunto de nível 2
#------------------------------------------------------------------------------
assuntosMacroClasse1_Teste = solr.query('classificacaoDeDocumentos_hierarquiaCompleta',{'q':queryMacroClasse1_Teste,'rows':'1000000','fl':'cd_assunto_nivel_2','sort':'id asc'})
assuntosMacroClasse1_Teste = pd.DataFrame(assuntosMacroClasse1_Teste.docs)    

assuntosMacroClasse1_Teste_outros = solr.query('classificacaoDeDocumentos_hierarquiaCompleta',{'q':queryMacroClasse1_Teste_outros,'rows':'1000000','fl':'cd_assunto_nivel_2','sort':'id asc'})
assuntosMacroClasse1_Teste_outros = pd.DataFrame(assuntosMacroClasse1_Teste_outros.docs)    
assuntosMacroClasse1_Teste_outros['cd_assunto_nivel_2'] = 0

assuntosMacroClasse1_Teste = assuntosMacroClasse1_Teste.append(assuntosMacroClasse1_Teste_outros)


#verifica se o numero de documentos bate com o numero de assuntos
#assuntosMacroClasse1_Teste.shape
#row_count = sum(1 for line in open("/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/listaProcessadaFinal_mc1_Teste.csv"))

assuntosMacroClasse1_Teste.reset_index(inplace=True)
assuntosMacroClasse1_Teste.cd_assunto_nivel_2

assuntosMacroClasse1_Teste = assuntosMacroClasse1_Teste['cd_assunto_nivel_2'].astype('category').values

del(assuntosMacroClasse1_Treinamento_outros,assuntosMacroClasse1_Teste_outros)
gc.collect()
################################################################################################################################
# INDUÇÃO DE MODELOS 
################################################################################################################################

# =============================================================================
# TF-IDF
# =============================================================================

avaliacaoFinal_MC1_TFIDF = pd.DataFrame(columns=['Model','Features','Macro Precisão', 'Macro Revocação', 'Macro F1-Measure','Micro Precisão', 'Micro Revocação', 'Micro F1-Measure'])
avaliacaoFinal_MC1_TFIDF.columns=['Model','Features','Macro Precision', 'Macro Recall', 'Macro F1-Measure','Micro Precision', 'Micro Recall', 'Micro F1-Measure']
#//TODO: fazer validacao cruzada
#------------------------------------------------------------------------------
# Multinomial Naive Bayes MC1
#------------------------------------------------------------------------------

naive_bayes(corpus_treinamento_mc1_tfidf_sparse,assuntosMacroClasse1_Treinamento.cd_assunto_nivel_2,corpus_teste_mc1_tfidf_sparse, assuntosMacroClasse1_Teste.cd_assunto_nivel_2, 1,classes_mc1,'TFIDF')


#------------------------------------------------------------------------------
# SVM MC1
#------------------------------------------------------------------------------
svm(corpus_treinamento_mc1_tfidf_sparse,assuntosMacroClasse1_Treinamento.cd_assunto_nivel_2,corpus_teste_mc1_tfidf_sparse, assuntosMacroClasse1_Teste.cd_assunto_nivel_2, 1,classes_mc1,'TFIDF')



#------------------------------------------------------------------------------
# RANDOM FOREST MC1
#------------------------------------------------------------------------------
 random_forest(corpus_treinamento_mc1_tfidf_sparse,assuntosMacroClasse1_Treinamento.cd_assunto_nivel_2,corpus_teste_mc1_tfidf_sparse, assuntosMacroClasse1_Teste.cd_assunto_nivel_2, 1,classes_mc1,'TFIDF')




#-----------------------------------------------------------------------------
# Rede neural MC1
#-----------------------------------------------------------------------------

mlp(corpus_treinamento_mc1_tfidf_sparse,assuntosMacroClasse1_Treinamento.cd_assunto_nivel_2,corpus_teste_mc1_tfidf_sparse, assuntosMacroClasse1_Teste.cd_assunto_nivel_2, 1,classes_mc1,'TFIDF')

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

plt.savefig('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/imagens/confusion_matrix_SVM_mc1_lsi.png') 

macro_precision,macro_recall,macro_fscore,macro_support=score(assuntosMacroClasse1_Teste['cd_assunto_nivel_2'].astype('category').values,predicted_SVM_mc1_lsi,average='macro')
micro_precision,micro_recall,micro_fscore,micro_support=score(assuntosMacroClasse1_Teste['cd_assunto_nivel_2'].astype('category').values,predicted_SVM_mc1_lsi,average='weighted')

avaliacaoFinal_MC1_TFIDF.loc[4]= ['SVM','LSI',macro_precision,macro_recall,macro_fscore,micro_precision,micro_recall,micro_fscore]

print(classification_report(assuntosMacroClasse1_Teste['cd_assunto_nivel_2'].astype('category').values,predicted_SVM_mc1_lsi))
print('Micro average precision = {:.2f} (dâ o mesmo peso para cada instância)'.format(precision_score(assuntosMacroClasse1_Teste['cd_assunto_nivel_2'].astype('category').values,predicted_SVM_mc1_lsi,average = 'weighted')))
print('Macro average precision = {:.2f} (dâ o mesmo peso para cada classe)'.format(precision_score(assuntosMacroClasse1_Teste['cd_assunto_nivel_2'].astype('category').values,predicted_SVM_mc1_lsi,average = 'macro')))



############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
#
# MACRO CLASSE 2: Menos que 15 mil dados e mais que 1000
#
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################

codigosMacroClasse2 = " OR ".join([str(codigo) for codigo in codigosMacroClasse2.cd_assunto_nivel_2])
queryMacroClasse2 = 'cd_assunto_nivel_2:('  + codigosMacroClasse2  + ')' 
#******************************************************************************************************************************
# Analisa o conjunto de Treinamento e Teste para a Macro Classe 2
#******************************************************************************************************************************
#-----------------------------------------------------------------------------
# Treinamento
#----------------------------------------------------------------------------

queryTreinamento_mc2 = 'tx_conteudo_documento:[* TO *] AND dt_juntada:[* TO 2027-07-01T00:00:00Z ] AND ' + queryMacroClasse2

#busca os dados da mc2 com seus respectivos códigos
solrDataAnalise = solr.query('classificacaoDeDocumentos_hierarquiaCompleta',{
'q':queryTreinamento_mc2,'fl':'id_processo_documento,cd_assunto_nivel_2', 'rows':'300000'
})
dfTreinamento_mc2 = pd.DataFrame(solrDataAnalise.docs)    


#busca os dados que não são da mc2 e traz todo mundo com um código -1
queryTreinamento_mc2_outros = 'tx_conteudo_documento:[* TO *] AND dt_juntada:[* TO 2017-07-01T00:00:00Z ] AND NOT ' + queryMacroClasse2 + ' AND NOT ' + queryMacroClasse1
solrDataAnalise = solr.query('classificacaoDeDocumentos_hierarquiaCompleta',{
'q':queryTreinamento_mc2_outros,'fl':'id_processo_documento,cd_assunto_nivel_2', 'rows':'300000'
})
dfTreinamento_mc2_outros = pd.DataFrame(solrDataAnalise.docs)    
dfTreinamento_mc2_outros['cd_assunto_nivel_2'] = -1 

dfTreinamento_mc2 = dfTreinamento_mc2.append(dfTreinamento_mc2_outros)

fig = plt.figure(figsize=(18,13))
dfTreinamento_mc2.groupby('cd_assunto_nivel_2').id_processo_documento.count().plot.bar(ylim=0)
plt.title('Distribuição dos dados de treinamento')
plt.ylabel('Quantidade de Documentos')
plt.xlabel('Código do assunto')
plt.show()
fig.savefig('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/imagens/TRT15_2GRAU_DistribuicaoClasses_2Nivel_Treinamento_mc2.png')  

#-----------------------------------------------------------------------------
# Teste
#-----------------------------------------------------------------------------

queryTeste_mc2 = 'tx_conteudo_documento:[* TO *] AND dt_juntada:[2017-07-01T00:00:00Z TO 2018-06-30T23:59:59Z] AND ' + queryMacroClasse2

#busca os dados da mc2 com seus respectivos códigos
solrDataAnalise = solr.query('classificacaoDeDocumentos_hierarquiaCompleta',{
'q':queryTeste_mc2,'fl':'id_processo_documento,cd_assunto_nivel_1,cd_assunto_nivel_2,cd_assunto_nivel_3,cd_assunto_nivel_4,cd_assunto_nivel_5', 'rows':'300000'
})
dfTeste_mc2 = pd.DataFrame(solrDataAnalise.docs)    

#busca os dados que não são da mc2 e traz todo mundo com um código -1
queryTeste_mc2_outros = 'tx_conteudo_documento:[* TO *] AND dt_juntada:[* TO 2017-07-01T00:00:00Z ] AND NOT ' + queryMacroClasse2  + ' AND NOT ' + queryMacroClasse1
solrDataAnalise = solr.query('classificacaoDeDocumentos_hierarquiaCompleta',{
'q':queryTeste_mc2_outros,'fl':'id_processo_documento,cd_assunto_nivel_2', 'rows':'300000'
})
dfTeste_mc2_outros = pd.DataFrame(solrDataAnalise.docs)    
dfTeste_mc2_outros['cd_assunto_nivel_2'] = -1 

dfTeste_mc2 = dfTeste_mc2.append(dfTeste_mc2_outros)


fig = plt.figure(figsize=(18,13))
dfTeste_mc2.groupby('cd_assunto_nivel_2').id_processo_documento.count().plot.bar(ylim=0)
plt.title('Distribuição dos dados de teste')
plt.ylabel('Quantidade de Documentos')
plt.xlabel('Código do assunto')
plt.show()
fig.savefig('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/imagens/TRT15_2GRAU_DistribuicaoClasses_2Nivel_Teste_mc2.png')  


del(dfTreinamento_mc2,dfTreinamento_mc2_outros,dfTeste_mc2,dfTeste_mc2_outros,fig)

################################################################################################################################
# BUSCA DADOS E CRIA O DICIONÁRIO
################################################################################################################################
dicionarioFinal_mc2 = corpora.Dictionary('')
#------------------------------------------------------------------------------
# Busca dados de Treinamento
#------------------------------------------------------------------------------
queryMacroClasse2_Treinamento = queryMacroClasse2 + ' AND ' + queryTreinamento

#busca todos os documentos que de fato são da classe 2
start_time = time.time()
listaProcessada = []
for resCursor in solr.cursor_query('classificacaoDeDocumentos_hierarquiaCompleta',{'q':queryMacroClasse2_Treinamento,'rows':'10000','fl':'tx_conteudo_documento','sort':'id asc'}):  
    listaProcessada = Parallel(n_jobs = 7)(delayed(processa_texto)(documento.get('tx_conteudo_documento')) for documento in resCursor.docs)
    dicionarioParcial = corpora.Dictionary(listaProcessada)
    dicionarioFinal_mc2.merge_with(dicionarioParcial)
    print(time.time() - start_time)
    with open("/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/listaProcessadaFinal_mc2_Treinamento.csv", "a") as fp:
        wr = csv.writer(fp, dialect='excel')
        for row in listaProcessada:
            wr.writerow(row)       
    with open("/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/listaProcessadaFinal_mc2_Treinamento backup.csv", "a") as fp:
        wr = csv.writer(fp, dialect='excel')
        for row in listaProcessada:
            wr.writerow(row)       
print(time.time() - start_time)

#busca todos os documentos que de fato NÃO SÃO da classe 1, para serem tratados como 'outros'
queryMacroClasse2_Treinamento_outros = 'NOT ' + queryMacroClasse2_Treinamento + ' AND NOT ' + queryMacroClasse1
start_time = time.time()
listaProcessada = []
for resCursor in solr.cursor_query('classificacaoDeDocumentos_hierarquiaCompleta',{'q':queryMacroClasse2_Treinamento_outros,'rows':'10000','fl':'tx_conteudo_documento','sort':'id asc'}):  
    listaProcessada = Parallel(n_jobs = 7)(delayed(processa_texto)(documento.get('tx_conteudo_documento')) for documento in resCursor.docs)
    dicionarioParcial = corpora.Dictionary(listaProcessada)
    dicionarioFinal_mc2.merge_with(dicionarioParcial)
    print(time.time() - start_time)
    with open("/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/listaProcessadaFinal_mc2_Treinamento.csv", "a") as fp:
        wr = csv.writer(fp, dialect='excel')
        for row in listaProcessada:
            wr.writerow(row)       
print(time.time() - start_time)

#------------------------------------------------------------------------------
# Busca dados de Teste
#------------------------------------------------------------------------------
#busca todos os documentos que de fato são da classe 1
queryMacroClasse2_Teste = queryMacroClasse2 + ' AND ' + queryTeste
start_time = time.time()
listaProcessada = []
for resCursor in solr.cursor_query('classificacaoDeDocumentos_hierarquiaCompleta',{'q':queryMacroClasse2_Teste,'rows':'10000','fl':'tx_conteudo_documento','sort':'id asc'}):  
    listaProcessada = Parallel(n_jobs = 7)(delayed(processa_texto)(documento.get('tx_conteudo_documento')) for documento in resCursor.docs)
    dicionarioParcial = corpora.Dictionary(listaProcessada)
    dicionarioFinal_mc2.merge_with(dicionarioParcial)
    with open("/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/listaProcessadaFinal_mc2_Teste.csv", "a") as fp:
        wr = csv.writer(fp, dialect='excel')
        for row in listaProcessada:
            wr.writerow(row)     
print(time.time() - start_time)


#busca todos os documentos que de fato NÃO SÃO da classe 1, para serem tratados como 'outros'
queryMacroClasse2_Teste_outros = 'NOT ' + queryMacroClasse2_Teste + ' AND NOT ' + queryMacroClasse1
start_time = time.time()
listaProcessada = []
for resCursor in solr.cursor_query('classificacaoDeDocumentos_hierarquiaCompleta',{'q':queryMacroClasse2_Teste_outros,'rows':'10000','fl':'tx_conteudo_documento','sort':'id asc'}):  
    listaProcessada = Parallel(n_jobs = 7)(delayed(processa_texto)(documento.get('tx_conteudo_documento')) for documento in resCursor.docs)
    dicionarioParcial = corpora.Dictionary(listaProcessada)
    dicionarioFinal_mc2.merge_with(dicionarioParcial)
    with open("/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/listaProcessadaFinal_mc2_Teste.csv", "a") as fp:
        wr = csv.writer(fp, dialect='excel')
        for row in listaProcessada:
            wr.writerow(row)
print(time.time() - start_time)
#------------------------------------------------------------------------------
# Salva dicionario
#------------------------------------------------------------------------------

dicionarioFinal_mc2.save('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/dicionarioFinal_mc2.dict')    
#aqui fiz o load do outro dicionario. acho que é que faria mais sentido.
dicionarioFinal_mc2=corpora.Dictionary.load('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/dicionarioFinal_mc1.dict', mmap='r')
#dicionarioFinal_mc2=corpora.Dictionary.load('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/dicionarioFinal_mc2.dict', mmap='r')
print(dicionarioFinal_mc2)
tamanho_dicionario = 181846
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
class MyCorpus_Treinamento_MacroClasse2(object):
    def __iter__(self):
        for line in open('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/listaProcessadaFinal_mc2_Treinamento.csv'):
            yield dicionarioFinal_mc2.doc2bow(line.split(','))
corpora.MmCorpus.serialize('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc2_BOW.mm', MyCorpus_Treinamento_MacroClasse2())
print(time.time() - start_time)


corpus_treinamento_mc2_bow_sparse = matutils.corpus2csc(corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc2_BOW.mm'), tamanho_dicionario).transpose()
corpus_treinamento_mc2_bow_sparse.shape

#------------------------------------------------------------------------------
# Cria o corpus TF-IDF
#------------------------------------------------------------------------------
start_time = time.time()
modeloTfidfTreinamento_mc2 = TfidfModel(corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc2_BOW.mm') , id2word=dicionarioFinal_mc2, normalize=True)
modeloTfidfTreinamento_mc2.save('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc2_TFIDF.tfidf_model')
MmCorpus.serialize('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc2_TFIDF.mm', modeloTfidfTreinamento_mc2[corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc2_BOW.mm')], progress_cnt=10000)
del(modeloTfidfTreinamento_mc2)
print(time.time() - start_time)


corpus_treinamento_mc2_tfidf_sparse = matutils.corpus2csc(corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc2_TFIDF.mm'), tamanho_dicionario).transpose()
corpus_treinamento_mc2_tfidf_sparse.shape

#------------------------------------------------------------------------------
# Cria o corpus LSI
#------------------------------------------------------------------------------
num_topics_mc2=300
start_time = time.time()
modeloLSITreinamento_mc2 = LsiModel(corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc2_TFIDF.mm'), id2word=dicionarioFinal_mc2, num_topics=num_topics_mc2)
modeloLSITreinamento_mc2.save('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc2_LSI.lsi_model')
MmCorpus.serialize('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc2_LSI.mm', modeloLSITreinamento_mc2[corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc2_TFIDF.mm')], progress_cnt=10000)
del(modeloLSITreinamento_mc2)
print(time.time() - start_time)

corpus_treinamento_mc2_lsi_sparse = matutils.corpus2csc(corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc2_LSI.mm'), num_topics_mc2).transpose()
corpus_treinamento_mc2_lsi_sparse.shape

#------------------------------------------------------------------------------
# Busca o target do conjunto de treinamento: assunto de nível 2
#------------------------------------------------------------------------------
 
assuntosMacroClasse2_Treinamento =       solr.query('classificacaoDeDocumentos_hierarquiaCompleta',{'q':queryMacroClasse2_Treinamento,'rows':'1000000','fl':'cd_assunto_nivel_2','sort':'id asc'})
assuntosMacroClasse2_Treinamento = pd.DataFrame(assuntosMacroClasse2_Treinamento.docs)    
assuntosMacroClasse2_Treinamento.shape

assuntosMacroClasse2_Treinamento_outros = solr.query('classificacaoDeDocumentos_hierarquiaCompleta',{'q':queryMacroClasse2_Treinamento_outros,'rows':'1000000','fl':'cd_assunto_nivel_2','sort':'id asc'})
assuntosMacroClasse2_Treinamento_outros = pd.DataFrame(assuntosMacroClasse2_Treinamento_outros.docs)    
assuntosMacroClasse2_Treinamento_outros['cd_assunto_nivel_2'] = 0 

assuntosMacroClasse2_Treinamento = assuntosMacroClasse2_Treinamento.append(assuntosMacroClasse2_Treinamento_outros)

#verifica se o numero de documentos bate com o numero de assuntos
#assuntosMacroClasse2_Treinamento.shape
#row_count = sum(1 for line in open("/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/listaProcessadaFinal_mc2_Treinamento.csv"))

assuntosMacroClasse2_Treinamento.reset_index(inplace=True)

assuntosMacroClasse2_Treinamento = assuntosMacroClasse2_Treinamento['cd_assunto_nivel_2'].astype('category').values



gc.collect()
#******************************************************************************************************************************
# TESTE
#******************************************************************************************************************************

#------------------------------------------------------------------------------
# Cria o corpus de Bag of Words
#------------------------------------------------------------------------------
start_time = time.time()
class MyCorpus_Teste(object):
    def __iter__(self):
        for line in open('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/listaProcessadaFinal_mc2_Teste.csv'):
            yield dicionarioFinal_mc2.doc2bow(line.split(','))
corpora.MmCorpus.serialize('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc2_BOW.mm', MyCorpus_Teste())
print(time.time() - start_time)       

corpus_teste_mc2_bow_sparse = matutils.corpus2csc(corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc2_BOW.mm'), tamanho_dicionario).transpose()
corpus_teste_mc2_bow_sparse.shape
  
#------------------------------------------------------------------------------
# Cria o corpus TF-IDF
#------------------------------------------------------------------------------
start_time = time.time()
modeloTfidfTeste_mc2 = TfidfModel(corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc2_BOW.mm'), id2word=dicionarioFinal_mc2, normalize=True)
modeloTfidfTeste_mc2.save('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc2_TFIDF.tfidf_model')
MmCorpus.serialize('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc2_TFIDF.mm', modeloTfidfTeste_mc2[corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc2_BOW.mm')], progress_cnt=10000)
print(time.time() - start_time)

corpus_teste_mc2_tfidf_sparse = matutils.corpus2csc(corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc2_TFIDF.mm'), tamanho_dicionario).transpose()
corpus_teste_mc2_tfidf_sparse.shape
     


#------------------------------------------------------------------------------
# Cria o corpus LSI
#------------------------------------------------------------------------------
num_topics_mc2=300
start_time = time.time()
modeloLSITeste_mc2 = LsiModel(corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc2_TFIDF.mm'), id2word=dicionarioFinal_mc2, num_topics=num_topics_mc2)
modeloLSITeste_mc2.save('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc2_LSI.lsi_model')
MmCorpus.serialize('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc2_LSI.mm', modeloLSITeste_mc2[corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc2_TFIDF.mm')], progress_cnt=10000)
del(modeloLSITeste_mc2)
print(time.time() - start_time)

corpus_teste_mc2_lsi_sparse = matutils.corpus2csc(corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc2_TFIDF.mm'), num_topics_mc2).transpose()
corpus_teste_mc2_lsi_sparse.shape

#------------------------------------------------------------------------------
# Busca o target do conjunto de teste: assunto de nível 2
#------------------------------------------------------------------------------
assuntosMacroClasse2_Teste = solr.query('classificacaoDeDocumentos_hierarquiaCompleta',{'q':queryMacroClasse2_Teste,'rows':'1000000','fl':'cd_assunto_nivel_2','sort':'id asc'})
assuntosMacroClasse2_Teste = pd.DataFrame(assuntosMacroClasse2_Teste.docs)    

assuntosMacroClasse2_Teste_outros = solr.query('classificacaoDeDocumentos_hierarquiaCompleta',{'q':queryMacroClasse2_Teste_outros,'rows':'1000000','fl':'cd_assunto_nivel_2','sort':'id asc'})
assuntosMacroClasse2_Teste_outros = pd.DataFrame(assuntosMacroClasse2_Teste_outros.docs)    
assuntosMacroClasse2_Teste_outros['cd_assunto_nivel_2'] = 0

assuntosMacroClasse2_Teste = assuntosMacroClasse2_Teste.append(assuntosMacroClasse2_Teste_outros)


#verifica se o numero de documentos bate com o numero de assuntos
#assuntosMacroClasse2_Teste.shape
#row_count = sum(1 for line in open("/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/listaProcessadaFinal_mc2_Teste.csv"))

assuntosMacroClasse2_Teste.reset_index(inplace=True)
assuntosMacroClasse2_Teste = assuntosMacroClasse2_Teste['cd_assunto_nivel_2'].astype('category').values

del(assuntosMacroClasse2_Treinamento_outros,assuntosMacroClasse2_Teste_outros)
gc.collect()
################################################################################################################################
# INDUÇÃO DE MODELOS 
################################################################################################################################

# =============================================================================
# TF-IDF
# =============================================================================

avaliacaoFinal_MC2_TFIDF = pd.DataFrame(columns=['Model','Features','Macro Precision', 'Macro Recall', 'Macro F1-Measure','Micro Precision', 'Micro Recall', 'Micro F1-Measure'])


#//TODO: fazer validacao cruzada
#------------------------------------------------------------------------------
# Multinomial Naive Bayes MC2
#------------------------------------------------------------------------------
   
naive_bayes(corpus_treinamento_mc2_tfidf_sparse,assuntosMacroClasse2_Treinamento.cd_assunto_nivel_2,corpus_teste_mc2_tfidf_sparse, assuntosMacroClasse2_Teste.cd_assunto_nivel_2, 2,classes_mc2,'TFIDF')

 
#------------------------------------------------------------------------------
# SVM MC2
#------------------------------------------------------------------------------

svm(corpus_treinamento_mc2_tfidf_sparse,assuntosMacroClasse2_Treinamento.cd_assunto_nivel_2,corpus_teste_mc2_tfidf_sparse, assuntosMacroClasse2_Teste.cd_assunto_nivel_2, 2,classes_mc2,'TFIDF')


#------------------------------------------------------------------------------
# RANDOM FOREST MC2
#------------------------------------------------------------------------------
#AQUI HAVIA UM ERRO... ESTA USANDO A MC1.

random_forest(corpus_treinamento_mc2_tfidf_sparse,assuntosMacroClasse2_Treinamento.cd_assunto_nivel_2,corpus_teste_mc2_tfidf_sparse, assuntosMacroClasse2_Teste.cd_assunto_nivel_2, 2,classes_mc2,'TFIDF')


#-----------------------------------------------------------------------------
# Rede neural MC2
#-----------------------------------------------------------------------------
    
mlp(corpus_treinamento_mc2_tfidf_sparse,assuntosMacroClasse2_Treinamento.cd_assunto_nivel_2,corpus_teste_mc2_tfidf_sparse, assuntosMacroClasse2_Teste.cd_assunto_nivel_2, 2,classes_mc2,'TFIDF')

# =============================================================================
# LSI
# =============================================================================

#------------------------------------------------------------------------------
# SVM MC2
#------------------------------------------------------------------------------
from sklearn import grid_search
from numpy.random import random, random_integers
param_grid = {
    'loss': [ 'modified_huber', 'squared_hinge'],
    'penalty': ['elasticnet','l2'],
    'alpha': [1e-4,1e-3]
    #'l1_ratio': [0, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1],
}
clf_SVM = SGDClassifier(random_state=0, class_weight='balanced',n_jobs=7)
clf_SVM_grid = grid_search.GridSearchCV(estimator=clf_SVM, param_grid=param_grid,
                                     scoring='f1_weighted',cv = 3)
clf_SVM_grid.fit(corpus_treinamento_mc2_lsi_sparse, assuntosMacroClasse2_Treinamento.cd_assunto_nivel_2)


print(clf_SVM_grid.best_score_)
print(clf_SVM_grid.best_params_)
teste_svm_c2_LSI= pd.DataFrame(clf_SVM_grid.grid_scores_)

clf_SVM = clf_SVM_grid.best_estimator_
clf_SVM.fit(corpus_treinamento_mc2_lsi_sparse, assuntosMacroClasse2_Treinamento.cd_assunto_nivel_2)
predicted_SVM_mc2_lsi =  clf_SVM.predict(corpus_teste_mc2_lsi_sparse)
np.mean(predicted_SVM_mc2_lsi == assuntosMacroClasse2_Teste.cd_assunto_nivel_2)


codigos= pd.DataFrame(assuntosMacroClasse2_Treinamento['cd_assunto_nivel_2'].astype('category').values.describe())
codigos.reset_index(inplace=True)
codigos = codigos.categories.tolist()

confusion_matrix_SVM_mc2_lsi = confusion_matrix(assuntosMacroClasse2_Teste['cd_assunto_nivel_2'].astype('category').values,predicted_SVM_mc2_lsi)
fig = plt.figure(figsize=(10,10))

plot_confusion_matrix(confusion_matrix_SVM_mc2_lsi, codigos,
                      title='SVM \nMacro Class 2 - LSI\nAccuracy: {0:.3f}'.format(accuracy_score(assuntosMacroClasse2_Teste['cd_assunto_nivel_2'].astype('category').values,predicted_SVM_mc2_lsi)))

plt.savefig('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/imagens/confusion_matrix_SVM_mc2_lsi.png') 

macro_precision,macro_recall,macro_fscore,macro_support=score(assuntosMacroClasse2_Teste['cd_assunto_nivel_2'].astype('category').values,predicted_SVM_mc2_lsi,average='macro')
micro_precision,micro_recall,micro_fscore,micro_support=score(assuntosMacroClasse2_Teste['cd_assunto_nivel_2'].astype('category').values,predicted_SVM_mc2_lsi,average='weighted')

avaliacaoFinal_MC2_TFIDF.loc[4]= ['SVM','LSI',macro_precision,macro_recall,macro_fscore,micro_precision,micro_recall,micro_fscore]

print(classification_report(assuntosMacroClasse2_Teste['cd_assunto_nivel_2'].astype('category').values,predicted_SVM_mc2_lsi))
print('Micro average precision = {:.2f} (dâ o mesmo peso para cada instância)'.format(precision_score(assuntosMacroClasse2_Teste['cd_assunto_nivel_2'].astype('category').values,predicted_SVM_mc2_lsi,average = 'weighted')))
print('Macro average precision = {:.2f} (dâ o mesmo peso para cada classe)'.format(precision_score(assuntosMacroClasse2_Teste['cd_assunto_nivel_2'].astype('category').values,predicted_SVM_mc2_lsi,average = 'macro')))







############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
#
# MACRO CLASSE 3: Mais que 15 mil dados
#
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################

codigosMacroClasse3 = quantidadesPorAssunto[(quantidadesPorAssunto['quantidadeDocumentos'] < 1000)]
#retirando os que tem menos que tem menos que 40
codigosMacroClasse3 = quantidadesPorAssunto[(quantidadesPorAssunto['quantidadeDocumentos'] < 1000) & (quantidadesPorAssunto['quantidadeDocumentos'] > 50)]
codigosMacroClasse3_removidos = quantidadesPorAssunto[(quantidadesPorAssunto['quantidadeDocumentos'] < 51)]
#13 assuntos, 173  removidos

codigosMacroClasse3 = " OR ".join([str(codigo) for codigo in codigosMacroClasse3.cd_assunto_nivel_2])
queryMacroClasse3 = 'cd_assunto_nivel_2:('  + codigosMacroClasse3  + ')' 
#******************************************************************************************************************************
# Analisa o conjunto de Treinamento e Teste para a Macro Classe 3
#******************************************************************************************************************************
#-----------------------------------------------------------------------------
# Treinamento
#----------------------------------------------------------------------------



queryTreinamento_mc3 = 'tx_conteudo_documento:[* TO *] AND dt_juntada:[* TO 2027-07-01T00:00:00Z ] AND ' + queryMacroClasse3

#busca os dados da mc3 com seus respectivos códigos
solrDataAnalise = solr.query('classificacaoDeDocumentos_hierarquiaCompleta',{
'q':queryTreinamento_mc3,'fl':'id_processo_documento,cd_assunto_nivel_2', 'rows':'300000'
})
dfTreinamento_mc3 = pd.DataFrame(solrDataAnalise.docs)    


#busca os dados que não são da mc3 e traz todo mundo com um código -1
queryTreinamento_mc3_outros = 'tx_conteudo_documento:[* TO *] AND dt_juntada:[* TO 2017-07-01T00:00:00Z ] AND NOT ' + queryMacroClasse3 + ' AND NOT ' + queryMacroClasse1 + ' AND NOT ' + queryMacroClasse2
solrDataAnalise = solr.query('classificacaoDeDocumentos_hierarquiaCompleta',{
'q':queryTreinamento_mc3_outros,'fl':'id_processo_documento,cd_assunto_nivel_2', 'rows':'300000'
})
dfTreinamento_mc3_outros = pd.DataFrame(solrDataAnalise.docs)    
dfTreinamento_mc3_outros['cd_assunto_nivel_2'] = -1 

dfTreinamento_mc3 = dfTreinamento_mc3.append(dfTreinamento_mc3_outros)

fig = plt.figure(figsize=(18,13))
dfTreinamento_mc3.groupby('cd_assunto_nivel_2').id_processo_documento.count().plot.bar(ylim=0)
plt.title('Distribuição dos dados de treinamento')
plt.ylabel('Quantidade de Documentos')
plt.xlabel('Código do assunto')
plt.show()
fig.savefig('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/imagens/TRT15_2GRAU_DistribuicaoClasses_2Nivel_Treinamento_mc3.png')  

#-----------------------------------------------------------------------------
# Teste
#-----------------------------------------------------------------------------

queryTeste_mc3 = 'tx_conteudo_documento:[* TO *] AND dt_juntada:[2017-07-01T00:00:00Z TO 2018-06-30T23:59:59Z] AND ' + queryMacroClasse3

#busca os dados da mc3 com seus respectivos códigos
solrDataAnalise = solr.query('classificacaoDeDocumentos_hierarquiaCompleta',{
'q':queryTeste_mc3,'fl':'id_processo_documento,cd_assunto_nivel_1,cd_assunto_nivel_2,cd_assunto_nivel_3,cd_assunto_nivel_4,cd_assunto_nivel_5', 'rows':'300000'
})
dfTeste_mc3 = pd.DataFrame(solrDataAnalise.docs)    

#busca os dados que não são da mc3 e traz todo mundo com um código -1
queryTeste_mc3_outros = 'tx_conteudo_documento:[* TO *] AND dt_juntada:[* TO 2017-07-01T00:00:00Z ] AND NOT ' + queryMacroClasse3  + ' AND NOT ' + queryMacroClasse1 + ' AND NOT ' + queryMacroClasse2
solrDataAnalise = solr.query('classificacaoDeDocumentos_hierarquiaCompleta',{
'q':queryTeste_mc3_outros,'fl':'id_processo_documento,cd_assunto_nivel_2', 'rows':'300000'
})
dfTeste_mc3_outros = pd.DataFrame(solrDataAnalise.docs)    
dfTeste_mc3_outros['cd_assunto_nivel_2'] = -1 

dfTeste_mc3 = dfTeste_mc3.append(dfTeste_mc3_outros)


fig = plt.figure(figsize=(18,13))
dfTeste_mc3.groupby('cd_assunto_nivel_2').id_processo_documento.count().plot.bar(ylim=0)
plt.title('Distribuição dos dados de teste')
plt.ylabel('Quantidade de Documentos')
plt.xlabel('Código do assunto')
plt.show()
fig.savefig('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/imagens/TRT15_2GRAU_DistribuicaoClasses_2Nivel_Teste_mc3.png')  


del(dfTreinamento_mc3,dfTreinamento_mc3_outros,dfTeste_mc3,dfTeste_mc3_outros,fig)

################################################################################################################################
# BUSCA DADOS E CRIA O DICIONÁRIO
################################################################################################################################
dicionarioFinal_mc3 = corpora.Dictionary('')
#------------------------------------------------------------------------------
# Busca dados de Treinamento
#------------------------------------------------------------------------------
queryMacroClasse3_Treinamento = queryMacroClasse3 + ' AND ' + queryTreinamento

#busca todos os documentos que de fato são da classe 2
start_time = time.time()
listaProcessada = []
for resCursor in solr.cursor_query('classificacaoDeDocumentos_hierarquiaCompleta',{'q':queryMacroClasse3_Treinamento,'rows':'10000','fl':'tx_conteudo_documento','sort':'id asc'}):  
    listaProcessada = Parallel(n_jobs = 7)(delayed(processa_texto)(documento.get('tx_conteudo_documento')) for documento in resCursor.docs)
    #dicionarioParcial = corpora.Dictionary(listaProcessada)
    #dicionarioFinal_mc3.merge_with(dicionarioParcial)
    print(time.time() - start_time)
    with open("/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/listaProcessadaFinal_mc3_Treinamento.csv", "a") as fp:
        wr = csv.writer(fp, dialect='excel')
        for row in listaProcessada:
            wr.writerow(row)           
print(time.time() - start_time)

#busca todos os documentos que de fato NÃO SÃO da classe 1, para serem tratados como 'outros'
queryMacroClasse3_Treinamento_outros = 'NOT ' + queryMacroClasse3_Treinamento + ' AND NOT ' + queryMacroClasse1  + ' AND NOT ' + queryMacroClasse2
start_time = time.time()
listaProcessada = []
for resCursor in solr.cursor_query('classificacaoDeDocumentos_hierarquiaCompleta',{'q':queryMacroClasse3_Treinamento_outros,'rows':'10000','fl':'tx_conteudo_documento','sort':'id asc'}):  
    listaProcessada = Parallel(n_jobs = 7)(delayed(processa_texto)(documento.get('tx_conteudo_documento')) for documento in resCursor.docs)
    #dicionarioParcial = corpora.Dictionary(listaProcessada)
    #dicionarioFinal_mc3.merge_with(dicionarioParcial)
    print(time.time() - start_time)
    with open("/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/listaProcessadaFinal_mc3_Treinamento.csv", "a") as fp:
        wr = csv.writer(fp, dialect='excel')
        for row in listaProcessada:
            wr.writerow(row)       
print(time.time() - start_time)

#------------------------------------------------------------------------------
# Busca dados de Teste
#------------------------------------------------------------------------------
#busca todos os documentos que de fato são da classe 1
queryMacroClasse3_Teste = queryMacroClasse3 + ' AND ' + queryTeste
start_time = time.time()
listaProcessada = []
for resCursor in solr.cursor_query('classificacaoDeDocumentos_hierarquiaCompleta',{'q':queryMacroClasse3_Teste,'rows':'10000','fl':'tx_conteudo_documento','sort':'id asc'}):  
    listaProcessada = Parallel(n_jobs = 7)(delayed(processa_texto)(documento.get('tx_conteudo_documento')) for documento in resCursor.docs)
    #dicionarioParcial = corpora.Dictionary(listaProcessada)
    #dicionarioFinal_mc3.merge_with(dicionarioParcial)
    with open("/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/listaProcessadaFinal_mc3_Teste.csv", "a") as fp:
        wr = csv.writer(fp, dialect='excel')
        for row in listaProcessada:
            wr.writerow(row)     
print(time.time() - start_time)


#busca todos os documentos que de fato NÃO SÃO da classe 1, para serem tratados como 'outros'
queryMacroClasse3_Teste_outros = 'NOT ' + queryMacroClasse3_Teste + ' AND NOT ' + queryMacroClasse1   + ' AND NOT ' + queryMacroClasse2
start_time = time.time()
listaProcessada = []
for resCursor in solr.cursor_query('classificacaoDeDocumentos_hierarquiaCompleta',{'q':queryMacroClasse3_Teste_outros,'rows':'10000','fl':'tx_conteudo_documento','sort':'id asc'}):  
    listaProcessada = Parallel(n_jobs = 7)(delayed(processa_texto)(documento.get('tx_conteudo_documento')) for documento in resCursor.docs)
    #dicionarioParcial = corpora.Dictionary(listaProcessada)
    #dicionarioFinal_mc3.merge_with(dicionarioParcial)
    with open("/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/listaProcessadaFinal_mc3_Teste.csv", "a") as fp:
        wr = csv.writer(fp, dialect='excel')
        for row in listaProcessada:
            wr.writerow(row)
print(time.time() - start_time)
#------------------------------------------------------------------------------
# Salva dicionario
#------------------------------------------------------------------------------

dicionarioFinal_mc3.save('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/dicionarioFinal_mc3.dict')    
#aqui fiz o load do outro dicionario. acho que é que faria mais sentido.
dicionarioFinal_mc3=corpora.Dictionary.load('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/dicionarioFinal_mc1.dict', mmap='r')
#dicionarioFinal_mc3=corpora.Dictionary.load('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/dicionarioFinal_mc3.dict', mmap='r')
print(dicionarioFinal_mc3)
tamanho_dicionario = 181846
del(listaProcessada,row,stopwords)
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
class MyCorpus_Treinamento_MacroClasse3(object):
    def __iter__(self):
        for line in open('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/listaProcessadaFinal_mc3_Treinamento.csv'):
            yield dicionarioFinal_mc3.doc2bow(line.split(','))
corpora.MmCorpus.serialize('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc3_BOW.mm', MyCorpus_Treinamento_MacroClasse3())
print(time.time() - start_time)


corpus_treinamento_mc3_bow_sparse = matutils.corpus2csc(corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc3_BOW.mm'), tamanho_dicionario).transpose()
corpus_treinamento_mc3_bow_sparse.shape

#------------------------------------------------------------------------------
# Cria o corpus TF-IDF
#------------------------------------------------------------------------------
start_time = time.time()
modeloTfidfTreinamento_mc3 = TfidfModel(corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc3_BOW.mm') , id2word=dicionarioFinal_mc3, normalize=True)
modeloTfidfTreinamento_mc3.save('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc3_TFIDF.tfidf_model')
MmCorpus.serialize('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc3_TFIDF.mm', modeloTfidfTreinamento_mc3[corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc3_BOW.mm')], progress_cnt=10000)
del(modeloTfidfTreinamento_mc3)
print(time.time() - start_time)


corpus_treinamento_mc3_tfidf_sparse = matutils.corpus2csc(corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc3_TFIDF.mm'), tamanho_dicionario).transpose()
corpus_treinamento_mc3_tfidf_sparse.shape

#------------------------------------------------------------------------------
# Cria o corpus LSI
#------------------------------------------------------------------------------
num_topics_mc3=300
start_time = time.time()
modeloLSITreinamento_mc3 = LsiModel(corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc3_TFIDF.mm'), id2word=dicionarioFinal_mc3, num_topics=num_topics_mc3)
modeloLSITreinamento_mc3.save('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc3_LSI.lsi_model')
MmCorpus.serialize('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc3_LSI.mm', modeloLSITreinamento_mc3[corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc3_TFIDF.mm')], progress_cnt=10000)
del(modeloLSITreinamento_mc3)
print(time.time() - start_time)

corpus_treinamento_mc3_lsi_sparse = matutils.corpus2csc(corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTreinamento_mc3_LSI.mm'), num_topics_mc3).transpose()
corpus_treinamento_mc3_lsi_sparse.shape

#------------------------------------------------------------------------------
# Busca o target do conjunto de treinamento: assunto de nível 2
#------------------------------------------------------------------------------
 
assuntosMacroClasse3_Treinamento =       solr.query('classificacaoDeDocumentos_hierarquiaCompleta',{'q':queryMacroClasse3_Treinamento,'rows':'1000000','fl':'cd_assunto_nivel_2','sort':'id asc'})
assuntosMacroClasse3_Treinamento = pd.DataFrame(assuntosMacroClasse3_Treinamento.docs)    
assuntosMacroClasse3_Treinamento.shape

assuntosMacroClasse3_Treinamento_outros = solr.query('classificacaoDeDocumentos_hierarquiaCompleta',{'q':queryMacroClasse3_Treinamento_outros,'rows':'1000000','fl':'cd_assunto_nivel_2','sort':'id asc'})
assuntosMacroClasse3_Treinamento_outros = pd.DataFrame(assuntosMacroClasse3_Treinamento_outros.docs)    
assuntosMacroClasse3_Treinamento_outros['cd_assunto_nivel_2'] = 0 

assuntosMacroClasse3_Treinamento = assuntosMacroClasse3_Treinamento.append(assuntosMacroClasse3_Treinamento_outros)

assuntosMacroClasse3_Treinamento.reset_index(inplace=True)
#verifica se o numero de documentos bate com o numero de assuntos
#assuntosMacroClasse3_Treinamento.shape
#row_count = sum(1 for line in open("/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/listaProcessadaFinal_mc3_Treinamento.csv"))


assuntosMacroClasse3_Treinamento = assuntosMacroClasse3_Treinamento['cd_assunto_nivel_2'].astype('category').values



gc.collect()
#******************************************************************************************************************************
# TESTE
#******************************************************************************************************************************

#------------------------------------------------------------------------------
# Cria o corpus de Bag of Words
#------------------------------------------------------------------------------
start_time = time.time()
class MyCorpus_Teste(object):
    def __iter__(self):
        for line in open('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/listaProcessadaFinal_mc3_Teste.csv'):
            yield dicionarioFinal_mc3.doc2bow(line.split(','))
corpora.MmCorpus.serialize('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc3_BOW.mm', MyCorpus_Teste())
print(time.time() - start_time)       

corpus_teste_mc3_bow_sparse = matutils.corpus2csc(corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc3_BOW.mm'), tamanho_dicionario).transpose()
corpus_teste_mc3_bow_sparse.shape
  
#------------------------------------------------------------------------------
# Cria o corpus TF-IDF
#------------------------------------------------------------------------------
start_time = time.time()
modeloTfidfTeste_mc3 = TfidfModel(corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc3_BOW.mm'), id2word=dicionarioFinal_mc3, normalize=True)
modeloTfidfTeste_mc3.save('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc3_TFIDF.tfidf_model')
MmCorpus.serialize('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc3_TFIDF.mm', modeloTfidfTeste_mc3[corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc3_BOW.mm')], progress_cnt=10000)
print(time.time() - start_time)


corpus_teste_mc3_tfidf_sparse = matutils.corpus2csc(corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc3_TFIDF.mm'), tamanho_dicionario).transpose()
corpus_teste_mc3_tfidf_sparse.shape
     


#------------------------------------------------------------------------------
# Cria o corpus LSI
#------------------------------------------------------------------------------
num_topics_mc3=300
start_time = time.time()
modeloLSITeste_mc3 = LsiModel(corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc3_TFIDF.mm'), id2word=dicionarioFinal_mc3, num_topics=num_topics_mc3)
modeloLSITeste_mc3.save('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc3_LSI.lsi_model')
MmCorpus.serialize('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc3_LSI.mm', modeloLSITeste_mc3[corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc3_TFIDF.mm')], progress_cnt=10000)
del(modeloLSITeste_mc3)
print(time.time() - start_time)

corpus_teste_mc3_lsi_sparse = matutils.corpus2csc(corpora.MmCorpus('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/corpusTeste_mc3_TFIDF.mm'), num_topics_mc3).transpose()
corpus_teste_mc3_lsi_sparse.shape

#------------------------------------------------------------------------------
# Busca o target do conjunto de teste: assunto de nível 2
#------------------------------------------------------------------------------
assuntosMacroClasse3_Teste = solr.query('classificacaoDeDocumentos_hierarquiaCompleta',{'q':queryMacroClasse3_Teste,'rows':'1000000','fl':'cd_assunto_nivel_2','sort':'id asc'})
assuntosMacroClasse3_Teste = pd.DataFrame(assuntosMacroClasse3_Teste.docs)    

assuntosMacroClasse3_Teste_outros = solr.query('classificacaoDeDocumentos_hierarquiaCompleta',{'q':queryMacroClasse3_Teste_outros,'rows':'1000000','fl':'cd_assunto_nivel_2','sort':'id asc'})
assuntosMacroClasse3_Teste_outros = pd.DataFrame(assuntosMacroClasse3_Teste_outros.docs)    
assuntosMacroClasse3_Teste_outros['cd_assunto_nivel_2'] = 0

assuntosMacroClasse3_Teste = assuntosMacroClasse3_Teste.append(assuntosMacroClasse3_Teste_outros)


assuntosMacroClasse3_Teste.reset_index(inplace=True)
#verifica se o numero de documentos bate com o numero de assuntos
#assuntosMacroClasse3_Teste.shape
#row_count = sum(1 for line in open("/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/Data/corpus/listaProcessadaFinal_mc3_Teste.csv"))

assuntosMacroClasse3_Teste = assuntosMacroClasse3_Teste['cd_assunto_nivel_2'].astype('category').values

del(assuntosMacroClasse3_Treinamento_outros,assuntosMacroClasse3_Teste_outros)
gc.collect()
################################################################################################################################
# INDUÇÃO DE MODELOS 
################################################################################################################################

# =============================================================================
# TF-IDF
# =============================================================================

avaliacaoFinal_MC3_TFIDF = pd.DataFrame(columns=['Model','Features','Macro Precision', 'Macro Recall', 'Macro F1-Measure','Micro Precision', 'Micro Recall', 'Micro F1-Measure'])

#//TODO: fazer validacao cruzada
#------------------------------------------------------------------------------
# Multinomial Naive Bayes
#------------------------------------------------------------------------------
naive_bayes(corpus_treinamento_mc3_tfidf_sparse,assuntosMacroClasse3_Treinamento.cd_assunto_nivel_2,corpus_teste_mc3_tfidf_sparse, assuntosMacroClasse3_Teste.cd_assunto_nivel_2, 3,classes_mc3,'TFIDF')
#------------------------------------------------------------------------------
# SVM
#------------------------------------------------------------------------------
svm(corpus_treinamento_mc3_tfidf_sparse,assuntosMacroClasse3_Treinamento.cd_assunto_nivel_2,corpus_teste_mc3_tfidf_sparse, assuntosMacroClasse3_Teste.cd_assunto_nivel_2, 3,classes_mc3,'TFIDF')
#------------------------------------------------------------------------------
# RANDOM FOREST
#------------------------------------------------------------------------------
random_forest(corpus_treinamento_mc3_tfidf_sparse,assuntosMacroClasse3_Treinamento.cd_assunto_nivel_2,corpus_teste_mc3_tfidf_sparse, assuntosMacroClasse3_Teste.cd_assunto_nivel_2, 3,classes_mc3,'TFIDF')
#-----------------------------------------------------------------------------
# Rede neural
#-----------------------------------------------------------------------------

mlp(corpus_treinamento_mc3_tfidf_sparse,assuntosMacroClasse3_Treinamento.cd_assunto_nivel_2,corpus_teste_mc3_tfidf_sparse, assuntosMacroClasse3_Teste.cd_assunto_nivel_2, 3,classes_mc3,'TFIDF')


# =============================================================================
# LSI
# =============================================================================

#------------------------------------------------------------------------------
# SVM
#------------------------------------------------------------------------------
from numpy.random import random, random_integers
param_grid = {
    'loss': [ 'modified_huber', 'squared_hinge'],
    'penalty': ['elasticnet','l2'],
    'alpha': [1e-4,1e-3]
    #'l1_ratio': [0, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1],
}
clf_SVM = SGDClassifier(random_state=0, class_weight='balanced',n_jobs=7)
clf_SVM_grid = grid_search.GridSearchCV(estimator=clf_SVM, param_grid=param_grid,
                                     scoring='f1_weighted',cv = 3)
clf_SVM_grid.fit(corpus_treinamento_mc3_lsi_sparse, assuntosMacroClasse3_Treinamento.cd_assunto_nivel_2)


print(clf_SVM_grid.best_score_)
print(clf_SVM_grid.best_params_)
teste_svm_c3_segunda_execucao = pd.DataFrame(clf_SVM_grid.grid_scores_)

clf_SVM = clf_SVM_grid.best_estimator_
clf_SVM.fit(corpus_treinamento_mc3_lsi_sparse, assuntosMacroClasse3_Treinamento.cd_assunto_nivel_2)
predicted_SVM_mc3_lsi =  clf_SVM.predict(corpus_teste_mc3_lsi_sparse)
np.mean(predicted_SVM_mc3_lsi == assuntosMacroClasse3_Teste.cd_assunto_nivel_2)


codigos= pd.DataFrame(assuntosMacroClasse3_Treinamento['cd_assunto_nivel_2'].astype('category').values.describe())
codigos.reset_index(inplace=True)
codigos = codigos.categories.tolist()

confusion_matrix_SVM_mc3_lsi = confusion_matrix(assuntosMacroClasse3_Teste['cd_assunto_nivel_2'].astype('category').values,predicted_SVM_mc3_lsi)
fig = plt.figure(figsize=(10,10))

plot_confusion_matrix(confusion_matrix_SVM_mc3_lsi, codigos,
                      title='SVM \nMacro Class 3 - LSI\nAccuracy: {0:.3f}'.format(accuracy_score(assuntosMacroClasse3_Teste['cd_assunto_nivel_2'].astype('category').values,predicted_SVM_mc3_lsi)))

plt.savefig('/media/anarocha/DATA/Documentos/Mestrado/Matérias/MDM/Trabalho/imagens/confusion_matrix_SVM_mc3_lsi.png') 

macro_precision,macro_recall,macro_fscore,macro_support=score(assuntosMacroClasse3_Teste['cd_assunto_nivel_2'].astype('category').values,predicted_SVM_mc3_lsi,average='macro')
micro_precision,micro_recall,micro_fscore,micro_support=score(assuntosMacroClasse3_Teste['cd_assunto_nivel_2'].astype('category').values,predicted_SVM_mc3_lsi,average='weighted')

avaliacaoFinal_MC3_TFIDF.loc[4]= ['SVM','LSI',macro_precision,macro_recall,macro_fscore,micro_precision,micro_recall,micro_fscore]

print(classification_report(assuntosMacroClasse3_Teste['cd_assunto_nivel_2'].astype('category').values,predicted_SVM_mc3_lsi))
print('Micro average precision = {:.2f} (dâ o mesmo peso para cada instância)'.format(precision_score(assuntosMacroClasse3_Teste['cd_assunto_nivel_2'].astype('category').values,predicted_SVM_mc3_lsi,average = 'weighted')))
print('Macro average precision = {:.2f} (dâ o mesmo peso para cada classe)'.format(precision_score(assuntosMacroClasse3_Teste['cd_assunto_nivel_2'].astype('category').values,predicted_SVM_mc3_lsi,average = 'macro')))

# =============================================================================
# Configuração geral
# =============================================================================

avaliacaoFinal = pd.DataFrame(columns=['Macro Class','Model','Features','Macro Precision', 'Macro Recall', 'Macro F1-Measure','Micro Precision', 'Micro Recall', 'Micro F1-Measure'])


classes_mc3 = pd.DataFrame(assuntosMacroClasse3_Treinamento['cd_assunto_nivel_2'].astype('category').values.describe())
classes_mc3 .reset_index(inplace=True)
classes_mc3  = classes_mc3.categories.tolist()

classes_mc2 = pd.DataFrame(assuntosMacroClasse2_Treinamento['cd_assunto_nivel_2'].astype('category').values.describe())
classes_mc2.reset_index(inplace=True)
classes_mc2  = classes_mc2.categories.tolist()

classes_mc1 = pd.DataFrame(assuntosMacroClasse1_Treinamento['cd_assunto_nivel_2'].astype('category').values.describe())
classes_mc1 .reset_index(inplace=True)
classes_mc1  = classes_mc1.categories.tolist()

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