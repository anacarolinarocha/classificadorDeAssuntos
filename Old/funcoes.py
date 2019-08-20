#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 00:13:33 2018

@author: anarocha
"""

import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import nltk
from unicodedata import normalize
import re
import pandas as pd
import numpy as np
import itertools
import time
import os
from datetime import timedelta

import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict

from gensim import  matutils
from gensim import corpora
from gensim.models import LsiModel, TfidfModel
from gensim.corpora import MmCorpus

from joblib import Parallel, delayed

from modelo import Modelo

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('portuguese')
stopwords.extend(['microsoftinternetexplorer','false','none','trabalho','juiz',
                  'reclamado','reclamada','autos','autor','excelentissimo',
                  'senhor','normal'])
stopwords = [normalize('NFKD', palavra).encode('ASCII','ignore').decode('ASCII') for palavra in stopwords]


nltk.download('rslp')
stemmer = nltk.stem.RSLPStemmer()


avaliacaoFinal = pd.DataFrame(columns=['Experimento','Model','Features','Macro Precisão', 'Macro Revocação', 'Macro F1-Measure','Micro Precisão', 'Micro Revocação', 'Micro F1-Measure','Tempo'])

modelos=[]

assuntos = pd.read_csv('./hierarquia_de_assuntos.csv')
assuntos = assuntos.replace(np.nan, 0, regex=True)
assuntosNivel1 = pd.Series(assuntos['cd_assunto_nivel_1'])
assuntosNivel2 = pd.Series(assuntos['cd_assunto_nivel_2'])
assuntosNivel3 = pd.Series(assuntos['cd_assunto_nivel_3'])
assuntosNivel4 = pd.Series(assuntos['cd_assunto_nivel_4'])
assuntosNivel5 = pd.Series(assuntos['cd_assunto_nivel_5'])

# =============================================================================
# Função para preencher avaliação final
# =============================================================================
def preencheAvaliacaoFinal(nomeExperimento,nomePasta,nomeArquivoResultadosCompilados,featureType):
    
    i=0
    for modelo in modelos:
        avaliacaoFinal.loc[i]=[nomeExperimento,
                            modelo.getNome(),
                          featureType,
                           modelo.getMacroPrecision(),
                           modelo.getMacroRecall(),
                           modelo.getMacroFscore(),
                           modelo.getMicroPrecision(),
                           modelo.getMicroRecall(),
                           modelo.getMicroFscore(),
                           modelo.getTempoProcessamento()]
        i+=1
        nomeArquivo = nomePasta+'/avaliacaoFinal_'+nomeExperimento +'.csv'
        avaliacaoFinal.to_csv(nomeArquivo, sep=';')
        avaliacaoFinal.to_csv(nomeArquivoResultadosCompilados, sep=';',mode='a', header=False)

# =============================================================================
# Função para recuperar o nível de um assunto
# =============================================================================
def recuperaNivelAssunto(codigo):
    global assuntos,assuntosNivel1,assuntosNivel2,assuntosNivel3,assuntosNivel4,assuntosNivel5
    nivel = -1
    if not assuntosNivel1[assuntosNivel1.isin([int(codigo)])].empty:
        nivel=1
    if not assuntosNivel2[assuntosNivel2.isin([int(codigo)])].empty:
        nivel=2
    if not assuntosNivel3[assuntosNivel3.isin([int(codigo)])].empty:
        nivel=3
    if not assuntosNivel4[assuntosNivel4.isin([int(codigo)])].empty:
        nivel=4
    if not assuntosNivel5[assuntosNivel5.isin([int(codigo)])].empty:
        nivel=5
    if(nivel==-1):
        print('NIVEL NAO ENCONTRADO: ' + str(codigo))
    return nivel


# =============================================================================
# Função para recupearar o codigo de nivel X  de um assunto
# =============================================================================
def recuperaAssuntoNivelEspecifico(codigo, nivel):
    global assuntos,assuntosNivel1,assuntosNivel2,assuntosNivel3,assuntosNivel4,assuntosNivel5
    nivelInicial = recuperaNivelAssunto(codigo)
    coluna='cd_assunto_nivel_'+str(nivelInicial)
    index = int(assuntos[assuntos[coluna]==codigo].index[0])
    cd_assunto = assuntos['cd_assunto_nivel_' + str(nivel)][index]
    return int(cd_assunto)

# =============================================================================
# Função que recupera a hieraria de assuntos d eum datafram que contenha um cd_assunto_trf
# =============================================================================
    

def recuperaHierarquiaAssuntos(df):
    start_time = time.time()
    for i, row in df.iterrows():
        df.set_value(i,'cd_assunto_nivel_1',recuperaAssuntoNivelEspecifico(int(row['cd_assunto_trf']),1))
        df.set_value(i,'cd_assunto_nivel_2',recuperaAssuntoNivelEspecifico(int(row['cd_assunto_trf']),2))
        df.set_value(i,'cd_assunto_nivel_3',recuperaAssuntoNivelEspecifico(int(row['cd_assunto_trf']),3))
        df.set_value(i,'cd_assunto_nivel_4',recuperaAssuntoNivelEspecifico(int(row['cd_assunto_trf']),4))
        df.set_value(i,'cd_assunto_nivel_5',recuperaAssuntoNivelEspecifico(int(row['cd_assunto_trf']),5))
    end_time = time.time() - start_time
    print('Tempo para montar a hierarquia de assuntos:' + str(timedelta(seconds=end_time)))   
         

# =============================================================================
# Função que recupera os filhos de um assunto de nivel 3
# =============================================================================
    
def recuperaFilhosDoNivel3(codigo):
    filhos =  assuntos.query('cd_assunto_nivel_3==' + str(codigo))
    codigosFilhos = []
    for i, row in filhos.iterrows():
        codigosFilhos.append(int(row['cd_assunto_nivel_4']))
        codigosFilhos.append(int(row['cd_assunto_nivel_5']))
    codigosFilhos =  [x for x in codigosFilhos if x != 0]
    codigosFilhos = list(set(codigosFilhos))
    return codigosFilhos


def recuperaFilhosDoNivel1(codigo):
    filhos =  assuntos.query('cd_assunto_nivel_1==' + str(codigo))
    codigosFilhos = []
    for i, row in filhos.iterrows():
        codigosFilhos.append(int(row['cd_assunto_nivel_2']))
        codigosFilhos.append(int(row['cd_assunto_nivel_3']))
        codigosFilhos.append(int(row['cd_assunto_nivel_4']))
        codigosFilhos.append(int(row['cd_assunto_nivel_5']))
    codigosFilhos =  [x for x in codigosFilhos if x != 0]
    codigosFilhos = list(set(codigosFilhos))
    return codigosFilhos



# =============================================================================
# Função que analisa todos plota graficos de barras para os 5 níveis
# =============================================================================

def analisaTodosOsNiveis (dfGeral, path, title):
    
    fig = plt.figure()
    ax1 = fig.add_subplot(5,1,1)
    ax1.title.set_text(title + '\n Nível 1')
    dfGeral.groupby('cd_assunto_nivel_1').id_processo_documento.count().plot.bar(ylim=0)
    plt.gca().axes.get_xaxis().set_visible(False)
    
    ax1 = fig.add_subplot(5,1,2)
    ax1.title.set_text('Nível 2')
    dfGeral.groupby('cd_assunto_nivel_2').id_processo_documento.count().plot.bar(ylim=0)
    plt.gca().axes.get_xaxis().set_visible(False)
    
    ax1 = fig.add_subplot(5,1,3)
    ax1.title.set_text('Nível 3')
    plt.subplot(513)
    dfGeral.groupby('cd_assunto_nivel_3').id_processo_documento.count().plot.bar(ylim=0)
    plt.gca().axes.get_xaxis().set_visible(False)
    
    ax1 = fig.add_subplot(5,1,4)
    ax1.title.set_text('Nível 4')
    plt.subplot(514)
    dfGeral.groupby('cd_assunto_nivel_4').id_processo_documento.count().plot.bar(ylim=0)
    plt.gca().axes.get_xaxis().set_visible(False)
    
    ax1 = fig.add_subplot(5,1,5)
    ax1.title.set_text('Nível 5')
    plt.subplot(515)
    dfGeral.groupby('cd_assunto_nivel_5').id_processo_documento.count().plot.bar(ylim=0)
    plt.gca().axes.get_xaxis().set_visible(False)
    
    fig.set_figheight(15)
    fig.set_figwidth(12)
    fig.tight_layout(pad=3)    
    plt.show()
    fig.savefig(path) 


# =============================================================================
# Funçao de processamento do texto
# =============================================================================
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
    plt.figure(figsize=(35,35))    
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
# Cria matrix de confusão - versao simplificada
# =============================================================================
    
def plot_simple_confusion_matrix(cm, classes, title):
    plt.rcParams.update({'font.size': 15})
    plt.title(title)
    plt.tight_layout(pad=1.4)
    sns.heatmap(cm,cmap="YlOrRd")
    
#        
#    labels=classes
#    fig = plt.figure(figsize=(25,25))
#    
#    ax = fig.add_subplot(111)
#    cax = ax.matshow(confusion_matrix_NB)
#    plt.title('Confusion matrix of the classifier')
#    fig.colorbar(cax)
#    ax.set_xticklabels([''] + labels)
#    ax.set_yticklabels([''] + labels)
#    plt.xlabel('Predicted')
#    plt.ylabel('True')
#    plt.show()
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
# SEM GRID    
# =============================================================================
    
# =============================================================================
# NAIVE BAYES (SEM GRID)
# =============================================================================

def naive_bayes(x, y, classes,featureType,nomePasta):
    start_time = time.time()
    clf_NB = ComplementNB()
    predicted_NB = cross_val_predict(clf_NB, x, y, cv=5, n_jobs=6)
    
    macro_precision,macro_recall,macro_fscore,macro_support=score(y,predicted_NB,average='macro')
    micro_precision,micro_recall,micro_fscore,micro_support=score(y,predicted_NB,average='weighted')
    confusion_matrix_NB = confusion_matrix(y,predicted_NB)

    matrixHeaderString = 'Naive Bayes - ' +featureType + '\nMacro/Micro Precisão: {0:.3f}/{0:.3f}'.format(macro_precision,micro_precision)
    
    plot_simple_confusion_matrix(confusion_matrix_NB, classes, title=matrixHeaderString)
    figureFile = nomePasta + '/imagens/simple_confusion_matrix_NB_'+ str(featureType) +'.png'
    plt.savefig(figureFile) 
    
    plot_confusion_matrix(confusion_matrix_NB, classes, title=matrixHeaderString)
    figureFile =nomePasta + '/imagens/confusion_matrix_NB_'+ str(featureType) +'.png'
    plt.savefig(figureFile) 
    
    end_time = time.time() - start_time
    print('Tempo da execução do Naive Bayes:' + str(timedelta(seconds=end_time)))
    
    global modelos
    modeloNB  = Modelo('Multinomial Naive Bayes', featureType,str(timedelta(seconds=end_time)), None, clf_NB, None, macro_precision,macro_recall,macro_fscore,micro_precision,micro_recall,micro_fscore)
    modelos.append(modeloNB)
    
    
    
    
# =============================================================================
# SVM (SEM GRID)
# =============================================================================
def svm(x, y, classes,featureType, nomePasta):
    start_time = time.time()
    
    clf_SVM = LinearSVC(random_state=0, class_weight='balanced')
    predicted_SVM = cross_val_predict(clf_SVM, x, y, cv=5, n_jobs=6)
    
    macro_precision,macro_recall,macro_fscore,macro_support=score(y,predicted_SVM,average='macro')
    micro_precision,micro_recall,micro_fscore,micro_support=score(y,predicted_SVM,average='weighted')
    confusion_matrix_SVM = confusion_matrix(y,predicted_SVM)
    
    matrixHeaderString = 'SVM  - ' +featureType + '\nMacro/Micro Precisão: {0:.3f}/{0:.3f}'.format(macro_precision,micro_precision)
    
    plot_simple_confusion_matrix(confusion_matrix_SVM, classes, title=matrixHeaderString)
    figureFile = nomePasta + '/imagens/simple_confusion_matrix_SVM_'+ str(featureType) +'.png'
    plt.savefig(figureFile) 
    
    
    plot_confusion_matrix(confusion_matrix_SVM, classes, title=matrixHeaderString)
    figureFile = nomePasta + '/imagens/confusion_matrix_SVM_'+ featureType+'.png'
    plt.savefig(figureFile) 
    
    end_time = time.time() - start_time
    print('Tempo da execução do SVM:' + str(timedelta(seconds=end_time)))
    
    modeloSVM  = Modelo('SVM', featureType,str(timedelta(seconds=end_time)), None, clf_SVM, None, macro_precision,macro_recall,macro_fscore,micro_precision,micro_recall,micro_fscore)
    
    global  modelos
    modelos.append(modeloSVM)
    
# =============================================================================
# SVM (SEM GRID)
# =============================================================================
from sklearn.ensemble import BaggingClassifier
def svm_bagging(x, y, classes,featureType, nomePasta):
    start_time = time.time()
    
    clf_SVM = BaggingClassifier(LinearSVC(random_state=0, class_weight='balanced'),bootstrap=False)
    predicted_SVM = cross_val_predict(clf_SVM, x, y, cv=5, n_jobs=8)
    
    macro_precision,macro_recall,macro_fscore,macro_support=score(y,predicted_SVM,average='macro')
    micro_precision,micro_recall,micro_fscore,micro_support=score(y,predicted_SVM,average='weighted')
    confusion_matrix_SVM = confusion_matrix(y,predicted_SVM)
    
    matrixHeaderString = 'SVM  - ' +featureType + '\nMacro/Micro Precisão: {0:.3f}/{0:.3f}'.format(macro_precision,micro_precision)
    
    plot_simple_confusion_matrix(confusion_matrix_SVM, classes, title=matrixHeaderString)
    figureFile = nomePasta + '/imagens/simple_confusion_matrix_SVM_'+ str(featureType) +'.png'
    plt.savefig(figureFile) 
    
    
    plot_confusion_matrix(confusion_matrix_SVM, classes, title=matrixHeaderString)
    figureFile = nomePasta + '/imagens/confusion_matrix_SVM_'+ featureType+'.png'
    plt.savefig(figureFile) 
    
    end_time = time.time() - start_time
    print('Tempo da execução do SVM_bagging:' + str(timedelta(seconds=end_time)))
    
    modeloSVM  = Modelo('SVM_bagging', featureType,str(timedelta(seconds=end_time)), None, clf_SVM, None, macro_precision,macro_recall,macro_fscore,micro_precision,micro_recall,micro_fscore)
    
    global  modelos
    modelos.append(modeloSVM)
    
    
    
# =============================================================================
# Random Forest (SEM GRID)
# =============================================================================
def random_forest(x, y, classes,featureType, nomePasta):
    start_time = time.time()
    clf_RF = RandomForestClassifier()
    predicted_RF =  cross_val_predict(clf_RF, x, y, cv=5, n_jobs=6)
    
    macro_precision,macro_recall,macro_fscore,macro_support=score(y,predicted_RF,average='macro')
    micro_precision,micro_recall,micro_fscore,micro_support=score(y,predicted_RF,average='weighted')
    confusion_matrix_RF = confusion_matrix(y,predicted_RF)
    
    matrixHeaderString = 'Random Forest - ' +featureType + '\nMacro/Micro Precisão: {0:.3f}/{0:.3f}'.format(macro_precision,micro_precision)
    
    plot_simple_confusion_matrix(confusion_matrix_RF, classes, title=matrixHeaderString)
    figureFile = nomePasta + '/imagens/simple_confusion_matrix_RF_'+ str(featureType) +'.png'
    plt.savefig(figureFile) 
    
    plot_confusion_matrix(confusion_matrix_RF, classes, title=matrixHeaderString)
    figureFile = nomePasta + '/imagens/confusion_matrix_RF_'+ featureType+'.png'
    plt.savefig(figureFile) 
    
    end_time = time.time() - start_time
    print('Tempo da execução do Random Forest:' + str(timedelta(seconds=end_time)))
    
    modeloRF  = Modelo('Random Forest', featureType,str(timedelta(seconds=end_time)), None, clf_RF, None, macro_precision,macro_recall,macro_fscore,micro_precision,micro_recall,micro_fscore)
    
    global modelos
    modelos.append(modeloRF)
# =============================================================================
# Multilayer Perceptron (SEM GRID)
# =============================================================================

def mlp(x, y, classes,featureType, nomePasta):
    start_time = time.time()
    clf_MLP = MLPClassifier()
    predicted_MLP = cross_val_predict(clf_MLP, x, y, cv=5, n_jobs=5)
    
    macro_precision,macro_recall,macro_fscore,macro_support=score(y,predicted_MLP,average='macro')
    micro_precision,micro_recall,micro_fscore,micro_support=score(y,predicted_MLP,average='weighted')
    confusion_matrix_MLP = confusion_matrix(y,predicted_MLP)
    
    matrixHeaderString = 'Multilayer Perceptron - ' +featureType + '\nMacro/Micro Precisão: {0:.3f}/{0:.3f}'.format(macro_precision,micro_precision)
    
    
    plot_simple_confusion_matrix(confusion_matrix_MLP, classes, title=matrixHeaderString)
    figureFile = nomePasta + '/imagens/simple_confusion_matrix_MLP_'+ str(featureType) +'.png'
    plt.savefig(figureFile) 
    
    
    plot_confusion_matrix(confusion_matrix_MLP, classes, title=matrixHeaderString)
    figureFile = nomePasta + '/imagens/confusion_matrix_MLP_'+ featureType+'.png'
    plt.savefig(figureFile) 
    
    end_time = time.time() - start_time
    print('Tempo da execução do MLP:' + str(timedelta(seconds=end_time)))
    modeloMLP  = Modelo('Multilayer Perceptron', featureType,str(timedelta(seconds=end_time)), None, clf_MLP, None, macro_precision,macro_recall,macro_fscore,micro_precision,micro_recall,micro_fscore)
    
    global  modelos
    modelos.append(modeloMLP)
# =============================================================================
# COM GRID
# =============================================================================

# =============================================================================
# Multinomial Nayve Bayes
# =============================================================================
def naive_bayes_GRID(training_corpus,training_classes,test_corpus, test_classes, classNumber,classes,featureType):
    classesCM = []
    classesCM = classes
    
#    param_grid = {
#        'fit_prior':[True, False],
#        'alpha':[1]
#    }
    
    cv = StratifiedKFold(n_splits=5)
    clf_NB = MultinomialNB()
    
#    clf_NB_grid = GridSearchCV(estimator=clf_NB, param_grid=param_grid,
#                                         scoring='f1_weighted',n_jobs=1,cv=5, verbose=4)
    
#    clf_NB_grid.fit(training_corpus, training_classes)
    
#    clf_NB = clf_NB_grid.best_estimator_
    clf_NB.fit(training_corpus, training_classes)

    predicted_NB = clf_NB.predict(test_corpus)
    np.mean(predicted_NB == test_classes)
    
    confusion_matrix_NB = confusion_matrix(test_classes,predicted_NB)
    
    
    matrixHeaderString = 'Naive Bayes \nMacro Class ' + str(classNumber) +' - ' +featureType + '\nAccuracy: {0:.3f}'.format(accuracy_score(test_classes,predicted_NB))
    plot_confusion_matrix(confusion_matrix_NB, classesCM, title=matrixHeaderString)
    figureFile = '/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/imagens/confusion_matrix_NB_mc'+ str(classNumber) +'_tfidf_testeFuncao.png'
    plt.savefig(figureFile) 
    
    macro_precision,macro_recall,macro_fscore,macro_support=score(test_classes,predicted_NB,average='macro')
    micro_precision,micro_recall,micro_fscore,micro_support=score(test_classes,predicted_NB,average='weighted')
#    modeloNB  = Modelo('Multinomial Naive Bayes', clf_NB_grid.best_params_, clf_NB_grid.best_estimator_, clf_NB_grid.cv_results_, macro_precision,macro_recall,macro_fscore,micro_precision,micro_recall,micro_fscore)
    modeloNB  = Modelo('Multinomial Naive Bayes', featureType, None, clf_NB, None, macro_precision,macro_recall,macro_fscore,micro_precision,micro_recall,micro_fscore)
    
    global avaliacaoFinal, modelos
    modelos.append(modeloNB)
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
def svm_GRID(training_corpus,training_classes,test_corpus, test_classes, classNumber,classes,featureType):
#    param_grid = {
#        'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
#        'C': [0.1,10],
#        'gamma':[0.1,1]
#    }
    clf_SVM = SVC(random_state=0, class_weight='balanced')
    
    clf_SVM.fit(training_corpus, training_classes)
#    clf_SVM_grid = GridSearchCV(estimator=clf_SVM, param_grid=param_grid,
#                                         scoring='precision',cv=5, verbose=4,n_jobs=1)
#    clf_SVM_grid.fit(training_corpus, training_classes)
    
    
#    clf_SVM = clf_SVM_grid.best_estimator_
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
    modeloSVM  = Modelo('SVM', featureType, None, clf_SVM, None, macro_precision,macro_recall,macro_fscore,micro_precision,micro_recall,micro_fscore)
    
    global avaliacaoFinal, modelos
    modelos.append(modeloSVM)
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
def random_forest_GRID(training_corpus,training_classes,test_corpus, test_classes, classNumber,classes,featureType):
    
#    param_grid = {
#       'max_features':[0.3,0.7],     
#       'n_estimators':[200,500],
#       'max_depth': [50,100],
#       'class_weight':['balanced','balanced_subsample']        
#    }
    clf_RF = RandomForestClassifier(random_state=0)
#    clf_RF_grid = GridSearchCV(estimator=clf_RF, param_grid=param_grid,
#                                         scoring='precision',n_jobs=1,verbose=4,cv=5)
#    clf_RF_grid.fit(training_corpus, training_classes)
#
#    clf_RF = clf_RF_grid.best_estimator_
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
    modeloRF  = Modelo('Random Forest', featureType, None, clf_RF, None, macro_precision,macro_recall,macro_fscore,micro_precision,micro_recall,micro_fscore)
    
    global avaliacaoFinal, modelos
    modelos.append(modeloRF)
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

def mlp_GRID(training_corpus,training_classes,test_corpus, test_classes, classNumber,classes,featureType):
    classesCM = []
    classesCM = classes
    
#    param_grid = {
#        'hidden_layer_sizes':[(5,5), (5)],
#        'activation': ['identity', 'logistic', 'tanh', 'relu'],
#        'solver':['lbfgs', 'sgd', 'adam']
#     }
    
    clf_MLP = MLPClassifier()
    
#    clf_MLP_grid = GridSearchCV(estimator = clf_MLP, param_grid = param_grid, 
#                             verbose = 4,cv=5,n_jobs=5)
#    
#    clf_MLP_grid.fit(training_corpus, training_classes)    
#    
#    clf_MLP = clf_MLP_grid.best_estimator_
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
    modeloMLP  = Modelo('Multilayer Perceptron', featureType, None, clf_MLP, None, macro_precision,macro_recall,macro_fscore,micro_precision,micro_recall,micro_fscore)
    
    global avaliacaoFinal, modelos
    modelos.append(modeloMLP)
    avaliacaoFinal = avaliacaoFinal.append({'Macro Class':classNumber,
                                            'Model':'Multilayer Perceptron',
                                            'Features':featureType,
                                            'Macro Precision':macro_precision,
                                            'Macro Recall':macro_recall,
                                            'Macro F1-Measure':macro_fscore,
                                            'Micro Precision':micro_precision,
                                            'Micro Recall':micro_recall,
                                            'Micro F1-Measure':micro_fscore}, ignore_index=True)   