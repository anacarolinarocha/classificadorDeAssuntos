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

import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from gensim import  matutils
from gensim import corpora
from gensim.models import LsiModel, TfidfModel
from gensim.corpora import MmCorpus

from joblib import Parallel, delayed

from modelo import Modelo

stopwords = nltk.corpus.stopwords.words('portuguese')
stopwords.extend(['microsoftinternetexplorer','false','none','trabalho','juiz',
                  'reclamado','reclamada','autos','autor','excelentissimo',
                  'senhor','normal'])
stopwords = [normalize('NFKD', palavra).encode('ASCII','ignore').decode('ASCII') for palavra in stopwords]


nltk.download('rslp')
stemmer = nltk.stem.RSLPStemmer()


avaliacaoFinal = pd.DataFrame(columns=['Model','Features','Macro Precisão', 'Macro Revocação', 'Macro F1-Measure','Micro Precisão', 'Micro Revocação', 'Micro F1-Measure'])
avaliacaoFinal.columns=['Model','Features','Macro Precision', 'Macro Recall', 'Macro F1-Measure','Micro Precision', 'Micro Recall', 'Micro F1-Measure']


modelos=[]


def analisaTodosOsNiveis (dfGeral, path):
    
    fig = plt.figure()
    ax1 = fig.add_subplot(5,1,1)
    ax1.title.set_text('Nível 1')
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
        'alpha':[1]
    }
    
    clf_NB = MultinomialNB()
    clf_NB_grid = GridSearchCV(estimator=clf_NB, param_grid=param_grid,
                                         scoring='f1_weighted',n_jobs=4,cv=5, verbose=4)
    
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
    modeloNB  = Modelo('Multinomial Naive Bayes', clf_NB_grid.best_params_, clf_NB_grid.best_estimator_, clf_NB_grid.cv_results_, macro_precision,macro_recall,macro_fscore,micro_precision,micro_recall,micro_fscore)
    
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
def svm(training_corpus,training_classes,test_corpus, test_classes, classNumber,classes,featureType):
    param_grid = {
        'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [0.1,10,100],
        'gamma':[0.1,5,10]
    }
    clf_SVM = SVC(random_state=0, class_weight='balanced')
    clf_SVM_grid = GridSearchCV(estimator=clf_SVM, param_grid=param_grid,
                                         scoring='f1_weighted',cv=5, verbose=4)
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
    modeloSVM  = Modelo('SVM', clf_SVM_grid.best_params_, clf_SVM_grid.best_estimator_, clf_SVM_grid.cv_results_, macro_precision,macro_recall,macro_fscore,micro_precision,micro_recall,micro_fscore)
    
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
def random_forest(training_corpus,training_classes,test_corpus, test_classes, classNumber,classes,featureType):
    
    param_grid = {
       'max_features':[0.3,0.7],     
       'n_estimators':[200,500],
       'min_samples_leaf':[10,50],
       'max_depth': [50,100],
       'class_weight':['balanced','balanced_subsample']        
    }
    clf_RF = RandomForestClassifier(random_state=1986,bootstrap=False)
    clf_RF_grid = GridSearchCV(estimator=clf_RF, param_grid=param_grid,
                                         scoring='f1_weighted',n_jobs=4,verbose=4,cv=5)
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
    modeloRF  = Modelo('Random Forest', clf_RF_grid.best_params_, clf_RF_grid.best_estimator_, clf_RF_grid.cv_results_, macro_precision,macro_recall,macro_fscore,micro_precision,micro_recall,micro_fscore)
    
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

def mlp(training_corpus,training_classes,test_corpus, test_classes, classNumber,classes,featureType):
    classesCM = []
    classesCM = classes
    
    param_grid = {
        'learning_rate_init':[0.01,0.001,0.0001],
        'hidden_layer_sizes':[(5,5), (5)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'momentum':[0.4,0.8],
        'solver':['lbfgs', 'sgd', 'adam'],
        'learning_rate':['constant', 'invscaling', 'adaptive'],
     }
    
    clf_MLP = MLPClassifier( batch_size='auto',
           beta_1=0.9, beta_2=0.999, early_stopping=False,
            momentum=0.9,   random_state=1, 
           solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
           warm_start=False)
    
    clf_MLP_grid = GridSearchCV(estimator = clf_MLP, param_grid = param_grid, 
                             verbose = 4,cv=5,n_jobs=5)
    
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
    modeloMLP  = Modelo('Multilayer Perceptron', clf_MLP_grid.best_params_, clf_MLP_grid.best_estimator_, clf_MLP_grid.cv_results_, macro_precision,macro_recall,macro_fscore,micro_precision,micro_recall,micro_fscore)
    
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