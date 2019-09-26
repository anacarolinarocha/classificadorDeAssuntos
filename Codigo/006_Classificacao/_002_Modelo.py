#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 16:08:24 2018

@author: anarocha
"""
import pandas as pd
import os
import pickle

class Modelo:
    def __init__(self, nome, feature_type = None, tempo_processamento = None, best_params_ = None, best_estimator_ = None, grid_scores_ = None, accuracy = None,balanced_accuracy = None,
                 macro_precision = None, macro_recall = None, macro_fscore = None, micro_precision = None, micro_recall = None, micro_fscore = None, confusion_matrix = None,
                 num_estimators = None, max_samples = None, tamanho_conjunto_treinamento = None, id_execucao = None, data = None, log_loss = None,
                 classification_report = None, grid_cv_results = None):
        self.nome = nome
        self.feature_type = feature_type
        self.tempo_processamento = tempo_processamento
        self.best_params_ = best_params_
        self.best_estimator_ = best_estimator_
        self.grid_scores_ = grid_scores_
        self.accuracy = accuracy
        self.balanced_accuracy = balanced_accuracy
        self.macro_precision = macro_precision
        self.macro_recall = macro_recall
        self.macro_fscore = macro_fscore
        self.micro_precision = micro_precision
        self.micro_recall = micro_recall
        self.micro_fscore = micro_fscore
        self.confusion_matrix = confusion_matrix
        self.num_estimators = num_estimators
        self.max_samples = max_samples
        self.tamanho_conjunto_treinamento = tamanho_conjunto_treinamento
        self.id_execucao = id_execucao
        self.data = data
        self.log_loss = log_loss
        self.classification_report = classification_report
        self.grid_cv_results = grid_cv_results

    def setNome(self, nome):
        self.nome = nome

    def setFeatureType(self, feature_type):
        self.feature_type = feature_type

    def setTempoProcessamento(self, tempo_processamento):
        self.tempo_processamento = tempo_processamento

    def setBestParams(self, best_params_):
        self.best_params_ = best_params_

    def setBestEstimator(self, best_estimator_):
        self.best_estimator_ = best_estimator_

    def setGridScores(self, grid_scores_):
        self.grid_scores_ = grid_scores_

    def setAccuracy (self, accuracy):
        self.accuracy = accuracy

    def setBalancedAccuracy(self, balanced_accuracy):
        self.balanced_accuracy = balanced_accuracy

    def setMacroPrecision(self, macro_precision):
        self.macro_precision = macro_precision

    def setMacroRecall(self, macro_recall):
        self.macro_recall = macro_recall

    def setMacroFscore(self, macro_fscore):
        self.macro_fscore = macro_fscore

    def setMicroPrecision(self, micro_precision):
        self.micro_precision = micro_precision

    def setMicroRecall(self, micro_recall):
        self.micro_recall = micro_recall

    def setMicroFscore(self, micro_fscore):
        self.micro_fscore = micro_fscore

    def setConfusionMatrix(self, confusion_matrix):
        self.confusion_matrix = confusion_matrix

    def setNumEstimators(self, num_estimators):
        self.num_estimators = num_estimators

    def setMaxSamples(self, max_samples):
        self.max_samples = max_samples

    def setTamanhoConjuntoTreinamento(self, tamanho_conjunto_treinamento):
        self.tamanho_conjunto_treinamento = tamanho_conjunto_treinamento

    def setIdExecucao(self, id_execucao):
        self.id_execucao = id_execucao

    def setData(self, data):
        self.data = data

    def setClassificationReport(self, classification_report):
        self.classification_report = classification_report

    def setGridCVResults(self,grid_cv_results):
        self.grid_cv_results = grid_cv_results

    def getNome(self):
        return self.nome

    def getLogLoss(self):
        return self.log_loss

    def getFeatureType(self):
        return self.feature_type

    def getTempoProcessamento(self):
        return self.tempo_processamento

    def getBestParams(self):
        return self.best_params_

    def getBestEstimator(self):
        return self.best_estimator_

    def getGridScores(self, grid_scores_):
        return self.grid_scores_

    def getAccuracy (self):
        return self.accuracy

    def getBalancedAccuracy (self):
        return self.balanced_accuracy

    def getMacroPrecision(self):
        return self.macro_precision

    def getMacroRecall(self):
        return self.macro_recall

    def getMacroFscore(self):
        return self.macro_fscore

    def getMicroPrecision(self):
        return self.micro_precision

    def getMicroRecall(self):
        return self.micro_recall

    def getMicroFscore(self):
        return self.micro_fscore

    def getConfusionMatrix(self):
        return self.confusion_matrix

    def getNumEstimators(self):
        return self.num_estimators

    def getMaxSamples(self):
        return self.max_samples

    def getTamanhoConjuntoTreinamento(self):
        return self.tamanho_conjunto_treinamento

    def getIdExecucao (self):
        return self.id_execucao

    def getData (self):
        return self.data

    def getLogLoss(self):
        return self.log_loss

    def getClassificationReport(self):
        return self.classification_report

    def getGridCVResults(self):
        return self.grid_cv_results

    def imprime(self):
        print(" ")
        print("Nome modelo: " + self.nome)
        print("Quantidade de elementos de treinamento: " + str(self.tamanho_conjunto_treinamento))
        print("Tempo de treinamento: " + str(self.tempo_processamento))
        print("Feature Type: " + self.feature_type)
        print("Accuracy: " + str(self.accuracy))
        print("Balanced Accuracy: " + str(self.balanced_accuracy))
        print('macro_precision %s \nmacro_recall    %s \nmacro_fscore    %s' % (self.macro_precision, self.macro_recall, self.macro_fscore))
        print('micro_precision %s \nmicro_recall    %s \nmicro_fscore    %s' % (self.micro_precision, self.micro_recall, self.micro_fscore))
        print("Log loss: " + str(self.log_loss))

    def salvaClassificationReport(self, arquivo):
        df = pd.DataFrame.from_dict(self.classification_report)
        df['nome_algoritmo']=self.nome
        df['id_execucao']=self.getIdExecucao()
        if not (os.path.isfile(arquivo)):
            with open(arquivo, 'a') as f:
                df.to_csv(f, header=True)
                f.close()
        else:
            with open(arquivo, 'a') as f:
                df.to_csv(f, header=False)
                f.close()

    def salvaModelo(self, path):
        nomePicke = path + 'Modelo_' + self.nome.replace(' ', '') + '.p'
        arquivoPickle = open(nomePicke, 'wb')
        pickle.dump(self.getBestEstimator(), arquivoPickle)
        arquivoPickle.close()
