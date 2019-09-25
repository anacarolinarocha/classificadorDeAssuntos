from sklearn.ensemble import BaggingClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import accuracy_score
from datetime import timedelta
import numpy as np
import time
import sys
sys.path.insert(1, '/home/anarocha/myGit/classificadorDeAssuntos/Codigo/006_Classificacao')
from _002_Modelo import *

# ---------------------------------------------------------------------------------------------------------------------
# Função que recebe um tipo de modelo, os dados de entrada e faz o treinamento OneVersusRest, com Balanceamento
#----------------------------------------------------------------------------------------------------------------------
def treina_modelo(x_tfidf_train,y_train, classificador, nomeModelo):
    # Chama o classificador
    print(">> Treinando classificador " + nomeModelo)
    n_estimators = 3
    max_samples=round(x_tfidf_train.shape[0] * 0.6)
    start_time = time.time()
    # clf = OneVsRestClassifier(BaggingClassifier(modelo, max_samples=max_samples, n_estimators=n_estimators, n_jobs=5))
    clf = OneVsRestClassifier(BalancedBaggingClassifier(classificador, max_samples=max_samples, n_estimators=n_estimators, n_jobs=6, bootstrap =False, verbose=1),n_jobs=4)
    clf.fit(x_tfidf_train, y_train)
    total_time = time.time() - start_time
    print("Tempo para a criação do modelo OVR Balanced Bagging " + nomeModelo + " para " + str(x_tfidf_train.shape[0]) + " elementos: ", str(timedelta(seconds=total_time)))
    modelo = Modelo(nomeModelo)
    modelo.setMaxSamples(max_samples)
    modelo.setNumEstimators(n_estimators)
    modelo.setTamanhoConjuntoTreinamento(x_tfidf_train.shape[0])
    modelo.setTempoProcessamento(str(timedelta(seconds=total_time)))
    modelo.setFeatureType('TF-IDF')
    modelo.setBestEstimator(clf)
    # modelo.setBestParams(self, best_params_)
    return modelo

# ---------------------------------------------------------------------------------------------------------------------
# Função que testa um modelo de classificação
#----------------------------------------------------------------------------------------------------------------------
def testa_modelo( x_tfidf_test,y_test, modelo):
    print(">> Testando classificador " + modelo.getNome())
    start_time = time.time()
    y_pred = modelo.getBestEstimator().predict(x_tfidf_test)
    total_time = time.time() - start_time
    print("Tempo para fazer a predicao de  " + str(x_tfidf_test.shape[0]) + " elementos: ", str(timedelta(seconds=total_time)))
    # clf_nb_bagged.score(x_tfidf_test,y_test)
    start_time = time.time()
    accuracy = accuracy_score(y_test, y_pred)
    macro_precision, macro_recall, macro_fscore =  score(y_test,y_pred,average='macro',labels=np.unique(y_pred))[:3]
    micro_precision, micro_recall, micro_fscore = score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))[:3]
    confusion_matrix = multilabel_confusion_matrix(y_true=y_test, y_pred=y_pred)
    total_time = time.time() - start_time
    # print('Confusion matrix:\n', conf_mat)
    print("Tempo para recuperar métricas:  "+    str(timedelta(seconds=total_time)))

    modelo.setAccuracy(accuracy)
    modelo.setMacroPrecision(macro_precision)
    modelo.setMacroRecall(macro_recall)
    modelo.setMacroFscore(macro_fscore)
    modelo.setMicroPrecision(micro_precision)
    modelo.setMicroRecall(micro_recall)
    modelo.setMicroFscore(micro_fscore)
    modelo.setConfusionMatrix(confusion_matrix)

    #Metrica escolhida para análise: Micro f-score (leva em consideracao o peso de cada classe, a precisao e o recall)
    return modelo
