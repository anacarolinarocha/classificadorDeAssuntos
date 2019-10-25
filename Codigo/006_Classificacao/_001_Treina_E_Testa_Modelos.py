from sklearn.ensemble import BaggingClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from datetime import timedelta
import numpy as np
import time
import sys
sys.path.insert(1, '/home/anarocha/myGit/classificadorDeAssuntos/Codigo/006_Classificacao')
from _002_Modelo import *
import os
# os.environ["KMP_AFFINITY"] = 'FALSE' #"Use "0",  ".F.", "off",// ja tentei "FALSE". "no"
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
    clf = OneVsRestClassifier(BalancedBaggingClassifier(classificador, max_samples=max_samples, n_estimators=n_estimators, n_jobs=1, bootstrap =False, verbose=1),n_jobs=7)
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

def treina_modelo_grid_search(x_tfidf_train,y_train, classificador, nomeModelo,param_grid , n_iterations_grid_search, n_jobs):
    print(">> Fazendo Grid Search para classificador " + nomeModelo)
    # max_samples=round(x_tfidf_train.shape[0] * 0.6)
    stratify_5_folds = StratifiedKFold(n_splits=5,random_state=42)
    start_time = time.time()
    classificadorBag = BalancedBaggingClassifier(classificador,n_jobs=1, bootstrap=False,random_state=42)
    classificadorOVR = OneVsRestClassifier(classificadorBag, n_jobs=1)
    grid_search = RandomizedSearchCV(estimator=classificadorOVR, param_distributions=param_grid, cv=stratify_5_folds, n_jobs=n_jobs, verbose=2, refit=True, n_iter = n_iterations_grid_search, scoring = 'precision_weighted')
    grid_search.fit(x_tfidf_train, y_train)
    grid_results = ""
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        grid_results += "%0.3f (+/-%0.03f) for %r \n" % (mean, std * 2, params)

    total_time = time.time() - start_time
    print("Tempo para execução do GridSearch para OVR Balanced Bagging " + nomeModelo + " para " + str(x_tfidf_train.shape[0]) + " elementos: ", str(timedelta(seconds=total_time)))
    modelo = Modelo(nomeModelo)
    # modelo.setMaxSamples(max_samples)
    modelo.setTamanhoConjuntoTreinamento(x_tfidf_train.shape[0])
    modelo.setTempoProcessamento(str(timedelta(seconds=grid_search.refit_time_)))
    modelo.setFeatureType('TF-IDF')
    modelo.setBestEstimator(grid_search.best_estimator_)
    modelo.setBestParams(grid_search.best_params_)
    modelo.setGridCVResults(grid_results)
    return modelo

# ---------------------------------------------------------------------------------------------------------------------
# Função que testa um modelo de classificação
#----------------------------------------------------------------------------------------------------------------------
from sklearn.metrics import precision_recall_fscore_support as score

def testa_modelo( x_tfidf_test,y_test, modelo):
    print(">> Testando classificador " + modelo.getNome())
    start_time = time.time()
    y_pred = modelo.getBestEstimator().predict(x_tfidf_test)
    y_pred_proba = modelo.getBestEstimator().predict_proba(x_tfidf_test)
    y_pred_proba_df = pd.DataFrame(y_pred_proba, columns =  modelo.getBestEstimator().classes_)
    total_time = time.time() - start_time
    print("Tempo para fazer a predicao de  " + str(x_tfidf_test.shape[0]) + " elementos: ", str(timedelta(seconds=total_time)))

    start_time = time.time()
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    macro_precision, macro_recall, macro_fscore =  score(y_test,y_pred,average='macro',labels=np.unique(y_pred))[:3]
    micro_precision, micro_recall, micro_fscore = score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))[:3]
    confusion_matrix = multilabel_confusion_matrix(y_true=y_test, y_pred=y_pred)
    classes = y_test.unique().astype(str).tolist()
    print(classification_report(y_test, y_pred, target_names=classes))
    classification_report_dict = classification_report(y_test, y_pred,target_names=classes,output_dict=True)
    total_time = time.time() - start_time
    # print('Confusion matrix:\n', conf_mat)
    print("Tempo para recuperar métricas:  "+    str(timedelta(seconds=total_time)))

    modelo.setAccuracy(accuracy)
    modelo.setBalancedAccuracy(balanced_accuracy)
    modelo.setMacroPrecision(macro_precision)
    modelo.setMacroRecall(macro_recall)
    modelo.setMacroFscore(macro_fscore)
    modelo.setMicroPrecision(micro_precision)
    modelo.setMicroRecall(micro_recall)
    modelo.setMicroFscore(micro_fscore)
    modelo.setConfusionMatrix(confusion_matrix)
    modelo.setClassificationReport(classification_report_dict)

    return modelo, y_pred,y_pred_proba_df
