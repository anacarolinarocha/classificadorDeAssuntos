from docutils.nodes import header
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC,LinearSVC
from datetime import timedelta
import time
import sys
from datetime import datetime
import uuid
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
sys.path.insert(1, '/home/anarocha/myGit/classificadorDeAssuntos/Codigo/003_EncontraTamanhoAmostra')
from _001_Recupera_Amostras import *
sys.path.insert(1, '/home/anarocha/myGit/classificadorDeAssuntos/Codigo/006_Classificacao')
from _001_Treina_E_Testa_Modelos import *
sys.path.insert(1, '/home/anarocha/myGit/classificadorDeAssuntos/Codigo/005_FeatureEngineering')
from _002_Extrai_Features import *

# ---------------------------------------------------------------------------------------------------------------------
# Setup
#----------------------------------------------------------------------------------------------------------------------
path='/home/anarocha/myGit/classificadorDeAssuntos/Resultados/EXP10_GridSearch_semOverSampler/'
if not os.path.exists(path):
    os.makedirs(path)

float_formatter = lambda x: "%.4f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

class PrintPythonConsoleOnFileAlso(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()

f = open(path + 'out.txt', 'w')
original = sys.stdout
sys.stdout = PrintPythonConsoleOnFileAlso(sys.stdout, f)

# sys.stdout = original
# print "This won't appear on file"  # Only on stdout
# f.close()
columnsResultados=['id_execucao', 'data', 'nome','feature_type','tempo_processamento','tamanho_conjunto_treinamento','accuracy','balanced_accuracy','micro_precision','micro_recall','micro_fscore','macro_precision','macro_recall','macro_fscore','best_params_','best_estimator_','grid_scores_','grid_cv_results','confusion_matrix','classification_report','num_estimators','max_samples']
df_resultados = pd.DataFrame(columns = columnsResultados)
nome_arquivo_destino = path + "Metricas.csv"
if  not (os.path.isfile(nome_arquivo_destino)):
    with open(nome_arquivo_destino, 'a') as f:
        df_resultados.to_csv(f, header=True)
nome_classification_reports = path + 'ClassificationReports.csv'

# ---------------------------------------------------------------------------------------------------------------------
# Função que divide conjunto de treinamento e teste de stratificado por assunto
#----------------------------------------------------------------------------------------------------------------------
def splitTrainTest(df_amostra_final):
    X_train, X_test, y_train, y_test = train_test_split(df_amostra_final[['sigla_trt','nr_processo','id_processo_documento','texto_stemizado']],
                                                        df_amostra_final['cd_assunto_nivel_3'], test_size=0.2,
                                                        random_state=42,
                                                        stratify=df_amostra_final['cd_assunto_nivel_3'])
    return X_train, X_test, y_train, y_test

# ---------------------------------------------------------------------------------------------------------------------
# Função que transforma matrizes textuais em matrizes processadas em tfidf
#----------------------------------------------------------------------------------------------------------------------
def extraiFeaturesTFIDF(df_amostra_final,X_train,X_test ):
    tfidf_transformer = recupera_tfidf_transformer(df_amostra_final)
    x_tfidf_train = tfidf_transformer.transform(X_train)
    x_tfidf_test = tfidf_transformer.transform(X_test)
    return x_tfidf_train, x_tfidf_test

# ---------------------------------------------------------------------------------------------------------------------
# Imprime evolucao de um algoritmo em grafico
#----------------------------------------------------------------------------------------------------------------------
def plota_evolucao_algoritmo(df_resultados, nomeAlgoritmo):
    plt.clf()
    plt.cla()
    plt.close()
    df_resultados_algoritmo = df_resultados[(df_resultados.nome == nomeAlgoritmo)]
    plt.title(nomeAlgoritmo)
    plt.plot('tamanho_conjunto_treinamento', 'micro_precision', data=df_resultados_algoritmo, marker='o',
             markerfacecolor='blue', markersize=10, color='skyblue', linewidth=4, label="Micro Precision")
    plt.plot('tamanho_conjunto_treinamento', 'micro_recall', data=df_resultados_algoritmo, marker='', color='olive',
             linewidth=2, linestyle='dashed', label="Micro Recall")
    plt.plot('tamanho_conjunto_treinamento', 'micro_fscore', data=df_resultados_algoritmo, marker='', color='gray',
             linewidth=2, linestyle='dashed', label="Micro FScore")
    plt.plot('tamanho_conjunto_treinamento', 'accuracy', data=df_resultados_algoritmo, marker='', color='green', linewidth=2,
             linestyle='dashed', label="Accuracy")
    plt.legend()
    plt.savefig("{0}{1}.png".format(path, nomeAlgoritmo.replace(' ', '')))
    # plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# Salva os valores preditos
#----------------------------------------------------------------------------------------------------------------------
def salvaPredicao(modelo, X_test, y_true, y_pred, y_pred_proba_df):
    global df_resultados
    nome_arquivo_predicao = path + 'predicao_' + modelo.getNome() + '.csv'
    df_pred = X_test[['sigla_trt','nr_processo','id_processo_documento']]
    df_pred['y_true'] = y_true
    df_pred['y_pred'] = y_pred
    df_pred = df_pred.reset_index(drop=True)
    #y_pred_proba_df = y_pred_proba_df.reset_index(drop=True)
    df_pred = df_pred.join(y_pred_proba_df)
    df_pred['modelo'] = modelo.getNome()
    df_pred.to_csv(nome_arquivo_predicao)

# ---------------------------------------------------------------------------------------------------------------------
# Grava métricas de execucao e modelo
#----------------------------------------------------------------------------------------------------------------------
def salvaModelo(modelo):
    global df_resultados
    modelo.salvaClassificationReport(nome_classification_reports)
    modelo.salvaModelo(path)
    df_resultados = df_resultados.append(modelo.__dict__, ignore_index=True)
    with open(nome_arquivo_destino, 'a') as f:
        df_resultados.tail(1).to_csv(f, header=False)

# ---------------------------------------------------------------------------------------------------------------------
# Chamada principal
#----------------------------------------------------------------------------------------------------------------------
# listaAssuntos=[2546,2086,1855]
# listaAssuntos=[2546,2086,1855,2594]
listaAssuntos=[2546,2086,1855,2594,2458,2029,2140,2478,2704,2021,2426,2656,8808,1844,1663,2666,2506,55220,2055,1806,2139,1888,2435,2215,5280,2554,2583,55170,2019,2117,1661,1904,2540,55345]

#Classificadores iniais para ter um baseline antes do GridSearch
classificadorNB = MultinomialNB()
classificadorRF = RandomForestClassifier()
# https://www.svm-tutorial.com/2014/10/svm-linear-kernel-good-text-classification/
classificadorSVM = CalibratedClassifierCV(LinearSVC(class_weight='balanced', max_iter=10000),method='sigmoid', cv=5)
# classificadorSVM = LinearSVC(class_weight='balanced', max_iter=10000)
classificadorMLP = MLPClassifier(early_stopping= True)

nomeAlgoritmoNB='Multinomial Naive Bayes'
nomeAlgoritmoRF='Random Forest'
nomeAlgoritmoSVM='SVM'
nomeAlgoritmoMLP="Multi-Layer Perceptron"

id_execucao = str(uuid.uuid1())[:7]
data = datetime.now().strftime("%d/%m/%Y %H:%M:%S")


for qtdElementosPorAssunto in range(1000000,1000001, 1000000):
# for qtdElementosPorAssunto in range(10, 11, 10):
    qtdElementosPorAssunto = 10
    df_amostra = recupera_amostras_de_todos_regionais(listaAssuntos, qtdElementosPorAssunto)
    X_train, X_test, y_train, y_test = splitTrainTest(df_amostra)
    print("=========================================================================")
    print("Amostra de treinamento de " + str(X_train.shape[0]) + " elementos")
    print("=========================================================================")
    title = "Balanceamento de assuntos na amostra de "  + str(X_train.shape[0])
    mostra_balanceamento_assunto(y_train.value_counts(), title, "Quantidade Elementos", "Código Assunto", path, y_train.shape[0])
    start_time = time.time()
    x_tfidf_train, x_tfidf_test = extraiFeaturesTFIDF(df_amostra,X_train['texto_stemizado'],X_test['texto_stemizado'] )
    total_time = time.time() - start_time
    print("Tempo para montar matrizes TF-IDF (features:  "+ str(x_tfidf_train.shape[1]) + ") :" +   str(timedelta(seconds=total_time)))
# df_amostra_grande = df_amostra

    from keras.models import Sequential
    from keras.layers import Dense, Activation
    from keras.wrappers.scikit_learn import KerasClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import BaggingClassifier
    from imblearn.ensemble import BalancedBaggingClassifier
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.model_selection import StratifiedKFold
    def create_model( hidden_layers=1):
        # create model
        model = Sequential()
        # model.add(Dense(20, input_dim=x_tfidf_train.shape[1], activation='relu'))
        model.add(Dense(10, input_shape=(x_tfidf_train.shape[1],), activation='relu'))

        for i in range(hidden_layers):
            # Add one hidden layer
            model.add(Dense(10, activation='relu'))

        # model.add(Dense(10, input_shape=(x_tfidf_train.shape[1],), activation='relu'))
        model.add(Dense(1, input_shape = (x_tfidf_train.shape[1],), activation='sigmoid'))

        # model.add(Dense(5, input_dim=5, activation='sigmoid'))
        # model.add(Activation('softmax'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])#class_mode='categorical',
        return model
    seed = 7
    np.random.seed(seed)
    model = KerasClassifier(build_fn=create_model, verbose=0)
    batch_size = [30,50,100]
    epochs = [2]
    hidden_layers = [1]
    param_grid = dict(estimator__base_estimator__batch_size=batch_size, estimator__base_estimator__epochs=epochs, estimator__base_estimator__hidden_layers=hidden_layers)
    # grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_jobs=-1)
    # grid_result = grid.fit(x_tfidf_train, y_train)
    stratify_5_folds = StratifiedKFold(n_splits=5)
    classificadorBag = BalancedBaggingClassifier(model,n_jobs=1, bootstrap=False)
    classificadorOVR = OneVsRestClassifier(classificadorBag, n_jobs=1)
    grid_search = RandomizedSearchCV(estimator=classificadorOVR, param_distributions=param_grid, cv=stratify_5_folds, n_jobs=1, verbose=2, refit=True, n_iter = 1, scoring = 'balanced_accuracy')
    grid_search.fit(x_tfidf_train, y_train)


    grid_results = ""
    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        grid_results += "%0.3f (+/-%0.03f) for %r \n" % (mean, std * 2, params)

    grid_search.best_estimator_



    y_pred = grid_search.best_estimator_.predict(x_tfidf_test)
    y_pred_proba = grid_search.best_estimator_.predict_proba(x_tfidf_test)
    y_pred_proba_df = pd.DataFrame(y_pred_proba, columns =  grid_search.best_estimator_.classes_)
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



    print("-------------------------------------------------------------------------")
    print(nomeAlgoritmoNB)
    print("-------------------------------------------------------------------------")
    param_grid_NB = {
        'estimator__n_estimators': [3,5,7,9],
	'estimator__max_samples': [round(x_tfidf_train.shape[0] * 0.7),round(x_tfidf_train.shape[0] * 0.4)],
        'estimator__base_estimator__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1]
    }
    n_iterations_grid_search_NB=10
    #modeloNB = treina_modelo_grid_search(x_tfidf_train, y_train, classificadorNB, nomeAlgoritmoNB,param_grid_NB,n_iterations_grid_search_NB)
    #modeloNB, y_pred, y_pred_proba_df = testa_modelo(x_tfidf_test, y_test, modeloNB)
    #modeloNB.setIdExecucao(id_execucao)
    #modeloNB.setData(data)
    #modeloNB.imprime()
    #salvaModelo(modeloNB)
    #salvaPredicao(modeloNB, X_test, y_test, y_pred,y_pred_proba_df)


    print("-------------------------------------------------------------------------")
    print(nomeAlgoritmoRF)
    print("-------------------------------------------------------------------------")
    param_grid_RF = {
        'estimator__n_estimators': [3],
	    'estimator__max_samples': [round(x_tfidf_train.shape[0] * 0.7),round(x_tfidf_train.shape[0] * 0.4)],
        'estimator__base_estimator__max_depth': [30,50,100],
        'estimator__base_estimator__n_estimators': [100,200,300],
        'estimator__base_estimator__min_samples_leaf': [0.05, 0.1, 0.5],
        'estimator__base_estimator__min_samples_split': [0.05, 0.1, 0.5],
        'estimator__base_estimator__max_features': [0.3, 0.5, 0.8]
    }
    n_iterations_grid_search_RF = 10
    #modeloRF = treina_modelo_grid_search(x_tfidf_train, y_train, classificadorRF, nomeAlgoritmoRF,param_grid_RF, n_iterations_grid_search_RF)
    #modeloRF, y_pred , y_pred_proba_df= testa_modelo(x_tfidf_test, y_test, modeloRF)
    #modeloRF.setIdExecucao(id_execucao)
    #modeloRF.setData(data)
    #modeloRF.imprime()
    #salvaModelo(modeloRF)
    #salvaPredicao(modeloRF, X_test, y_test, y_pred, y_pred_proba_df)


    print("-------------------------------------------------------------------------")
    print(nomeAlgoritmoMLP)
    print("-------------------------------------------------------------------------")
    param_grid_MLP = {
        'estimator__n_estimators': [3,5],
	    'estimator__max_samples': [round(x_tfidf_train.shape[0] * 0.7),round(x_tfidf_train.shape[0] * 0.4)],
        'estimator__base_estimator__hidden_layer_sizes': [(10,10)],
        'estimator__base_estimator__activation': ['identity', 'logistic', 'tanh', 'relu'],
        'estimator__base_estimator__solver': ['sgd', 'adam','lbfgs'],
        'estimator__base_estimator__alpha': [0.0001,0.001, 0.01, 0.05, 0.1],
        'estimator__base_estimator__learning_rate': ['constant','adaptive','invscaling'],
        'estimator__base_estimator__max_iter': [200,300,500]
    }
    n_iterations_grid_search_MLP = 10
    modeloMLP = treina_modelo_grid_search(x_tfidf_train, y_train, classificadorMLP, nomeAlgoritmoMLP,param_grid_MLP,n_iterations_grid_search_MLP)
    modeloMLP,y_pred , y_pred_proba_df= testa_modelo(x_tfidf_test, y_test, modeloMLP)
    modeloMLP.setIdExecucao(id_execucao)
    modeloMLP.setData(data)
    modeloMLP.imprime()
    salvaModelo(modeloMLP)
    (modeloMLP, X_test, y_test, y_pred, y_pred_proba_df)

    print("-------------------------------------------------------------------------")
    print(nomeAlgoritmoSVM)
    print("-------------------------------------------------------------------------")
    # param_grid_SVM = {
    #     'estimator__n_estimators': [1],
    #     'estimator__max_samples': [round(x_tfidf_train.shape[0] * 0.2),round(x_tfidf_train.shape[0] * 0.1)],
    #     'estimator__base_estimator__C':[1,10,100,1000],
    #     'estimator__base_estimator__gamma':[1,0.1,0.001,0.0001],
    #     'estimator__base_estimator__kernel':['linear','rbf']
    # }

    param_grid_SVM = {
        'estimator__n_estimators': [3,5,7,9],
        'estimator__max_samples': [round(x_tfidf_train.shape[0] * 0.5),round(x_tfidf_train.shape[0] * 0.4)],
        'estimator__base_estimator__base_estimator__C':[0.01,0.1,1,10]
    }

    # n_iterations_grid_search_SVM = 10
    # modeloSVM = treina_modelo_grid_search(x_tfidf_train, y_train, classificadorSVM, nomeAlgoritmoSVM,param_grid_SVM,n_iterations_grid_search_SVM)
    # modeloSVM, y_pred, y_pred_proba_df = testa_modelo(x_tfidf_test, y_test, modeloSVM)
    # modeloSVM.imprime()
    # modeloSVM.setIdExecucao(id_execucao)
    # modeloSVM.setData(data)
    # salvaModelo(modeloSVM)
    # salvaPredicao(modeloSVM, X_test, y_test, y_pred, y_pred_proba_df)


print("-------------------------------------------------------------------------")


# import pandas as pd
# df_resultados.to_csv(path + "Metricas", header=True)
# df_resultados = pd.read_csv(path + "Metricas.csv")
# df_resultados.columns = ['id_execucao', 'data', 'nome','feature_type','tempo_processamento','tamanho_conjunto_treinamento','accuracy','micro_precision','micro_recall','micro_fscore','macro_precision','macro_recall','macro_fscore','best_params_','best_estimator_','grid_scores_','grid_cv_results','confusion_matrix','classification_report','num_estimators','max_samples']
# df_resultados.shape
# df_teste = df_resultados.drop_duplicates()
# df_teste.shape
# df_resultados = df_teste
# plota_evolucao_algoritmo(df_resultados,nomeAlgoritmoNB )
# plota_evolucao_algoritmo(df_resultados,nomeAlgoritmoRF )
# plota_evolucao_algoritmo(df_resultados,nomeAlgoritmoSVM )
# plota_evolucao_algoritmo(df_resultados,nomeAlgoritmoMLP )




# listaRegionais = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']
# # listaRegionais = ['01','02']
#
# listaAssuntos=[2546,2086,1855,2594,2458,2029,2140,2478,2704,2021,2426,2656,8808,1844,1663,2666,2506,55220,2055,1806,2139,1888,2435,2215,5280,2554,2583,55170,2019,2117,1661,1904,2540,55345]
# # listaAssuntos=[2546,2086,1855,2594,2458,2029,2140,2478,2704,2021]
# listaResultados = []
# for qtdElementosPorAssunto in range(10, 201, 10):
#     listaResultados.append(recupera_resultados_modelo(qtdElementosPorAssunto,listaAssuntos,listaRegionais, MultinomialNB(), 'Multinomial Naive Bayes'))
# plt.plot(*zip(*listaResultados))
# plt.title("NB")
# plt.savefig("{0}{1}.png".format(path, str("NB").replace(' ', '')))
# plt.show()
# SVC(kernel='linear', probability=True, class_weight='balanced')
#
# def getKey(item):
#     return item[0]
# teste = sorted(listaResultados, key=getKey)
# listaResultados = teste
# listaResultadosSVMBackup = listaResultados
