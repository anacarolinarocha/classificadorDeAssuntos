from docutils.nodes import header
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from datetime import timedelta
import time
import sys
from datetime import datetime
import uuid
import matplotlib.pyplot as plt
sys.path.insert(1, '/home/anarocha/myGit/classificadorDeAssuntos/Codigo/003_EncontraTamanhoAmostra')
from _001_Recupera_Amostras import *
sys.path.insert(1, '/home/anarocha/myGit/classificadorDeAssuntos/Codigo/006_Classificacao')
from _001_Treina_E_Testa_Modelos import *
sys.path.insert(1, '/home/anarocha/myGit/classificadorDeAssuntos/Codigo/005_FeatureEngineering')
from _002_Extrai_Features import *

path='/home/anarocha/myGit/classificadorDeAssuntos/Resultados/EXP10_32GigasOnTheHouse/'
if not os.path.exists(path):
    os.makedirs(path)

# ---------------------------------------------------------------------------------------------------------------------
# Função que divide conjunto de treinamento e teste de stratificado por assunto
#----------------------------------------------------------------------------------------------------------------------
def splitTrainTest(df_amostra_final):
    X_train, X_test, y_train, y_test = train_test_split(df_amostra_final['texto_stemizado'],
                                                        df_amostra_final['cd_assunto_nivel_3'], test_size=0.3,
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
# Chamada principal
#----------------------------------------------------------------------------------------------------------------------
# listaAssuntos=[2546,2086,1855]
listaAssuntos=[2546,2086,1855,2594,2458,2029,2140,2478,2704,2021,2426,2656,8808,1844,1663,2666,2506,55220,2055,1806,2139,1888,2435,2215,5280,2554,2583,55170,2019,2117,1661,1904,2540,55345]

#Classificadores iniais para ter um baseline antes do GridSearch
classificadorNB = MultinomialNB()
classificadorRF = RandomForestClassifier(n_estimators=100, max_depth=10,random_state=0)
classificadorSVM = SVC(kernel='linear', probability=True, class_weight='balanced')
classificadorMLP = MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=500)

nomeAlgoritmoNB='Multinomial Naive Bayes'
nomeAlgoritmoRF='Random Forest'
nomeAlgoritmoSVM='SVM'
nomeAlgoritmoMLP="Multi-Layer Perceptron"

id_execucao = str(uuid.uuid1())[:7]
data = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

df_resultados = pd.DataFrame(columns=['id_execucao', 'data', 'nome','feature_type','tempo_processamento','tamanho_conjunto_treinamento','accuracy','micro_precision','micro_recall','micro_fscore','macro_precision','macro_recall','macro_fscore','best_params_','best_estimator_','grid_scores_','confusion_matrix','num_estimators','max_samples'])
nome_arquivo_destino = path + "Metricas.csv"
if  not (os.path.isfile(nome_arquivo_destino)):
    with open(nome_arquivo_destino, 'a') as f:
        df_resultados.to_csv(f, header=True)

for qtdElementosPorAssunto in range(10000000,10000001, 10000000):
# for qtdElementosPorAssunto in range(10, 11, 10):
    df_amostra = recupera_amostras_de_todos_regionais(listaAssuntos, qtdElementosPorAssunto)
    X_train, X_test, y_train, y_test = splitTrainTest(df_amostra)
    print("=========================================================================")
    print("Amostra de treinamento de " + str(X_train.shape[0]) + " elementos")
    print("=========================================================================")
    title = "Balanceamento de assuntos na amostra de "  + str(X_train.shape[0])
    mostra_balanceamento_assunto(y_train.value_counts(), title, "Quantidade Elementos", "Código Assunto", path, y_train.shape[0])
    x_tfidf_train, x_tfidf_test = extraiFeaturesTFIDF(df_amostra,X_train,X_test )

    print("-------------------------------------------------------------------------")
    print(nomeAlgoritmoSVM)
    print("-------------------------------------------------------------------------")
    modeloSVM = treina_modelo(x_tfidf_train, y_train, classificadorSVM, nomeAlgoritmoSVM)
    modeloSVM = testa_modelo(x_tfidf_test, y_test, modeloSVM)
    modeloSVM.imprime()
    modeloSVM.setIdExecucao(id_execucao)
    modeloSVM.setData(data)
    df_resultados = df_resultados.append(modeloSVM.__dict__, ignore_index=True)

    print("-------------------------------------------------------------------------")
    print(nomeAlgoritmoMLP)
    print("-------------------------------------------------------------------------")
    modeloMLP = treina_modelo(x_tfidf_train, y_train, classificadorMLP, nomeAlgoritmoMLP)
    modeloMLP = testa_modelo(x_tfidf_test, y_test, modeloMLP)
    modeloMLP.imprime()
    modeloMLP.setIdExecucao(id_execucao)
    modeloMLP.setData(data)
    df_resultados = df_resultados.append(modeloMLP.__dict__, ignore_index=True)

    print("-------------------------------------------------------------------------")
    print(nomeAlgoritmoNB)
    print("-------------------------------------------------------------------------")
    modeloNB = treina_modelo(x_tfidf_train, y_train, classificadorNB, nomeAlgoritmoNB)
    modeloNB = testa_modelo(x_tfidf_test, y_test, modeloNB)
    modeloNB.imprime()
    modeloNB.setIdExecucao(id_execucao)
    modeloNB.setData(data)
    df_resultados = df_resultados.append(modeloNB.__dict__, ignore_index=True)

    print("-------------------------------------------------------------------------")
    print(nomeAlgoritmoRF)
    print("-------------------------------------------------------------------------")
    modeloRF = treina_modelo(x_tfidf_train, y_train, classificadorRF, nomeAlgoritmoRF)
    modeloRF = testa_modelo(x_tfidf_test, y_test, modeloRF)
    modeloRF.imprime()
    modeloRF.setIdExecucao(id_execucao)
    modeloRF.setData(data)
    df_resultados = df_resultados.append(modeloRF.__dict__, ignore_index=True)


    with open(nome_arquivo_destino, 'a') as f:
        df_resultados.to_csv(f, header=False)
	
    print("-------------------------------------------------------------------------")
# import pandas as pd
# df_resultados.to_csv(path + "Metricas", header=True)
# df_resultados = pd.read_csv(path + "Metricas.csv")
# df_resultados.columns = ['index','id_execucao', 'data', 'nome','feature_type','tempo_processamento','tamanho_conjunto_treinamento','accuracy','micro_precision','micro_recall','micro_fscore','macro_precision','macro_recall','macro_fscore','best_params_','best_estimator_','grid_scores_','confusion_matrix','num_estimators','max_samples']
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
