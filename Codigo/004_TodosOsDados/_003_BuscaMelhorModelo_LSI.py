from docutils.nodes import header
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC,LinearSVC
from datetime import timedelta
import time
import sys
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import uuid
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
sys.path.insert(1, '/home/anarocha/myGit/classificadorDeAssuntos/Codigo/003_EncontraTamanhoAmostra')
from _001_Recupera_Amostras import *
sys.path.insert(1, '/home/anarocha/myGit/classificadorDeAssuntos/Codigo/006_Classificacao')
from _001_Treina_E_Testa_Modelos import *
sys.path.insert(1, '/home/anarocha/myGit/classificadorDeAssuntos/Codigo/005_FeatureEngineering')
from _002_Extrai_Features import *
from _003_BM25_Transformer import *

# ---------------------------------------------------------------------------------------------------------------------
# Setup
#----------------------------------------------------------------------------------------------------------------------
path_resultados='/media/DATA/classificadorDeAssuntos/Dados/Resultados/EXP30_MelhoresModelos_TextsoReduzidos_LSI500/'
if not os.path.exists(path_resultados):
    os.makedirs(path_resultados)

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

f = open(path_resultados + 'out.txt', 'w')
original = sys.stdout
sys.stdout = PrintPythonConsoleOnFileAlso(sys.stdout, f)

# sys.stdout = original
# print "This won't appear on file"  # Only on stdout
# f.close()
columnsResultados=['id_execucao', 'data', 'nome','feature_type','tempo_processamento','tamanho_conjunto_treinamento','accuracy','balanced_accuracy','micro_precision','micro_recall','micro_fscore','macro_precision','macro_recall','macro_fscore','best_params_','best_estimator_','grid_scores_','grid_cv_results','confusion_matrix','classification_report','num_estimators','max_samples']
df_resultados = pd.DataFrame(columns = columnsResultados)
nome_arquivo_destino = path_resultados + "Metricas.csv"
if  not (os.path.isfile(nome_arquivo_destino)):
    with open(nome_arquivo_destino, 'a') as f:
        df_resultados.to_csv(f, header=True)
nome_classification_reports = path_resultados + 'ClassificationReports.csv'

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
    plt.savefig("{0}{1}.png".format(path_resultados, nomeAlgoritmo.replace(' ', '')))
    # plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# Salva os valores preditos
#----------------------------------------------------------------------------------------------------------------------
def salvaPredicao(modelo, X_test, y_true, y_pred, y_pred_proba_df):
    global df_resultados
    nome_arquivo_predicao = path_resultados + 'predicao_' + modelo.getNome() + '.csv'
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
    modelo.salvaModelo(path_resultados)
    df_resultados = df_resultados.append(modelo.__dict__, ignore_index=True)
    with open(nome_arquivo_destino, 'a') as f:
        df_resultados.tail(1).to_csv(f, header=False)


# ---------------------------------------------------------------------------------------------------------------------
# Chamada principal
#----------------------------------------------------------------------------------------------------------------------
listaAssuntos=[2546,2086,1855,2594,2458,2704,2656,2140,2435,2029,2583,2554,8808,2117,2021,5280,1904,1844,2055,1907,1806,55220,2506,
                        4437,10570,1783,1888,2478,5356,1773,1663,5272,2215,1767,1661,1690]


classificadorNB = MultinomialNB()
classificadorRF = RandomForestClassifier(random_state=42)
classificadorSVM = CalibratedClassifierCV(LinearSVC(class_weight='balanced', max_iter=10000,random_state=42),method='sigmoid', cv=5)
classificadorMLP = MLPClassifier(early_stopping= True,random_state=42)

nomeAlgoritmoNB='Multinomial Naive Bayes'
nomeAlgoritmoRF='Random Forest'
nomeAlgoritmoSVM='SVM'
nomeAlgoritmoMLP="Multi-Layer Perceptron"

id_execucao = str(uuid.uuid1())[:7]
data = datetime.now().strftime("%d/%m/%Y %H:%M:%S")


# for qtdElementosPorAssunto in range(1000000,1000001, 1000000):
# for qtdElementosPorAssunto in range(10, 11, 10):

# ---------------------------------------------------------
# Recuperando dados
# ---------------------------------------------------------
qtdElementosPorAssunto=1000000
df_amostra = recupera_amostras_de_todos_regionais(listaAssuntos, qtdElementosPorAssunto,'/media/DATA/classificadorDeAssuntos/Dados/naoPublicavel/ConferenciaDeAssuntos/OK/')

#Juntando os assuntos 55220 e 1855, ambos Indenização por Dano Moral
df_amostra.loc[df_amostra['cd_assunto_nivel_3'] == 55220, 'cd_assunto_nivel_3'] = 1855
df_amostra.loc[df_amostra['cd_assunto_nivel_2'] == 55218, 'cd_assunto_nivel_3'] = 2567

print('Total de textos recuperados: ' + str(len(df_amostra)))
df_amostra = df_amostra.dropna(subset=['texto_stemizado'])
print('Total de textos recuperados com conteúdo: ' + str(len(df_amostra)))

# ---------------------------------------------------------
#Analisando tamanho dos textos
# ---------------------------------------------------------
df_amostra['quantidade_de_palavras'] = [len(x.split()) for x in df_amostra['texto_processado'].tolist()]
sns.boxplot(df_amostra['quantidade_de_palavras'])
plt.savefig("{0}{1}.png".format(path_resultados, "Distribuicao_Tamanho_Textos_Original"))

df_amostra_f = df_amostra[((df_amostra.quantidade_de_palavras < 100) & (df_amostra.quantidade_de_palavras > 0))]
print('Quantidade de textos entro 0 e 100 palavras: ' + str(len(df_amostra_f)))
# df_amostra_f[['texto_processado','quantidade_de_palavras']].head(1000).to_csv(path_resultados + 'teste.csv',sep='#', quoting=csv.QUOTE_ALL)
df_amostra_f = df_amostra[((df_amostra.quantidade_de_palavras < 200) & (df_amostra.quantidade_de_palavras > 100))]
print('Quantidade de textos entro 100 e 200 palavras: ' + str(len(df_amostra_f)))
df_amostra_f = df_amostra[((df_amostra.quantidade_de_palavras < 300) & (df_amostra.quantidade_de_palavras > 200))]
print('Quantidade de textos entro 200 e 300 palavras: ' + str(len(df_amostra_f)))
df_amostra_f = df_amostra[((df_amostra.quantidade_de_palavras < 400) & (df_amostra.quantidade_de_palavras > 300))]
print('Quantidade de textos entro 300 e 400 palavras: ' + str(len(df_amostra_f)))
df_amostra.shape
df_amostra_f = df_amostra[((df_amostra.quantidade_de_palavras < 10000) & (df_amostra.quantidade_de_palavras > 400))]
df_amostra_f= df_amostra_f.sort_values(by='quantidade_de_palavras', ascending=True)
df_amostra_f.shape
df_amostra = df_amostra_f
plt.clf()
plt.cla()
plt.close()
sns.boxplot(df_amostra['quantidade_de_palavras'])
plt.savefig("{0}{1}.png".format(path_resultados, "Distribuicao_Tamanho_Textos_Depois_Da_Remocao_De_Textos_Com_Mais_De_400_e_Menos_de_10000"))


print("=========================================================================")
print('Total de textos utilizados: ' + str(len(df_amostra)))
X_train, X_test, y_train, y_test = splitTrainTest(df_amostra)
print("Amostra de treinamento de " + str(X_train.shape[0]) + " elementos")
print("=========================================================================")
title = "Balanceamento de assuntos na amostra de "  + str(X_train.shape[0])
mostra_balanceamento_assunto(y_train.value_counts(), title, "Quantidade Elementos", "Código Assunto", path_resultados, y_train.shape[0])

# start_time = time.time()
# lsi100_transformer,x_lsi100_train, x_lsi100_test = extraiFeaturesLSI(df_amostra, X_train['texto_stemizado'], X_test['texto_stemizado'], 100, path_resultados)
# total_time = time.time() - start_time
# print("Tempo para montar matrizes LSI 100 (features:  "+ str(x_lsi100_train.shape[1]) + ") :" +   str(timedelta(seconds=total_time)))
#
# start_time = time.time()
# lsi250_transformer,x_lsi250_train, x_lsi250_test = extraiFeaturesLSI(df_amostra, X_train['texto_stemizado'], X_test['texto_stemizado'], 250, path_resultados)
# total_time = time.time() - start_time
# print("Tempo para montar matrizes LSI 250 (features:  "+ str(x_lsi250_train.shape[1]) + ") :" +   str(timedelta(seconds=total_time)))

start_time = time.time()
lsi500_transformer,x_lsi500_train, x_lsi500_test = extraiFeaturesLSI(df_amostra, X_train['texto_stemizado'], X_test['texto_stemizado'], 500, path_resultados)
total_time = time.time() - start_time
print("Tempo para montar matrizes LSI 250 (features:  "+ str(x_lsi500_train.shape[1]) + ") :" +   str(timedelta(seconds=total_time)))


del(df_amostra)


print("=========================================================================")
print("LSI 500")
print("=========================================================================")


print("-------------------------------------------------------------------------")
print(nomeAlgoritmoSVM)
print("-------------------------------------------------------------------------")
param_grid_SVM = {
    'estimator__n_estimators': [5],
    'estimator__max_samples': [0.8],
    'estimator__base_estimator__base_estimator__C': [1]
}
n_iterations_grid_search_SVM = 1
modeloSVM = treina_modelo_grid_search(x_lsi500_train, y_train, classificadorSVM, nomeAlgoritmoSVM,'LSI500', param_grid_SVM,
                                      n_iterations_grid_search_SVM, 5)
modeloSVM, y_pred, y_pred_proba_df = testa_modelo(x_lsi500_test, y_test, modeloSVM)
modeloSVM.imprime()
modeloSVM.setIdExecucao(id_execucao)
modeloSVM.setData(data)
salvaModelo(modeloSVM)
salvaPredicao(modeloSVM, X_test, y_test, y_pred, y_pred_proba_df)

print("-------------------------------------------------------------------------")
print(nomeAlgoritmoRF)
print("-------------------------------------------------------------------------")
param_grid_RF = {
    'estimator__n_estimators': [3],
    'estimator__max_samples': [0.5],
    'estimator__base_estimator__max_depth': [100],
    'estimator__base_estimator__n_estimators': [200],
    'estimator__base_estimator__min_samples_leaf': [0.05],
    'estimator__base_estimator__min_samples_split': [ 0.1],
    'estimator__base_estimator__max_features': [0.3]
}
n_iterations_grid_search_RF = 1
modeloRF = treina_modelo_grid_search(x_lsi500_train, y_train, classificadorRF, nomeAlgoritmoRF,'LSI500',param_grid_RF, n_iterations_grid_search_RF, 5)
modeloRF, y_pred , y_pred_proba_df= testa_modelo(x_lsi500_test, y_test, modeloRF)
modeloRF.setIdExecucao(id_execucao)
modeloRF.setData(data)
modeloRF.imprime()
salvaModelo(modeloRF)
salvaPredicao(modeloRF, X_test, y_test, y_pred, y_pred_proba_df)

print("-------------------------------------------------------------------------")
print(nomeAlgoritmoMLP)
print("-------------------------------------------------------------------------")
param_grid_MLP = {
    'estimator__n_estimators': [3],
    'estimator__max_samples': [0.8],
    'estimator__base_estimator__hidden_layer_sizes': [(10,10)],
    'estimator__base_estimator__activation': ['logistic'],
    'estimator__base_estimator__solver': ['lbfgs'],
    'estimator__base_estimator__alpha': [0.05],
    'estimator__base_estimator__learning_rate': ['constant'],
    'estimator__base_estimator__max_iter': [400]
}
n_iterations_grid_search_MLP = 1
modeloMLP = treina_modelo_grid_search(x_lsi500_train, y_train, classificadorMLP, nomeAlgoritmoMLP,'LSI500',param_grid_MLP,n_iterations_grid_search_MLP, 3)
modeloMLP,y_pred , y_pred_proba_df= testa_modelo(x_lsi500_test, y_test, modeloMLP)
modeloMLP.setIdExecucao(id_execucao)
modeloMLP.setData(data)
modeloMLP.imprime()
salvaModelo(modeloMLP)
salvaPredicao(modeloMLP, X_test, y_test, y_pred, y_pred_proba_df)

print("-------------------------------------------------------------------------")

print("=========================================================================")
print("LSI 250")
print("=========================================================================")


print("-------------------------------------------------------------------------")
print(nomeAlgoritmoSVM)
print("-------------------------------------------------------------------------")
n_iterations_grid_search_SVM = 1
modeloSVM = treina_modelo_grid_search(x_lsi250_train, y_train, classificadorSVM, nomeAlgoritmoSVM,'LSI250', param_grid_SVM,
                                      n_iterations_grid_search_SVM, 5)
modeloSVM, y_pred, y_pred_proba_df = testa_modelo(x_lsi250_test, y_test, modeloSVM)
modeloSVM.imprime()
modeloSVM.setIdExecucao(id_execucao)
modeloSVM.setData(data)
salvaModelo(modeloSVM)
salvaPredicao(modeloSVM, X_test, y_test, y_pred, y_pred_proba_df)

print("-------------------------------------------------------------------------")
print(nomeAlgoritmoRF)
print("-------------------------------------------------------------------------")
n_iterations_grid_search_RF = 1
modeloRF = treina_modelo_grid_search(x_lsi250_train, y_train, classificadorRF, nomeAlgoritmoRF,'LSI250',param_grid_RF, n_iterations_grid_search_RF, 5)
modeloRF, y_pred , y_pred_proba_df= testa_modelo(x_lsi250_test, y_test, modeloRF)
modeloRF.setIdExecucao(id_execucao)
modeloRF.setData(data)
modeloRF.imprime()
salvaModelo(modeloRF)
salvaPredicao(modeloRF, X_test, y_test, y_pred, y_pred_proba_df)

print("-------------------------------------------------------------------------")
print(nomeAlgoritmoMLP)
print("-------------------------------------------------------------------------")
n_iterations_grid_search_MLP = 1
modeloMLP = treina_modelo_grid_search(x_lsi250_train, y_train, classificadorMLP, nomeAlgoritmoMLP,'LSI250',param_grid_MLP,n_iterations_grid_search_MLP, 3)
modeloMLP,y_pred , y_pred_proba_df= testa_modelo(x_lsi250_test, y_test, modeloMLP)
modeloMLP.setIdExecucao(id_execucao)
modeloMLP.setData(data)
modeloMLP.imprime()
salvaModelo(modeloMLP)
salvaPredicao(modeloMLP, X_test, y_test, y_pred, y_pred_proba_df)

print("-------------------------------------------------------------------------")
#
# print("=========================================================================")
# print("LSI 100")
# print("=========================================================================")
#
#
# print("-------------------------------------------------------------------------")
# print(nomeAlgoritmoSVM)
# print("-------------------------------------------------------------------------")
# param_grid_SVM = {
#     'estimator__n_estimators': [5],
#     'estimator__max_samples': [0.8],
#     'estimator__base_estimator__base_estimator__C': [1]
# }
# n_iterations_grid_search_SVM = 1
# modeloSVM = treina_modelo_grid_search(x_lsi100_train, y_train, classificadorSVM, nomeAlgoritmoSVM,'LSI100', param_grid_SVM,
#                                       n_iterations_grid_search_SVM, 5)
# modeloSVM, y_pred, y_pred_proba_df = testa_modelo(x_lsi100_test, y_test, modeloSVM)
# modeloSVM.imprime()
# modeloSVM.setIdExecucao(id_execucao)
# modeloSVM.setData(data)
# salvaModelo(modeloSVM)
# salvaPredicao(modeloSVM, X_test, y_test, y_pred, y_pred_proba_df)
#
# print("-------------------------------------------------------------------------")
# print(nomeAlgoritmoRF)
# print("-------------------------------------------------------------------------")
# param_grid_RF = {
#     'estimator__n_estimators': [3],
#     'estimator__max_samples': [0.5],
#     'estimator__base_estimator__max_depth': [100],
#     'estimator__base_estimator__n_estimators': [200],
#     'estimator__base_estimator__min_samples_leaf': [0.05],
#     'estimator__base_estimator__min_samples_split': [ 0.1],
#     'estimator__base_estimator__max_features': [0.3]
# }
# n_iterations_grid_search_RF = 1
# modeloRF = treina_modelo_grid_search(x_lsi100_train, y_train, classificadorRF, nomeAlgoritmoRF,'LSI100',param_grid_RF, n_iterations_grid_search_RF, 5)
# modeloRF, y_pred , y_pred_proba_df= testa_modelo(x_lsi100_test, y_test, modeloRF)
# modeloRF.setIdExecucao(id_execucao)
# modeloRF.setData(data)
# modeloRF.imprime()
# salvaModelo(modeloRF)
# salvaPredicao(modeloRF, X_test, y_test, y_pred, y_pred_proba_df)
#
# print("-------------------------------------------------------------------------")
# print(nomeAlgoritmoMLP)
# print("-------------------------------------------------------------------------")
# param_grid_MLP = {
#     'estimator__n_estimators': [3],
#     'estimator__max_samples': [0.8],
#     'estimator__base_estimator__hidden_layer_sizes': [(10,10)],
#     'estimator__base_estimator__activation': ['logistic'],
#     'estimator__base_estimator__solver': ['lbfgs'],
#     'estimator__base_estimator__alpha': [0.05],
#     'estimator__base_estimator__learning_rate': ['constant'],
#     'estimator__base_estimator__max_iter': [400]
# }
# n_iterations_grid_search_MLP = 1
# modeloMLP = treina_modelo_grid_search(x_lsi100_train, y_train, classificadorMLP, nomeAlgoritmoMLP,'LSI100',param_grid_MLP,n_iterations_grid_search_MLP, 3)
# modeloMLP,y_pred , y_pred_proba_df= testa_modelo(x_lsi100_test, y_test, modeloMLP)
# modeloMLP.setIdExecucao(id_execucao)
# modeloMLP.setData(data)
# modeloMLP.imprime()
# salvaModelo(modeloMLP)
# salvaPredicao(modeloMLP, X_test, y_test, y_pred, y_pred_proba_df)
#
# print("-------------------------------------------------------------------------")
#
# print("=========================================================================")
# print("LSI 250")
# print("=========================================================================")
#
#
# print("-------------------------------------------------------------------------")
# print(nomeAlgoritmoSVM)
# print("-------------------------------------------------------------------------")
# n_iterations_grid_search_SVM = 1
# modeloSVM = treina_modelo_grid_search(x_lsi250_train, y_train, classificadorSVM, nomeAlgoritmoSVM,'LSI250', param_grid_SVM,
#                                       n_iterations_grid_search_SVM, 5)
# modeloSVM, y_pred, y_pred_proba_df = testa_modelo(x_lsi250_test, y_test, modeloSVM)
# modeloSVM.imprime()
# modeloSVM.setIdExecucao(id_execucao)
# modeloSVM.setData(data)
# salvaModelo(modeloSVM)
# salvaPredicao(modeloSVM, X_test, y_test, y_pred, y_pred_proba_df)
#
# print("-------------------------------------------------------------------------")
# print(nomeAlgoritmoRF)
# print("-------------------------------------------------------------------------")
# n_iterations_grid_search_RF = 1
# modeloRF = treina_modelo_grid_search(x_lsi250_train, y_train, classificadorRF, nomeAlgoritmoRF,'LSI250',param_grid_RF, n_iterations_grid_search_RF, 5)
# modeloRF, y_pred , y_pred_proba_df= testa_modelo(x_lsi250_test, y_test, modeloRF)
# modeloRF.setIdExecucao(id_execucao)
# modeloRF.setData(data)
# modeloRF.imprime()
# salvaModelo(modeloRF)
# salvaPredicao(modeloRF, X_test, y_test, y_pred, y_pred_proba_df)
#
# print("-------------------------------------------------------------------------")
# print(nomeAlgoritmoMLP)
# print("-------------------------------------------------------------------------")
# n_iterations_grid_search_MLP = 1
# modeloMLP = treina_modelo_grid_search(x_lsi250_train, y_train, classificadorMLP, nomeAlgoritmoMLP,'LSI250',param_grid_MLP,n_iterations_grid_search_MLP, 3)
# modeloMLP,y_pred , y_pred_proba_df= testa_modelo(x_lsi250_test, y_test, modeloMLP)
# modeloMLP.setIdExecucao(id_execucao)
# modeloMLP.setData(data)
# modeloMLP.imprime()
# salvaModelo(modeloMLP)
# salvaPredicao(modeloMLP, X_test, y_test, y_pred, y_pred_proba_df)
#
# print("-------------------------------------------------------------------------")
#
#
# print("=========================================================================")
# print("LSI 500")
# print("=========================================================================")
#
#
# print("-------------------------------------------------------------------------")
# print(nomeAlgoritmoSVM)
# print("-------------------------------------------------------------------------")
# n_iterations_grid_search_SVM = 1
# modeloSVM = treina_modelo_grid_search(x_lsi500_train, y_train, classificadorSVM, nomeAlgoritmoSVM,'LSI500', param_grid_SVM,
#                                       n_iterations_grid_search_SVM, 5)
# modeloSVM, y_pred, y_pred_proba_df = testa_modelo(x_lsi500_test, y_test, modeloSVM)
# modeloSVM.imprime()
# modeloSVM.setIdExecucao(id_execucao)
# modeloSVM.setData(data)
# salvaModelo(modeloSVM)
# salvaPredicao(modeloSVM, X_test, y_test, y_pred, y_pred_proba_df)
#
# print("-------------------------------------------------------------------------")
# print(nomeAlgoritmoRF)
# print("-------------------------------------------------------------------------")
# n_iterations_grid_search_RF = 1
# modeloRF = treina_modelo_grid_search(x_lsi500_train, y_train, classificadorRF, nomeAlgoritmoRF,'LSI500',param_grid_RF, n_iterations_grid_search_RF, 5)
# modeloRF, y_pred , y_pred_proba_df= testa_modelo(x_lsi500_test, y_test, modeloRF)
# modeloRF.setIdExecucao(id_execucao)
# modeloRF.setData(data)
# modeloRF.imprime()
# salvaModelo(modeloRF)
# salvaPredicao(modeloRF, X_test, y_test, y_pred, y_pred_proba_df)
#
# print("-------------------------------------------------------------------------")
# print(nomeAlgoritmoMLP)
# print("-------------------------------------------------------------------------")
# n_iterations_grid_search_MLP = 1
# modeloMLP = treina_modelo_grid_search(x_lsi500_train, y_train, classificadorMLP, nomeAlgoritmoMLP,'LSI500',param_grid_MLP,n_iterations_grid_search_MLP, 3)
# modeloMLP,y_pred , y_pred_proba_df= testa_modelo(x_lsi500_test, y_test, modeloMLP)
# modeloMLP.setIdExecucao(id_execucao)
# modeloMLP.setData(data)
# modeloMLP.imprime()
# salvaModelo(modeloMLP)
# salvaPredicao(modeloMLP, X_test, y_test, y_pred, y_pred_proba_df)
#
# print("-------------------------------------------------------------------------")