#######################################################################################################################
# Script com funções auxiliares utilizadas no projeto Classificador de Assuntos
# Por Ana Carolina Pereira Rocha
# Data: 10/12/2019
#######################################################################################################################
from modelo import *

#######################################################################################################################
#######################################################################################################################
# FUNÇÕES DE PRÉ-PROCESSAMENTO DE TEXTOS
#######################################################################################################################
#######################################################################################################################
import ssl
import nltk
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
#nltk.download('stopwords')
#nltk.download('rslp')


# ---------------------------------------------------------------------------------------------------------------------
# Função que remove marcacoes HTML de um texto
#----------------------------------------------------------------------------------------------------------------------
from bs4 import BeautifulSoup
def removeHTML(texto):
    """
    Função para remover HTML
    :param texto:
    :return: texto sem tags HTML
    """
    texto = texto.replace('\n', ' ')
    texto = texto.replace('\t', ' ')
    return BeautifulSoup(texto, 'lxml').get_text(" ", strip=True)

# ---------------------------------------------------------------------------------------------------------------------
# Função que remove acentos, numeros, palavras menores que 3 caracteres, caracteres especiais e tranforma em minusculo
#----------------------------------------------------------------------------------------------------------------------
import re
stemmer = nltk.stem.RSLPStemmer()
def processa_stemiza_texto(texto):
    """
    Função para remover caracteres especiais, acentos, pontuações, números, stopwords e  palavras menores que 3 caracteres. 
    :param texto:
    :return: texto processado
    """
    global stopwords_processadas
    textoProcessado = normalize('NFKD', texto).encode('ASCII','ignore').decode('ASCII')
    textoProcessado = re.sub('[^a-zA-Z]',' ',textoProcessado)
    textoProcessado = textoProcessado.lower()
    textoProcessado = textoProcessado.split()
    textoProcessado = [palavra for palavra in textoProcessnome_arquivo_destinoado if not palavra in stopwords_processadas]
    textoProcessado = [palavra for palavra in textoProcessado if len(palavra)>3]
    textoProcessado =  [stemmer.stem(palavra) for palavra in textoProcessado]
    return ' '.join(word for word in textoProcessado)

# ---------------------------------------------------------------------------------------------------------------------
# Função que faz o processamento dos textos para cada regional
#----------------------------------------------------------------------------------------------------------------------
from unicodedata import normalize
import pandas as pd
import os
import csv
import time
import multiprocessing as mp
from datetime import timedelta


# [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
stopwords_processadas = []  
def processaDocumentos(path_fonte_de_dados, path_destino_de_dados, regionais=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]):
    """
    Função para processar os documentos de todos os regionais. Ao final do processamento, serão armazenados arquivos csv
        com o conteúdo dos documentos processados e seus metadados.
    :param path_fonte_de_dados: Diretório onde serão buscados os arquivos csv com o conteúdo dos documentos a serem
         processados. É esperado que os documentos contenham o nome no seguinte padrão: 'TRT_XX_documentosSelecionados.csv', 
         onde XX se refere à sigla do Tribunal Regional, sempre com dois dígitos (01,02,10,20...)
    :param path_destino_de_dados: Diretório onde serão gravados os documentos que foram processados. Os documentos serão 
        armazenados com o seguinte nome: 'TRT_XX_documentosSelecionadosProcessados.csv'
    :param regionais: lista dos Tribunais Regionais a serem buscados (apenas um dígito). Por padrão, irá buscar todos
        os regionais, da 1ª à 24ª região    
    """
    global stopwords_processadas 
    print("Passando stopwords pelo pre processamento....")
    stopwords = nltk.corpus.stopwords.words('portuguese')
    for row in stopwords:
        palavraProcessada = normalize('NFKD', row).encode('ASCII', 'ignore').decode('ASCII')
        stopwords_processadas.append(palavraProcessada)    

    for regional in regionais:
        sigla_trt="{:02d}".format(regional)
        print("----------------------------------------------------------------------------")
        print('Processando texto dos documentos do TRT ' + sigla_trt)
        nome_arquivo_origem = path_fonte_de_dados + 'TRT_' + sigla_trt + '_documentosSelecionados.csv'
        nome_arquivo_destino = path_destino_de_dados + 'TRT_' + sigla_trt + '_documentosSelecionadosProcessados.csv'

        if not os.path.exists(nome_arquivo_origem):
            print("Não foi encontrado o arquivo de documentos do TRT " + sigla_trt + ". Buscou-se pelo arquivo " + nome_arquivo_origem)
            continue

        colnames = ['index','nr_processo', 'id_processo_documento', 'cd_assunto_nivel_5', 'cd_assunto_nivel_4','cd_assunto_nivel_3','cd_assunto_nivel_2','cd_assunto_nivel_1','tx_conteudo_documento','ds_identificador_unico','ds_identificador_unico_simplificado','ds_orgao_julgador','ds_orgao_julgador_colegiado','dt_juntada']
        df_trt = pd.read_csv(nome_arquivo_origem, sep=',', names=colnames, index_col=0, header=None, quoting=csv.QUOTE_ALL)

        #remove as tags HTML
        start_time = time.time()
        pool = mp.Pool(7)
        df_trt = df_trt.dropna(subset=['tx_conteudo_documento'])
        df_trt['texto_processado'] = pool.map(removeHTML, [row for row in df_trt['tx_conteudo_documento']])
        pool.close()
        total_time = time.time() - start_time
        print('Tempo para processamento do texto:' + str(timedelta(seconds=total_time)))

        #faz a stemizacao
        start_time = time.time()
        pool = mp.Pool(7)
        df_trt = df_trt.dropna(subset=['texto_processado'])
        df_trt['texto_stemizado'] = pool.map(processa_stemiza_texto, [row for row in df_trt['texto_processado']])
        pool.close()
        total_time = time.time() - start_time
        print('Tempo para stemização do texto:' + str(timedelta(seconds=total_time)))

        #----------------------------------------------------------
        # VERIFICA O CONTEUDO DE UM DOCUMENTO
        # f = open("./teste.html", "w")
        # f.write(df_trt.iloc[0]['tx_conteudo_documento'])
        # f.close()
        # import webbrowser
        # webbrowser.get('firefox').open_new_tab('./teste.html')
        # ----------------------------------------------------------

        df_trt = df_trt.drop(columns=['tx_conteudo_documento'])
        print("Encontrados " + str(df_trt.shape[0]) + " documentos para o TRT " + sigla_trt)

        if os.path.isfile(nome_arquivo_destino):
            os.remove(nome_arquivo_destino)

        df_trt.to_csv(nome_arquivo_destino, sep='#', quoting=csv.QUOTE_ALL)

#######################################################################################################################
#######################################################################################################################
# FUNÇÕES DE RECUPERAÇÃO DOS DADOS
#######################################################################################################################
#######################################################################################################################


# -----------------------------------------------------------------------------------------------------
# Função que recuperar amostra estratificada pelo codigo de assunto dentre todos os codigos existentes no dataset. Nao faz bootstrapping..
#----------------------------------------------------------------------------------------------------------------------
def stratified_sample_df(df, col, n_samples):
    """
    Função que recupera n_samples de cada documento baseado. 
    :param df: data frame que contem os dados
    :param col: nome da coluna para fazer a stratificação
    :param n_samples: quantidade de elementos a ser recuperado de cada classe. O código abaixo pode funcionar com o 
    oversampling, recuperando sempre n_samples de um assunto, ou sem oversampling, recuperando n_samples ou a 
    quantidade disponível de exemplos, caso não exista n_samples exemplos.
    :return: dataframe estratificado pelos valores apresentados na coluna col
    """
    # ------------------------------
    #COM OVER SAMPLING
    # min_accepted = 50
    # df_ = df.groupby(col).apply(lambda x: x.sample(calcularValorMinimo(x.shape[0], n_samples,min_accepted), random_state=42, replace = isResampling(x.shape[0], min_accepted)))

    #------------------------------
    #SEM OVER SAMPLING
    df_ = df.groupby(col).apply(lambda x: x.sample(min(x.shape[0], n_samples),random_state=42))

    # ------------------------------
    df_.index = df_.index.droplevel(0)
    return df_

def isResampling(value, min_accepted):
    if(value > min_accepted):
        return False
    else:
        return True

def calcularValorMinimo(value,n_samples, min_accepted):
    minimoEncontrado = min(value, n_samples)
    if(minimoEncontrado < min_accepted):
        return min_accepted
    else:
        return minimoEncontrado


# ---------------------------------------------------------------------------------------------------------------------
# Função que recupera os documentos de cada csv
#----------------------------------------------------------------------------------------------------------------------

def collect_results(result):
    """Uses apply_async's callback to setup up a separate Queue for each process"""
    results.extend(result)

def recupera_n_amostras_por_assunto_por_regional(sigla_trt, assuntos, nroElementos,path, sufixo):

    """
    Função que, dado um regional, uma lista de assuntos, e definida a a quantidade de amostras de cada item,
    busca o arquivo com os documentos do regional informado e retira o número de elementos de dado assunto deste regional
    :param regional: sigla do regional onde se deve buscar os dados
    :param assuntos: lista de assuntos a se buscar
    :param quantidadeAmostras: quantidade de elementos de cada assunto. Se não existir a quantidade demandada, irá limitar a quantidade retornada em cada classe
    ao mínimo existente
    :return:
    """
    nome_arquivo = path + 'TRT_' + sigla_trt + '_documentosSelecionadosProcessados' + sufixo + '.csv'
    #nome_arquivo = path + 'TRT_' + sigla_trt + '_2G_2010-2019_documentosSelecionadosProcessados' + sufixo + '.csv'
    if not os.path.exists(nome_arquivo):
        print( "Não foi encontrado o arquivo de documentos do TRT " + sigla_trt + ". Buscou-se pelo arquivo " + nome_arquivo)
        return []
    df_trt_csv = pd.read_csv(nome_arquivo, sep='#', quoting=csv.QUOTE_ALL)
    df_trt_csv.loc[:,'sigla_trt'] = "TRT"+sigla_trt;
    
    #Removendo dados que não serao necessarios nessa iteracao
    df_trt_csv.cd_assunto_nivel_3 = pd.to_numeric(df_trt_csv.cd_assunto_nivel_3)
    df_trt_filtrado = df_trt_csv[df_trt_csv.cd_assunto_nivel_3.isin(assuntos)]
    
    del(df_trt_csv)
    #Estratificando
    df_amostra = stratified_sample_df(df_trt_filtrado,'cd_assunto_nivel_3',nroElementos)
    
    # df_amostra['cd_assunto_nivel_3'].value_counts()
    print("Quantidade de documentos recuperados no TRT " + sigla_trt + ": " + str(df_amostra.shape[0]))

    return df_amostra.values.tolist()

def recupera_amostras_de_todos_regionais(listaAssuntos, nroElementos,path, sufixo='',regionais=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]):
    """
    Função que busca n (nroElementos) documentos (ou tantos quanto disponíveis) em arquivos CSVs para os 24 Tribunais Regionais
    :param listaAssuntos: assuntos a serem buscados
    :param nroElementos: quantidade de elementos a ser recuperada
    :param regionais: lista dos regionais nos quais se vai buscar os documentos. Por padrão, irá buscar todos
        os regionais, da 1ª à 24ª região    
    :param path: local onde recuperar os documentos dos regionais
    :return: data frame com o conteúdo de documentos e os metadadaos correpondentes de todos os regionais
    """
    global results
    results = []
    print("Buscando " + str(nroElementos) + " elementos de cada assunto em cada regional")
    start_time = time.time()

    pool = mp.Pool(processes=mp.cpu_count())
    # for i in range (1,25):
    for regional in regionais:
        pool.apply_async(recupera_n_amostras_por_assunto_por_regional, args=("{:02d}".format(regional),listaAssuntos,nroElementos,path,sufixo), callback=collect_results)
    pool.close()
    pool.join()

    df = pd.DataFrame(results, columns=['index','nr_processo','id_processo_documento','cd_assunto_nivel_1','cd_assunto_nivel_2','cd_assunto_nivel_3','cd_assunto_nivel_4','cd_assunto_nivel_5','ds_identificador_unico',
                                         'ds_identificador_unico_simplificado','ds_orgao_julgador', 'ds_orgao_julgador_colegiado','dt_juntada','texto_processado', 'texto_stemizado','sigla_trt'])
    print(df.shape)
    total_time = time.time() - start_time
    print("Tempo para recuperar amostra de todos os regionais ", str(timedelta(seconds=total_time)))
    return df

results = []

#----------------------------------------------------------------------------------------------------------------------
# Função que mostra a distribuição de elementos por assunto
#----------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
def mostra_balanceamento_assunto(data, title, ylabel, xlabel, path, qnt_elem):
    """
    Função que cria um grafico de barras a partir de um conjunto de dados para mostrar a quantidade de elementos por 
        assunto
    :param data: dados a serem processados
    :param title: título do grafico
    :param ylabel: label do eixo y
    :param xlabel: label do eixo x
    :param path: local onde sera gravado o grafico gerado
    :param qnt_elem: quantidade de ementos em data.    
    """
    plt.clf()
    plt.cla()
    plt.close()
    data.plot.bar(ylim=0)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    #plt.bar(data)
    plt.savefig("{0}{1}.png".format(path, "Balanceamento_Assuntos_" + str(qnt_elem) + "_Elementos"))
    #plt.show()

    #df = pd.DataFrame(y_train.value_counts())
    #df = df.reset_index()
    #df.columns =  ['assunto_nivel_3', 'qnt_documentos']
    #plt.bar(df['assunto_nivel_3'], df['qnt_documentos'], align='center', alpha=0.5)

# ---------------------------------------------------------------------------------------------------------------------
# Função que divide conjunto de treinamento e teste de stratificado por assunto
#----------------------------------------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
def splitTrainTest(df_amostra_final):
    X_train, X_test, y_train, y_test = train_test_split(df_amostra_final[['sigla_trt','nr_processo','id_processo_documento','texto_stemizado']],
                                                        df_amostra_final['cd_assunto_nivel_3'], test_size=0.2,
                                                        random_state=42,
                                                        stratify=df_amostra_final['cd_assunto_nivel_3'])
    return X_train, X_test, y_train, y_test


#######################################################################################################################
#######################################################################################################################
# FUNÇÕES AUXILIARES DE MODELOS
#######################################################################################################################
#######################################################################################################################

# ---------------------------------------------------------------------------------------------------------------------
# Salva os valores preditos
#----------------------------------------------------------------------------------------------------------------------
def salvaPredicao(modelo, X_test, y_true, y_pred, y_pred_proba_df,df_resultados,path_resultados):
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
def salvaModelo(modelo, path_resultados,df_resultados,nome_arquivo_destino):    
    modelo.salvaClassificationReport(path_resultados + 'ClassificationReport_' + modelo.getNome() + '.csv')
    modelo.salvaModelo(path_resultados)
    df_resultados = df_resultados.append(modelo.__dict__, ignore_index=True)
    with open(nome_arquivo_destino, 'a') as f:
        df_resultados.tail(1).to_csv(f, header=False)

# ---------------------------------------------------------------------------------------------------------------------
# Chama o treinamento do modelo
#----------------------------------------------------------------------------------------------------------------------
def chama_treinamento_modelo(x_trainamento, y_train, x_teste, y_test, modelo,nomeModelo,feature_type, param_grid, n_iteracoes, n_jobs, id_execucao ,data,path_resultados,df_resultados,nome_arquivo_destino,X_test_original):
    modelo = treina_modelo_grid_search(x_trainamento, y_train,modelo , nomeModelo, feature_type,param_grid,n_iteracoes, n_jobs)
    modelo, y_pred, y_pred_proba_df = testa_modelo(x_teste, y_test, modelo)
    modelo.setIdExecucao(id_execucao)
    modelo.setData(data)
    modelo.imprime()
    salvaModelo(modelo,path_resultados,df_resultados,nome_arquivo_destino)
    salvaPredicao(modelo, X_test_original, y_test, y_pred, y_pred_proba_df,df_resultados,path_resultados)
    return modelo

def teste():
    print('it works')

# ---------------------------------------------------------------------------------------------------------------------
# Grava modelo de transformação de features
#----------------------------------------------------------------------------------------------------------------------
import pickle
def salvaTransformer(transformer, nome, path):
    nomePicke = path + nome + '.p'
    arquivoPickle = open(nomePicke, 'wb')
    pickle.dump(transformer, arquivoPickle)
    arquivoPickle.close()


def carregaModelo(arquivo):
    with open(arquivo, "rb") as input_file:
        return pickle.load(input_file)

#######################################################################################################################
#######################################################################################################################
# FUNÇÕES DE GERAÇÃO DE FEATURES
#######################################################################################################################
#######################################################################################################################

#----------------------------------------------------------------------------------------------------------------------
# Função de geração de matriz TF-IDF
#----------------------------------------------------------------------------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
def extraiFeaturesTFIDF_train_test(df,X_train,X_test ,path):
    tfidf_vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b[A-Za-z]+\b', max_df=0.8, min_df=5)
    tfidf_transformer = tfidf_vectorizer.fit(df.texto_stemizado.astype(str))
    salvaTransformer(tfidf_transformer, 'TFIDF', path)
    x_tfidf_train = tfidf_transformer.transform(X_train)
    x_tfidf_test = tfidf_transformer.transform(X_test)
    return tfidf_transformer,x_tfidf_train, x_tfidf_test


#----------------------------------------------------------------------------------------------------------------------
# Função de geração de matriz BM25
#----------------------------------------------------------------------------------------------------------------------
from BM25_Transformer import *

def extraiFeaturesBM25(df_amostra_final,tfidf_transformer,x_tfidf_train,x_tfidf_test ,path):
    df_amostra_final_tfidf = tfidf_transformer.transform(df_amostra_final)
    bm25_transformer = BM25Transformer()
    bm25_transformer.fit(df_amostra_final_tfidf)
    salvaTransformer(bm25_transformer, 'BM25', path)
    x_bm25_train = bm25_transformer.transform(x_tfidf_train)
    x_bm25_test = bm25_transformer.transform(x_tfidf_test)
    return bm25_transformer,x_bm25_train, x_bm25_test


#----------------------------------------------------------------------------------------------------------------------
# Função de geração de matriz LSA
#----------------------------------------------------------------------------------------------------------------------
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
def recupera_lsi_transformer(df, topics):
    tfidf_vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b[A-Za-z]+\b', max_df=0.8, min_df=5)
    svd_model = TruncatedSVD(n_components=topics, algorithm='randomized',n_iter=10, random_state=42)
    svd_transformer = Pipeline([('tfidf', tfidf_vectorizer),
                                ('svd', svd_model)])
    svd_transformer = svd_transformer.fit(df.texto_stemizado.astype(str))
    return svd_transformer

def extraiFeaturesLSI(df_amostra_final,X_train,X_test,topics ,path):
    svd_transformer = recupera_lsi_transformer(df_amostra_final,topics)
    salvaTransformer(svd_transformer, 'LSI' + str(topics), path)
    x_lsi_train = svd_transformer.transform(X_train)
    x_lsi_test = svd_transformer.transform(X_test)
    return svd_transformer,x_lsi_train, x_lsi_test


#######################################################################################################################
#######################################################################################################################
# FUNÇÕES DE MODELAGEM
#######################################################################################################################
#######################################################################################################################

#----------------------------------------------------------------------------------------------------------------------
# Função que faz a busca de hiper-parametros para um modelo
#----------------------------------------------------------------------------------------------------------------------
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
def treina_modelo_grid_search(x_tfidf_train,y_train, classificador, nomeModelo, feature_type, param_grid ,n_iterations_grid_search, n_jobs):
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
    modelo = Modelo(feature_type + '_' + nomeModelo)
    # modelo.setMaxSamples(max_samples)
    modelo.setTamanhoConjuntoTreinamento(x_tfidf_train.shape[0])
    modelo.setTempoProcessamento(str(timedelta(seconds=grid_search.refit_time_)))
    modelo.setFeatureType(feature_type)
    modelo.setBestEstimator(grid_search.best_estimator_)
    modelo.setBestParams(grid_search.best_params_)
    modelo.setGridCVResults(grid_results)
    return modelo

#----------------------------------------------------------------------------------------------------------------------
# Função que faz o teste de um modelo
#----------------------------------------------------------------------------------------------------------------------

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
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
    #print(classification_report(y_test, y_pred, target_names=classes))
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