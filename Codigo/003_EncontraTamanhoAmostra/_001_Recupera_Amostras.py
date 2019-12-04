# ### ====================================================================================
# Script que recuperar amostras de um dataset
# ### ====================================================================================
import csv
import multiprocessing as mp
import sys
import time
import nltk
import os
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.model_selection import train_test_split
import warnings


# -----------------------------------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------
# Função que recuperar amostra estratificada pelo codigo de assunto dentre todos os codigos existentes no dataset. Nao faz bootstrapping..
#----------------------------------------------------------------------------------------------------------------------
def stratified_sample_df(df, col, n_samples):
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
#----------------------------------------------------------------------------------------------------------------------
# Função que mostra a distribuição de elementos por tribunal para cada assunto
#----------------------------------------------------------------------------------------------------------------------
def mostra_representatividade_regional_por_amostra(df, path):
    df_amostra_final_trunc = df[['sigla_trt','cd_assunto_nivel_3']]
    df_tmp = df_amostra_final_trunc.groupby(['sigla_trt','cd_assunto_nivel_3']).cd_assunto_nivel_3.count().to_frame()
    df_tmp = df_tmp.unstack()
    df_tmp.columns = df_tmp.columns.droplevel()  # Drop `cd_assunto_nivel_3` label.
    df_tmp = df_tmp.div(df_tmp.sum())
    df_tmp.T.plot(kind='bar', stacked=True, rot=1, figsize=(8, 8),
                  title="Distribuição de assuntos na amostra por tribunal")
    plt.legend()
    plt.savefig("{0}{1}.png".format(path, "Representatividade_Regional_" + str(df.shape[0]) + "_Elementos"))
    # plt.show()
#----------------------------------------------------------------------------------------------------------------------
# Função que mostra a distribuição de elementos por assunto
#----------------------------------------------------------------------------------------------------------------------
def mostra_balanceamento_assunto(data, title, ylabel, xlabel, path, qnt_elem):
    plt.clf()
    plt.cla()
    plt.close()
    data.plot.bar(ylim=0)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.bar(data)
    plt.savefig("{0}{1}.png".format(path, "Balanceamento_Assuntos_" + str(qnt_elem) + "_Elementos"))
    #plt.show()

    df = pd.DataFrame(y_train.value_counts())
    df = df.reset_index()
    df.columns =  ['assunto_nivel_3', 'qnt_documentos']
    plt.bar(df['assunto_nivel_3'], df['qnt_documentos'], align='center', alpha=0.5)



#//TODO: Ajustar esse grafico para que ele traga o nome do assunto ao inves do codigo, ou adicionar legenda
#----------------------------------------------------------------------------------------------------------------------
#Recuperando amostra
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
    Função que busca documentos em arquivos CSVs para os 24 Tribunais Regionais
    :param listaAssuntos: assuntos a serem buscados
    :param nroElementos: quantidade de elementos a ser recuperada
    :param regionais: lista dos regionais nos quais se vai buscar os documentos
    :param path local onde recuperar os documentos dos regionais
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

# ---------------------------------------------------------------------------------------------------------------------
# Imprime evolucao de um algoritmo em grafico em funcao da quantidade de amostras
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

# Código de chamada de teste
# listaAssuntos=[2546,2086,1855]
# regionais=[20,21]
# path = '/media/DATA/classificadorDeAssuntos/Dados/naoPublicavel/ConferenciaDeAssuntos/OK/'
# for qtdElementosPorAssunto in range(10,21, 10):
#     print("------------------------------------------------")
#     print('Buscando ' + str(qtdElementosPorAssunto) + " elementos")
#     df_recuperado = recupera_amostras_de_todos_regionais(listaAssuntos, qtdElementosPorAssunto,path,regionais )