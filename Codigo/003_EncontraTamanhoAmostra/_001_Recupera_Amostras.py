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
    data.plot.bar(ylim=0)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.savefig("{0}{1}.png".format(path, "Balanceamento_Assuntos_" + str(qnt_elem) + "_Elementos"))
    # plt.show()

#//TODO: Ajustar esse grafico para que ele traga o nome do assunto ao inves do codigo, ou adicionar legenda
#----------------------------------------------------------------------------------------------------------------------
#Recuperando amostra
#----------------------------------------------------------------------------------------------------------------------
def collect_results(result):
    """Uses apply_async's callback to setup up a separate Queue for each process"""
    results.extend(result)

def recupera_n_amostras_por_assunto_por_regional(sigla_trt, assuntos, nroElementos):

    """
    Função que, dado um regional, uma lista de assuntos, e definida a a quantidade de amostras de cada item,
    busca o arquivo com os documentos do regional informado e retira o número de elementos de dada assunto deste regional
    :param regional: sigla do regional onde se deve buscar os dados
    :param assuntos: lista de assuntos a se buscar
    :param quantidadeAmostras: quantidade de elementos de cada assunto. Se não existir a quantidade demandada, irá limitar a quantidade retornada em cada classe
    ao mínimo existente
    :return:
    """
    path = '/media/DATA/classificadorDeAssuntos/Dados/naoPublicavel/ConferenciaDeAssuntos/OK/'
    # sigla_trt='19'
    # assuntos = [1690,1661]
    # nroElementos=20
    # print('Buscando dados para o TRT  '+ sigla_trt)
    nome_arquivo = path + 'TRT_' + sigla_trt + '_2G_2010-2019_documentosSelecionadosProcessados.csv'
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

def recupera_amostras_de_todos_regionais(listaAssuntos, nroElementos):
    global results
    results = []
    print("Buscando " + str(nroElementos) + " elementos de cada assunto em cada regional")
    start_time = time.time()

    pool = mp.Pool(processes=mp.cpu_count())
    # pool = mp.Pool(4)
    # listaAssuntos=[2546]
    # nroElementos=10000000
    for i in range (1,25):
        pool.apply_async(recupera_n_amostras_por_assunto_por_regional, args=("{:02d}".format(i),listaAssuntos,nroElementos), callback=collect_results)
    pool.close()
    pool.join()

    # Converts list of lists to a data frame
    df = pd.DataFrame(results, columns=['index','nr_processo','id_processo_documento','cd_assunto_nivel_1','cd_assunto_nivel_2','cd_assunto_nivel_3','cd_assunto_nivel_4','cd_assunto_nivel_5','ds_identificador_unico',
                                         'ds_identificador_unico_simplificado','ds_orgao_julgador', 'ds_orgao_julgador_colegiado','dt_juntada','texto_processado', 'texto_stemizado','sigla_trt'])
    print(df.shape)
    total_time = time.time() - start_time
    print("Tempo para recuperar amostra de todos os regionais ", str(timedelta(seconds=total_time)))
    return df

results = []

# for qtdElementosPorAssunto in range(10,401, 10):
#     print("------------------------------------------------")
#     print('Buscando ' + str(qtdElementosPorAssunto) + " elementos")
#     df_recuperado = recupera_amostras_de_todos_regionais([], qtdElementosPorAssunto, path = '/media/DATA/classificadorDeAssuntos/Dados/naoPublicavel/ConferenciaDeAssuntos/OK/')