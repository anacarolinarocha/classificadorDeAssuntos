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
    min_accepted = 5
    # df = df_trt_filtrado
    # col = 'cd_assunto_nivel_3'
    # n_samples = 100000
    # len(df_trt_filtrado)
    # df_trt_filtrado.shape[0]
    df_ = df.groupby(col).apply(lambda x: x.sample(min(x.shape[0], n_samples),random_state=42))
    # df_ = df.groupby(col).apply(lambda x: x.sample(calcularValorMinimo(x.shape[0], n_samples,min_accepted), random_state=42, replace = isBootstraping(x.shape[0], min_accepted)))
    df_.index = df_.index.droplevel(0)
    return df_
#
# def isBootstraping(value, min_accepted):
#     if(value > min_accepted):
#         return False
#     else:
#         return True
#
# def calcularValorMinimo(value,n_samples, min_accepted):
#     minimoEncontrado = min(value, n_samples)
#     if(minimoEncontrado < min_accepted):
#         return min_accepted
#     else:
#         return minimoEncontrado
#----------------------------------------------------------------------------------------------------------------------
# Função que mostra a distribuição de elementos por tribunal para cada assunto
#----------------------------------------------------------------------------------------------------------------------
def mostra_representatividade_regional_por_amostra(df):
    df_amostra_final_trunc = df[['sigla_trt','cd_assunto_nivel_3']]
    df_tmp = df_amostra_final_trunc.groupby(['sigla_trt','cd_assunto_nivel_3']).cd_assunto_nivel_3.count().to_frame()
    df_tmp = df_tmp.unstack()
    df_tmp.columns = df_tmp.columns.droplevel()  # Drop `cd_assunto_nivel_3` label.
    df_tmp = df_tmp.div(df_tmp.sum())
    df_tmp.T.plot(kind='bar', stacked=True, rot=1, figsize=(8, 8),
                  title="Distribuição de assuntos na amostra por tribunal")
    plt.show()
#----------------------------------------------------------------------------------------------------------------------
# Função que mostra a distribuição de elementos por assunto
#----------------------------------------------------------------------------------------------------------------------
def mostra_balanceamento_assunto(data, title, ylabel, xlabel):
    data.plot.bar(ylim=0)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()

#//TODO: Ajustar esse grafico para que ele traga o nome do assunto ao inves do codigo
#----------------------------------------------------------------------------------------------------------------------
#Recuperando amostra
#----------------------------------------------------------------------------------------------------------------------
# nome_arquivo= path_base_dados_retificados_processados + 'listaDocumentosNaoSigilososRetificadosProcessados_MultiClasse_TRT' + '21' + '_2G_2010-2019.csv'
# df_trt_csv = pd.read_csv(nome_arquivo, sep=',', quoting=csv.QUOTE_ALL)
# df_trt_csv = df_trt_csv.head(1000)
# df_trt_csv.to_csv(
#             path_base_dados_retificados_processados + 'listaDocumentosNaoSigilososRetificadosProcessados_MultiClasse_TRT' +  '21'  + '_2G_2010-2019.csv',
#             sep='#', quoting=csv.QUOTE_ALL)

def collect_results(result):
    """Uses apply_async's callback to setup up a separate Queue for each process"""
    results.extend(result)

def recupera_n_amostras_por_assunto_por_regional(sigla_trt, assuntos, nroElementos):

    """
    Função que, dada um regional, uma lista de assuntos, e definida a a quantidade de amostras de cada item,
    busca o arquivo com os documentos do regional informado e retira o número de elementos de dada assunto deste regional
    :param regional: sigla do regional onde se deve buscar os dados
    :param assuntos: lista de assuntos a se buscar
    :param quantidadeAmostras: quantidade de elementos de cada assunto. Se não existir a quantidade demandada, irá limitar a quantidade retornada em cada classe
    ao mínimo existente
    :return:
    """
    path = '/media/DATA/classificadorDeAssuntos/Dados/naoPublicavel/ConferenciaDeAssuntos/OK/'
    # sigla_trt='01'
    # assuntos = [2546, 2086, 1855]
    # nroElementos=100
    print('Buscando dados para o TRT  '+ sigla_trt)
    nome_arquivo = path + 'TRT_' + sigla_trt + '_2G_2010-2019_documentosSelecionadosProcessados.csv'
    df_trt_csv = pd.read_csv(nome_arquivo, sep='#', quoting=csv.QUOTE_ALL)
    df_trt_csv.loc[:,'sigla_trt'] = "TRT"+sigla_trt;

    #Removendo dados que não serao necessarios nessa iteracao
    df_trt_csv.cd_assunto_nivel_3 = pd.to_numeric(df_trt_csv.cd_assunto_nivel_3)
    df_trt_filtrado = df_trt_csv[df_trt_csv.cd_assunto_nivel_3.isin(assuntos)]

    del(df_trt_csv)
    #Estratificando
    df_amostra = stratified_sample_df(df_trt_filtrado,'cd_assunto_nivel_3',nroElementos)

    return df_amostra.values.tolist()

results = []
def recupera_amostras_de_todos_regionais(listaAssuntos, nroElementos):

    start_time = time.time()

    pool = mp.Pool(processes=mp.cpu_count())
    # pool = mp.Pool(4)

    listaAssuntos=[2546]
    nroElementos=10000000
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

# listaAssuntos=[2546,2086,1855,2594,2458,2029,2140,2478,2704,2021,2426,2656,8808,1844,1663,2666,2506,55220,2055,1806,2139,1888,2435,2215,5280,2554,2583,55170,2019,2117,1661,1904,2540,55345]
# listaRegionais = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']
# df_amostra_final = recupera_amostras_de_todos_regionais(listaAssuntos,10)
# mostra_representatividade_regional_por_amostra(df_amostra_final)
# mostra_balanceamento_assunto(df_amostra_final['cd_assunto_nivel_3'].value_counts(), 'Balanceamento de classes -', 'Quantidade de documentos', 'Código Assunto')