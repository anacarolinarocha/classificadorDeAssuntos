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

def recupera_n_amostras_por_assunto_por_regional(regionais, assuntos, nroElementos, percentualTeste):

    """
    Função que, dada uma lista de regionais, uma lista de assuntos, e definida a a quantidade de amostras de cada item,
    busca o arquivo com os documentos do regional informado e retira o número de elementos de dada assunto de cada regional.
    Será armazenado no arquivo quais foram os elementos selecinados, indetificando entre elementos separados para treinamento
    e elementos separados para teste
    :param regionais: sigla dos regionais onde se deve buscar os dados
    :param assuntos: lista de assuntos a se buscar
    :param quantidadeAmostras: quantidade de elementos de cada assunto de cada regional. Se não existir a quantidade demandada, irá limitar a quantidade retornada por classe
    em funçao do assunto que tiver o menor número de representantes.
    :param percentualTeste: indica qual percentual da amostra sera usado para teste
    //TODO: implementar bootstrap para fazer oversamplig
    :return:
    """
    path = '/media/DATA/classificadorDeAssuntos/Dados/naoPublicavel/ConferenciaDeAssuntos/OK/'
    df_amostras_trts = pd.DataFrame()
    print('Buscando dados para amostra....')
    for  sigla_trt in regionais:
        # sigla_trt='01'
        # assuntos = [2546, 2086, 1855]
        # nroElementos=100
        # percentualTeste=0.3
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

        # mostra_representatividade_regional_por_amostra(df_amostra)

        #Marcado treinamento e teste
        # train, test = train_test_split(df_amostra, test_size=percentualTeste, stratify=df_amostra['cd_assunto_nivel_3'])

        # warnings.filterwarnings("ignore")
        # train.loc[:,'in_selecionando_para_amostra'] ='Treinamento'
        # test.loc[:,'in_selecionando_para_amostra'] ='Teste'
        # df_amostras_trts = df_amostras_trts.append(train)
        # df_amostras_trts = df_amostras_trts.append(test)
        # warnings.filterwarnings("default")

        df_amostras_trts = df_amostras_trts.append(df_amostra)



    return df_amostras_trts
# warnings.filterwarnings("default")

# listaAssuntos=[2546,2086,1855,2594,2458,2029,2140,2478,2704,2021,2426,2656,8808,1844,1663,2666,2506,55220,2055,1806,2139,1888,2435,2215,5280,2554,2583,55170,2019,2117,1661,1904,2540,55345]
# listaRegionais = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']
# # listaAssuntos=[2546,2086]
# df_amostra_final = recupera_n_amostras_por_assunto_por_regional(listaRegionais,listaAssuntos,10,0.3)
# mostra_representatividade_regional_por_amostra(df_amostra_final)
# mostra_balanceamento_assunto(df_amostra_final['cd_assunto_nivel_3'].value_counts(), 'Balanceamento de classes -', 'Quantidade de documentos', 'Código Assunto')