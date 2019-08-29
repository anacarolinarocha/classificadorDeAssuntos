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
path_base_dados_retificados_processados= '/mnt/04E61847E6183AFE/classificadorDeAssuntos/Dados/naoPublicavel/DocumenosTRTsRetificadosProcessados/'
# -----------------------------------------------------------------------------------------------------
# Função que recuperar amostra estratificada pelo codigo de assunto dentre todos os codigos existentes no dataset
def stratified_sample_df(df, col, n_samples):
    n = min(n_samples, df[col].value_counts().min())
    df_ = df.groupby(col).apply(lambda x: x.sample(n))
    df_.index = df_.index.droplevel(0)
    return df_


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
    df_amostras_trts = pd.DataFrame()
    for  sigla_trt in regionais:
        # sigla_trt='21'
        # assuntos = [2546, 2086, 1855]
        # nroElementos=10
        # percentualTeste=0.3
        nome_arquivo= path_base_dados_retificados_processados + 'listaDocumentosNaoSigilososRetificadosProcessados_MultiClasse_TRT' + sigla_trt + '_2G_2010-2019.csv'
        df_trt_csv = pd.read_csv(nome_arquivo, sep='#', quoting=csv.QUOTE_ALL)
        df_trt_csv.loc[:,'in_selecionando_para_amostra']='N'
        df_trt_csv.loc[:,'sigla_trt'] = "TRT"+sigla_trt;

        df_trt_filtrado = df_trt_csv[df_trt_csv.in_selecionando_para_amostra == 'N']
        df_trt_filtrado = df_trt_filtrado[df_trt_filtrado.cd_assunto_nivel_3.isin(assuntos)]

        df_amostra = stratified_sample_df(df_trt_filtrado,'cd_assunto_nivel_3',nroElementos)
        df_amostra = df_amostra.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])

        train, test = train_test_split(df_amostra, test_size=percentualTeste, stratify=df_amostra['cd_assunto_nivel_3'])

        warnings.filterwarnings("ignore")

        train.loc[:,'in_selecionando_para_amostra'] ='Treinamento'
        df_trt_csv.loc[df_trt_csv.index.isin(train.index), 'in_selecionando_para_amostra'] = 'Treinamento'

        test.loc[:,'in_selecionando_para_amostra'] ='Teste'
        df_trt_csv.loc[df_trt_csv.index.isin(test.index), 'in_selecionando_para_amostra'] = 'Teste'


        df_amostras_trts = df_amostras_trts.append(train)
        df_amostras_trts = df_amostras_trts.append(test)

        warnings.filterwarnings("default")
        df_trt_csv.to_csv(
            path_base_dados_retificados_processados + 'listaDocumentosNaoSigilososRetificadosProcessados_MultiClasse_TRT' + sigla_trt + '_2G_2010-2019.csv',
            sep='#', quoting=csv.QUOTE_ALL)

    return df_amostras_trts
warnings.filterwarnings("ignore")
# warnings.filterwarnings("default")

#listaAssuntos=['2546','2086','1855','2594','2458','2029','2140','2478','2704','2021','2426','2656','8808','1844','1663','2666','2506','55220','2055','1806','2139','1888','2435','2215','5280','2554','2583','55170','2019','2117','1661','1904','2540','55345']
listaAssuntos=[2546,2086,1855]
df_amostra_final = recupera_n_amostras_por_assunto_por_regional(['21','18'],listaAssuntos,50,0.3)

# pd.DataFrame.from_dict(Counter(df_amostra_final['in_selecionando_para_amostra']), orient='index').sort_values(by=[0], ascending=False).plot(kind='bar')
# plt.show()

df_amostra_final_trunc = df_amostra_final[['sigla_trt','cd_assunto_nivel_3']]
df_tmp = df_amostra_final_trunc.groupby(['sigla_trt','cd_assunto_nivel_3']).cd_assunto_nivel_3.count().to_frame()
df_tmp = sub_teste.unstack()
df_tmp.columns = df_tmp.columns.droplevel()  # Drop `cd_assunto_nivel_3` label.
df_tmp = df_tmp.div(df_tmp.sum())
df_tmp.T.plot(kind='bar', stacked=True, rot=1, figsize=(8, 8),
              title="Distribuição de assuntos na amostra por tribunal")
plt.show()

#//TODO: Ajustar esse grafico para que ele traga o nome do assunto ao inves do codigo