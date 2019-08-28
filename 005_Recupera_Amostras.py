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

# -----------------------------------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------------------------------
path_base_dados_retificados_processados= '/mnt/04E61847E6183AFE/classificadorDeAssuntos/Dados/naoPublicavel/DocumenosTRTsRetificadosProcessados/'
listaAssuntos=['2546','2086','1855','2594','2458','2029','2140','2478','2704','2021','2426','2656','8808','1844','1663','2666','2506','55220','2055','1806','2139','1888','2435','2215','5280','2554','2583','55170','2019','2117','1661','1904','2540','55345']
listaAssuntos=[2546,2086,1855]

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

def recupera_n_amostras_por_assunto_por_regional(regionais):
    for  sigla_trt in regionais:
        # sigla_trt='20'
        nome_arquivo= path_base_dados_retificados_processados + 'listaDocumentosNaoSigilososRetificadosProcessados_MultiClasse_TRT' + sigla_trt + '_2G_2010-2019.csv'
        df_trt = pd.read_csv(nome_arquivo, sep='#', quoting=csv.QUOTE_ALL)
        # df_trt = df_trt.dropna()
        df_trt = df_trt[df_trt.cd_assunto_nivel_3.isin(listaAssuntos)]
        df_trt.shape

        df_trt['in_selecionando_para_amostra']='N'
        df_amostras_trts=pd.DataFrame()
        #df_temp=df_trt[(df_trt['cd_assunto_nivel_3'] == assunto) & (df_trt['in_selecionando_para_amostra'] == 'N')].head(100)
        #df_trt.loc[(df_trt['cd_assunto_nivel_3'] == assunto) & (df_trt['in_selecionando_para_amostra'] == 'N'), 'in_selecionando_para_amostra'] = 'S'
        df_amostra= stratified_sample_df(df_trt,'cd_assunto_nivel_3',10)
        df_amostra.shape

        df_amostras_trts=df_amostras_trts.append(df_amostra)

        pd.DataFrame.from_dict(Counter(df_amostras_trts['cd_assunto_nivel_3']), orient='index').sort_values(by=[0], ascending=False).plot(kind='bar')
        plt.show()


recupera_n_amostras_por_assunto_por_regional(['20'])