# ### ====================================================================================
# Script que processa os textos recuperados
# ### ====================================================================================
import csv
import multiprocessing as mp
import sys
import time
import nltk
import os
import numpy as np
import pandas as pd
from datetime import timedelta
import lxml
from bs4 import BeautifulSoup

# -----------------------------------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------------------------------
path_base_dados_originais= '/mnt/04E61847E6183AFE/classificadorDeAssuntos/Dados/naoPublicavel/DocumenosTRTs/'
path_base_dados_retificados_processados= '/mnt/04E61847E6183AFE/classificadorDeAssuntos/Dados/naoPublicavel/DocumenosTRTsRetificadosProcessados/'

if not os.path.exists(path_base_dados_retificados_processados):
    os.makedirs(path_base_dados_retificados_processados)

#----------------------------------------------------------------------------------------------------------------------
def removeHTML(texto):
    """
    Função para remover HTML
    :param texto:
    :return: texto sem tags HTML
    """
    return BeautifulSoup(texto, 'lxml').get_text(strip=True)

#----------------------------------------------------------------------------------------------------------------------
# Organiza colunas que ficaram com o nome errado....
#----------------------------------------------------------------------------------------------------------------------
sigla_trt='01'
nome_arquivo= path_base_dados_originais + 'listaDocumentosNaoSigilosos_MultiLabel_TRT' + sigla_trt + '_2G_2010-2019.csv'
df_trt = pd.read_csv(nome_arquivo, sep=',', quoting=csv.QUOTE_ALL)
df_trt.shape
df_trt.columns
df_trt = df_trt.set_axis([ 'Unnamed: 0','ds_orgao_julgador', 'ds_orgao_julgador_colegiado', 'nr_processo', 'id_processo_documento', 'codigo_documento', 'dt_juntada', 'ds_modelo_documento', 'cd_assunto_nivel_3', 'in_processo_retificado'], axis=1, inplace=False)
df_trt = df_trt.dropna(subset=['Unnamed: 0'])
df_trt.shape
df_trt.columns
df_trt = df_trt.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
df_trt = df_trt.drop(columns=['Unnamed: 0'])
df_trt.columns
df_trt['in_selecionando_para_amostra']='N'
df_trt_csv['sigla_trt'] = "TRT"+sigla_trt;
df_trt.to_csv(path_base_dados_originais + 'listaDocumentosNaoSigilosos_MultiLabel_TRT' + sigla_trt + '_2G_2010-2019.csv',sep=',', quoting=csv.QUOTE_ALL)
#----------------------------------------------------------------------------------------------------------------------
#Processando texto
#----------------------------------------------------------------------------------------------------------------------

def processaTextosRegional(regionais):
    for  sigla_trt in regionais:

        nome_arquivo= path_base_dados_originais + 'listaDocumentosNaoSigilosos_MultiLabel_TRT' + sigla_trt + '_2G_2010-2019.csv'
        df_trt = pd.read_csv(nome_arquivo, sep=',', quoting=csv.QUOTE_ALL)
        df_trt = df_trt[df_trt['in_processo_retificado'] == 'S']
        df_trt.shape
        #pool = mp.Pool(mp.cpu_count())
        pool = mp.Pool(7)
        start_time = time.time()
        df_trt = df_trt.dropna(subset=['ds_modelo_documento'])
        df_trt['texto_processado'] = pool.map(removeHTML, [row for row in df_trt['ds_modelo_documento']])
        pool.close()
        total_time = time.time() - start_time
        print('Tempo para processamento do texto:' + str(timedelta(seconds=total_time)))
        df_trt.shape

        df_trt.to_csv(path_base_dados_retificados_processados + 'listaDocumentosNaoSigilososRetificadosProcessados_MultiClasse_TRT' + sigla_trt + '_2G_2010-2019.csv', sep='#', quoting=csv.QUOTE_ALL)


        del(df_trt)

processaTextosRegional(['21'])