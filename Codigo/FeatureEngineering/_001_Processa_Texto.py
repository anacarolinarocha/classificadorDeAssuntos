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
import warnings

# nltk.download('rslp')
stemmer = nltk.stem.RSLPStemmer()
# -----------------------------------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------------------------------
path = '/media/DATA/classificadorDeAssuntos/Dados/naoPublicavel/ConferenciaDeAssuntos/OK/'

if not os.path.exists(path):
    os.makedirs(path)

#----------------------------------------------------------------------------------------------------------------------
def removeHTML(texto):
    """
    Função para remover HTML
    :param texto:
    :return: texto sem tags HTML
    """
    texto = texto.replace('\n', ' ')
    texto = texto.replace('\t', ' ')
    return BeautifulSoup(texto, 'lxml').get_text(" ", strip=True)

def stemiza(texto):
    textoProcessado = [stemmer.stem(palavra) for palavra in texto.split()]
    return ' '.join(word for word in textoProcessado)
#----------------------------------------------------------------------------------------------------------------------
# Dropa coluna de HTML
#----------------------------------------------------------------------------------------------------------------------
# s#igla_trt='18'
# nome_arquivo= path_base_dados_retificados_processados + 'listaDocumentosNaoSigilososRetificadosProcessados_MultiClasse_TRT' + sigla_trt + '_2G_2010-2019.csv'
# df_trt = pd.read_csv(nome_arquivo, sep='#', quoting=csv.QUOTE_ALL)
# df_trt = df_trt.drop(columns=['ds_modelo_documento'])
# df_trt.to_csv(nome_arquivo,sep='#', quoting=csv.QUOTE_ALL)
#----------------------------------------------------------------------------------------------------------------------
# Organiza colunas que ficaram com o nome errado....
#----------------------------------------------------------------------------------------------------------------------
#sigla_trt='22'
# nome_arquivo= path_base_dados_originais + 'listaDocumentosNaoSigilosos_MultiLabel_TRT' + sigla_trt + '_2G_2010-2019.csv'
# df_trt = pd.read_csv(nome_arquivo, sep=',', quoting=csv.QUOTE_ALL)
# df_trt.shape
# df_trt.columns
# df_trt = df_trt.drop(columns=['ds_modelo_documento'])
# df_trt = df_trt.set_axis([ 'Unnamed: 0','ds_orgao_julgador', 'ds_orgao_julgador_colegiado', 'nr_processo', 'id_processo_documento', 'codigo_documento', 'dt_juntada', 'ds_modelo_documento', 'cd_assunto_nivel_3', 'in_processo_retificado'], axis=1, inplace=False)
# df_trt = df_trt.dropna(subset=['Unnamed: 0'])
# df_trt.shape
# df_trt.columns
# df_trt = df_trt.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
# df_trt = df_trt.drop(columns=['Unnamed: 0'])
# df_trt.columns
# df_trt['in_selecionando_para_amostra']='N'
# df_trt_csv['sigla_trt'] = "TRT"+sigla_trt;
# df_trt.to_csv(nome_arquivo,sep=',', quoting=csv.QUOTE_ALL)
#----------------------------------------------------------------------------------------------------------------------
#Processando texto
#----------------------------------------------------------------------------------------------------------------------

def processaTextosRegional(regionais):
    for  sigla_trt in regionais:
        # sigla_trt='07'
        print("----------------------------------------------------------------------------")
        print('Processando texto dos documentos do TRT ' + sigla_trt)
        nome_arquivo_origem = path + 'TRT_' + sigla_trt + '_2G_2010-2019_documentosSelecionados.csv'
        nome_arquivo_destino = path + 'TRT_' + sigla_trt + '_2G_2010-2019_documentosSelecionadosProcessados.csv'


        colnames = ['index','nr_processo', 'id_processo_documento', 'cd_assunto_nivel_5', 'cd_assunto_nivel_4','cd_assunto_nivel_3','cd_assunto_nivel_2','cd_assunto_nivel_1','tx_conteudo_documento','ds_identificador_unico','ds_identificador_unico_simplificado','ds_orgao_julgador','ds_orgao_julgador_colegiado','dt_juntada']
        df_trt = pd.read_csv(nome_arquivo_origem, sep=',', names=colnames, index_col=0, header=None, quoting=csv.QUOTE_ALL)
        # df_trt.head()
        # df_trt = df_trt.head(100)
        # df_trt.drop(df_trt[df_trt['cd_assunto_nivel_3'] == str('cd_assunto_nivel_3')].index, inplace=True)
        #pool = mp.Pool(mp.cpu_count())

        #remove as tags HTML
        start_time = time.time()
        pool = mp.Pool(1)
        df_trt = df_trt.dropna(subset=['tx_conteudo_documento'])
        df_trt['texto_processado'] = pool.map(removeHTML, [row for row in df_trt['tx_conteudo_documento']])
        pool.close()
        total_time = time.time() - start_time
        print('Tempo para processamento do texto:' + str(timedelta(seconds=total_time)))

        #remove textos muito pequenos
        df_trt = df_trt[df_trt['texto_processado'].map(len) > 100]

        #faz a stemizacao
        start_time = time.time()
        pool = mp.Pool(1)
        df_trt = df_trt.dropna(subset=['texto_processado'])
        df_trt['texto_stemizado'] = pool.map(stemiza, [row for row in df_trt['texto_processado']])
        pool.close()
        total_time = time.time() - start_time
        print('Tempo para stemização do texto:' + str(timedelta(seconds=total_time)))

        # # verifica o conteudo de um documento
        # f = open("./teste.html", "w")
        # f.write(df_trt.iloc[0]['tx_conteudo_documento'])
        # f.close()
        # import webbrowser
        # webbrowser.get('firefox').open_new_tab('./teste.html')


        df_trt = df_trt.drop(columns=['tx_conteudo_documento'])
        print("Encontrados " + str(df_trt.shape[0]) + " documentos para o TRT " + sigla_trt)

        if os.path.isfile(nome_arquivo_destino):
            os.remove(nome_arquivo_destino)

        df_trt.to_csv(nome_arquivo_destino, sep='#', quoting=csv.QUOTE_ALL)

# processaTextosRegional(['16','17'])
# for i in range (2,7):
#     processaTextosRegional([("{:02d}".format(i))])
