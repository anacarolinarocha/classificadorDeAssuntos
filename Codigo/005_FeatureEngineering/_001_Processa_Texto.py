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
from unicodedata import normalize
import re
import warnings

# nltk.download('rslp')
# nltk.download('stopwords')
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

def processa_stemiza_texto(texto):
        textoProcessado = normalize('NFKD', texto).encode('ASCII','ignore').decode('ASCII')
        textoProcessado = re.sub('[^a-zA-Z]',' ',textoProcessado)
        textoProcessado = textoProcessado.lower()
        textoProcessado = textoProcessado.split()
        textoProcessado = [palavra for palavra in textoProcessado if not palavra in stopwords_processadas]
        textoProcessado = [palavra for palavra in textoProcessado if len(palavra)>3]
        textoProcessado =  [stemmer.stem(palavra) for palavra in textoProcessado]
        return ' '.join(word for word in textoProcessado)


def stemiza(texto):
    textoProcessado = [stemmer.stem(palavra) for palavra in texto.split()]
    return ' '.join(word for word in textoProcessado)
#----------------------------------------------------------------------------------------------------------------------
#Processando texto
#----------------------------------------------------------------------------------------------------------------------

def processaTextosRegional(regionais):
    for  sigla_trt in regionais:
        sigla_trt='02'
        print("----------------------------------------------------------------------------")
        print('Processando texto dos documentos do TRT ' + sigla_trt)
        nome_arquivo_origem = path + 'TRT_' + sigla_trt + '_2G_2010-2019_documentosSelecionados.csv'
        nome_arquivo_destino = path + 'TRT_' + sigla_trt + '_2G_2010-2019_documentosSelecionadosProcessados.csv'
        # for i in range (1,25):
        #     sigla_trt = "{:02d}".format(i)
        #     sigla_trt = '05'
        #     print('Processando texto dos documentos do TRT ' + sigla_trt)
        #     nome_arquivo_destino = path + 'TRT_' + sigla_trt + '_2G_2010-2019_documentosSelecionadosProcessados.csv'
        #     df = pd.read_csv(nome_arquivo_destino,sep='#', quoting=csv.QUOTE_ALL)
        #     print(df.shape)
        #     df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        #     df.to_csv(nome_arquivo_destino, sep='#', quoting=csv.QUOTE_ALL, index=False)
            # df = df.dropna(subset=['texto_stemizado'])
            # teste = df.texto_stemizado.isna()
            # df = df[]
            # print(df.shape)


        colnames = ['index','nr_processo', 'id_processo_documento', 'cd_assunto_nivel_5', 'cd_assunto_nivel_4','cd_assunto_nivel_3','cd_assunto_nivel_2','cd_assunto_nivel_1','tx_conteudo_documento','ds_identificador_unico','ds_identificador_unico_simplificado','ds_orgao_julgador','ds_orgao_julgador_colegiado','dt_juntada']
        df_trt = pd.read_csv(nome_arquivo_origem, sep=',', names=colnames, index_col=0, header=None, quoting=csv.QUOTE_ALL)
        # df_trt.head()
        # df_trt = df_trt.head(100)
        # df_trt.drop(df_trt[df_trt['cd_assunto_nivel_3'] == str('cd_assunto_nivel_3')].index, inplace=True)
        #pool = mp.Pool(mp.cpu_count())

        #remove as tags HTML
        start_time = time.time()
        pool = mp.Pool(7)
        df_trt = df_trt.dropna(subset=['tx_conteudo_documento'])
        df_trt['texto_processado'] = pool.map(removeHTML, [row for row in df_trt['tx_conteudo_documento']])
        pool.close()
        total_time = time.time() - start_time
        print('Tempo para processamento do texto:' + str(timedelta(seconds=total_time)))

        #remove textos muito pequenos
        df_trt = df_trt[df_trt['texto_processado'].map(len) > 100]

        #faz a stemizacao
        start_time = time.time()
        pool = mp.Pool(7)
        df_trt = df_trt.dropna(subset=['texto_processado'])
        df_trt['texto_stemizado'] = pool.map(processa_stemiza_texto, [row for row in df_trt['texto_processado']])
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

# print("Passando stopwords pelo pre processamento....")
# stopwords = nltk.corpus.stopwords.words('portuguese')
#
# stopwords_processadas = []
# for row in stopwords:
#     palavraProcessada = normalize('NFKD', row).encode('ASCII', 'ignore').decode('ASCII')
#     stopwords_processadas.append(palavraProcessada)



# processaTextosRegional(['19'])
#for i in range (5,25):
#    processaTextosRegional([("{:02d}".format(i))])
