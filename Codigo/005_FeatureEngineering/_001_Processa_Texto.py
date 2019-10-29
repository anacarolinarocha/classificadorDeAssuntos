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
        global stopwords_processadas
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


def processaTextosRegional(regionais,path_fonte_de_dados , path_destino_de_dados):
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


#----------------------------------------------------------------------------------------------------------------------
#Processando texto
#----------------------------------------------------------------------------------------------------------------------
stopwords_processadas = []
def processaDocumentos(path_fonte_de_dados, path_destino_de_dados, regionais=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]):
    print("Passando stopwords pelo pre processamento....")
    stopwords = nltk.corpus.stopwords.words('portuguese')
    global stopwords_processadas
    for row in stopwords:
        palavraProcessada = normalize('NFKD', row).encode('ASCII', 'ignore').decode('ASCII')
        stopwords_processadas.append(palavraProcessada)

    if not os.path.exists(path_destino_de_dados):
        os.makedirs(path_destino_de_dados)

    processaTextosRegional(regionais, path_fonte_de_dados, path_destino_de_dados)


# CODIGO DE TESTE
# path  = '/media/DATA/classificadorDeAssuntos/Dados/naoPublicavel/Documentos/'
# processaDocumenos(path,[8,9])

