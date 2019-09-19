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
import pandas as pd

tamanhos = []
path = '/media/DATA/classificadorDeAssuntos/Dados/naoPublicavel/ConferenciaDeAssuntos/OK/'

df_count_assuntos = pd.DataFrame()
for i in range(1, 25):
    sigla_trt="{:02d}".format(i)
    # sigla_trt = '08'
    # assuntos = [2546, 2086, 1855]
    # nroElementos=100
    print('Buscando dados para o TRT  ' + sigla_trt)
    nome_arquivo = path + 'TRT_' + sigla_trt + '_2G_2010-2019_documentosSelecionadosProcessados.csv'
    df_trt_csv = pd.read_csv(nome_arquivo, sep='#', quoting=csv.QUOTE_ALL)
    df_trt_csv.loc[:, 'sigla_trt'] = "TRT" + sigla_trt
    tamanhos.append((df_trt_csv.shape[0], 'TRT ' + sigla_trt))
    df_count = pd.DataFrame(df_trt_csv.groupby(['cd_assunto_nivel_3'], as_index  =False).count())
    df_count = df_count[['cd_assunto_nivel_3', 'index']]
    df_count_assuntos = df_count_assuntos.append(df_count)


df_trunc = df_count_assuntos.groupby(['cd_assunto_nivel_3']).index.sum()
df_trunc = pd.DataFrame(df_trunc)
df_trunc.to_csv("/home/anarocha/myGit/classificadorDeAssuntos/Codigo/Analises/contagem_assuntos.csv")

df_count_assuntos_n2 = pd.DataFrame()
for i in range(1, 25):
    sigla_trt="{:02d}".format(i)
    sigla_trt = '08'
    assuntos = [2546, 2086, 1855]
    nroElementos=100
    print('Buscando dados para o TRT  ' + sigla_trt)
    nome_arquivo = path + 'TRT_' + sigla_trt + '_2G_2010-2019_documentosSelecionadosProcessados.csv'
    df_trt_csv = pd.read_csv(nome_arquivo, sep='#', quoting=csv.QUOTE_ALL)
    df_trt_csv.loc[:, 'sigla_trt'] = "TRT" + sigla_trt
    tamanhos.append((df_trt_csv.shape[0], 'TRT ' + sigla_trt))
    df_count = pd.DataFrame(df_trt_csv.groupby(['cd_assunto_nivel_2'], as_index  =False).count())
    df_count = df_count[['cd_assunto_nivel_2', 'index']]
    df_count_assuntos_n2 = df_count_assuntos.append(df_count)


df_trunc_n2 = df_count_assuntos_n2.groupby(['cd_assunto_nivel_2']).index.sum()
df_trunc_n2 = pd.DataFrame(df_trunc_n2)
df_trunc_n2.to_csv("/home/anarocha/myGit/classificadorDeAssuntos/Codigo/Analises/contagem_assuntos.csv")