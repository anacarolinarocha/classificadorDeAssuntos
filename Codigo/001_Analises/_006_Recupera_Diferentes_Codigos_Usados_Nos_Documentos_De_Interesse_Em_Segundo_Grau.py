# ### ====================================================================================
# Script que varre tdoas as bases buscando os codigos dos documentos de interesse e verifica
# quais sao os códigos que se deve buscar de maneira global
# ### ====================================================================================

from datetime import timedelta
import sys
import pandas as pd
import pandas.io.sql as psql
import time
import csv
import psycopg2
import os
import matplotlib.pyplot as plt

sys.path.insert(1, '/home/anarocha/Documents/credentials')
from credentials import *


#SELECT QUE BUSCA OS PROCESSOS QUE FORAM REMETIDOS AO 2 GRAU E SEUS ASSUNTOS EM PRIMEIRO GRAU
sql_original = """
SELECT ds_tipo_processo_documento, cd_documento,in_ativo FROM tb_tipo_processo_documento WHERE ds_tipo_processo_documento ilike any (array['Peti__o inicial', 'Recurso Ordin_rio', 'Agravo de peti__o','Recurso adesivo','Agravo de Instrumento em Agravo de Peti__o','Agravo de Instrumento em Recurso Ordin_rio']);
"""
sql_original = sql_original.replace('\n', ' ')
sql_original = sql_original.replace('\t', ' ')


df = pd.DataFrame()
for i in range(1, 25):
    sigla_trt = "{:02d}".format(i)

    print("----------------------------------------------------------------------------")
    print("PROCESSANDO DADOS DO TRT {} - 2o GRAU".format(sigla_trt))
    start_time = time.time()
    porta = '5' + sigla_trt + '2'

    try:
        if (sigla_trt != '20'):
            conn = psycopg2.connect(dbname=dbname_2g, user=userbugfix, password=senhabugfix, host=ipbugfix,
                                    port=porta)
        else:
            conn = psycopg2.connect(dbname='pje_2grau_consulta', user=userbugfix, password=senhabugfix,
                                    host=ipbugfix,
                                    port=porta)
        # conn = psycopg2.connect(dbname='pje_1grau_consulta', user=userbugfix, password=senhabugfix, host=ipbugfix, port=porta)
        start_time = time.time()
        df_temp = psql.read_sql(sql_original, conn)
        df = df.append(df_temp)
        total_time = time.time() - start_time
        print('\nTempo para recuperar dados: ' + str(timedelta(seconds=(total_time))))
    except Exception as e:
        print("\033[91mNão foi possível se conectar na base do  TRT " + sigla_trt + "\033[0m")
        print(e)

# TRT 15  E TRT 24 ESTAVAM FORA
print(df.cd_documento.unique())
#['7149' '7154' '58' '7152' '63' '69']
df.ds_tipo_processo_documento.unique()
#['Agravo de Instrumento em Agravo de Petição', 'Agravo de Petição','Petição Inicial', 'Agravo de Instrumento em Recurso Ordinário', 'Recurso Adesivo', 'Recurso Ordinário']

