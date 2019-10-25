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
select (SELECT cd_sigla_tribunal from tb_tribunal) AS tribunal, tpd.cd_documento, tpd.ds_tipo_processo_documento,count(doc.id_processo_documento)
from tb_processo_documento doc 
inner join tb_tipo_processo_documento tpd on doc.id_tipo_processo_documento = tpd.id_tipo_processo_documento
where ((tpd.cd_documento in ('7149', '7154' , '7152' ,'63' ,'69') AND doc.ds_instancia = '1')
or (tpd.cd_documento = '58' AND doc.ds_instancia = '2'))
and dt_juntada IS NOT null
group by tribunal, tpd.cd_documento,tpd.ds_tipo_processo_documento
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


total = df.groupby(['ds_tipo_processo_documento']).sum()
total.plot(kind='bar')
plt.show()

total.to_csv('/home/anarocha/myGit/classificadorDeAssuntos/Codigo/001_Analises/Outputs/Total_Documentos_2_Grau.csv')

