# ### ====================================================================================
# Script que verifica qual é o percentual de processos que contem somente um assunto e
# qual é o percentual que contem mais de um assunto
# ### ====================================================================================

from datetime import timedelta
import sys
import pandas as pd
import pandas.io.sql as psql
import time
import csv
import psycopg2
import os

sys.path.insert(1, '/home/anarocha/Documents/credentials')
from credentials import *

sql_original = """
select (select cd_sigla_tribunal from tb_tribunal) as sigla_trt, (select count(id_processo_documento) as total
from tb_processo_documento doc 
inner join tb_processo_documento_bin docbin on doc.id_processo_documento_bin = docbin.id_processo_documento_bin
where id_tipo_processo_documento = (select id_tipo_processo_documento from tb_tipo_processo_documento where ds_tipo_processo_documento like 'Recurso Ordin_rio')
and docbin.ds_extensao is null) as total_html,
(select count(id_processo_documento) as total
from tb_processo_documento doc 
inner join tb_processo_documento_bin docbin on doc.id_processo_documento_bin = docbin.id_processo_documento_bin
where id_tipo_processo_documento = (select id_tipo_processo_documento from tb_tipo_processo_documento where ds_tipo_processo_documento like 'Recurso Ordin_rio')
and docbin.ds_extensao is not null) as total_pdf

"""
sql_original = sql_original.replace('\n', ' ')
sql_original = sql_original.replace('\t', ' ')

#SELECT QUE BUSCA OS PROCESSOS QUE FORAM REMETIDOS AO 2 GRAU E SEUS ASSUNTOS EM PRIMEIRO GRAU


df_count_processos=pd.DataFrame()

for  sigla_trt in range (1,25):
    sigla_trt="{:02d}".format(sigla_trt)
    print("----------------------------------------------------------------------------")
    print("PROCESSANDO DADOS DO TRT {} - 2o GRAU".format(sigla_trt))
    porta = '5' + sigla_trt + '2'
    try:
        if(sigla_trt != '20'):
            conn = psycopg2.connect(dbname=dbname_2g, user=userbugfix, password=senhabugfix, host=ipbugfix, port=porta)
        else:
            conn = psycopg2.connect(dbname='pje_2grau_consulta', user=userbugfix, password=senhabugfix, host=ipbugfix,
                                    port=porta)
        # conn = psycopg2.connect(dbname='pje_1grau_consulta', user=userbugfix, password=senhabugfix, host=ipbugfix, port=porta)
        start_time = time.time()
        df_temp = psql.read_sql(sql_original, conn)
        df_count_processos = df_count_processos.append(df_temp)
        total_time = time.time() - start_time
        print('\nTempo para recuperar dados: ' + str(timedelta(seconds=(total_time))))
    except Exception as e:
        print("\033[91mNão foi possível se conectar na base do  TRT " + sigla_trt + "\033[0m")
        print(e)

df_count_processos.to_csv('/home/anarocha/myGit/classificadorDeAssuntos/Codigo/001_Analises/Outputs/Total_De_Processos_X_Total_Elegivel.csv')