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


#SELECT QUE BUSCA OS PROCESSOS QUE FORAM REMETIDOS AO 2 GRAU E SEUS ASSUNTOS EM PRIMEIRO GRAU
sql_original = """
with totalprocessos as 
	(select  count(*) as total from tb_processo where nr_processo is not null and nr_processo <> '')
select (select cd_sigla_tribunal from tb_tribunal) as tribunal,
	'Total de processsos com um assunto', 
	count(*) as quantidade, 
	(select total from totalprocessos) as totalprocessos, 
	round(CAST(float8 (count(*)/(select  total::numeric from totalprocessos)) as numeric), 2) as percentual
	FROM (SELECT id_processo_trf, count(id_assunto_trf) FROM tb_processo_assunto GROUP BY id_processo_trf HAVING count(id_assunto_trf) = 1) as t
union all
(with totalprocessos as 
	(select  count(*) as total from tb_processo where nr_processo is not null and nr_processo <> '')
select (select cd_sigla_tribunal from tb_tribunal) as tribunal,
	'Total de processsos com mais de um assunto', 
	count(*) as quantidade, 
	(select total from totalprocessos) as totalprocessos, 
	round(CAST(float8 (count(*)/(select  total::numeric from totalprocessos)) as numeric), 2) as percentual
	FROM (SELECT id_processo_trf, count(id_assunto_trf) FROM tb_processo_assunto GROUP BY id_processo_trf HAVING count(id_assunto_trf) > 1) as v)
"""
sql_original = sql_original.replace('\n', ' ')
sql_original = sql_original.replace('\t', ' ')


nomeArquivo = '/home/anarocha/myGit/classificadorDeAssuntos/Planilhas/ContagemPercentualProcessosComUmAssuntoNoSegundoGrau.xlsx'
    if os.path.isfile(nomeArquivo):
        os.remove(nomeArquivo)
df = pd.DataFrame()
def recuperaDadosRegional(regionais):
    for  sigla_trt in regionais:
        global df
        # sigla_trt='19'
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
            df = df.append(df_temp)
            total_time = time.time() - start_time
            print('\nTempo para recuperar dados: ' + str(timedelta(seconds=(total_time))))
        except Exception as e:
            print("\033[91mNão foi possível se conectar na base do  TRT " + sigla_trt + "\033[0m")
            print(e)


for i in range (1,25):
    recuperaDadosRegional([("{:02d}".format(i))])
# dados = recuperaDadosRegional(['19','20']) # 11, 22
df.to_excel(nomeArquivo)
