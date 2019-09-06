# ### ====================================================================================
# Script que recupara os dados dos regionais e salva em CSV
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
sql_original = """select p1g.nr_processo as processo_1g, a1g.cd_assunto_trf as cd_assunto_1g
from pje.tb_processo_assunto pa1g
inner join pje.tb_assunto_trf a1g on a1g.id_assunto_trf = pa1g.id_assunto_trf
inner join pje.tb_processo p1g on p1g.id_processo = pa1g.id_processo_trf
where p1g.id_processo in (select id_processo_trf from pje.tb_manifestacao_processual where cd_origem ilike '%envio' )"""
sql_original = sql_original.replace('\n', ' ')

def recuperaDadosRegional(regionais):

    for  sigla_trt in regionais:
        # sigla_trt='19'
        print("----------------------------------------------------------------------------")
        print("PROCESSANDO DADOS DO TRT {} - 1o GRAU".format(sigla_trt))
        start_time = time.time()
        porta = '5' + sigla_trt + '1'
        nomeArquivo = '/media/DATA/classificadorDeAssuntos/Dados/naoPublicavel/ConferenciaDeAssuntos/TRT_' + sigla_trt  + '_1G_2010-2019_listaAssuntosProcessosRemetidosAoSegundoGrauComAssuntos.csv'

        try:
            conn = psycopg2.connect(dbname=dbname_1g, user=userbugfix, password=senhabugfix, host=ipbugfix, port=porta)

            if os.path.isfile(nomeArquivo):
                os.remove(nomeArquivo)
            chunk_size = 1000
            offset = 0
            dfs=[]
            while True:
            # for i in range(1,5):
                sql = sql_original +"limit %d offset %d" % (chunk_size,offset)
                dfs.append(psql.read_sql(sql, conn))
                if offset == 0 :
                    print('Primeiros dados recuperados ...' + sql[-50:])
                    dfs[-1].to_csv(nomeArquivo, mode='a', header=True, quoting=csv.QUOTE_ALL)
                else:
                    dfs[-1].to_csv(nomeArquivo, mode='a', header=False, quoting=csv.QUOTE_ALL)
                offset += chunk_size
                if len(dfs[-1]) < chunk_size:
                    print('Dados recuperados com sucesso.' )
                    break;
            total_time = time.time() - start_time
            print('\nTempo para recuperar dados: ' + str(timedelta(seconds=(total_time))))
        except Exception as e:
            print("\033[91mNão foi possível se conectar na base do  TRT " + sigla_trt + "\033[0m")
            print(e)
            continue;

for i in range (1,5):
    recuperaDadosRegional([("{:02d}".format(i))])
recuperaDadosRegional(['18','19','20'])

