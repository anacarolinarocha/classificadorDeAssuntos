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

#Carrega SQLs
#SQl que cria a conexão com a base log
with open('/home/anarocha/myGit/classificadorDeAssuntos/Scripts/001-Consultas/002-SelectDocumentosMultiClasse.sql',
          'r') as file:
    sql_original = file.read().replace('\n', ' ')

#SQL com select
with open('/home/anarocha/myGit/classificadorDeAssuntos/Scripts/001-Consultas/005_CriaExtensaoParaBaseLog.sql',
          'r') as file:
    sql_fdw_original = file.read().replace('\n', ' ')


#6 ok (talvez nao tenha ido tudo...)
#7 erro de conversao
#8 indisponivel

def recuperaDadosRegional(regionais):

    for  sigla_trt in regionais:
        # sigla_trt='14'
        print("----------------------------------------------------------------------------")
        print("PROCESSANDO DADOS DO TRT {} - 2o GRAU".format(sigla_trt))
        start_time = time.time()
        porta = '5' + sigla_trt + '2'
        try:
            # Cria conexao com banco
            sql_fdw = sql_fdw_original.format(ipbugfix, porta, dbname_2g_log, userbugfix, senhabugfix)
            conn = psycopg2.connect(dbname=dbname_2g, user=userbugfix, password=senhabugfix, host=ipbugfix, port=porta)
            cur = conn.cursor()
            cur.execute(sql_fdw)
            conn.commit()

            if os.path.isfile('/mnt/04E61847E6183AFE/classificadorDeAssuntos/Dados/naoPublicavel/DocumenosTRTs/listaDocumentosNaoSigilosos_MultiLabel_TRT' + sigla_trt  + '_2G_2010-2019.csv'):
                os.remove('/mnt/04E61847E6183AFE/classificadorDeAssuntos/Dados/naoPublicavel/DocumenosTRTs/listaDocumentosNaoSigilosos_MultiLabel_TRT' + sigla_trt  + '_2G_2010-2019.csv')
            chunk_size = 5000
            offset = 0
            dfs=[]
            while True:
            # for i in range(1,5):
                sql = sql_original +"limit %d offset %d" % (chunk_size,offset)
                dfs.append(psql.read_sql(sql, conn))
                if offset == 0 :
                    print(sql[-50:])
                    dfs[-1].to_csv('/mnt/04E61847E6183AFE/classificadorDeAssuntos/Dados/naoPublicavel/DocumenosTRTs/listaDocumentosNaoSigilosos_MultiLabel_TRT' + sigla_trt  + '_2G_2010-2019.csv', mode='a', header=False, quoting=csv.QUOTE_ALL)
                else:
                    dfs[-1].to_csv('/mnt/04E61847E6183AFE/classificadorDeAssuntos/Dados/naoPublicavel/DocumenosTRTs/listaDocumentosNaoSigilosos_MultiLabel_TRT' + sigla_trt  + '_2G_2010-2019.csv', mode='a', header=True, quoting=csv.QUOTE_ALL)
                offset += chunk_size
                if len(dfs[-1]) < chunk_size:
                    print(sql[-50:])
                    break;
            print('\nTempo para recuperar dados: ' + str(timedelta(seconds=(total_time))))
        except:
            print("\033[91mNão foi possível se conectar na base do  TRT " + sigla_trt + "\033[0m")
            continue;

recuperaDadosRegional(['19','05','08','12'])

