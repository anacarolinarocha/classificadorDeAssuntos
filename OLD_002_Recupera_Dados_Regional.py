# ### ====================================================================================
# Script que recupara os dados dos regionais e salva em CSV
# ### ====================================================================================

from datetime import timedelta
import sys
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
import time
import csv

sys.path.insert(1, '/home/anarocha/Documents/credentials')
from credentials import *

with open('/home/anarocha/myGit/classificadorDeAssuntos/Scripts/001-Consultas/002-SelectDocumentosMultiClasse_com_fdw.sql',
          'r') as file:
    sql = file.read().replace('\n', ' ')

def recuperaDadosRegional():

    for sigla_trt in range(21, 25):
        sigla_trt='{:02d}'.format(sigla_trt)
        print("----------------------------------------------------------------------------")
        print("PROCESSANDO DADOS DO TRT {} - 2o GRAU".format(sigla_trt))

        porta = '5' + sigla_trt + '2'

        engine = create_engine(credentials["TRT" + sigla_trt + "-2G"])

        try:
            engine.connect();
        except:
            print("\033[91mNão foi possível se conectar na base do  TRT " + sigla_trt + "\033[0m")
            continue

        start_time = time.time()
        try:
            documentosRecuperados = pd.read_sql_query(sqlalchemy.text(sql.format(ipbugfix,porta,dbname,userbugfix,senhabugfix)), engine)
        except:
            print("\033[91mErro ao recuperar dados \033[0m")
            continue
        total_time = time.time() - start_time
        print('\nTempo para recuperar dados: '+ str(timedelta(seconds=total_time)))
        print(str(documentosRecuperados.shape[0])+ ' documentos recuperados! ')
        documentosRecuperados.to_csv('/mnt/04E61847E6183AFE/classificadorDeAssuntos/Dados/naoPublicavel/DocumenosTRTs/listaDocumentosNaoSigilosos_MultiLabel_TRT' + sigla_trt  + '_2G_2010-2019.csv', sep='#',
                     quoting=csv.QUOTE_ALL)

recuperaDadosRegional()

