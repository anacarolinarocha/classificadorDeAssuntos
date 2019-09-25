

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import timedelta
import pandas.io.sql as psql
import time
import psycopg2
sys.path.insert(1, '/home/anarocha/Documents/credentials')
from credentials import *
import os
import csv

sys.path.insert(1, '/home/anarocha/myGit/classificadorDeAssuntos/Codigo/ETL')
from _004_Cria_Tabela_Postgres_From_Dataframe import *

path = '/media/DATA/classificadorDeAssuntos/Dados/naoPublicavel/Contingencia/Documentos2Grau/'

sql_original = """select p.nr_processo, 
doc.id_processo_documento,
tpd.id_tipo_processo_documento, 
tpd.ds_tipo_processo_documento,
docbin.ds_modelo_documento,
doc.ds_identificador_unico, 
right(doc.ds_identificador_unico,7)::text as codigo_documento_simplificado,
oj.ds_orgao_julgador,
ojc.ds_orgao_julgador_colegiado,
doc.dt_juntada
from tb_processo p
inner join tb_processo_documento doc on doc.id_processo = p.id_processo
inner join tb_processo_documento_bin docbin on docbin.id_processo_documento_bin = doc.id_processo_documento_bin
inner join tb_processo_trf ptrf on ptrf.id_processo_trf = p.id_processo
inner join pje.tb_orgao_julgador oj on oj.id_orgao_julgador = ptrf.id_orgao_julgador
inner join pje.tb_orgao_julgador_colgiado ojc on ojc.id_orgao_julgador_colegiado = ptrf.id_orgao_julgador_colegiado
inner join pje.tb_tipo_processo_documento tpd on tpd.id_tipo_processo_documento = doc.id_tipo_processo_documento
where doc.in_documento_sigiloso = 'N' /*filtrei os sigilosos*/
and doc.dt_juntada is not null /*filtrei os nao assinados*/
and docbin.ds_extensao is null /*filtrei os PDFs*/
"""
sql_original = sql_original.replace('\n', ' ')
sql_original = sql_original.replace('\t', ' ')

def recuperaDocumentos(regionais):

    for  sigla_trt in regionais:
        # sigla_trt='07'
        print("----------------------------------------------------------------------------")
        print('Recuperando TODOS os documentos nao sigilosos para o TRT ' + sigla_trt)
        porta = '5' + sigla_trt + '2'
        nome_arquivo_documentos_selecionados = 'TRT_' + sigla_trt + '_2G_2010-2019_documentosSelecionados.csv'

        try:
            # -----------------------------------------------------------------------------------------------------------------------
            # Recupera os documentos
            conn = psycopg2.connect(dbname=dbname_2g, user=userbugfix, password=senhabugfix, host=ipbugfix, port=porta)
            # conn = psycopg2.connect(dbname='pje_2grau_consulta', user=userbugfix, password=senhabugfix, host=ipbugfix, port=porta)
            start_time = time.time()

            sql_count = """select max(id_processo_documento) from tb_processo_documento"""
            total_registros = (psql.read_sql(sql_count, conn))
            total_registros = total_registros['max'][0]
            print(
                'Encontrados ' + str(total_registros) + ' documentos no total na tabela tb_processo_documento do TRT ' + sigla_trt)

            if os.path.isfile(path + nome_arquivo_documentos_selecionados):
                os.remove(path + nome_arquivo_documentos_selecionados)
            chunk_size = 1000
            offset = 1000
            dfs=[]
            while True:
                # for i in range(1,5):
                sql = sql_original + " and doc.id_processo_documento > %d and doc.id_processo_documento < %d  limit %d " % (offset-chunk_size,offset, chunk_size)
                dfs.append(psql.read_sql(sql, conn))
                if offset == 1000 :
                    print('Primeiros dados recuperados ...' + sql[-100:])
                    dfs[-1].to_csv(path + nome_arquivo_documentos_selecionados, mode='a', header=True, quoting=csv.QUOTE_ALL)
                    a=dfs[-1]
                else:
                    dfs[-1].to_csv(path + nome_arquivo_documentos_selecionados, mode='a', header=False, quoting=csv.QUOTE_ALL)
                offset += chunk_size
                if offset > total_registros + chunk_size:
                    print('Ultimo sql executado ...' + sql[-100:])
                    print('Dados recuperados com sucesso.')
                    break;
            total_time = time.time() - start_time
            print('\nTempo para recuperar dados: ' + str(timedelta(seconds=(total_time))))
        except Exception as e:
            print("\033[91mNão foi possível se conectar na base do  TRT " + sigla_trt + "\033[0m")
            print(e)
            continue;
for i in range (3,25):
    recuperaDocumentos([("{:02d}".format(i))])
# recuperaDocumentos(['07'])

#'07','24',