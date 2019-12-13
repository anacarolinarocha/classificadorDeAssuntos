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
sql_original = """WITH processos_com_um_ro as
(SELECT id_processo FROM
			(SELECT id_processo, count(id_processo_documento) as total_recursos_ordinarios
			 FROM tb_processo_documento
			 WHERE id_tipo_processo_documento = (select id_tipo_processo_documento from tb_tipo_processo_documento where ds_tipo_processo_documento = 'Recurso Ordinário')
			GROUP BY id_processo
			HAVING count(id_processo_documento) = 1) AS t 
		)
select p2g.nr_processo as processo_2g, a2g.cd_assunto_trf as cd_assunto_2g
from tb_processo_assunto pa2g
inner join pje.tb_assunto_trf a2g on a2g.id_assunto_trf = pa2g.id_assunto_trf
inner join tb_processo p2g on p2g.id_processo = pa2g.id_processo_trf
inner join tb_processo_trf ptrf on ptrf.id_processo_trf = pa2g.id_processo_trf
where pa2g.id_processo_trf in (select * from processos_com_um_ro) 
and ptrf.in_segredo_justica = 'N'
"""
sql_original = sql_original.replace('\n', ' ')

def recuperaDadosRegional(regionais):

    for  sigla_trt in regionais:
        # sigla_trt='01'
        print("----------------------------------------------------------------------------")
        print("PROCESSANDO DADOS DO TRT {} - 2o GRAU".format(sigla_trt))
        start_time = time.time()
        porta = '5' + sigla_trt + '2'
        nomeArquivo = '/media/DATA/classificadorDeAssuntos/Dados/naoPublicavel/ConferenciaDeAssuntos/TRT_' + sigla_trt  + '_2G_2010-2019_listaAssuntosProcessosNoSegundoGrauSemSegredoComUmRO.csv'

        try:
            # conn = psycopg2.connect(dbname=dbname_2g, user=userbugfix, password=senhabugfix, host=ipbugfix, port=porta)
            conn = psycopg2.connect(dbname='pje_2grau_consulta', user=userbugfix, password=senhabugfix, host=ipbugfix, port=porta)

            sql_count = """select max(id_processo_assunto) from tb_processo_assunto"""
            total_registros = (psql.read_sql(sql_count,conn))
            total_registros = total_registros['max'][0]
            print('Encontrados ' + str(total_registros) + ' registros na tabela tb_processo_assunto do TRT ' + sigla_trt)
            if os.path.isfile(nomeArquivo):
               os.remove(nomeArquivo)
            chunk_size = 50000
            offset = 50000
            dfs=[]
            while True:
            # for i in range(1,5):
                sql = sql_original +" and pa2g.id_processo_assunto > %d and pa2g.id_processo_assunto < %d  limit %d " % (offset-chunk_size,offset, chunk_size)
                dfs.append(psql.read_sql(sql, conn))
                if offset == 50000 :
                    print('Primeiros dados recuperados ...' + sql[-100:])
                    dfs[-1].to_csv(nomeArquivo, mode='a', header=True, quoting=csv.QUOTE_ALL)
                else:
                    dfs[-1].to_csv(nomeArquivo, mode='a', header=False, quoting=csv.QUOTE_ALL)
                offset += chunk_size
                # if len(dfs[-1]) < chunk_size:
                #     print('Dados recuperados com sucesso.' )
                #     break;
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

# for i in range (15,25):
#     recuperaDadosRegional([("{:02d}".format(i))])
recuperaDadosRegional(['20'])

# import multiprocessing as mp
# pool = mp.Pool(4)
# results = [pool.apply(recuperaDadosRegional, args=([("{:02d}".format(i))])) for i in range (1,25)]