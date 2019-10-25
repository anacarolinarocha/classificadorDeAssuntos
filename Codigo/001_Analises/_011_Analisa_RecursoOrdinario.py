# ### ====================================================================================
# Script que recupera o conteúdo de um RO
# ### ====================================================================================

from datetime import timedelta
import sys
import pandas as pd
import time
import csv
import os
import matplotlib.pyplot as plt

sys.path.insert(1, '/home/anarocha/Documents/credentials')
from credentials import *

path = '/media/DATA/classificadorDeAssuntos/Dados/naoPublicavel/ConferenciaDeAssuntos/OK/'
path_outputs =  '/home/anarocha/myGit/classificadorDeAssuntos/Codigo/001_Analises/Outputs/ROs/'
sigla_trt = '19'
nome_arquivo_origem = path + 'TRT_' + sigla_trt + '_2G_2010-2019_documentosSelecionados.csv'
nome_arquivo_2g = path + 'TRT_' + sigla_trt + '_2G_2010-2019_listaAssuntosProcessosNoSegundoGrauSemSegredoComUmRO.csv'

nomes_assuntos = '/home/anarocha/myGit/classificadorDeAssuntos/Dados/nomes_assuntos.csv'
lista_assuntos = pd.read_csv(nomes_assuntos, sep=";")

colnames = ['index','nr_processo', 'id_processo_documento', 'cd_assunto_nivel_5', 'cd_assunto_nivel_4','cd_assunto_nivel_3','cd_assunto_nivel_2','cd_assunto_nivel_1','tx_conteudo_documento','ds_identificador_unico','ds_identificador_unico_simplificado','ds_orgao_julgador','ds_orgao_julgador_colegiado','dt_juntada']
df_trt = pd.read_csv(nome_arquivo_origem, sep=',', names=colnames, index_col=0, header=None, quoting=csv.QUOTE_ALL)

df_trt['quantidade_de_palavras'] = [len(x.split()) for x in df_trt['tx_conteudo_documento'].tolist()]
df_trt= df_trt.sort_values(by='quantidade_de_palavras', ascending=False)


# verifica o conteudo de um documento
for i in range (1880,1899,1):
    nome_ro = "TRT_" + sigla_trt + "_RO_" + str(df_trt.iloc[i]['nr_processo']) + "_id_" + str(df_trt.iloc[i]['id_processo_documento']) + ".html"
    f = open(path_outputs + nome_ro, "w")
    f.write(df_trt.iloc[i]['tx_conteudo_documento'])
    f.close()
    import webbrowser
    webbrowser.get('firefox').open_new_tab(path_outputs + nome_ro)

df_2g = pd.read_csv(nome_arquivo_2g, sep=',')
numero_processo = '0000759-40.2014.5.19.0059'
df_2g_processo = df_2g[(df_2g.processo_2g == numero_processo)]['cd_assunto_2g']

documento = df_trt[(df_trt.nr_processo == numero_processo)]['cd_assunto_nivel_5']

assuntos_processo = set(df_2g_processo)

assunto_principal =  str(df_trt[(df_trt.nr_processo == numero_processo)]['cd_assunto_nivel_1'].iloc[0]) + '/' + str(df_trt[(df_trt.nr_processo == numero_processo)]['cd_assunto_nivel_2'].iloc[0]) + '/' + str(df_trt[(df_trt.nr_processo == numero_processo)]['cd_assunto_nivel_3'].iloc[0]) + '/' + str(df_trt[(df_trt.nr_processo == numero_processo)]['cd_assunto_nivel_4'].iloc[0]) + '/' + str(df_trt[(df_trt.nr_processo == numero_processo)]['cd_assunto_nivel_5'].iloc[0]) + '/'
print('------------------------------------------------------------------')
print('PROCESSO ' + numero_processo)
print('..................................................................')
for assunto in assuntos_processo:
    nome = lista_assuntos[(lista_assuntos.cd_assunto_trf == assunto)]['ds_assunto_completo'].iloc[0]
    print(str(assunto) + ' - ' +  nome)
print('ASSUNTO PRINCIPAL: ' + assunto_principal)
print('------------------------------------------------------------------')

