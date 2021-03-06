

import pandas as pd
import numpy as np
from datetime import timedelta
import time

path = '/media/DATA/classificadorDeAssuntos/Dados/naoPublicavel/ConferenciaDeAssuntos/OK/'
import pandas as pd
df = pd.DataFrame(columns=['sigla_trt','total_inicial','total_1_assunto_removido','total_grupos_distintos_removido'])

# def cruzaAssuntos(regionais):
#     for  sigla_trt in regionais:
#         sigla_trt='09'

for sigla_trt in range(1, 25):
    sigla_trt = "{:02d}".format(sigla_trt)
    sigla_trt='09'
    print("================================")
    print('Cruzando assuntos para o TRT ' + sigla_trt)
    nome_arquivo_1g = 'TRT_' + sigla_trt + '_1G_2010-2019_listaAssuntosProcessosRemetidosAoSegundoGrauComAssuntos.csv'
    nome_arquivo_2g = 'TRT_' + sigla_trt + '_2G_2010-2019_listaAssuntosProcessosNoSegundoGrauSemSegredoComUmRO.csv'
    nome_arquivo_processos_selecionados = 'TRT_' + sigla_trt + '_2G_2010-2019_listaProcessosCurados.csv'
    df_1g = pd.read_csv(path + nome_arquivo_1g, sep=',')
    df_2g = pd.read_csv(path + nome_arquivo_2g, sep=',')

    df_1g.shape
    df_2g.shape

    df_1g['processo_1g'].value_counts()
    df_2g['processo_2g'].value_counts()

    print("Total de processos no segundo grau: " + str(len(df_2g['processo_2g'].value_counts())))

    total_inicial = len(df_2g['processo_2g'].value_counts())
    #-----------------------------------------------------------------------------------------------------------------------
    #Seleciona apenas os processos que tenham mais de um assunto no segundo grau
    df_2g_mais_de_um_assunto = df_2g.groupby(['processo_2g'],as_index=False)[['cd_assunto_2g']].count()
    #//TODO: verificar se ha embasamento legal para que eu remova de fato esses caras.... se nao eu estaria enviezando o dataset
    df_2g_mais_de_um_assunto = df_2g_mais_de_um_assunto[df_2g_mais_de_um_assunto['cd_assunto_2g']>1]

    i1 = df_2g.set_index('processo_2g').index
    i2 = df_2g_mais_de_um_assunto.set_index('processo_2g').index
    processos_filtrados = set(df_2g[i1.isin(i2)]['processo_2g'])
    print("Total de processos que sobraram, com mais de um assunto: " + str(len(processos_filtrados)))

    total_1_assunto_removido = total_inicial - len(processos_filtrados)
    #-----------------------------------------------------------------------------------------------------------------------
    #Seleciona apenas os processos em comum entre os dois graus

    i1 = df_1g.set_index('processo_1g').index
    i2 = processos_filtrados
    processos_filtrados = set(df_1g[i1.isin(i2)]['processo_1g'])
    len(processos_filtrados)
    total_em_comum_nos_dois_graus = len(processos_filtrados)

    #-----------------------------------------------------------------------------------------------------------------------
    #Seleciona apenas os processos cujo grupo de assuntos no segundo grau é diferente do grupo de assuntos no primeiro grau
    start_time = time.time()
    processos_selecionados  = []
    total_grupos_distintos_removido = 0
    for processo in processos_filtrados:
        df_grupo_1 = set(df_1g[(df_1g.processo_1g == processo)]['cd_assunto_1g'])
        df_grupo_2 = set(df_2g[(df_2g.processo_2g == processo)]['cd_assunto_2g'])
        # print('-----------------------------')
        # print('Processo: ' + processo)
        # print('Grupo 1: '+ str(df_grupo_1))
        # print('Grupo 2: ' + str(df_grupo_2))
        if(df_grupo_1 != df_grupo_2):
            processos_selecionados.append(processo)
        else:
            total_grupos_distintos_removido = total_grupos_distintos_removido + 1
    len(processos_selecionados)
    total_time = time.time() - start_time
    print("Total de processos que sobraram, com mais de um assunto e com grupos diferentes: " + str(len(processos_filtrados)))
    df = df.append([[sigla_trt, total_inicial,total_1_assunto_removido,total_grupos_distintos_removido]])
    print("------------------------------")
    print('Tempo para cruzamento dos assuntos no TRT ' + sigla_trt + ':  ' + str(timedelta(seconds=total_time)))
    print('Encontrados '+ str(len(processos_selecionados)) + 'no TRT ' + sigla_trt)
    # pd.DataFrame(processos_selecionados, columns=['processo']).to_csv(path+nome_arquivo_processos_selecionados, index=False)
    print("------------------------------")


df.to_csv('/home/anarocha/myGit/classificadorDeAssuntos/Codigo/001_Analises/Outputs/Quantidade_processos_removidos_pela_limpeza_de_ruido.csv')
# for i in range (12,15):
#     cruzaAssuntos([("{:02d}".format(i))])
# cruzaAssuntos(['22']) # 11, 22

