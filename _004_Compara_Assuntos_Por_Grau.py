

import pandas as pd
import numpy as np

path = '/media/DATA/classificadorDeAssuntos/Dados/naoPublicavel/ConferenciaDeAssuntos/sample'
nome_arquivo_1g = 'TRT_01_1G_2010-2019_listaAssuntosProcessosRemetidosAoSegundoGrauComAssuntos.csv'
nome_arquivo_2g = 'TRT_01_2G_2010-2019_listaAssuntosProcessosNoSegundoGrauSemSegredoComUmRO.csv'
nome_arquivo_processos_selecionados = 'TRT_01_2G_2010-2019_listaAssuntosProcessosNoSegundoGrauSemSegredoComUmRO.csv'
df_1g = pd.read_csv(path + nome_arquivo_1g, sep=',')
df_2g = pd.read_csv(path + nome_arquivo_2g, sep=',')

df_1g.shape
df_2g.shape


df_1g['processo_1g'].value_counts()
df_2g['processo_2g'].value_counts()

#-----------------------------------------------------------------------------------------------------------------------
#Seleciona apenas os processos que tenham mais de um assunto
procs_2grau_com_mais_de_um_assunto = df_2g.groupby(['processo_2g'],as_index=False)[['cd_assunto_2g']].count()
procs_2grau_com_mais_de_um_assunto = procs_2grau_com_mais_de_um_assunto[procs_2grau_com_mais_de_um_assunto['cd_assunto_2g']>1]

i1 = procs_2grau_com_mais_de_um_assunto.set_index('processo_2g').index
i2 = df_1g.set_index('processo_1g').index
processos_filtrados = set(df_1g[i2.isin(i1)]['processo_1g'])


i1 = df_1g.set_index('processo_1g').index
i2 = procs_2grau_com_mais_de_um_assunto.set_index('processo_2g').index
processos_filtrados = set(df_1g[i1.isin(i2)]['processo_1g'])
a = i1.isin(i2)
a.shape
#-----------------------------------------------------------------------------------------------------------------------
#Seleciona apenas os processos cujo grupo de assuntos no segundo grau é diferente do grupo de assuntos no primeiro grau
processos_selecionados  = []
for processo in processos_filtrados:
    df_grupo_1 = set(df_1g[(df_1g.processo_1g == processo)]['cd_assunto_1g'])
    df_grupo_2 = set(df_2g[(df_2g.processo_2g == processo)]['cd_assunto_2g'])
    if(df_grupo_1 != df_grupo_2):
        processos_selecionados.append(processo)





