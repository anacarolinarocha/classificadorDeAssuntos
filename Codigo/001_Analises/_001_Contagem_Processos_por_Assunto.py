# ### ====================================================================================
# Script que faz a contagem dos assuntos de nivel 3 que são utilizados na maior parte dos processos de interesse
# Calcula o percentual cumulativo dos assuntos.
# No momento, os dados de entrada levam em consideracao apenas o assunto principal
# ### ====================================================================================

import pandas as pd
import numpy as np
import openpyxl


# df = pd.read_csv('/media/DATA/classificadorDeAssuntos/Dados/naoPublicavel/DadosUsadosParaComporATabelaDeQuantidadeDeAssuntosANivelNacional/contagem_qtd_RO_por_assunto_nivel_3_por_TRT.csv')

df = pd.read_csv('/home/anarocha/myGit/classificadorDeAssuntos/Codigo/001_Analises/Outputs/Total_De_Processos_Por_Nivel_3_Assunto_Principal.csv')

df_count = df[['cd_assunto_nivel_3','count']]
# df_count = df_count.groupby(['cd_assunto_nivel_3'], as_index=False)['count'].sum()

total_documentos = df_count.sum()['count']


df_count['percent'] = df_count['count'].apply(lambda x: round(x/total_documentos,6))
df_count = df_count.sort_values('percent', ascending=False)

df_count['cumulative_percentage'] = 100*df_count.total_recursos_ordinarios.cumsum()/df_count.total_recursos_ordinarios.sum()

df_count=df_count.reset_index(drop=True)

df_count.to_excel('/home/anarocha/myGit/classificadorDeAssuntos/Codigo/001_Analises/Outputs/Frequencia_Cumulativa_Processo_Com_Assunto_Nivel_3.xlsx')
