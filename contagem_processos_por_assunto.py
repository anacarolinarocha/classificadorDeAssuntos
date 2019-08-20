import pandas as pd
import numpy as np

df = pd.read_csv('/home/anarocha/myGit/classificadorDeAssuntos/Dados/naoPublicavel/contagem_qtd_RO_por_assunto_nivel_3_por_TRT.csv')
#TODO: acrescentar no dataset os dados do TRT 15 E 21. Ficou faltando.

df_count = df[['cd_assunto_nivel_3','total_recursos_ordinarios']]
df_count = df_count.groupby(['cd_assunto_nivel_3'], as_index=False)['total_recursos_ordinarios'].sum()

total_documentos = df_count.sum()['total_recursos_ordinarios']


df_count['percent'] = df_count['total_recursos_ordinarios'].apply(lambda x: round(x/total_documentos,6))
df_count = df_count.sort_values('percent', ascending=False)

df_count['cumulative_percentage'] = 100*df_count.total_recursos_ordinarios.cumsum()/df_count.total_recursos_ordinarios.sum()

df_count=df_count.reset_index(drop=True)

