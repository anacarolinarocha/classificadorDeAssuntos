# ### ====================================================================================
# Script analisa a quantidade de palavras nos textos
# ### ====================================================================================


import pandas as pd
import numpy as np
from datetime import timedelta
import time

import seaborn as sns
import matplotlib.pyplot as plt

path = '/media/DATA/classificadorDeAssuntos/Dados/naoPublicavel/ConferenciaDeAssuntos/OK/'

df_todos_os_tamanhos = pd.DataFrame()
for i in range (1,25):
# for i in listaregionais:
    sigla_trt = ("{:02d}".format(i))
    print('Verificando textos do TRT ' + sigla_trt)
    nome_arquivo = 'TRT_' + sigla_trt + '_2G_2010-2019_documentosSelecionadosProcessados.csv'
    df = pd.read_csv(path + nome_arquivo, sep='#')
    # df.columns
    # df=df.head(5)
    df['totalwords'] = [len(x.split()) for x in df['texto_processado'].tolist()]
    df_todos_os_tamanhos = df_todos_os_tamanhos.append(df[['totalwords']])

df_todos_os_tamanhos['totalwords'].mean()


import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(df_todos_os_tamanhos['totalwords'])
plt.show()

df_todos_os_tamanhos_f = df_todos_os_tamanhos[((df_todos_os_tamanhos.totalwords < 10000) & (df_todos_os_tamanhos.totalwords > 600))]
df_todos_os_tamanhos_f = df_todos_os_tamanhos_f.sort_values(by='totalwords', ascending=True)

sns.boxplot(df_todos_os_tamanhos_f['totalwords'])
plt.show()



sns.boxplot(y=df_final["n10_percent_dentro_do_escopo"], color='green')
plt.show()

total_time = time.time() - start_time
print('Tempo para cruzamento dos assuntos no TRT ' + sigla_trt + ':  ' + str(timedelta(seconds=total_time)))
print('Encontrados '+ str(len(processos_selecionados)) + 'no TRT ' + sigla_trt)
pd.DataFrame(processos_selecionados, columns=['processo']).to_csv(path+nome_arquivo_processos_selecionados, index=False)


# for i in range (12,15):
#     cruzaAssuntos([("{:02d}".format(i))])
cruzaAssuntos(['22']) # 11, 22

