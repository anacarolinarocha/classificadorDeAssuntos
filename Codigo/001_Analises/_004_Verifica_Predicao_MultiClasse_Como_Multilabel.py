# ### ====================================================================================
# Script que verifica quais assuntos dentre os 5 mais provavies e os 10 mais provaveis estão contidos
# nos assuntos do processo. Ou seja, qual seria o percentual de acerto multilabel
# ### ====================================================================================

from datetime import timedelta
import sys
import pandas as pd
import time
import csv
import os
import numpy as np
from datetime import timedelta
import seaborn as sns
import matplotlib.pyplot as plt

float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
#------------------------------------------------------------------------------------------
#Carrega os assuntos e funcoes auxiliares
#------------------------------------------------------------------------------------------
arquivo_hierarquia_assuntos = '/home/anarocha/myGit/classificadorDeAssuntos/Dados/hierarquia_de_assuntos.csv'
assuntos = pd.read_csv(arquivo_hierarquia_assuntos)

assuntos = assuntos.replace(np.nan, 0, regex=True)
assuntosNivel1 = pd.Series(assuntos['cd_assunto_nivel_1'])
assuntosNivel2 = pd.Series(assuntos['cd_assunto_nivel_2'])
assuntosNivel3 = pd.Series(assuntos['cd_assunto_nivel_3'])
assuntosNivel4 = pd.Series(assuntos['cd_assunto_nivel_4'])
assuntosNivel5 = pd.Series(assuntos['cd_assunto_nivel_5'])

def recuperaNivelAssunto(codigo):
    global assuntos,assuntosNivel1,assuntosNivel2,assuntosNivel3,assuntosNivel4,assuntosNivel5
    nivel = -1
    if not assuntosNivel1[assuntosNivel1.isin([int(codigo)])].empty:
        nivel=1
    if not assuntosNivel2[assuntosNivel2.isin([int(codigo)])].empty:
        nivel=2
    if not assuntosNivel3[assuntosNivel3.isin([int(codigo)])].empty:
        nivel=3
    if not assuntosNivel4[assuntosNivel4.isin([int(codigo)])].empty:
        nivel=4
    if not assuntosNivel5[assuntosNivel5.isin([int(codigo)])].empty:
        nivel=5
    if(nivel==-1):
        print('NIVEL NAO ENCONTRADO: ' + str(codigo))
    return nivel


def recuperaAssuntoNivelEspecifico(codigo):
    nivel=3
    global assuntos,assuntosNivel1,assuntosNivel2,assuntosNivel3,assuntosNivel4,assuntosNivel5
    nivelInicial = recuperaNivelAssunto(codigo)
    coluna='cd_assunto_nivel_'+str(nivelInicial)
    index = int(assuntos[assuntos[coluna]==codigo].index[0])
    cd_assunto = assuntos['cd_assunto_nivel_' + str(nivel)][index]
    return int(cd_assunto)


#------------------------------------------------------------------------------------------
#Verifica...
#------------------------------------------------------------------------------------------
path_predicao = '/home/anarocha/myGit/classificadorDeAssuntos/Resultados/EXP20_MelhoresModelos_LSI250_TextosReduzidos_v2/'


nome_arquivo_predicao_MLP = path_predicao + 'predicao_Multi-Layer Perceptron.csv'
# nome_arquivo_predicao_MNB = path_predicao + 'predicao_Multinomial Naive Bayes.csv'
# nome_arquivo_predicao_RF = path_predicao + 'predicao_Random Forest.csv'
# nome_arquivo_predicao_SVM = path_predicao + 'predicao_SVM.csv'

df_predito_MLP = pd.read_csv(nome_arquivo_predicao_MLP, sep=',')
# df_predito_MNB = pd.read_csv(nome_arquivo_predicao_MNB, sep=',')
# df_predito_RF = pd.read_csv(nome_arquivo_predicao_RF, sep=',')
# df_predito_SVM = pd.read_csv(nome_arquivo_predicao_SVM, sep=',')

path = '/media/DATA/classificadorDeAssuntos/Dados/naoPublicavel/ConferenciaDeAssuntos/OK/'


df_resultado_analise_MLP = []
# df_resultado_analise_MNB = []
# df_resultado_analise_RF = []
# df_resultado_analise_SVM = []

# listaregionais=[20]
listaAssuntosCorrigida=[2546,2086,1855,2594,2458,2704,2656,2140,2435,2029,2583,2554,8808,2117,2021,5280,1904,1844,2055,1907,1806,55220,2506,
                        4437,10570,1783,1888,2478,5356,1773,1663,5272,2215,1767,1661,1690]
listaAssuntos =  listaAssuntosCorrigida

for i in range (1,25):
# for i in listaregionais:
    sigla_trt = ("{:02d}".format(i))
    # sigla_trt = '22'
    nome_arquivo_2g_trt = 'TRT_' + sigla_trt + '_2G_2010-2019_listaAssuntosProcessosNoSegundoGrauSemSegredoComUmRO.csv'
    df_assuntos_2g_trt =  pd.read_csv(path + nome_arquivo_2g_trt, sep=',')


    df_processos_preditos_trt_MLP = df_predito_MLP[(df_predito_MLP.sigla_trt == 'TRT' + sigla_trt)]
    # df_processos_preditos_trt_MNB = df_predito_MNB[(df_predito_MNB.sigla_trt == 'TRT' + sigla_trt)]
    # df_processos_preditos_trt_RF = df_predito_RF[(df_predito_RF.sigla_trt == 'TRT' + sigla_trt)]
    # df_processos_preditos_trt_SVM = df_predito_SVM[(df_predito_SVM.sigla_trt == 'TRT' + sigla_trt)]

    # df_processos_preditos_trt_MLP = df_processos_preditos_trt_MLP.head(5)
    # df_processos_preditos_trt_MNB = df_processos_preditos_trt_MNB.head(5)
    # df_processos_preditos_trt_RF = df_processos_preditos_trt_RF.head(5)
    # df_processos_preditos_trt_SVM = df_processos_preditos_trt_SVM.head(5)

    start_time = time.time()
    for index, row in df_processos_preditos_trt_MLP.iterrows():
        # print('----------------------------------------------------------------------')
        # print('PROCESSO: ' + df_processos_preditos_trt_MLP.loc[index]['nr_processo'])

        df_assuntos_2g_temp = df_assuntos_2g_trt[(df_assuntos_2g_trt.processo_2g == row['nr_processo'])]
        df_assuntos_2g_temp['cd_assunto_nivel_3'] = df_assuntos_2g_temp['cd_assunto_2g'].map(recuperaAssuntoNivelEspecifico)
        assuntos_existentes = set(df_assuntos_2g_temp['cd_assunto_nivel_3'])
        qnd_assuntos_no_2_grau = len(assuntos_existentes)
        assuntos_existentes_dentro_do_escopo = set([i for i in assuntos_existentes if i  in listaAssuntos])
        qnd_assuntos_no_2_grau_dentro_do_escopo = len(assuntos_existentes_dentro_do_escopo)

        #pega os n assuntos preditos mais provaveis
        row_predictions_MLP = pd.to_numeric(df_processos_preditos_trt_MLP.loc[index].tail(36).head(35))
        # row_predictions_MNB = pd.to_numeric(df_processos_preditos_trt_MNB.loc[index].tail(36).head(35))
        # row_predictions_RF = pd.to_numeric(df_processos_preditos_trt_RF.loc[index].tail(36).head(35))
        # row_predictions_SVM = pd.to_numeric(df_processos_preditos_trt_SVM.loc[index].tail(36).head(35))

        row_predictions = [(row_predictions_MLP,df_resultado_analise_MLP,df_processos_preditos_trt_MLP.loc[index])
                            # ,
                           # (row_predictions_MNB, df_resultado_analise_MNB, df_processos_preditos_trt_MNB.loc[index]),
                           # (row_predictions_RF, df_resultado_analise_RF, df_processos_preditos_trt_RF.loc[index]),
                           # (row_predictions_SVM, df_resultado_analise_SVM, df_processos_preditos_trt_SVM.loc[index])
                           ]

        for prediction in row_predictions:
            n5_assuntos_mais_provaveis = set(pd.to_numeric(prediction[0].nlargest(5).index.tolist()))
            n5_assuntos_que_acertou = [i for i in n5_assuntos_mais_provaveis if i in assuntos_existentes]
            if qnd_assuntos_no_2_grau > 0:
                n5_percent = len(n5_assuntos_que_acertou) / qnd_assuntos_no_2_grau
            else:
                n5_percent = 0
            n5_assuntos_que_acertou_dentro_do_escopo = [i for i in n5_assuntos_mais_provaveis if i in assuntos_existentes_dentro_do_escopo]
            if qnd_assuntos_no_2_grau_dentro_do_escopo > 0:
                n5_percent_dentro_do_escopo = len(n5_assuntos_que_acertou_dentro_do_escopo) / qnd_assuntos_no_2_grau_dentro_do_escopo
            else:
                n10_percent_dentro_do_escopo = 0
            n5_acerto = len(n5_assuntos_que_acertou)

            n10_assuntos_mais_provaveis = set(pd.to_numeric(prediction[0].nlargest(10).index.tolist()))
            n10_assuntos_que_acertou = [i for i in n10_assuntos_mais_provaveis if i in assuntos_existentes]
            if qnd_assuntos_no_2_grau > 0:
                n10_percent = len(n10_assuntos_que_acertou) / qnd_assuntos_no_2_grau
            else:
                n10_percent = 0
            n10_assuntos_que_acertou_dentro_do_escopo = [i for i in n10_assuntos_mais_provaveis if i in assuntos_existentes_dentro_do_escopo]
            if qnd_assuntos_no_2_grau_dentro_do_escopo > 0:
                n10_percent_dentro_do_escopo = len(n10_assuntos_que_acertou_dentro_do_escopo) / qnd_assuntos_no_2_grau_dentro_do_escopo
            else:
                n10_percent_dentro_do_escopo = 0
            n10_acerto = len(n10_assuntos_que_acertou)


            prediction[1].append(
                [prediction[2]['sigla_trt'], prediction[2]['nr_processo'], prediction[2]['id_processo_documento'], qnd_assuntos_no_2_grau,
                 qnd_assuntos_no_2_grau_dentro_do_escopo, n5_acerto, n5_percent, n5_percent_dentro_do_escopo,
                 n10_acerto,
                 n10_percent, n10_percent_dentro_do_escopo, assuntos_existentes, assuntos_existentes_dentro_do_escopo,
                 n5_assuntos_mais_provaveis, n5_assuntos_que_acertou, n10_assuntos_mais_provaveis,
                 n10_assuntos_que_acertou])


    total_time = time.time() - start_time
    print("Tempo para recuperar dados do TRT " + sigla_trt + ': ' + str(timedelta(seconds=total_time)))

colunas = ['sigla_trt', 'nr_processo', 'id_processo_documento','qnd_assuntos_no_2_grau','qnd_assuntos_no_2_grau_dentro_do_escopo','n5_acerto','n5_percent','n5_percent_dentro_do_escopo','n10_acerto','n10_percent','n10_percent_dentro_do_escopo','assuntos_existentes','assuntos_existentes_dentro_do_escopo','n5_assuntos_mais_provaveis','n5_assuntos_que_acertou','n10_assuntos_mais_provaveis','n10_assuntos_que_acertou']
df_final_MLP = pd.DataFrame(df_resultado_analise_MLP, columns=colunas)
# df_final_MNB = pd.DataFrame(df_resultado_analise_MNB, columns=colunas)
# df_final_RF = pd.DataFrame(df_resultado_analise_RF, columns=colunas)
# df_final_SVM = pd.DataFrame(df_resultado_analise_SVM, columns=colunas)

nome_arquivo_predicao_avaliado_MLP = path_predicao + 'predicao_multilabel_avaliada_MLP.csv'
# nome_arquivo_predicao_avaliado_MNB = path_predicao + 'predicao_multilabel_avaliada_MNB.csv'
# nome_arquivo_predicao_avaliado_RF = path_predicao + 'predicao_multilabel_avaliada_RF.csv'
# nome_arquivo_predicao_avaliado_SVM = path_predicao + 'predicao_multilabel_avaliada_SVM.csv'

df_final_MLP.to_csv(nome_arquivo_predicao_avaliado_MLP, sep='#',decimal=",")
# df_final_MNB.to_csv(nome_arquivo_predicao_avaliado_MNB, sep='#',decimal=",")
# df_final_RF.to_csv(nome_arquivo_predicao_avaliado_RF, sep='#',decimal=",")
# df_final_SVM.to_csv(nome_arquivo_predicao_avaliado_SVM, sep='#',decimal=",")

# df_final = pd.read_csv(nome_arquivo_predicao_avaliado, sep='#',decimal=",")
# df_final.columns


df_finais = [(df_final_MLP,'MLP')
                # ,
            # (df_final_MNB,'MNB'),
            # (df_final_RF,'RF'),
            # (df_final_SVM,'SVM')
                         ]

for df_final in df_finais:

    df_filtrado = df_final[0][['n5_percent_dentro_do_escopo','n10_percent_dentro_do_escopo']]
    df_filtrado = df_filtrado.rename(columns={"n5_percent_dentro_do_escopo": "Percentual de acerto com 5 chutes", "n10_percent_dentro_do_escopo": "Percentual de acerto com 10 chutes"})

    plt.cla()
    plt.clf
    sns.boxplot(y=df_filtrado["Percentual de acerto com 5 chutes"],color='blue').set_title(df_final[1])
    plt.savefig("{0}{1}.png".format(path_predicao, 'boxplot_n5_chutes_' + df_final[1]))

    plt.cla()
    plt.clf
    sns.boxplot(y=df_filtrado["Percentual de acerto com 10 chutes"],color='green').set_title(df_final[1])
    plt.savefig("{0}{1}.png".format(path_predicao, 'boxplot_n10_chutes_' + df_final[1]))

