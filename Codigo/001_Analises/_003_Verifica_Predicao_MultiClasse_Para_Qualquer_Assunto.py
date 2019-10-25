# ### ====================================================================================
# Script que verifica se o assunto principal predito na classificacao multiclasse está
# contido nos demais assuntos do processo
# ### ====================================================================================

from datetime import timedelta
import sys
import pandas as pd
import time
import csv
import os
import numpy as np
from datetime import timedelta
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

nome_base_arquivo_predicao = '/home/anarocha/myGit/classificadorDeAssuntos/Resultados/EXP20_MelhoresModelos_LSI250_TextosReduzidos_v2/predicao_'


path = '/media/DATA/classificadorDeAssuntos/Dados/naoPublicavel/ConferenciaDeAssuntos/OK/'
modelos=['Multi-Layer Perceptron','Multinomial Naive Bayes','Random Forest','SVM']


modelo='Multi-Layer Perceptron'
nome_arquivo_predicao = nome_base_arquivo_predicao + modelo + '.csv'
df_predito_MLP = pd.read_csv(nome_arquivo_predicao, sep=',')
df_resultado_analise_MLP = []

# modelo='Multinomial Naive Bayes'
# nome_arquivo_predicao = nome_base_arquivo_predicao + modelo + '.csv'
# df_predito_MNB = pd.read_csv(nome_arquivo_predicao, sep=',')
# df_resultado_analise_MNB = []
#
# modelo='Random Forest'
# nome_arquivo_predicao = nome_base_arquivo_predicao + modelo + '.csv'
# df_predito_RF = pd.read_csv(nome_arquivo_predicao, sep=',')
# df_resultado_analise_RF = []
#
# modelo='SVM'
# nome_arquivo_predicao = nome_base_arquivo_predicao + modelo + '.csv'
# df_predito_SVM = pd.read_csv(nome_arquivo_predicao, sep=',')
# df_resultado_analise_SVM = []




for i in range (1,25):
    sigla_trt = ("{:02d}".format(i))
#     sigla_trt = '22'
    nome_arquivo_2g_trt = 'TRT_' + sigla_trt + '_2G_2010-2019_listaAssuntosProcessosNoSegundoGrauSemSegredoComUmRO.csv'
    df_assuntos_2g_trt =  pd.read_csv(path + nome_arquivo_2g_trt, sep=',')
    df_processos_preditos_trt_MLP = df_predito_MLP[(df_predito_MLP.sigla_trt == 'TRT' + sigla_trt)]
    # df_processos_preditos_trt_MNB = df_predito_MNB[(df_predito_MNB.sigla_trt == 'TRT' + sigla_trt)]
    # df_processos_preditos_trt_RF = df_predito_RF[(df_predito_RF.sigla_trt == 'TRT' + sigla_trt)]
    # df_processos_preditos_trt_SVM = df_predito_SVM[(df_predito_SVM.sigla_trt == 'TRT' + sigla_trt)]
    #

    # df_processos_preditos_trt = df_processos_preditos_trt.head(5)
    start_time = time.time()

    #AQUI POSSO PEGAR QUALQUER UM DOS 4 DE REFERENCIA, PORQUE SAO OS MESMOS SEMPRE....
    # df_processos_preditos_trt_RF = df_processos_preditos_trt_RF.head(50)
    for index, row in df_processos_preditos_trt_MLP.iterrows():
        # print('----------------------------------------------------------------------')
        # print('PROCESSO: ' + row['nr_processo'])
        # print(str(df_processos_preditos_trt_MLP.loc[index]['id_processo_documento']) + ' = ' + str(df_processos_preditos_trt_MNB.loc[index]['id_processo_documento']) + ' = ' +
        #       str(df_processos_preditos_trt_RF.loc[index]['id_processo_documento']) + ' = ' +    str(df_processos_preditos_trt_SVM.loc[index]['id_processo_documento']))

        df_assuntos_2g_temp = df_assuntos_2g_trt[(df_assuntos_2g_trt.processo_2g == row['nr_processo'])]
        df_assuntos_2g_temp['cd_assunto_nivel_3'] = df_assuntos_2g_temp['cd_assunto_2g'].map(recuperaAssuntoNivelEspecifico)

        assunto_predito_MLP = df_processos_preditos_trt_MLP.loc[index]['y_pred']
        # assunto_predito_MNB = df_processos_preditos_trt_MNB.loc[index]['y_pred']
        # assunto_predito_RF = df_processos_preditos_trt_RF.loc[index]['y_pred']
        # assunto_predito_SVM = df_processos_preditos_trt_SVM.loc[index]['y_pred']

        acertou_predicao_MLP = (df_processos_preditos_trt_MLP.loc[index]['y_true'] == df_processos_preditos_trt_MLP.loc[index]['y_pred'])
        # acertou_predicao_MNB = (df_processos_preditos_trt_MNB.loc[index]['y_true'] == df_processos_preditos_trt_MNB.loc[index]['y_pred'])
        # acertou_predicao_RF = (df_processos_preditos_trt_RF.loc[index]['y_true'] == df_processos_preditos_trt_RF.loc[index]['y_pred'])
        # acertou_predicao_SVM = (df_processos_preditos_trt_SVM.loc[index]['y_true'] == df_processos_preditos_trt_SVM.loc[index]['y_pred'])
        # print('----------------------------------------------------------------------')
        # if df_processos_preditos_trt_MLP.loc[index]['y_true'] == df_processos_preditos_trt_MLP.loc[index]['y_pred']:
        #      print('Acertou o assunto principal, que era ' + str(df_processos_preditos_trt_MLP.loc[index]['y_true']))
        # else:
        #     print('Errou o assunto principal, que era ' + str(df_processos_preditos_trt_MLP.loc[index]['y_true']))
        # print('Assunto Predito: ' + str(assunto_predito_MLP))
        # print(df_assuntos_2g_temp)


        assunto_predio_existe_no_processo_MLP = any(df_assuntos_2g_temp.cd_assunto_nivel_3 == assunto_predito_MLP)
        # assunto_predio_existe_no_processo_MNB = any(df_assuntos_2g_temp.cd_assunto_nivel_3 == assunto_predito_MNB)
        # assunto_predio_existe_no_processo_RF = any(df_assuntos_2g_temp.cd_assunto_nivel_3 == assunto_predito_RF)
        # assunto_predio_existe_no_processo_SVM = any(df_assuntos_2g_temp.cd_assunto_nivel_3 == assunto_predito_SVM)


        # print('O assunto predito existe dentre os demais assuntos do processo? ' + str(assunto_predio_existe_no_processo))
        df_resultado_analise_MLP.append([df_processos_preditos_trt_MLP.loc[index]['sigla_trt'],
                                         df_processos_preditos_trt_MLP.loc[index]['nr_processo'],
                                         df_processos_preditos_trt_MLP.loc[index]['id_processo_documento'],
                                         df_processos_preditos_trt_MLP.loc[index]['y_true'],
                                         df_processos_preditos_trt_MLP.loc[index]['y_pred'],
                                         acertou_predicao_MLP,assunto_predio_existe_no_processo_MLP])
        # df_resultado_analise_MNB.append([df_processos_preditos_trt_MNB.loc[index]['sigla_trt'],
        #                                  df_processos_preditos_trt_MNB.loc[index]['nr_processo'],
        #                                  df_processos_preditos_trt_MNB.loc[index]['id_processo_documento'],
        #                                  df_processos_preditos_trt_MNB.loc[index]['y_true'],
        #                                  df_processos_preditos_trt_MNB.loc[index]['y_pred'], acertou_predicao_MNB,
        #                                  assunto_predio_existe_no_processo_MNB])
        # df_resultado_analise_RF.append([df_processos_preditos_trt_RF.loc[index]['sigla_trt'],
        #                                 df_processos_preditos_trt_RF.loc[index]['nr_processo'],
        #                                 df_processos_preditos_trt_RF.loc[index]['id_processo_documento'],
        #                                 df_processos_preditos_trt_RF.loc[index]['y_true'],
        #                                 df_processos_preditos_trt_RF.loc[index]['y_pred'], acertou_predicao_RF,
        #                                 assunto_predio_existe_no_processo_RF])
        # df_resultado_analise_SVM.append([df_processos_preditos_trt_SVM.loc[index]['sigla_trt'],
        #                                  df_processos_preditos_trt_SVM.loc[index]['nr_processo'],
        #                                  df_processos_preditos_trt_SVM.loc[index]['id_processo_documento'],
        #                                  df_processos_preditos_trt_SVM.loc[index]['y_true'],
        #                                  df_processos_preditos_trt_SVM.loc[index]['y_pred'], acertou_predicao_SVM,
        #                                  assunto_predio_existe_no_processo_SVM])

    total_time = time.time() - start_time
    print("Tempo para recuperar dados do TRT " + sigla_trt + ': ' + str(timedelta(seconds=total_time)))

    df_final_MLP = pd.DataFrame(df_resultado_analise_MLP,
                                columns=['sigla_trt', 'nr_processo', 'id_processo_documento', 'y_true', 'y_pred',
                                         'acertou_predicao', 'assunto_predio_existe_no_processo'])
    # df_final_MNB = pd.DataFrame(df_resultado_analise_MNB,
    #                             columns=['sigla_trt', 'nr_processo', 'id_processo_documento', 'y_true', 'y_pred',
    #                                      'acertou_predicao', 'assunto_predio_existe_no_processo'])
    # df_final_RF = pd.DataFrame(df_resultado_analise_RF,
    #                             columns=['sigla_trt', 'nr_processo', 'id_processo_documento', 'y_true', 'y_pred',
    #                                      'acertou_predicao', 'assunto_predio_existe_no_processo'])
    # df_final_SVM = pd.DataFrame(df_resultado_analise_SVM,
    #                             columns=['sigla_trt', 'nr_processo', 'id_processo_documento', 'y_true', 'y_pred',
    #                                      'acertou_predicao', 'assunto_predio_existe_no_processo'])

    path_final =  '/home/anarocha/myGit/classificadorDeAssuntos/Resultados/EXP20_MelhoresModelos_LSI250_TextosReduzidos_v2/'
    nome_arquivo_predicao_avaliado_MLP = path_final + 'predicao_multiclasse_avaliada_assunto_principal_qualquer_assunto_' + 'MLP.csv'
    # nome_arquivo_predicao_avaliado_MNB = path_final + 'predicao_multiclasse_avaliada_assunto_principal_qualquer_assunto_' + 'MNB.csv'
    # nome_arquivo_predicao_avaliado_RF = path_final + 'predicao_multiclasse_avaliada_assunto_principal_qualquer_assunto_' + 'RF.csv'
    # nome_arquivo_predicao_avaliado_SVM = path_final + 'predicao_multiclasse_avaliada_assunto_principal_qualquer_assunto_' + 'SVM.csv'

    df_final_MLP.to_csv(nome_arquivo_predicao_avaliado_MLP)
    # df_final_MNB.to_csv(nome_arquivo_predicao_avaliado_MNB)
    # df_final_RF.to_csv(nome_arquivo_predicao_avaliado_RF)
    # df_final_SVM.to_csv(nome_arquivo_predicao_avaliado_SVM)
