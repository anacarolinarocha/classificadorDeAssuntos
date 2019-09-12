
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 17:09:43 2018

@author: anarocha
"""
#import multiprocessing
#multiprocessing.set_start_method('forkserver')
from SolrClient import SolrClient
import multiprocessing as mp



import csv
import time
import pandas as pd
import json
import sys
import gc
import os
import numpy as np
from numpy.random import random, random_integers
from datetime import timedelta


from modelo import Modelo
from funcoes import *

    

# =============================================================================
# Dinive variaveis globais
# =============================================================================
nomeDataSet = 'TRT13_2G_JT_REDUZIDO'
featureType = 'NAIVE_BAYES'
nomeExperimento = nomeDataSet+'_'+featureType+'_CorpusCompleto_CV5'
numeroExperimento = '15'
nomePastaRestultados='./Resultados'
nomeArqutivoResultadosCompilados=nomePastaRestultados+'/resultadosFinaisCompilados.csv'
nomePasta = nomePastaRestultados + '/Experimento ' + numeroExperimento + ' - ' + nomeExperimento

nomeCore='documentos_2g_trt3_2'

import os
if not os.path.exists(nomePasta):
    os.makedirs(nomePasta)
    os.makedirs(nomePasta+'/imagens')
    
quantidadeMinimaDocumentos = 500;
solr = SolrClient('http://localhost:8983/solr')


# =============================================================================
# Marca os elementos que serão usados para teste no Solr
# =============================================================================
def marcarDocumentosSolr(field,data, flag):
    documentos = []
    for ids in data:
        doc = {'id': ids, field:{'set': flag}}
        documentos.append(doc)
    solr.index_json(nomeCore,json.dumps(documentos))
    solr.commit(openSearcher=True, collection=nomeCore)

##############################################################################    
##############################################################################
# ANALISA OS DADOS
##############################################################################
##############################################################################

# =============================================================================
# Retira dos dados aqueles que tem menos de 50 exemplares no nivel 3.
# Além de ser necessário ter uma amostra representativa dos dados, o cross 
# validation não funcionará bem se não houver uma amostra suficientemente grande
# para popular cada fold.
# =============================================================================
query = 'tx_conteudo_documento:[* TO *]'
solrDataAnalise = solr.query(nomeCore,{
'q':query,'fl':'id_processo_documento,cd_assunto_trf', 'rows':'300000'
})
dfGeral = pd.DataFrame(solrDataAnalise.docs)

recuperaHierarquiaAssuntos(dfGeral)



#
#def setHierarquiaAssuntos(df,i):
#    df.set_value(i,'cd_assunto_nivel_1',recuperaAssuntoNivelEspecifico(int(row[0]),1))
#    df.set_value(i,'cd_assunto_nivel_2',recuperaAssuntoNivelEspecifico(int(row['cd_assunto_trf']),2))
#    df.set_value(i,'cd_assunto_nivel_3',recuperaAssuntoNivelEspecifico(int(row['cd_assunto_trf']),3))
#    df.set_value(i,'cd_assunto_nivel_4',recuperaAssuntoNivelEspecifico(int(row['cd_assunto_trf']),4))
#    df.set_value(i,'cd_assunto_nivel_5',recuperaAssuntoNivelEspecifico(int(row['cd_assunto_trf']),5))
#
#def recuperaHierarquiaAssuntos(df):
#    start_time = time.time()
#    Parallel(n_jobs=-1)(delayed(setHierarquiaAssuntos)(row,i) for i, row in dfGeral.iterrows())
#    end_time = time.time() - start_time
#    print('Tempo para montar a hierarquia de assuntos:' + str(timedelta(seconds=end_time)))   
         


dfGeral_CountNivel3 = dfGeral.groupby('cd_assunto_nivel_3')[['id_processo_documento']].count()
codigosAbaixoQuantidadeMinima = dfGeral_CountNivel3[dfGeral_CountNivel3['id_processo_documento'] < quantidadeMinimaDocumentos]
codigosAbaixoQuantidadeMinima.reset_index(inplace=True)

codigosAbaixoQuantidadeMinima = codigosAbaixoQuantidadeMinima['cd_assunto_nivel_3'].values.tolist()

filhosAbaixoQuantidadeMinima = []
for nivel3 in codigosAbaixoQuantidadeMinima:
    for filho in recuperaFilhosDoNivel3(int(nivel3)):
        filhosAbaixoQuantidadeMinima.append(filho)   
        
for filho in filhosAbaixoQuantidadeMinima:
    codigosAbaixoQuantidadeMinima.append(filho)

codigosAbaixoQuantidadeMinima = ' '.join(map(str, codigosAbaixoQuantidadeMinima)) 
codigosAbaixoQuantidadeMinima = codigosAbaixoQuantidadeMinima.replace('.0', '')
#um total de 92208 caiu para 91211 quando tiramos os com menos de 50.
# =============================================================================
# Recupera o conjunto de dados, já excluindo:
# 1) os que tem menos que a quantidade minima
# =============================================================================
query = query + ' AND NOT cd_assunto_trf:(' + codigosAbaixoQuantidadeMinima + ')'
solrDataAnalise = solr.query(nomeCore,{
'q':query,'fl':'id,id_processo_documento,cd_assunto_trf', 'rows':'300000'
})
dfGeral = pd.DataFrame(solrDataAnalise.docs)  
recuperaHierarquiaAssuntos(dfGeral)
    
analisaTodosOsNiveis(dfGeral, nomePasta+'/imagens/'+nomeDataSet+'Distribuicao_De_Processos_Por_Nivel_Assunto_ArvoreCompleta_2017.png', 'Distribuição de Processo Por Nível de Assunto - Árvore Completa')
# =============================================================================
# Fazendo uma análise específica da subarvore do DIREITO DO TRABALHO (Código 864)
# =============================================================================

dfGeralJT = dfGeral.query('cd_assunto_nivel_1==' + str(864))  
#recuperaHierarquiaAssuntos(dfGeralJT)
codigosJT =recuperaFilhosDoNivel1(864)
codigosJT = ' '.join(map(str, codigosJT)) 

analisaTodosOsNiveis(dfGeralJT, nomePasta+'/imagens/'+nomeDataSet+'Distribuicao_De_Processos_Por_Nivel_Assunto_DiretoDoTrabalho_2017.png', 'Distribuição de Processo Por Nível de Assunto - Árvore de Direito do Trabalho')

# =============================================================================
# Marca os elementos que são ja JT (como usa só o assunto filho, nao da pra saber direto la)
# =============================================================================

#
#queryJT = query + ' AND cd_assunto_trf:(' + codigosJT + ')'
#solrDataAnalise = solr.query(nomeCore,{
#'q':queryJT,'fl':'id,id_processo_documento,cd_assunto_trf', 'rows':'300000'
#})
#dfJT = pd.DataFrame(solrDataAnalise.docs)  
#marcarDocumentosSolr('isJT',dfJT['id'], 'true')
#
#
#queryNOT_JT = query + ' AND NOT cd_assunto_trf:(' + codigosJT + ')'
#solrDataAnalise = solr.query(nomeCore,{
#'q':queryNOT_JT,'fl':'id,id_processo_documento,cd_assunto_trf', 'rows':'300000'
#})
#df_NOTJT = pd.DataFrame(solrDataAnalise.docs)  
#marcarDocumentosSolr('isJT',df_NOTJT['id'], 'false')
#query = query + ' AND isJT:true'

# =============================================================================
# Analisa a representatividade de uma amostra
# =============================================================================
countGeral_nivel3 = dfGeral.groupby('cd_assunto_nivel_3').id_processo_documento.count()
countGeral_nivel3=pd.DataFrame(countGeral_nivel3)
countGeral_nivel3.sort_values(['id_processo_documento'], ascending=False)



#pega um ano pra traz, considerando que os ultimos dados foram os de outubro.
#query_2017_2018 = query + ' AND dt_juntada:[2017-11-01T00:00:58.518Z TO 2018-10-30T23:59:58.518Z]'
#solrDataAnalise = solr.query(nomeCore,{
#'q':query_2017_2018,'fl':'id,id_processo_documento,cd_assunto_trf', 'rows':'300000'
#})
#dfJT_2017_2018 = pd.DataFrame(solrDataAnalise.docs)  
#recuperaHierarquiaAssuntos(dfJT_2017_2018)
    

####################################################################################################################################
# =============================================================================
# CORPUS INTEIRO
# =============================================================================
dicionarioFinal = corpora.Dictionary('')
start_time = time.time()
listaProcessada = []
if os.path.exists('./Data/corpus/listaProcessadaFinal_'+nomeDataSet+'_CorpusCompleto.csv'):
  os.remove('./Data/corpus/listaProcessadaFinal_'+nomeDataSet+'_CorpusCompleto.csv')
for resCursor in solr.cursor_query(nomeCore,{'q':query,'rows':'100','fl':'tx_conteudo_documento','sort':'id asc'}):  
    listaProcessada = Parallel(n_jobs = 7)(delayed(processa_texto)(documento.get('tx_conteudo_documento')) for documento in resCursor.docs)
    dicionarioParcial = corpora.Dictionary(listaProcessada)
    dicionarioFinal.merge_with(dicionarioParcial)
    with open('./Data/corpus/listaProcessadaFinal_'+nomeDataSet+'_CorpusCompleto.csv', "a") as fp:
        wr = csv.writer(fp, dialect='excel')
        for row in listaProcessada:
            wr.writerow(row)           
end_time = time.time() - start_time
print('Tempo de processamento do texto:' + str(timedelta(seconds=end_time)))
#2029 segundos -> 33 minutos
print("--------------------------- passou aqui 7")

#------------------------------------------------------------------------------
# Salva dicionario
#------------------------------------------------------------------------------
if os.path.exists('./Data/corpus/dicionarioFinal_'+nomeDataSet+'_CorpusCompleto.dict'):
  os.remove('./Data/corpus/dicionarioFinal_'+nomeDataSet+'_CorpusCompleto.dict')
dicionarioFinal.save('./Data/corpus/dicionarioFinal_'+nomeDataSet+'_CorpusCompleto.dict')    


#carrega dicionaria
dicionarioFinal=corpora.Dictionary.load('./Data/corpus/dicionarioFinal_'+nomeDataSet+'_CorpusCompleto.dict', mmap='r')
tamanho_dicionario = len(dicionarioFinal.keys())
tamanho_dicionario

#------------------------------------------------------------------------------
# Cria o corpus de Bag of Words
#------------------------------------------------------------------------------
start_time = time.time()        
class MyCorpus_Treinamento_Doc2Bow(object):
    def __iter__(self):
        for line in open('./Data/corpus/listaProcessadaFinal_'+nomeDataSet+'_CorpusCompleto.csv'):
            yield dicionarioFinal.doc2bow(line.split(','))
corpora.MmCorpus.serialize('./Data/corpus/CorpusCompleto_BOW_'+nomeDataSet+'.mm', MyCorpus_Treinamento_Doc2Bow())
end_time = time.time() - start_time
print('Tempo de criação do matriz BOW:' + str(timedelta(seconds=end_time)))


corpus_bow_sparse = matutils.corpus2csc(corpora.MmCorpus('./Data/corpus/CorpusCompleto_BOW_'+nomeDataSet+'.mm'), tamanho_dicionario).transpose()
corpus_bow_sparse.shape


#------------------------------------------------------------------------------
# Cria o corpus TF-IDF
#------------------------------------------------------------------------------
start_time = time.time()
modeloTfidf = TfidfModel(corpora.MmCorpus('./Data/corpus/CorpusCompleto_BOW_'+nomeDataSet+'.mm') , id2word=dicionarioFinal, normalize=True)
modeloTfidf.save('./Data/corpus/CorpusCompleto_TDIDF_'+nomeDataSet+'.tfidf_model')
MmCorpus.serialize('./Data/corpus/CorpusCompleto_TDIDF_'+nomeDataSet+'.mm', modeloTfidf[corpora.MmCorpus('./Data/corpus/CorpusCompleto_BOW_'+nomeDataSet+'.mm')], progress_cnt=10000)
del(modeloTfidf)
end_time = time.time() - start_time
print('Tempo de criação do matriz TD-IDF:' + str(timedelta(seconds=end_time)))


corpus_tfidf_sparse = matutils.corpus2csc(corpora.MmCorpus('./Data/corpus/CorpusCompleto_TDIDF_'+nomeDataSet+'.mm'), tamanho_dicionario).transpose()
corpus_tfidf_sparse.shape

# =============================================================================
# Recupera todos os assuntos
# =============================================================================
assuntosCorpusInteiro = solr.query(nomeCore,{'q':query,'rows':'1000000','fl':'cd_assunto_trf','sort':'id asc'})
assuntosCorpusInteiro = pd.DataFrame(assuntosCorpusInteiro.docs)    
assuntosCorpusInteiro.shape
recuperaHierarquiaAssuntos(assuntosCorpusInteiro)

# =============================================================================
# Recupera as classes de acordo com o nível 3
# =============================================================================
counts = pd.DataFrame(assuntosCorpusInteiro['cd_assunto_trf'].astype('category').values.describe())
classes = pd.DataFrame(assuntosCorpusInteiro['cd_assunto_trf'].astype('category').values.describe())
classes.reset_index(inplace=True)
classes = classes['categories']
classes = pd.DataFrame({'cd_assunto_trf':classes})
recuperaHierarquiaAssuntos(classes)

classes = classes.query('cd_assunto_nivel_3!=' + str(0))
classes.reset_index(inplace=True)
classes = classes[['cd_assunto_nivel_3']]
classes = pd.DataFrame(classes)
classes = classes['cd_assunto_nivel_3'].astype('category').values.describe()
classes.reset_index(inplace=True)
classes = classes['categories']
classes = pd.DataFrame({'cd_assunto_nivel_3':classes})
classes = classes.cd_assunto_nivel_3.tolist()
classes = [int(i) for i in classes]
type(classes)

################################################################################################################################
# INDUÇÃO DE MODELOS 
################################################################################################################################
from funcoes import *
#------------------------------------------------------------------------------
# Modelos TD-IDF
#------------------------------------------------------------------------------

 

#TODO: fazer a curva de aprendizagem do ganho do algoritmo com a quantidade de elementos para verificar se precisa rodar com tudo etc. 
    #https://www.kaggle.com/residentmario/learning-curves-with-zillow-economics-data/

#
    
    
naive_bayes(corpus_tfidf_sparse, assuntosCorpusInteiro['cd_assunto_nivel_3'], classes,'TFIDF',nomePasta)
random_forest(corpus_tfidf_sparse, assuntosCorpusInteiro['cd_assunto_nivel_3'], classes,'TFIDF',nomePasta)
mlp(corpus_tfidf_sparse, assuntosCorpusInteiro['cd_assunto_nivel_3'], classes,'TFIDF',nomePasta)
svm_baggin(corpus_tfidf_sparse, assuntosCorpusInteiro['cd_assunto_nivel_3'], classes,'TFIDF',nomePasta)

preencheAvaliacaoFinal(nomeExperimento,nomePasta,nomeArqutivoResultadosCompilados, featureType)
# =============================================================================
# LSI
# =============================================================================
#------------------------------------------------------------------------------
# Cria o corpus LSI
#------------------------------------------------------------------------------
nomeDataSet = 'TRT13_2G_JT_REDUZIDO'
featureType = 'LSI_200'
nomeExperimento = nomeDataSet+'_'+featureType+'_CorpusCompleto_CV5'
numeroExperimento = '13'
nomePastaRestultados='./Resultados'
nomeArqutivoResultadosCompilados=nomePastaRestultados+'/resultadosFinaisCompilados.csv'
nomePasta = nomePastaRestultados + '/Experimento ' + numeroExperimento + ' - ' + nomeExperimento

nomeCore='documentos_2g_trt3'

import os
if not os.path.exists(nomePasta):
    os.makedirs(nomePasta)
    os.makedirs(nomePasta+'/imagens')

num_topics=200
start_time = time.time()
modeloLSITreinamento = LsiModel(corpora.MmCorpus('./Data/corpus/CorpusCompleto_TDIDF_'+nomeDataSet+'.mm'), id2word=dicionarioFinal, num_topics=num_topics)
modeloLSITreinamento.save('./Data/corpus/corpus_LSI_'+str(num_topics)+'_'+nomeDataSet+'.mm')
MmCorpus.serialize('./Data/corpus/corpus_LSI_'+str(num_topics)+'_'+nomeDataSet+'.mm', modeloLSITreinamento[corpora.MmCorpus('./Data/corpus/CorpusCompleto_TDIDF_'+nomeDataSet+'.mm')], progress_cnt=10000)
end_time = time.time() - start_time
print('Tempo de processamento do LSI:' + str(timedelta(seconds=end_time)))
#Tempo de processamento do LSI:0:13:00.940066
corpus_lsi_sparse = matutils.corpus2csc(corpora.MmCorpus('./Data/corpus/corpus_LSI_'+str(num_topics)+'_'+nomeDataSet+'.mm'), num_topics).transpose()
corpus_lsi_sparse.shape


teste = corpus_lsi_sparse.todense()
from sklearn.preprocessing import normalize, MinMaxScaler
scaler = MinMaxScaler()
scaled_teste = scaler.fit_transform(teste)
teste_normalized = normalize(teste, norm='max')
teste.head(4)

naive_bayes(scaled_teste, assuntosCorpusInteiro['cd_assunto_nivel_3'], classes,'LSI',nomePasta)
random_forest(corpus_lsi_sparse, assuntosCorpusInteiro['cd_assunto_nivel_3'], classes,'LSI',nomePasta)
mlp(corpus_lsi_sparse, assuntosCorpusInteiro['cd_assunto_nivel_3'], classes,'LSI',nomePasta)
svm_bagging(corpus_lsi_sparse, assuntosCorpusInteiro['cd_assunto_nivel_3'], classes,'LSI',nomePasta)

preencheAvaliacaoFinal(nomeExperimento,nomePasta,nomeArqutivoResultadosCompilados, featureType)

# =============================================================================
# LSI
# =============================================================================
#------------------------------------------------------------------------------
# Cria o corpus LSI
#------------------------------------------------------------------------------
nomeDataSet = 'TRT13_2G_JT_REDUZIDO'
featureType = 'LSI_300'
nomeExperimento = nomeDataSet+'_'+featureType+'_CorpusCompleto_CV5'
numeroExperimento = '15'
nomePastaRestultados='./Resultados'
nomeArqutivoResultadosCompilados=nomePastaRestultados+'/resultadosFinaisCompilados.csv'
nomePasta = nomePastaRestultados + '/Experimento ' + numeroExperimento + ' - ' + nomeExperimento

nomeCore='documentos_2g_trt3'

import os
if not os.path.exists(nomePasta):
    os.makedirs(nomePasta)
    os.makedirs(nomePasta+'/imagens')

num_topics=300
start_time = time.time()
modeloLSITreinamento = LsiModel(corpora.MmCorpus('./Data/corpus/CorpusCompleto_TDIDF_'+nomeDataSet+'.mm'), id2word=dicionarioFinal, num_topics=num_topics)
modeloLSITreinamento.save('./Data/corpus/corpus_LSI_'+str(num_topics)+'_'+nomeDataSet+'.mm')
MmCorpus.serialize('./Data/corpus/corpus_LSI_'+str(num_topics)+'_'+nomeDataSet+'.mm', modeloLSITreinamento[corpora.MmCorpus('./Data/corpus/CorpusCompleto_TDIDF_'+nomeDataSet+'.mm')], progress_cnt=10000)
end_time = time.time() - start_time
print('Tempo de processamento do LSI:' + str(timedelta(seconds=end_time)))
#Tempo de processamento do LSI:0:13:00.940066
corpus_lsi_sparse = matutils.corpus2csc(corpora.MmCorpus('./Data/corpus/corpus_LSI_'+str(num_topics)+'_'+nomeDataSet+'.mm'), num_topics).transpose()
corpus_lsi_sparse.shape


teste = corpus_lsi_sparse.todense()
from sklearn.preprocessing import normalize, MinMaxScaler
scaler = MinMaxScaler()
scaled_teste = scaler.fit_transform(teste)
teste_normalized = normalize(teste, norm='max')

naive_bayes(scaled_teste, assuntosCorpusInteiro['cd_assunto_nivel_3'], classes,'LSI',nomePasta)
random_forest(corpus_lsi_sparse, assuntosCorpusInteiro['cd_assunto_nivel_3'], classes,'LSI',nomePasta)
mlp(corpus_lsi_sparse, assuntosCorpusInteiro['cd_assunto_nivel_3'], classes,'LSI',nomePasta)
svm_bagging(corpus_lsi_sparse, assuntosCorpusInteiro['cd_assunto_nivel_3'], classes,'LSI',nomePasta)

preencheAvaliacaoFinal(nomeExperimento,nomePasta,nomeArqutivoResultadosCompilados, featureType)

# =============================================================================
# LSI
# =============================================================================
#------------------------------------------------------------------------------
# Cria o corpus LSI
#------------------------------------------------------------------------------
nomeDataSet = 'TRT13_2G_JT_REDUZIDO'
featureType = 'LSI_350'
nomeExperimento = nomeDataSet+'_'+featureType+'_CorpusCompleto_CV5'
numeroExperimento = '16'
nomePastaRestultados='./Resultados'
nomeArqutivoResultadosCompilados=nomePastaRestultados+'/resultadosFinaisCompilados.csv'
nomePasta = nomePastaRestultados + '/Experimento ' + numeroExperimento + ' - ' + nomeExperimento

nomeCore='documentos_2g_trt3'

import os
if not os.path.exists(nomePasta):
    os.makedirs(nomePasta)
    os.makedirs(nomePasta+'/imagens')

num_topics=350
start_time = time.time()
modeloLSITreinamento = LsiModel(corpora.MmCorpus('./Data/corpus/CorpusCompleto_TDIDF_'+nomeDataSet+'.mm'), id2word=dicionarioFinal, num_topics=num_topics)
modeloLSITreinamento.save('./Data/corpus/corpus_LSI_'+str(num_topics)+'_'+nomeDataSet+'.mm')
MmCorpus.serialize('./Data/corpus/corpus_LSI_'+str(num_topics)+'_'+nomeDataSet+'.mm', modeloLSITreinamento[corpora.MmCorpus('./Data/corpus/CorpusCompleto_TDIDF_'+nomeDataSet+'.mm')], progress_cnt=10000)
end_time = time.time() - start_time
print('Tempo de processamento do LSI:' + str(timedelta(seconds=end_time)))
#Tempo de processamento do LSI:0:13:00.940066
corpus_lsi_sparse = matutils.corpus2csc(corpora.MmCorpus('./Data/corpus/corpus_LSI_'+str(num_topics)+'_'+nomeDataSet+'.mm'), num_topics).transpose()
corpus_lsi_sparse.shape


teste = corpus_lsi_sparse.todense()
from sklearn.preprocessing import normalize, MinMaxScaler
scaler = MinMaxScaler()
scaled_teste = scaler.fit_transform(teste)
teste_normalized = normalize(teste, norm='max')

naive_bayes(scaled_teste, assuntosCorpusInteiro['cd_assunto_nivel_3'], classes,'LSI',nomePasta)
random_forest(corpus_lsi_sparse, assuntosCorpusInteiro['cd_assunto_nivel_3'], classes,'LSI',nomePasta)
mlp(corpus_lsi_sparse, assuntosCorpusInteiro['cd_assunto_nivel_3'], classes,'LSI',nomePasta)
svm_bagging(corpus_lsi_sparse, assuntosCorpusInteiro['cd_assunto_nivel_3'], classes,'LSI',nomePasta)

preencheAvaliacaoFinal(nomeExperimento,nomePasta,nomeArqutivoResultadosCompilados, featureType)

##############################################################################
# VERIFICANDO A DISTRIBUIÇÃO DOS DADOS
##############################################################################

#-----------------------------------------------------------------------------
# Quantidade média de palavras por classe
#-----------------------------------------------------------------------------
    
contador =  pd.DataFrame(columns=['qtd_palavras','cd_assunto_nivel_2'])
count = 0;
for y in range(0,len(df)):
    #normalizando o texto
        documento =  df['tx_conteudo_documento'][y].split()
        for palavra in documento :
            count += 1    
        contador.loc[y] = [count, df['cd_assunto_nivel_2'][y]]
        count = 0
        
for y in range(0,len(contador)):
    contador['qtd_palavras'][y] = float(contador['qtd_palavras'][y])

contador.groupby(['cd_assunto_nivel_2']).mean()  

grouped2=pd.to_numeric(contador['qtd_palavras']).groupby(contador['cd_assunto_nivel_2'])
grouped2.mean()
  
    


#Teste para ver se o parallel manteria a ordem        
df = pd.DataFrame(columns=['inicial','final'])
for i in range(0,10000):
    df.loc[i] = [i,i]

def quadrado(f):
    return f*f
df.inicial[15]

resultado = Parallel(n_jobs = 7)(delayed(quadrado)(item) for item in df.inicial)