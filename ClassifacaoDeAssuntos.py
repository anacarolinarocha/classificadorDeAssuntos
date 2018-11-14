
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
nomeDataSet = 'TRT13_2G_JT_2017'
featureType = 'LSI_200'
nomeExperimento = nomeDataSet+'_'+featureType+'_CorpusCompleto_CV5'
numeroExperimento = '3'
nomePastaRestultados='./Resultados'
nomeArqutivoResultadosCompilados=nomePastaRestultados+'/resultadosFinaisCompilados.csv'
nomePasta = nomePastaRestultados + '/Experimento ' + numeroExperimento + ' - ' + nomeExperimento

nomeCore='documentos_2g_trt3'

import os
if not os.path.exists(nomePasta):
    os.makedirs(nomePasta)
    os.makedirs(nomePasta+'/imagens')
    
quantidadeMinimaDocumentos = 50;
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
print("--------------------------- passou aqui 1")


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
print("--------------------------- passou aqui 2")        
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
print("--------------------------- passou aqui 3")
# =============================================================================
# Fazendo uma análise específica da subarvore do DIREITO DO TRABALHO (Código 864)
# =============================================================================

dfGeralJT = dfGeral.query('cd_assunto_nivel_1==' + str(864))  
#recuperaHierarquiaAssuntos(dfGeralJT)
codigosJT =recuperaFilhosDoNivel1(864)
codigosJT = ' '.join(map(str, codigosJT)) 

analisaTodosOsNiveis(dfGeralJT, nomePasta+'/imagens/'+nomeDataSet+'Distribuicao_De_Processos_Por_Nivel_Assunto_DiretoDoTrabalho_2017.png', 'Distribuição de Processo Por Nível de Assunto - Árvore de Direito do Trabalho')
print("--------------------------- passou aqui 4")
# =============================================================================
# Marca os elementos que são ja JT (como usa só o assunto filho, nao da pra saber direto la)
# =============================================================================



queryJT = query + ' AND cd_assunto_trf:(' + codigosJT + ')'
solrDataAnalise = solr.query(nomeCore,{
'q':queryJT,'fl':'id,id_processo_documento,cd_assunto_trf', 'rows':'300000'
})
dfJT = pd.DataFrame(solrDataAnalise.docs)  
marcarDocumentosSolr('isJT',dfJT['id'], 'true')
print("--------------------------- passou aqui 5")


queryNOT_JT = query + ' AND NOT cd_assunto_trf:(' + codigosJT + ')'
solrDataAnalise = solr.query(nomeCore,{
'q':queryNOT_JT,'fl':'id,id_processo_documento,cd_assunto_trf', 'rows':'300000'
})
df_NOTJT = pd.DataFrame(solrDataAnalise.docs)  
marcarDocumentosSolr('isJT',df_NOTJT['id'], 'false')
query = query + ' AND isJT:true'

print("--------------------------- passou aqui 6")

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
#dicionarioFinal=corpora.Dictionary.load('./Data/corpus/dicionarioFinal_CorpusCompleto_JT_2017.dict', mmap='r')
tamanho_dicionario = len(dicionarioFinal.keys())
tamanho_dicionario

#------------------------------------------------------------------------------
# Cria o corpus de Bag of Words
#------------------------------------------------------------------------------
start_time = time.time()        
class MyCorpus_Treinamento_Doc2Bow(object):
    def __iter__(self):
        for line in open('./Data/corpus/listaProcessadaFinal_CorpusCompleto_JT_2017.csv'):
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

#------------------------------------------------------------------------------
# Cria o corpus LSI
#------------------------------------------------------------------------------
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


corpus_lsi_sparse = matutils.corpus2csc(corpora.MmCorpus('./Data/corpus/corpus_LSI_'+str(200)+'_'+nomeDataSet+'.mm'), 200).transpose()
corpus_lsi_sparse.shape

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


#
#naive_bayes(corpus_tfidf_sparse, assuntosCorpusInteiro['cd_assunto_nivel_3'], classes,'TFIDF')
#mlp(corpus_tfidf_sparse, assuntosCorpusInteiro['cd_assunto_nivel_3'], classes,'TFIDF')
#svm(corpus_tfidf_sparse, assuntosCorpusInteiro['cd_assunto_nivel_3'], classes,'TFIDF')
#random_forest(corpus_tfidf_sparse, assuntosCorpusInteiro['cd_assunto_nivel_3'], classes,'TFIDF')

#------------------------------------------------------------------------------
# Modelos LSI
#------------------------------------------------------------------------------


naive_bayes(corpus_lsi_sparse, assuntosCorpusInteiro['cd_assunto_nivel_3'], classes,'LSI',nomePasta)
random_forest(corpus_lsi_sparse, assuntosCorpusInteiro['cd_assunto_nivel_3'], classes,'LSI',nomePasta)
mlp(corpus_lsi_sparse, assuntosCorpusInteiro['cd_assunto_nivel_3'], classes,'LSI',nomePasta)
svm(corpus_lsi_sparse, assuntosCorpusInteiro['cd_assunto_nivel_3'], classes,'LSI',nomePasta)

preencheAvaliacaoFinal(nomeExperimento,nomePasta,nomeArqutivoResultadosCompilados, featureType)




#TODO: fazer a curva de aprendizagem do ganho do algoritmo com a quantidade de elementos para verificar se precisa rodar com tudo etc. 
    #https://www.kaggle.com/residentmario/learning-curves-with-zillow-economics-data/



# =============================================================================
# Ensemble
# =============================================================================
from sklearn.ensemble import VotingClassifier
estimators = []

model1 = MultinomialNB(class_prior=None,fit_prior=True)
model2 = clf_SVM_mc1_tfidf = SGDClassifier(loss='modified_huber', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)
model3 = clf_RF_mc1_tfidf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=50, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=50, min_samples_split=50,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=7,
            oob_score=False, random_state=0, verbose=0, warm_start=False)
model4 = clf_MLP_mc1_tfidf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(10, 5), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9, 
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)

estimators.append(('Naive Bayes', model1))
estimators.append(('SVM', model2))
estimators.append(('Random Forest', model3))
#estimators.append(('Naive Bayes', model4))
# create the ensemble model
ensemble = VotingClassifier(estimators)

from sklearn import model_selection
kfold = model_selection.KFold(n_splits=5, random_state=12)
results = model_selection.cross_val_score(ensemble, corpus_treinamento_mc2_tfidf_sparse, assuntosMacroClasse2_Treinamento, cv=kfold)
print(results.mean())


type(corpus_treinamento_mc3_tfidf_sparse)
teste = corpus_treinamento_mc1_tfidf_sparse.todense()




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