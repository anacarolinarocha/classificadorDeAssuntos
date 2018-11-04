
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


quantidadeMinimaDocumentos = 50;
solr = SolrClient('http://localhost:8983/solr')


##############################################################################    
##############################################################################
# ANALISA OS DADOS
##############################################################################
##############################################################################



print(recuperaNivelAssunto(55327))
print(recuperaAssuntoNivelEspecifico(2554,4))

# =============================================================================
# Retira dos dados aqueles que tem menos de 50 exemplares no nivel 3.
# Além de ser necessário ter uma amostra representativa dos dados, o cross 
# validation não funcionará bem se não houver uma amostra suficientemente grande
# para popular cada fold.
# =============================================================================
query = 'tx_conteudo_documento:[* TO *]'
solrDataAnalise = solr.query('documentos',{
'q':query,'fl':'id_processo_documento,cd_assunto_trf', 'rows':'300000'
})
dfGeral = pd.DataFrame(solrDataAnalise.docs)

recuperaHierarquiaAssuntos(dfGeral)



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
solrDataAnalise = solr.query('documentos',{
'q':query,'fl':'id,id_processo_documento,cd_assunto_trf', 'rows':'300000'
})
dfGeral = pd.DataFrame(solrDataAnalise.docs)  
recuperaHierarquiaAssuntos(dfGeral)
    
analisaTodosOsNiveis(dfGeral, './imagens/TRT15_2GRAU_Distribuicao_De_Processos_Por_Nivel_Assunto_ArvoreCompleta_2017.png', 'Distribuição de Processo Por Nível de Assunto - Árvore Completa')

# =============================================================================
# Fazendo uma análise específica da subarvore do DIREITO DO TRABALHO (Código 864)
# =============================================================================
dfGeralJT = dfGeral.query('cd_assunto_nivel_1==' + str(864))  
recuperaHierarquiaAssuntos(dfGeralJT)
    
analisaTodosOsNiveis(dfGeralJT, './imagens/TRT15_2GRAU_Distribuicao_De_Processos_Por_Nivel_Assunto_DiretoDoTrabalho_2017.png', 'Distribuição de Processo Por Nível de Assunto - Árvore de Direito do Trabalho')


#USANDO FACETS.....
#solrDataAnalise = solr.query('classificacaoDeDocumentos_hierarquiaCompleta',{
#'q':'tx_conteudo_documento:[* TO *]',
#'rows':'10',
#'facet':True,
#'facet.field':'cd_assunto_nivel_1',
#'facet.mincount':'10'
#})
#columns = ['cd_assunto_nivel_1','count']
#facets = pd.DataFrame(columns=columns)
#facets['cd_assunto_nivel_2']= solrDataAnalise.get_facet_keys_as_list('cd_assunto_nivel_1')
#facets['count']= solrDataAnalise.get_facet_values_as_list('cd_assunto_nivel_1')
#dfFacet = pd.DataFrame(solrDataAnalise.facet_pivot)    


# =============================================================================
# Criando conjuntos de treinamento e teste estratificados
# =============================================================================
#Codigo a ser rodado na primeira vez. depois que o Solr já está 'marcado', nao se faz split denovo.
#solrDataAnalise = solr.query('documentos',{
#'q':query,'fl':'id,id_processo_documento,cd_assunto_trf', 'rows':'300000'
#})
#dfFinal = pd.DataFrame(solrDataAnalise.docs)    
#
#recuperaHierarquiaAssuntos(dfFinal)

df_idProcessos_treinamento, df_idProcessos_teste, df_codigoAssunto_treinamento, df_codigoAssunto_teste = train_test_split(dfGeralJT[['id','id_processo_documento']], dfGeralJT['cd_assunto_nivel_3'],
                                                    stratify=dfGeralJT['cd_assunto_nivel_3'], 
                                                    test_size=0.3)

# =============================================================================
# Marca os elementos que serão usados para teste no Solr
# =============================================================================
def marcarDocumentosSolr(data, flag):
    documentos = []
    for ids in data:
        doc = {'id': ids, 'isTeste':{'set': flag}}
        documentos.append(doc)
    solr.index_json('documentos',json.dumps(documentos))
    solr.commit(openSearcher=True, collection='documentos')

marcarDocumentosSolr(df_idProcessos_treinamento['id'], 'false')
marcarDocumentosSolr(df_idProcessos_teste['id'], 'true')



queryTreinamento = query + ' AND NOT isTeste:true'
queryTeste  = query + ' AND isTeste:true'

solrDataAnalise = solr.query('documentos',{
'q':queryTreinamento,'fl':'id, id_processo_documento', 'rows':'300000'
})
df_idProcessos_treinamento = pd.DataFrame(solrDataAnalise.docs)

solrDataAnalise = solr.query('documentos',{
'q':queryTeste,'fl':'id, id_processo_documento', 'rows':'300000'
})
df_idProcessos_teste = pd.DataFrame(solrDataAnalise.docs)    

solrDataAnalise = solr.query('documentos',{
'q':queryTreinamento,'fl':'cd_assunto_trf', 'rows':'300000'
})
df_codigoAssunto_treinamento = pd.DataFrame(solrDataAnalise.docs)
recuperaHierarquiaAssuntos(df_codigoAssunto_treinamento)
df_codigoAssunto_treinamento=df_codigoAssunto_treinamento['cd_assunto_nivel_3']

solrDataAnalise = solr.query('documentos',{
'q':queryTeste,'fl':'cd_assunto_trf', 'rows':'300000'
})
df_codigoAssunto_teste = pd.DataFrame(solrDataAnalise.docs)    
recuperaHierarquiaAssuntos(df_codigoAssunto_teste)
df_codigoAssunto_teste=df_codigoAssunto_teste['cd_assunto_nivel_3']


################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################

# =============================================================================
# Processa conjunto de treinamento e de teste
# =============================================================================

################################################################################################################################
# BUSCA DADOS E CRIA O DICIONÁRIO
################################################################################################################################
dicionarioFinal = corpora.Dictionary('')

start_time = time.time()
listaProcessada = []
if os.path.exists('./Data/corpus/listaProcessadaFinal_Treinamento.csv'):
  os.remove('./Data/corpus/listaProcessadaFinal_Treinamento.csv')
for resCursor in solr.cursor_query('classificacaoDeDocumentos_hierarquiaCompleta',{'q':queryTreinamento,'rows':'100','fl':'tx_conteudo_documento','sort':'id asc'}):  
    listaProcessada = Parallel(n_jobs = 7)(delayed(processa_texto)(documento.get('tx_conteudo_documento')) for documento in resCursor.docs)
    dicionarioParcial = corpora.Dictionary(listaProcessada)
    dicionarioFinal.merge_with(dicionarioParcial)
    with open("./Data/corpus/listaProcessadaFinal_Treinamento.csv", "a") as fp:
        wr = csv.writer(fp, dialect='excel')
        for row in listaProcessada:
            wr.writerow(row)           
end_time = time.time() - start_time
print('Tempo de processamento do texto de treinamento:' + str(timedelta(seconds=end_time)))
#2029 segundos -> 33 minutos

start_time = time.time()
listaProcessada = []
if os.path.exists('./Data/corpus/listaProcessadaFinal_Teste.csv'):
  os.remove('./Data/corpus/listaProcessadaFinal_Teste.csv')

for resCursor in solr.cursor_query('classificacaoDeDocumentos_hierarquiaCompleta',{'q':queryTeste,'rows':'1000','fl':'tx_conteudo_documento','sort':'id asc'}):  
    listaProcessada = Parallel(n_jobs = 7)(delayed(processa_texto)(documento.get('tx_conteudo_documento')) for documento in resCursor.docs)
    dicionarioParcial = corpora.Dictionary(listaProcessada)
    dicionarioFinal.merge_with(dicionarioParcial)
    with open("./Data/corpus/listaProcessadaFinal_Teste.csv", "a") as fp:
        wr = csv.writer(fp, dialect='excel')
        for row in listaProcessada:
            wr.writerow(row)           
end_time = time.time() - start_time
print('Tempo de processamento do texto de teste:' + str(timedelta(seconds=end_time)))

if os.path.exists('./Data/corpus/dicionarioFinal.dict'):
  os.remove('./Data/corpus/dicionarioFinal.dict')
dicionarioFinal.save('./Data/corpus/dicionarioFinal.dict')    
#436 segundos -> 7 minutos

#------------------------------------------------------------------------------
# Salva dicionario
#------------------------------------------------------------------------------


#carrega dicionaria
#dicionarioFinal=corpora.Dictionary.load('./Data/corpus/dicionarioFinal.dict', mmap='r')
tamanho_dicionario = len(dicionarioFinal.keys())

###############################################################################################################################
# CRIA VETORES DE TEXTO
###############################################################################################################################

#******************************************************************************************************************************
# Dados de Treinamento
#******************************************************************************************************************************

#------------------------------------------------------------------------------
# Cria o corpus de Bag of Words
#------------------------------------------------------------------------------
start_time = time.time()        
class MyCorpus_Treinamento_Doc2Bow(object):
    def __iter__(self):
        for line in open('./Data/corpus/listaProcessadaFinal_Treinamento.csv'):
            yield dicionarioFinal.doc2bow(line.split(','))
corpora.MmCorpus.serialize('./Data/corpus/corpusTreinamento_BOW.mm', MyCorpus_Treinamento_Doc2Bow())
end_time = time.time() - start_time
print('Tempo de criação do matriz BOW de treinamento:' + str(timedelta(seconds=end_time)))


corpus_treinamento_bow_sparse = matutils.corpus2csc(corpora.MmCorpus('./Data/corpus/corpusTreinamento_BOW.mm'), tamanho_dicionario).transpose()
corpus_treinamento_bow_sparse.shape

#------------------------------------------------------------------------------
# Cria o corpus TF-IDF
#------------------------------------------------------------------------------
start_time = time.time()
modeloTfidfTreinamento = TfidfModel(corpora.MmCorpus('./Data/corpus/corpusTreinamento_BOW.mm') , id2word=dicionarioFinal, normalize=True)
modeloTfidfTreinamento.save('./Data/corpus/corpusTreinamento_TFIDF.tfidf_model')
MmCorpus.serialize('./Data/corpus/corpusTreinamento_TFIDF.mm', modeloTfidfTreinamento[corpora.MmCorpus('./Data/corpus/corpusTreinamento_BOW.mm')], progress_cnt=10000)
del(modeloTfidfTreinamento)
end_time = time.time() - start_time
print('Tempo de criação do matriz TD-IDF de treinamento:' + str(timedelta(seconds=end_time)))


corpus_treinamento_tfidf_sparse = matutils.corpus2csc(corpora.MmCorpus('./Data/corpus/corpusTreinamento_TFIDF.mm'), tamanho_dicionario).transpose()
corpus_treinamento_tfidf_sparse.shape

#------------------------------------------------------------------------------
# Cria o corpus LSI
#------------------------------------------------------------------------------
#num_topics=300
#start_time = time.time()
#modeloLSITreinamento = LsiModel(corpora.MmCorpus('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/corpusTreinamento_TFIDF.mm'), id2word=dicionarioFinal, num_topics=num_topics)
#modeloLSITreinamento.save('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/corpusTreinamento_LSI.lsi_model')
#MmCorpus.serialize('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/corpusTreinamento_LSI.mm', modeloLSITreinamento[corpora.MmCorpus('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/corpusTreinamento_TFIDF.mm')], progress_cnt=10000)
#del(modeloLSITreinamento)
#print(time.time() - start_time)
#
#corpus_treinamento_lsi_sparse = matutils.corpus2csc(corpora.MmCorpus('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/corpusTreinamento_LSI.mm'), num_topics).transpose()
#corpus_treinamento_lsi_sparse.shape

#------------------------------------------------------------------------------
# Busca o target do conjunto de treinamento: assunto de nível 2
#------------------------------------------------------------------------------
 
assuntos_Treinamento = solr.query('classificacaoDeDocumentos_hierarquiaCompleta',{'q':queryTreinamento,'rows':'1000000','fl':'cd_assunto_nivel_2','sort':'id asc'})
assuntos_Treinamento = pd.DataFrame(assuntos_Treinamento.docs)    
assuntos_Treinamento.shape


#verifica se o numero de documentos bate com o numero de assuntos
#assuntosMacroClasse1_Treinamento.shape
#row_count = sum(1 for line in open("/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/listaProcessadaFinal_mc1_Treinamento.csv"))


gc.collect()
#******************************************************************************************************************************
# TESTE
#******************************************************************************************************************************

#------------------------------------------------------------------------------
# Cria o corpus de Bag of Words
#------------------------------------------------------------------------------
start_time = time.time()
class MyCorpus_Teste_Doc2Bow(object):
    def __iter__(self):
        for line in open('./Data/corpus/listaProcessadaFinal_Teste.csv'):
            yield dicionarioFinal.doc2bow(line.split(','))
corpora.MmCorpus.serialize('./Data/corpus/corpusTeste_BOW.mm', MyCorpus_Teste_Doc2Bow())
end_time = time.time() - start_time
print('Tempo de criação do matriz BOW de teste:' + str(timedelta(seconds=end_time)))

corpus_teste_bow_sparse = matutils.corpus2csc(corpora.MmCorpus('./Data/corpus/corpusTeste_BOW.mm'), tamanho_dicionario).transpose()
corpus_teste_bow_sparse.shape
  
#------------------------------------------------------------------------------
# Cria o corpus TF-IDF
#------------------------------------------------------------------------------
start_time = time.time()
modeloTfidfTeste = TfidfModel(corpora.MmCorpus('./Data/corpus/corpusTeste_BOW.mm'), id2word=dicionarioFinal, normalize=True)
modeloTfidfTeste.save('./Data/corpus/corpusTeste_TFIDF.tfidf_model')
MmCorpus.serialize('./Data/corpus/corpusTeste_TFIDF.mm', modeloTfidfTeste[corpora.MmCorpus('./Data/corpus/corpusTeste_BOW.mm')], progress_cnt=10000)
end_time = time.time() - start_time
print('Tempo de criação do matriz TD-IDF de teste:' + str(timedelta(seconds=end_time)))


corpus_teste_tfidf_sparse = matutils.corpus2csc(corpora.MmCorpus('./Data/corpus/corpusTeste_TFIDF.mm'), tamanho_dicionario).transpose()
corpus_teste_tfidf_sparse.shape

#------------------------------------------------------------------------------
# Cria o corpus LSI
#------------------------------------------------------------------------------
#num_topics_mc1=300
#start_time = time.time()
#modeloLSITeste = LsiModel(corpora.MmCorpus('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/corpusTeste_TFIDF.mm'), id2word=dicionarioFinal, num_topics=num_topics)
#modeloLSITeste.save('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/corpusTeste_LSI.lsi_model')
#MmCorpus.serialize('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/corpusTeste_LSI.mm', modeloLSITeste[corpora.MmCorpus('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/corpusTeste_TFIDF.mm')], progress_cnt=10000)
#del(modeloLSITeste)
#print(time.time() - start_time)
#
#corpus_teste_lsi_sparse = matutils.corpus2csc(corpora.MmCorpus('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/corpusTeste_TFIDF.mm'), num_topics).transpose()
#corpus_teste_lsi_sparse.shape

#------------------------------------------------------------------------------
# Busca o target do conjunto de teste: assunto de nível 2
#------------------------------------------------------------------------------
assuntos_Teste = solr.query('classificacaoDeDocumentos_hierarquiaCompleta',{'q':queryTeste,'rows':'1000000','fl':'cd_assunto_nivel_2','sort':'id asc'})
assuntos_Teste = pd.DataFrame(assuntos_Teste.docs)    
assuntos_Teste.shape

#verifica se o numero de documentos bate com o numero de assuntos
#assuntosMacroClasse1_Teste.shape
#row_count = sum(1 for line in open("/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/Data/corpus/listaProcessadaFinal_mc1_Teste.csv"))


################################################################################################################################
# INDUÇÃO DE MODELOS 
################################################################################################################################

classes = pd.DataFrame(assuntos_Teste['cd_assunto_nivel_2'].astype('category').values.describe())
classes.reset_index(inplace=True)
classes = classes.categories.tolist()

#//TODO: fazer validacao cruzada
#------------------------------------------------------------------------------
# Modelos
#------------------------------------------------------------------------------

from funcoes import *
    
start_time = time.time()
naive_bayes(corpus_treinamento_tfidf_sparse,assuntos_Treinamento['cd_assunto_nivel_2'],corpus_teste_tfidf_sparse, assuntos_Teste['cd_assunto_nivel_2'], 1,classes,'TFIDF')
end_time = time.time() - start_time
print('Tempo de criação do matriz BOW de teste:' + str(timedelta(seconds=end_time)))


start_time = time.time()
mlp(corpus_treinamento_tfidf_sparse,assuntos_Treinamento['cd_assunto_nivel_2'],corpus_teste_tfidf_sparse,  assuntos_Teste['cd_assunto_nivel_2'], 1,classes,'TFIDF')
end_time = time.time() - start_time
print('Tempo da execução co Grid MLP:' + str(timedelta(seconds=end_time)))

start_time = time.time()
svm(corpus_teste_tfidf_sparse,assuntos_Teste['cd_assunto_nivel_2'],corpus_teste_tfidf_sparse, assuntos_Teste['cd_assunto_nivel_2'], 1,classes,'TFIDF')
end_time = time.time() - start_time
print('Tempo da execução co Grid SVM:' + str(timedelta(seconds=end_time)))

start_time = time.time()
random_forest(corpus_treinamento_tfidf_sparse,assuntos_Treinamento['cd_assunto_nivel_2'],corpus_teste_tfidf_sparse,  assuntos_Teste['cd_assunto_nivel_2'], 1,classes,'TFIDF')
end_time = time.time() - start_time
print('RF' + str(time.time() - start_time))




    
naive_bayes(corpus_treinamento_bow_sparse,assuntos_Treinamento['cd_assunto_nivel_2'].astype('category').values,corpus_teste_bow_sparse, assuntos_Teste['cd_assunto_nivel_2'].astype('category').values, 1,classes,'BOW')
svm(corpus_treinamento_bow_sparse,assuntos_Treinamento['cd_assunto_nivel_2'],corpus_teste_bow_sparse, assuntos_Teste['cd_assunto_nivel_2'], 1,classes,'BOW')
random_forest(corpus_treinamento_bow_sparse,assuntos_Treinamento['cd_assunto_nivel_2'],corpus_teste_bow_sparse,  assuntos_Teste['cd_assunto_nivel_2'], 1,classes,'BOW')
mlp(corpus_treinamento_bow_sparse,assuntos_Treinamento['cd_assunto_nivel_2'],corpus_teste_bow_sparse,  assuntos_Teste['cd_assunto_nivel_2'], 1,classes,'BOW')




#TODO: fazer a curva de aprendizagem do ganho do algoritmo com a quantidade de elementos para verificar se precisa rodar com tudo etc. 
    #https://www.kaggle.com/residentmario/learning-curves-with-zillow-economics-data/

# =============================================================================
# LSI
# =============================================================================

#------------------------------------------------------------------------------
# SVM MC1
#------------------------------------------------------------------------------


param_grid = {
    'loss': [ 'modified_huber', 'squared_hinge'],
    'penalty': ['elasticnet','l2'],
    'alpha': [1e-4,1e-3]
    #'l1_ratio': [0, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1],
}
clf_SVM = SGDClassifier(random_state=0, class_weight='balanced',n_jobs=7)
clf_SVM_grid = grid_search.GridSearchCV(estimator=clf_SVM, param_grid=param_grid,
                                     scoring='f1_weighted',cv = 3)
clf_SVM_grid.fit(corpus_treinamento_mc1_lsi_sparse, assuntosMacroClasse1_Treinamento.cd_assunto_nivel_2)


print(clf_SVM_grid.best_score_)
print(clf_SVM_grid.best_params_)
teste_svm_c1_segunda_execucao = pd.DataFrame(clf_SVM_grid.grid_scores_)

clf_SVM = clf_SVM_grid.best_estimator_
clf_SVM.fit(corpus_treinamento_mc1_lsi_sparse, assuntosMacroClasse1_Treinamento.cd_assunto_nivel_2)
predicted_SVM_mc1_lsi =  clf_SVM.predict(corpus_teste_mc1_lsi_sparse)
np.mean(predicted_SVM_mc1_lsi == assuntosMacroClasse1_Teste.cd_assunto_nivel_2)


codigos= pd.DataFrame(assuntosMacroClasse1_Treinamento['cd_assunto_nivel_2'].astype('category').values.describe())
codigos.reset_index(inplace=True)
codigos = codigos.categories.tolist()

confusion_matrix_SVM_mc1_lsi = confusion_matrix(assuntosMacroClasse1_Teste['cd_assunto_nivel_2'].astype('category').values,predicted_SVM_mc1_lsi)
fig = plt.figure(figsize=(10,10))

plot_confusion_matrix(confusion_matrix_SVM_mc1_lsi, codigos,
                      title='SVM \nMacro Class 1 - TF-IDF\nAccuracy: {0:.3f}'.format(accuracy_score(assuntosMacroClasse1_Teste['cd_assunto_nivel_2'].astype('category').values,predicted_SVM_mc1_lsi)))

plt.savefig('/home/anarocha/Documentos/myGit/git/classificadorDeAssuntos/imagens/confusion_matrix_SVM_mc1_lsi.png') 

macro_precision,macro_recall,macro_fscore,macro_support=score(assuntosMacroClasse1_Teste['cd_assunto_nivel_2'].astype('category').values,predicted_SVM_mc1_lsi,average='macro')
micro_precision,micro_recall,micro_fscore,micro_support=score(assuntosMacroClasse1_Teste['cd_assunto_nivel_2'].astype('category').values,predicted_SVM_mc1_lsi,average='weighted')

avaliacaoFinal_MC1_TFIDF.loc[4]= ['SVM','LSI',macro_precision,macro_recall,macro_fscore,micro_precision,micro_recall,micro_fscore]

print(classification_report(assuntosMacroClasse1_Teste['cd_assunto_nivel_2'].astype('category').values,predicted_SVM_mc1_lsi))
print('Micro average precision = {:.2f} (dâ o mesmo peso para cada instância)'.format(precision_score(assuntosMacroClasse1_Teste['cd_assunto_nivel_2'].astype('category').values,predicted_SVM_mc1_lsi,average = 'weighted')))
print('Macro average precision = {:.2f} (dâ o mesmo peso para cada classe)'.format(precision_score(assuntosMacroClasse1_Teste['cd_assunto_nivel_2'].astype('category').values,predicted_SVM_mc1_lsi,average = 'macro')))




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