import sys
sys.path.insert(1, '/home/anarocha/myGit/classificadorDeAssuntos/Codigo/001_Analises/')
# from _WordCloud import *
sys.path.insert(1, '/home/anarocha/myGit/classificadorDeAssuntos/Codigo/003_EncontraTamanhoAmostra')
from _001_Recupera_Amostras import *
sys.path.insert(1, '/home/anarocha/myGit/classificadorDeAssuntos/Codigo/005_FeatureEngineering')
from _002_Extrai_Features import *
import nltk
import pandas as pd
import multiprocessing as mp
import re

from wordcloud import WordCloud, STOPWORDS

def processa_texto(texto):
    textoProcessado = normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')
    textoProcessado = re.sub('[^a-zA-Z]', ' ', textoProcessado)
    textoProcessado = textoProcessado.lower()
    return textoProcessado


stopwords = nltk.corpus.stopwords.words('portuguese')
stopwords_processadas = []
for row in stopwords:
    palavraProcessada = normalize('NFKD', row).encode('ASCII', 'ignore').decode('ASCII')
    stopwords_processadas.append(palavraProcessada)
stopwords_processadas.append('trabalho')
stopwords_processadas.append('juiz')
stopwords_processadas.append('vara')
stopwords_processadas.append('recurso')
stopwords_processadas.append('ordinario')
stopwords_processadas.append('reclamada')
stopwords_processadas.append('reclamante')
stopwords_processadas.append('reclamado')
stopwords_processadas.append('recorrente')
stopwords_processadas.append('trabalho')
stopwords_processadas.append('empregado')
stopwords_processadas.append('empregada')
stopwords_processadas.append('art')
stopwords_processadas.append('processo')
stopwords_processadas.append('trabalhista')
stopwords_processadas.append('tribunal')
stopwords_processadas.append('regional')
stopwords_processadas.append('regiao')
stopwords_processadas.append('autor')
stopwords_processadas.append('reu')
stopwords_processadas.append('turma')
stopwords_processadas.append('trt')
stopwords_processadas.append('advogado')
stopwords_processadas.append('advogada')
stopwords_processadas.append('oab')
stopwords_processadas.append('ser')
stopwords_processadas.append('tst')
stopwords_processadas.append('sentenca')
stopwords_processadas.append('pagamento')
stopwords_processadas.append('direito')
stopwords_processadas.append('trabalhista')
stopwords_processadas.append('assim')
stopwords_processadas.append('autos')
stopwords_processadas.append('justica')
stopwords_processadas.append('lei')
stopwords_processadas.append('clt')
stopwords_processadas.append('decisao')
stopwords_processadas.append('empresa')
stopwords_processadas.append('contratante')
stopwords_processadas.append('contratada')
stopwords_processadas.append('contratado')
stopwords_processadas.append('recorrido')
stopwords_processadas.append('recorrida')
stopwords_processadas.append('conforme')


def gera_word_cloud(cod_assunto, nome_assunto):

    docs = df_amostra[(df_amostra['cd_assunto_nivel_3'] == cod_assunto)]
    docs = docs.head(2000)

    pool = mp.Pool(7)
    docs['texto_processado_2'] = pool.map(processa_texto, [row for row in docs['texto_processado']])
    pool.close()

    texto_docs = ''
    for index, row in docs.iterrows():
        texto_docs = texto_docs + ' ' + row['texto_processado_2']

    wc = WordCloud(
        background_color="white",
        max_words=200,
        width = 1024,
        height = 350,
        stopwords = stopwords_processadas,
        collocations = False,
        colormap="twilight",
        normalize_plurals=False

    )
    t = wc.process_text(texto_docs)
    wc.generate_from_frequencies(t)
    wc.to_file("/media/DATA/classificadorDeAssuntos/Dados/Resultados/word_clouds/word_cloud_" + str(cod_assunto) + "_" + nome_assunto + ".png")


listaAssuntos=[2546,2086,1855,2594,2458,2704,2656,2140,2435,2029,2583,2554,8808,2117,2021,5280,1904,1844,2055,1907,1806,55220,2506,
                        4437,10570,1783,1888,2478,5356,1773,1663,5272,2215,1767,1661,1690]
# listaAssuntos=[2546]

path_documentos = '/media/DATA/classificadorDeAssuntos/Dados/naoPublicavel/ConferenciaDeAssuntos/OK/'
qtdElementosPorAssunto=1000
# regionais=[22,23,24]
df_amostra = recupera_amostras_de_todos_regionais(listaAssuntos, qtdElementosPorAssunto,path_documentos)
df_amostra = df_amostra.dropna(subset=['texto_processado'])
df_amostra['quantidade_de_palavras'] = [len(x.split()) for x in df_amostra['texto_processado'].tolist()]
df_amostra = df_amostra[((df_amostra.quantidade_de_palavras < 10000) & (df_amostra.quantidade_de_palavras > 400))]

gera_word_cloud(2086,'Horas_Extras')
gera_word_cloud(2458,'Salario___Diferenca_Salarial')
gera_word_cloud(4437,'Revisao_de_Sentenca_Normativa')
gera_word_cloud(2546,'Verbas_Rescisórias')
gera_word_cloud(1855,'Indenizacao_por_Dano_Moral')
gera_word_cloud(2594,'Adicional')
gera_word_cloud(2704,'Tomador_de_Servicos___Terceirizacao')
gera_word_cloud(2656,'Reintegracao___Readmissao_ou_Indenizacao')
gera_word_cloud(2140,'Intervalo_Intrajornada')
gera_word_cloud(2435,'Rescisao_Indireta')
gera_word_cloud(2029,'FGTS')
gera_word_cloud(2583,'Abono')
gera_word_cloud(2554,'Reconhecimento_de_Relacao_de_Emprego')
gera_word_cloud(8808,'Indenizacao_por_Dano_Material')
gera_word_cloud(2117,'Supressao___Reducao_de_Horas_Extras_Habituais_-_Indenizacao')
gera_word_cloud(2021,'Indenizacao___Dobra___Terco_Constitucional')
gera_word_cloud(5280,'Bancarios')
gera_word_cloud(1904,'Despedida___Dispensa_Imotivada')
gera_word_cloud(1844,'CTPS')
gera_word_cloud(2055,'Gratificacao')
gera_word_cloud(1907,'Justa_Causa___Falta_Grave')
gera_word_cloud(1806,'Alteracao_Contratual_ou_das_Condicoes_de_Trabalho')
gera_word_cloud(55220,'Indenizacao_por_Dano_Moral')
gera_word_cloud(2506,'Ajuda___Tíquete_Alimentacao')
gera_word_cloud(10570,'FGTS')
gera_word_cloud(1783,'Comissao')
gera_word_cloud(1888,'Descontos_Salariais_-_Devolucao')
gera_word_cloud(2478,'Seguro_Desemprego')
gera_word_cloud(5356,'Grupo_Econômico')
gera_word_cloud(1773,'Contribuicao_Sindical')
gera_word_cloud(1663,'Adicional_Noturno')
gera_word_cloud(5272,'Administracao_Publica')
gera_word_cloud(2215,'Multa_Prevista_em_Norma_Coletiva')
gera_word_cloud(1767,'Cesta_Basica')
gera_word_cloud(1661,'Horas_in_Itinere')