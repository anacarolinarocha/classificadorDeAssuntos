import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
sys.path.insert(1, '/home/anarocha/myGit/classificadorDeAssuntos/Codigo/FeatureEngineering/')
from _001_Processa_Texto import *
import multiprocessing as mp
from unicodedata import normalize

def removeAcento(texto):
    return normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')

def recupera_tfidf_transformer(df):
    """
        Método responsável por criar o modelo da matriz tf-idf
    :param df: dataframe com corpus de texto
    :return: modelo tfidf
    """
    stopwords = nltk.corpus.stopwords.words('portuguese')

    pool = mp.Pool(1)
    df_stopwords = pool.map(stemiza, [row for row in stopwords])
    df_stopwords = pool.map(removeAcento, [row for row in df_stopwords])
    df_stopwords

    pool.close()


    tfidf_vectorizer = TfidfVectorizer(strip_accents='ascii', stop_words=df_stopwords, token_pattern=r'(?u)\b[A-Za-z]+\b', max_df=0.8)
    tfidf_transformer = tfidf_vectorizer.fit(df['texto_stemizado'])
    return tfidf_transformer
