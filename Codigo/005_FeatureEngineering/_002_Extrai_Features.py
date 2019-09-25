import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
sys.path.insert(1, '/home/anarocha/myGit/classificadorDeAssuntos/Codigo/005_FeatureEngineering/')
from _001_Processa_Texto import *
import multiprocessing as mp
from unicodedata import normalize

def recupera_tfidf_transformer(df):
    """
        Método responsável por criar o modelo da matriz tf-idf
    :param df: dataframe com corpus de texto
    :return: modelo tfidf
    """

    tfidf_vectorizer = TfidfVectorizer( token_pattern=r'(?u)\b[A-Za-z]+\b', max_df=0.8)
    # tfidf_transformer = tfidf_vectorizer.fit(df['texto_stemizado'])
    tfidf_transformer = tfidf_vectorizer.fit(df.texto_stemizado.astype(str))
    return tfidf_transformer
