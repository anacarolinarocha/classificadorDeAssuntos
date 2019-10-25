import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import multiprocessing as mp
from unicodedata import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

def recupera_tfidf_transformer(df):
    """
        Método responsável por criar o modelo da matriz tf-idf
    :param df: dataframe com corpus de texto
    :return: modelo tfidf
    """

    tfidf_vectorizer = TfidfVectorizer( token_pattern=r'(?u)\b[A-Za-z]+\b', max_df=0.8, min_df=5)
    tfidf_transformer = tfidf_vectorizer.fit(df.texto_stemizado.astype(str))
    return tfidf_transformer

def recupera_lsi_transformer(df):
    tfidf_vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b[A-Za-z]+\b', max_df=0.8, min_df=5)
    svd_model = TruncatedSVD(n_components=250, algorithm='randomized',n_iter=10, random_state=42)
    svd_transformer = Pipeline([('tfidf', tfidf_vectorizer),
                                ('svd', svd_model)])
    svd_transformer = svd_transformer.fit(df.texto_stemizado.astype(str))
    return svd_transformer

