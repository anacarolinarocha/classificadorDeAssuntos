import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import multiprocessing as mp
from unicodedata import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import pickle
from _003_BM25_Transformer import *


def recupera_lsi_transformer(df, topics):
    tfidf_vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b[A-Za-z]+\b', max_df=0.8, min_df=5)
    svd_model = TruncatedSVD(n_components=topics, algorithm='randomized',n_iter=10, random_state=42)
    svd_transformer = Pipeline([('tfidf', tfidf_vectorizer),
                                ('svd', svd_model)])
    svd_transformer = svd_transformer.fit(df.texto_stemizado.astype(str))
    return svd_transformer

def extraiFeaturesBM25(df_amostra_final,tfidf_transformer,x_tfidf_train,x_tfidf_test ,path):
    df_amostra_final_tfidf = tfidf_transformer.transform(df_amostra_final)
    bm25_transformer = BM25Transformer()
    bm25_transformer.fit(df_amostra_final_tfidf)
    salvaTransformer(bm25_transformer, 'BM25', path)
    x_bm25_train = bm25_transformer.transform(x_tfidf_train)
    x_bm25_test = bm25_transformer.transform(x_tfidf_test)
    return bm25_transformer,x_bm25_train, x_bm25_test

def extraiFeaturesTFIDF(df,X_train,X_test ,path):
    tfidf_vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b[A-Za-z]+\b', max_df=0.8, min_df=5)
    tfidf_transformer = tfidf_vectorizer.fit(df.texto_stemizado.astype(str))
    salvaTransformer(tfidf_transformer, 'TFIDF', path)
    x_tfidf_train = tfidf_transformer.transform(X_train)
    x_tfidf_test = tfidf_transformer.transform(X_test)
    return tfidf_transformer,x_tfidf_train, x_tfidf_test


def extraiFeaturesTFIDF_ngrams(df,X_train,X_test ,path):
    tfidf_vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b[A-Za-z]+\b', max_df=0.8, min_df=5)
    tfidf_transformer = tfidf_vectorizer.fit(df.texto_stemizado.astype(str))
    salvaTransformer(tfidf_transformer, 'TFIDF', path)
    x_tfidf_train = tfidf_transformer.transform(X_train)
    x_tfidf_test = tfidf_transformer.transform(X_test)
    return tfidf_transformer,x_tfidf_train, x_tfidf_test

def extraiFeaturesLSI(df_amostra_final,X_train,X_test,topics ,path):
    svd_transformer = recupera_lsi_transformer(df_amostra_final,topics)
    salvaTransformer(svd_transformer, 'LSI100', path)
    x_lsi_train = svd_transformer.transform(X_train)
    x_lsi_test = svd_transformer.transform(X_test)
    return svd_transformer,x_lsi_train, x_lsi_test


# ---------------------------------------------------------------------------------------------------------------------
# Grava modelo de transformação de features
#----------------------------------------------------------------------------------------------------------------------
def salvaTransformer(transformer, nome, path):
    nomePicke = path + nome + '.p'
    arquivoPickle = open(nomePicke, 'wb')
    pickle.dump(transformer, arquivoPickle)
    arquivoPickle.close()


