import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import multiprocessing as mp
from unicodedata import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import pickle
from _003_BM25_Transformer import *
import time
from datetime import timedelta
import pandas as pd



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

def extraiFeaturesTFIDF_train_test(df,X_train,X_test ,path):
    tfidf_vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b[A-Za-z]+\b', max_df=0.8, min_df=5)
    tfidf_transformer = tfidf_vectorizer.fit(df.texto_stemizado.astype(str))
    salvaTransformer(tfidf_transformer, 'TFIDF', path)
    x_tfidf_train = tfidf_transformer.transform(X_train)
    x_tfidf_test = tfidf_transformer.transform(X_test)
    return tfidf_transformer,x_tfidf_train, x_tfidf_test

def extraiFeaturesTFIDF(df):
    tfidf_vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b[A-Za-z]+\b', max_df=0.8, min_df=5)
    tfidf_transformer = tfidf_vectorizer.fit(df.texto_stemizado.astype(str))
    df_tfidf = tfidf_transformer.transform(df)
    return tfidf_transformer,df_tfidf


def extraiFeaturesTFIDF_ngrams(df,X_train,X_test ,path):
    tfidf_vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b[A-Za-z]+\b', max_df=0.8, min_df=10, ngram_range=(1,3),max_features=150000)
    tfidf_transformer = tfidf_vectorizer.fit(df.texto_stemizado.astype(str))
    salvaTransformer(tfidf_transformer, 'TFIDF', path)
    x_tfidf_train = tfidf_transformer.transform(X_train)
    x_tfidf_test = tfidf_transformer.transform(X_test)
    return tfidf_transformer,x_tfidf_train, x_tfidf_test

def extraiFeaturesLSI(df_amostra_final,X_train,X_test,topics ,path):
    svd_transformer = recupera_lsi_transformer(df_amostra_final,topics)
    salvaTransformer(svd_transformer, 'LSI' + str(topics), path)
    x_lsi_train = svd_transformer.transform(X_train)
    x_lsi_test = svd_transformer.transform(X_test)
    return svd_transformer,x_lsi_train, x_lsi_test
#
#
# arquivo_glove = '/media/DATA/classificadorDeAssuntos/Dados/naoPublicavel/glove_s300.txt'
# from gensim.models import KeyedVectors
# start_time = time.time()
# embeddings = KeyedVectors.load_word2vec_format(arquivo_glove,
#                                                    binary=False,
#                                                    unicode_errors="ignore")
# total_time = time.time() - start_time
# print("Tempo para carregar embeddings:" +   str(timedelta(seconds=total_time)))
#
# def extraiFeaturesEmbeddings(X_train,X_test ):
#     embeddings_train = []
#     embeddings_test = []
#
#     pool = mp.Pool(7)
#     embeddings_train = pool.map(get_mean_vector, [row for row in X_train])
#     # embeddings_train = pool.map(get_sum_vector, [row for row in X_train])
#     pool.close()
#
#     start_time = time.time()
#     pool = mp.Pool(7)
#     embeddings_test = pool.map(get_mean_vector, [row for row in X_test])
#     # embeddings_test = pool.map(get_sum_vector, [row for row in X_test])
#     pool.close()
#
#     df_embeddings_train = pd.DataFrame(embeddings_train)
#     df_embeddings_test = pd.DataFrame(embeddings_test)
#
#     return df_embeddings_train,df_embeddings_test
#
# def get_mean_vector(words):
#     global embeddings
#     # remove out-of-vocabulary words
#     words = [word for word in words if word in embeddings.vocab]
#     if len(words) >= 1:
#         return np.mean(embeddings[words], axis=0)
#     else:
#         return []
#
#
# def get_sum_vector(words):
#     global embeddings
#     # remove out-of-vocabulary words
#     words = [word for word in words if word in embeddings.vocab]
#     if len(words) >= 1:
#         return np.sum(embeddings[words], axis=0)
#     else:
#         return []

# ---------------------------------------------------------------------------------------------------------------------
# Grava modelo de transformação de features
#----------------------------------------------------------------------------------------------------------------------
def salvaTransformer(transformer, nome, path):
    nomePicke = path + nome + '.p'
    arquivoPickle = open(nomePicke, 'wb')
    pickle.dump(transformer, arquivoPickle)
    arquivoPickle.close()


def carregaModelo(arquivo):
    with open(arquivo, "rb") as input_file:
        return pickle.load(input_file)

