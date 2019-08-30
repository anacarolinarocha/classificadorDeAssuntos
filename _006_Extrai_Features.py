import nltk
from sklearn.feature_extraction.text import TfidfVectorizer



def recupera_tfidf_transformer(df):
    """
        Método responsável por criar o modelo da matriz tf-idf
    :param df: dataframe com corpus de texto
    :return: modelo tfidf
    """
    stopwords = nltk.corpus.stopwords.words('portuguese')
    tfidf_vectorizer = TfidfVectorizer(strip_accents='ascii', stop_words=stopwords, token_pattern=r'(?u)\b[A-Za-z]+\b', max_df=0.8)
    tfidf_transformer = tfidf_vectorizer.fit(df['texto_processado'])
    return tfidf_transformer
