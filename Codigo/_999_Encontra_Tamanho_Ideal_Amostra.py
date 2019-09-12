from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from datetime import timedelta
import time
import matplotlib.pyplot as plt
sys.path.insert(1, '/home/anarocha/myGit/classificadorDeAssuntos/Codigo')
from _997_Recupera_Amostras import *
from _998_Extrai_Features import *

#Recupera elementos

def mede_fscore_NB(qt_exemplos,listaAssuntos,listaRegionais):
    # qt_exemplos=10
    # listaAssuntos = [2546,2086]
    # listaRegionais = ['22','21']

    df_amostra_final = recupera_n_amostras_por_assunto_por_regional(listaRegionais,listaAssuntos,qt_exemplos,0.3)
    mostra_grafico_amostra(df_amostra_final)
    tamanhoAmostra=len(df_amostra_final)

    # Divide treinamento e teste, fazendo a extração de features do texto, e encontrando a variável alvo y

    tfidf_transformer = recupera_tfidf_transformer(df_amostra_final)
    x_tfidf_train = tfidf_transformer.transform(df_amostra_final[df_amostra_final['in_selecionando_para_amostra']=='Treinamento']['texto_processado'])
    x_tfidf_test = tfidf_transformer.transform(df_amostra_final[df_amostra_final['in_selecionando_para_amostra']=='Teste']['texto_processado'])

    y_train = df_amostra_final[df_amostra_final['in_selecionando_para_amostra']=='Treinamento']['cd_assunto_nivel_3']
    y_test = df_amostra_final[df_amostra_final['in_selecionando_para_amostra']=='Teste']['cd_assunto_nivel_3']

    #//TODO: adicionar criterio de parada
    # Chama o classificador

    nome_classificador="Multinomial_Naive_Bayes"

    n_estimators = 5
    max_samples=round(x_tfidf_train.shape[0] * 0.5)
    start_time = time.time()
    # clf = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', probability=True, class_weight='balanced'), max_samples=max_samples, n_estimators=n_estimators, n_jobs=-1))
    clf_nb_bagged = OneVsRestClassifier(BaggingClassifier(MultinomialNB(), max_samples=max_samples, n_estimators=n_estimators, n_jobs=-1))
    clf.fit(x_tfidf_train, y_train)
    total_time = time.time() - start_time
    print("Tempo para a criação do modelo Bagging Naive Bayes Multinomial para " + str(tamanhoAmostra) + " elementos: ", str(timedelta(seconds=total_time)))


    y_pred = clf.predict(x_tfidf_test)
    # clf_nb_bagged.score(x_tfidf_test,y_test)
    # accuracy = clf_nb_bagged.score(x_tfidf_test,y_test)
    # print('macro_precision %s \nmacro_recall    %s \nmacro_fscore    %s' % score(y_test,y_pred,average='macro')[:3])
    # print('micro_precision %s \nmicro_recall    %s \nmicro_fscore    %s' % score(y_test,y_pred,average='weighted')[:3])
    # conf_mat = multilabel_confusion_matrix(y_true=y_test, y_pred=y_pred)
    # print('Confusion matrix:\n', conf_mat)


    #Metrica escolhida para análise: Micro f-score (leva em consideracao o peso de cada classe, a precisao e o recall)
    microFscore =  score(y_test,y_pred,average='weighted')[2:3][0]
    return tamanhoAmostra, microFscore


path='/home/anarocha/myGit/classificadorDeAssuntos/Resultados/EXP3_MedindoTamanhoDaAmostra/'
if not os.path.exists(path):
    os.makedirs(path)

listaRegionais = ['22','21','18','09','12','23']
listaAssuntos=[2546,2086,1855,2594,2458,2029,2140]
listaResultados = []
for qtdElementosPorAssunto in range(round(10000/len(listaRegionais)), 5000, round(10000/len(listaRegionais))):
    listaResultados.append(mede_fscore_NB(qtdElementosPorAssunto,listaAssuntos,listaRegionais))
plt.plot(*zip(*listaResultados))
plt.title("NB")
plt.savefig("{0}{1}.png".format(path, str("NB").replace(' ', '')))
plt.show()

#
# def getKey(item):
#     return item[0]
# teste = sorted(listaResultados, key=getKey)
# listaResultados = teste
# listaResultadosSVMBackup = listaResultados
