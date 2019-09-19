from docutils.nodes import header
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from datetime import timedelta
import time
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '/home/anarocha/myGit/classificadorDeAssuntos/Codigo/EncontraTamanhoAmostra')
from _002_Encontra_Tamanho_Ideal_Amostra import *

path='/home/anarocha/myGit/classificadorDeAssuntos/Resultados/EXP6_RodandoTudo/'
if not os.path.exists(path):
    os.makedirs(path)

listaRegionais = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']
# listaRegionais = ['08','09','10']
listaAssuntos=[2546,2086,1855,2594,2458,2029,2140,2478,2704,2021,2426,2656,8808,1844,1663,2666,2506,55220,2055,1806,2139,1888,2435,2215,5280,2554,2583,55170,2019,2117,1661,1904,2540,55345]
# listaAssuntos=[2546,2086,1855,2594,2458,2029,2140,2478,2704,2021]

#-----------------------------------------------------------------------------------------------------------------------
#Multinomial Nayve Bayes
# nomeAlgoritmo='Multinomial Naive Bayes'
# classificador =  MultinomialNB()
# listaResultados = []
# for qtdElementosPorAssunto in range(100000, 100001, 100):
#     listaResultados.append(recupera_resultados_modelo(qtdElementosPorAssunto,listaAssuntos,listaRegionais, classificador,nomeAlgoritmo ))
# resultados = pd.DataFrame(listaResultados, columns=['tamanhoAmostra','microPrecision','microRecall','microFscore','accuracy'])
# resultados['algoritmo'] = nomeAlgoritmo
# resultados.to_csv(path + "Metricas_" + nomeAlgoritmo.replace(' ', '') + "_TodosElementosDispioniveis", header=True)

#-----------------------------------------------------------------------------------------------------------------------
#Random Forest
nomeAlgoritmo='Random Forest'
classificador = RandomForestClassifier(n_estimators=100, max_depth=10,random_state=0)
listaResultados = []
for qtdElementosPorAssunto in range(100000, 100001, 100):
    listaResultados.append(recupera_resultados_modelo(qtdElementosPorAssunto,listaAssuntos,listaRegionais, classificador,nomeAlgoritmo ))
resultados2 = pd.DataFrame(listaResultados, columns=['tamanhoAmostra','microPrecision','microRecall','microFscore','accuracy'])
resultados2['algoritmo'] = nomeAlgoritmo
resultados2.to_csv(path + "Metricas_" + nomeAlgoritmo.replace(' ', '') + "_TodosElementosDispioniveis", header=True)

