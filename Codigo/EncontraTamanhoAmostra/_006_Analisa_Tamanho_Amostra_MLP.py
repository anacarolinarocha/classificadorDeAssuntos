from sklearn.ensemble import RandomForestClassifier
from datetime import timedelta
import time
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '/home/anarocha/myGit/classificadorDeAssuntos/Codigo/EncontraTamanhoAmostra')
from _002_Encontra_Tamanho_Ideal_Amostra import *

path='/home/anarocha/myGit/classificadorDeAssuntos/Resultados/EXP4_MedindoTamanhoDaAmostra/'
if not os.path.exists(path):
    os.makedirs(path)

listaRegionais = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']
# listaRegionais = ['08','09','10']
listaAssuntos=[2546,2086,1855,2594,2458,2029,2140,2478,2704,2021,2426,2656,8808,1844,1663,2666,2506,55220,2055,1806,2139,1888,2435,2215,5280,2554,2583,55170,2019,2117,1661,1904,2540,55345]
# listaAssuntos=[2546,2086,1855,2594,2458,2029,2140,2478,2704,2021]
nomeAlgoritmo='Random Forest'
classificador = RandomForestClassifier(n_estimators=100, max_depth=10,random_state=0)
listaResultados = []
for qtdElementosPorAssunto in range(10, 201, 10):
    listaResultados.append(recupera_resultados_modelo(qtdElementosPorAssunto,listaAssuntos,listaRegionais, classificador,nomeAlgoritmo ))
resultados = pd.DataFrame(listaResultados, columns=['tamanhoAmostra','microPrecision','microRecall','microFscore','accuracy'])

plt.title(nomeAlgoritmo)
plt.plot('tamanhoAmostra', 'microPrecision', data=resultados, marker='o', markerfacecolor='blue', markersize=10, color='skyblue', linewidth=4, label="Micro Precision")
plt.plot('tamanhoAmostra', 'microRecall', data=resultados, marker='', color='olive', linewidth=2, linestyle='dashed', label="Micro Recall")
plt.plot('tamanhoAmostra', 'microFscore', data=resultados, marker='', color='gray', linewidth=2, linestyle='dashed', label="Micro FScore")
plt.plot('tamanhoAmostra', 'accuracy', data=resultados, marker='', color='green', linewidth=2, linestyle='dashed', label="Accuracy")
plt.legend()
plt.savefig("{0}{1}.png".format(path, nomeAlgoritmo.replace(' ', '')))
plt.show()

#-------------------------------------------------------------------------
nomeAlgoritmo='SVM'
classificador = SVC(kernel='linear', probability=True, class_weight='balanced')
listaResultados = []
for qtdElementosPorAssunto in range(70, 201, 10):
    listaResultados.append(recupera_resultados_modelo(qtdElementosPorAssunto,listaAssuntos,listaRegionais, classificador,nomeAlgoritmo ))
resultados = pd.DataFrame(listaResultados, columns=['tamanhoAmostra','microPrecision','microRecall','microFscore','accuracy'])
listaResultadosBackup = listaResultados
plt.title(nomeAlgoritmo)
plt.plot('tamanhoAmostra', 'microPrecision', data=resultados, marker='o', markerfacecolor='blue', markersize=10, color='skyblue', linewidth=4, label="Micro Precision")
plt.plot('tamanhoAmostra', 'microRecall', data=resultados, marker='', color='olive', linewidth=2, linestyle='dashed', label="Micro Recall")
plt.plot('tamanhoAmostra', 'microFscore', data=resultados, marker='', color='gray', linewidth=2, linestyle='dashed', label="Micro FScore")
plt.plot('tamanhoAmostra', 'accuracy', data=resultados, marker='', color='green', linewidth=2, linestyle='dashed', label="Accuracy")
plt.legend()
plt.savefig("{0}{1}.png".format(path, nomeAlgoritmo.replace(' ', '')))
plt.show()

