from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import lxml

#----------------------------------------------------------------------------------------------------------------------
def removeHTML(texto):
    """
    Função para remover HTML
    :param texto:
    :return: texto sem tags HTML
    """
    return BeautifulSoup(texto, 'lxml').get_text(strip=True)

#----------------------------------------------------------------------------------------------------------------------
def plot_ys_mc(ys_series, ys_title, path):
    """
    Função para plotar um histograma que conta quantos elementos de cada tipo existem ali, ordenado pelo nome de cada tipo
    :param ys_series: dados
    :param ys_title: titulo
    :return:
    """

    pd.DataFrame.from_dict(Counter(ys_series), orient='index').sort_values(by=[0],ascending=False).plot(kind='bar')
    plt.title(ys_title)
    plt.savefig("{0}{1}.png".format(path, str(ys_title).replace(' ', '')))
    plt.show()

#----------------------------------------------------------------------------------------------------------------------
from types import ModuleType, FunctionType
from gc import get_referents

# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType

def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size
#----------------------------------------------------------------------------------------------------------------------
UNITS_MAPPING = [
    (1<<50, ' PB'),
    (1<<40, ' TB'),
    (1<<30, ' GB'),
    (1<<20, ' MB'),
    (1<<10, ' KB'),
    (1, (' byte', ' bytes')),
]

def pretty_size(bytes, units=UNITS_MAPPING):
    """Get human-readable file sizes.
    simplified version of https://pypi.python.org/pypi/hurry.filesize/
    """
    for factor, suffix in units:
        if bytes >= factor:
            break
    amount = int(bytes / factor)

    if isinstance(suffix, tuple):
        singular, multiple = suffix
        if amount == 1:
            suffix = singular
        else:
            suffix = multiple
    return str(amount) + suffix

#----------------------------------------------------------------------------------------------------------------------
def imprime_resultado_classificador(path, nome_experimento, nome_clf, total_time, acuracea, macro, micro):
    global f
    f = open(path + nome_clf + ".txt", "w")
    f.write(nome_experimento)
    f.write('\n')
    f.write(nome_clf)
    f.write('\n')
    f.write("Tempo de treinamento: ")
    f.write(str(total_time))
    f.write('\n')
    f.write("Acurácia:       ")
    f.write(str(acuracea))
    f.write('\n')
    f.write(macro)
    f.write('\n')
    f.write(micro)
    f.close()