#Projeto Classificador de Assuntos

Este projeto foi desenvolvido como um entregável de uma dissertação de Mestrado em Computação Aplicada pela Universidade
 de Brasilia. O código em questão tem o objetivo de testar vários modelos de aprendizagem de máquina para fazer a 
 classificação do assunto principal de processos judiciais trabalhistas de acordo com assuntos da Tabela Processual 
 Unificada de Assuntos do Conselho Nacional de Justiça e Tribunal Superior do Trabalho.  São comparados modelos Naive 
 Bayes (NB) , Support Vector Machine (SVM), Random Forest (RF) e Multi-layer Perceptron (MLP). Todos os algoritmos são 
 testados em 10 combinações diferentes de parâmetros, com validação cruzada de 5 folds. Os textos são testados em sua 
 represetanção no formato  TF-IDF, BM25 e LSA com 100 e 250 tópicos.
 
O código em questão executa os seguintes passos:

1) Busca em uma pasta os arquivos referentes aos documentos 
2) Faz o pré processamento dos textos
3) Transforma o texto em uma matriz numérica 
4) Analisa o balanceamento dos dados em função do assunto principal
5) Faz um GridSearch para verificar os melhores parâmetros e a melhor forma de se representar os textos nos algoritmos
6) Escolhe o melhor modelo de cada algoritmo em função da micro precisão. 
7) Testa o modelo vencedor de cada algoritmo como as diferentes formas de representação do texto
8) Escolhe a combinação vencedora em função da micro precisão

Os modelos de classificação foram treinados para reconhecer os assuntos abaixo, sendo todos assuntos do nível 3 da 
Tabela Processual Unificada:

* **2583** - Abono
* **2594** - Adicional
* **1663** - Adicional Noturno
* **5272** - Administração Pública
* **2506** - Ajuda / Tíquete Alimentação
* **1806** - Alteração Contratual ou das Condições de Trabalho
* **5280** - Bancários
* **1767** - Cesta Básica
* **1783** - Comissão
* **1690** - Contribuição / Taxa Assistencial
* **1773** - Contribuição Sindical
* **1844** - CTPS
* **1888** - Descontos Salariais - Devolução
* **1904** - Despedida / Dispensa Imotivada
* **2029** - FGTS
* **10570** - FGTS
* **2055** - Gratificação
* **5356** - Grupo Econômico
* **2086** - Horas Extras
* **1661** - Horas in Itinere
* **2021** - Indenização / Dobra / Terço Constitucional
* **8808** - Indenização por Dano Material
* **1855** - Indenização por Dano Moral
* **55220** - Indenização por Dano Moral
* **2140** - Intervalo Intrajornada
* **1907** - Justa Causa / Falta Grave
* **2215** - Multa Prevista em Norma Coletiva
* **2554** - Reconhecimento de Relação de Emprego
* **2656** - Reintegração / Readmissão ou Indenização
* **2435** - Rescisão Indireta
* **4437** - Revisão de Sentença Normativa
* **2458** - Salário / Diferença Salarial
* **2478** - Seguro Desemprego
* **2117** - Supressão / Redução de Horas Extras Habituais - Indenização
* **2704** - Tomador de Serviços / Terceirização
* **2546** - Verbas Rescisórias

PS: O assunto de código **55220** (Indenização por Dano Moral) será agrupado com o assunto **1855** (Indenização por 
Dano Moral)

### Exemplo de utilização 

1) Acessar o diretório **Código**
2) Rodar a linha de comando abaixo

**`python buscaMelhorModelo.py -dd /path/dos/documentos -dr /path/dos/resultados`**

Onde:

**-dd** é o Diretório fonte para se buscar os arquivos CSV

**-dr** é o Diretório de destino dos arquivos de saída



### Pré-requisitos

Para referência, os modelos treinados com este código foram treinados com 180.672 documentos. Como são trabalhados uma 
grande quantidade de textos por envolver grupos representativos de textos de 35 assuntos diferentes, recomenda-se a 
utilização mínima de 16 GB de memória RAM e 4 cores. Para cada quatro cores adicionais disponíveis, sugere-se acrescentar 
mais 16GB de memória. 

Os modelos são executados utilizando processamento paralelo, de forma que os modelos de Naive Bayes, SVM e M
### Entrada

Como entrada, são esperados arquvios csv que contenham as colunas abaixo, na ordem apresentada. Nem todas as informações 
são utilizadas para a tarefa de classificação, mas foram recuperadas pois são úteis na identificação do documento:

* **index**: número sequencial de cada linha
* **nr_processo**: número do processo no [formato](https://www.conjur.com.br/dl/resolucao-65-cnj.pdf) do CNJ 
(NNNNNNN-DD.AAAA.JTR.OOOO)  
* **id_processo_documento**: Chave primária da tb_processo_documento
* **cd_assunto_nivel_5**: Código do nível 5 do assunto principal, se houver. Se não houver, deve ficar vazio
* **cd_assunto_nivel_4**: Código do nível 4 do assunto principal, se houver. Se não houver, deve ficar vazio 
* **cd_assunto_nivel_3**: Código do nível 3 do assunto principal, se houver. Se não houver, deve ficar vazio
* **cd_assunto_nivel_2**: Código do nível 2 do assunto principal, se houver. Se não houver, deve ficar vazio
* **cd_assunto_nivel_1**: Código do nível 1 do assunto principal, se houver. Se não houver, deve ficar vazio
* **tx_conteudo_documento**: Conteúdo HTML completo do documento 
* **ds_identificador_unico**: Identificador único completo do documento
* **ds_identificador_unico_simplificado**: 7 últimos dígitos do identificador único do documento (forma de identificação no PJe)
* **ds_orgao_julgador**: Nome do órgão julgador do processo
* **ds_orgao_julgador_colegiado**: Nome do órgão julgador colegiado do processo
* **dt_juntada**: Data da juntada do documento

É esperado um arquivo CSV por cada regional, com o nome no seguinte padrão:
TRT_YY_documentosSelecionados.csv

Onde: 
YY representa a sigla de cada regional, sendo obrigatório 2 dígitos. 

Exemplos:
TRT_01_documentosSelecionados.csv
TRT_22_documentosSelecionados.csv

O diretório onde se deve buscar os arquivos deve ser passado com o parametro -dd no momento da chamada do código. 
Embora seja desejável, não é obrigatório ter arquivos de entrada de todos os regionais. 

### Saída 

Como saída deste pipeline, são gravados diversos arquivos. Seguem:

* **Balanceamento_Assuntos_XXX_Elementos.png**: Imagem que mostra o balancemento dos assuntos no dataset separado para treinamento
* **Distribuição_Tamanho_Textos_XXXX.png**: Imagens que mostram um boxplot com a distribuição da quantidade de palavras por documento com todos os docuementos e depois removendo-se os documentos com menos de 400 palavras e mais de 10.000 palavras
* **Modelo_XXX.p**: Modelos que foram gerados, levando da forma de representação do texto e o algortimo utilizado
* **Feature.p**: Modelo de transformação do texto
* **Metricas.csv**: Arquivo que guarda todas as métricas escolhidas de todos os modelos testados (para o primeiro GridSearch, usa-se apenas o modelo vencedor)
* **predicao_XXX**: Arquivo que contém a predição feita por cada combinação XXX de modelo e features, trazendo também o percentual probabilidade de cada um dos assuntos selecionados no escopo do projeto.
* **ClassificationReport_XXX.csv**: Arquivo que da a métrica individual de cada um dos assuntos selecionados 
* **MelhorModelo.p**: Modelo vencedor
* **MelhorModeloFeature.p**: Modelo de transformação de features vencedor

