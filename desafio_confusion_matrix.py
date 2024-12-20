# Cálculo de Métricas de Avaliação de Aprendizado 

# Neste projeto, vamos calcular as principais métricas para avaliação de modelos de classificação de dados, como acurácia, sensibilidade (recall), especificidade, precisão e F-score. Para que seja possível implementar estas funções, você deve utilizar os métodos e suas fórmulas correspondentes (Tabela 1). 

# Para a leitura dos valores de VP, VN, FP e FN, será necessário escolher uma matriz de confusão para a base dos cálculos. Essa matriz você pode escolher de forma arbitraria, pois nosso objetivo é entender como funciona cada métrica.  

 

# Tabela 1: Visão geral das métricas usadas para avaliar métodos de classificação. VP: verdadeiros positivos; FN: falsos negativos; FP: falsos positivos; VN: verdadeiros negativos; P: precisão; S: sensibilidade; N: total de elementos.


## IMPORTAR BIBLIOTECAS
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

# Dados de exemplo (substitua com seus próprios dados)
y_true = [0, 1, 2, 2, 0]  # Valores reais
y_pred = [0, 0, 2, 2, 1]  # Predições do modelo

# Classes (substitua conforme necessário)
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Calculando a matriz de confusão
con_mat = confusion_matrix(y_true, y_pred, labels=classes)

# Normalizando a matriz de confusão (com tratamento para evitar divisões por zero)
with np.errstate(divide='ignore', invalid='ignore'):
    con_mat_norm = np.divide(
        con_mat.astype('float'),
        con_mat.sum(axis=1)[:, np.newaxis],
        out=np.zeros_like(con_mat, dtype=float),
        where=(con_mat.sum(axis=1)[:, np.newaxis] != 0)
    )
con_mat_norm = np.round(con_mat_norm, decimals=2)

# Criando um DataFrame para a matriz
con_mat_df = pd.DataFrame(con_mat_norm, index=classes, columns=classes)

# Plotando a matriz de confusão
plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues, fmt='.2f')
plt.tight_layout()
plt.ylabel('Rótulo Verdadeiro')
plt.xlabel('Predicted Label')
plt.title('Matriz de Confusão Normalizada')
plt.show()

# Calculando as métricas
def calcula_metricas(conf_mat):
    VP = np.diag(conf_mat)  # Verdadeiros Positivos
    FP = np.sum(conf_mat, axis=0) - VP  # Falsos Positivos
    FN = np.sum(conf_mat, axis=1) - VP  # Falsos Negativos
    VN = np.sum(conf_mat) - (FP + FN + VP)  # Verdadeiros Negativos

    # Tratamento para evitar divisões por zero
    with np.errstate(divide='ignore', invalid='ignore'):
        sensibilidade = np.divide(VP, (VP + FN), out=np.zeros_like(VP, dtype=float), where=(VP + FN) != 0)
        especificidade = np.divide(VN, (FP + VN), out=np.zeros_like(VN, dtype=float), where=(FP + VN) != 0)
        acuracia = np.divide((VP + VN), np.sum(conf_mat), out=np.zeros_like(VP, dtype=float), where=np.sum(conf_mat) != 0)
        precisao = np.divide(VP, (VP + FP), out=np.zeros_like(VP, dtype=float), where=(VP + FP) != 0)
        f_score = np.divide(2 * (precisao * sensibilidade), (precisao + sensibilidade), out=np.zeros_like(precisao, dtype=float), where=(precisao + sensibilidade) != 0)

    return {
        'Sensibilidade': sensibilidade,
        'Especificidade': especificidade,
        'Acurácia': acuracia,
        'Precisão': precisao,
        'F-score': f_score
    }
# Chamando a função para calcular as métricas
metricas = calcula_metricas(con_mat)

# Exibindo as métricas
for metrica, valores in metricas.items():
    print(f"{metrica}: {valores}")
#======================================================================================================================================================================================================
#SAIDA: 

Sensibilidade: [0.5 0.  1.  0.  0.  0.  0.  0.  0.  0. ]
Especificidade: [0.66666667 0.75       1.         1.         1.         1.
 1.         1.         1.         1.        ]
Acurácia: [0.6 0.6 1.  1.  1.  1.  1.  1.  1.  1. ]
Precisão: [0.5 0.  1.  0.  0.  0.  0.  0.  0.  0. ]
F-score: [0.5 0.  1.  0.  0.  0.  0.  0.  0.  0. ]
