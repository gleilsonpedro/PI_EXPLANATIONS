import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def treinar_modelo(X, y, regularizacao=1.0, penalty='l1', max_iter=200, solver="liblinear"):
    """
    Treina um modelo de Regressão Logística usando os dados fornecidos.

    Parâmetros:
        X: DataFrame ou array contendo as features (atributos de entrada)
        y: array ou Series com os rótulos (0 ou 1)
        regularizacao: valor de C na Regressão Logística (inverso da força de regularização)
        penalty: tipo de penalização a aplicar ('l1' ou 'l2')
        max_iter: número máximo de iterações para convergência do otimizador
        solver: algoritmo de otimização usado (ex: 'liblinear' é ideal para problemas binários)

    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    modelo = LogisticRegression(
        C=regularizacao,
        penalty=penalty,
        max_iter=max_iter,
        solver=solver
    )
    
    modelo.fit(X_train, y_train)
    return modelo, X_test, y_test




#def treinar_modelo(X, y):
#    # Garantir que X é um DataFrame com nomes de features
#    if not isinstance(X, pd.DataFrame):
#        X = pd.DataFrame(X)
#    
#    # Manter como DataFrame durante toda a operação
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42    )
#    
#    modelo = LogisticRegression(C=10.0, max_iter=200, solver="liblinear")
#    # modelo = LogisticRegression(max_iter=200, solver="liblinear")
#    # C=10.0 → menos regularização (aumenta os coeficientes/pesos)
#    # solver="liblinear" → bom para conjuntos de dados pequenos e binários
#
#    modelo.fit(X_train, y_train)
#    
#    return modelo, X_test, y_test