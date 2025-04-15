import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def treinar_modelo(X, y, regularizacao=1.0, penalty='l1', max_iter=200, solver="liblinear"):
    """
    Treina modelo de regressão logística com opção de limitar explicações a top-N features
    
    Parâmetros:
        X: DataFrame ou array com features
        y: array com labels
        regularizacao: valor de regularização (inverso de C)
        top_n: número máximo de features a incluir nas explicações (None para sem limite)
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