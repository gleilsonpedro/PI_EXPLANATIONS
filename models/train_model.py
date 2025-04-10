import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def treinar_modelo(X, y):
    # Garantir que X é um DataFrame com nomes de features
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    # Manter como DataFrame durante toda a operação
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
    
    modelo = LogisticRegression(max_iter=200)
    modelo.fit(X_train, y_train)
    
    return modelo, X_test, y_test