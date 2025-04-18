from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

def treinar_modelo(X, y):
    """Treina modelo com os parâmetros solicitados"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    modelo = LogisticRegression(
        penalty='l2',  # Regularização L2 (padrão), (l1)= None pode incluir features irrelevantes
        C=0.01,         # Parâmetro de regularização se > 1mais features nas explicações
        max_iter=1000,  # Número suficiente para convergir
        solver='lbfgs' # Bom para datasets pequenos/médios
    )
    modelo.fit(X_train, y_train)
    # Garante tipos consistentes
    y_test = pd.Series(y_test, index=X_test.index, name='target')
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    return modelo, X_test, y_test