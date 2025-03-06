# models/train_model.py

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def transformar_problema_binario(y, classe_0):
    return [1 if label == classe_0 else 0 for label in y]

def treinar_modelo(X, y, classe_0=0):
    y_binario = transformar_problema_binario(y, classe_0) # AQUI PODE MUDAR A CLASSE PARA A PRINCIPAL (CLASSE A SER ANALISADA)
    X_train, X_test, y_train, y_test = train_test_split(X, y_binario, test_size=0.2, random_state=42)
    modelo = LogisticRegression(max_iter=200)
    modelo.fit(X_train, y_train)
    return modelo, X_test, y_test