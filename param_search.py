import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from data.load_datasets import carregar_dataset, selecionar_dataset_e_classe  # Importando do load_datasets.py

# Função para selecionar features com base no percentil
def selecionar_features_por_percentil(X, y, percentil):
    """
    Seleciona as features mais importantes com base no percentil.
    """
    selector = SelectPercentile(f_classif, percentile=percentil)
    X_selecionado = selector.fit_transform(X, y)
    return X_selecionado, selector.get_support(indices=True)

# Função para calcular a sensibilidade
def calcular_sensibilidade(y_true, y_pred):
    """
    Calcula a sensibilidade (recall) para a classe positiva (classe 0) em um problema One-vs-Rest.
    """
    return recall_score(y_true, y_pred, pos_label=0)  # Calcula o recall para a classe 0

# Função para calcular a métrica combinada
def calcular_metrica_comb(num_features, erro, features_relevantes, sensibilidade, delta_value, percentil, alpha=0.5):
    """
    Calcula a métrica combinada, penalizando muitas features relevantes e recompensando alta sensibilidade, delta value e percentil.
    """
    return (alpha * (features_relevantes / num_features)) + ((1 - alpha) * erro) + sensibilidade + (1 / delta_value) + (percentil / 100)

# Função principal do param_search
def param_search(X, y, alpha=0.5):
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Padronizar as features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Valores de percentil e delta value para testar
    percentis = [10, 25, 50, 75]
    valores_delta = [0.5, 1.0, 1.5]

    resultados = []

    for p in percentis:
        for delta_value in valores_delta:
            # Selecionar features com base no percentil
            X_train_selecionado, indices_features = selecionar_features_por_percentil(X_train, y_train, p)
            X_test_selecionado = X_test[:, indices_features]

            # Ajustar o parâmetro C com base no delta value
            C = 1.0 / delta_value  # Quanto maior o delta value, menor o C (mais regularização)

            # Treinar o modelo com as features selecionadas e o C ajustado
            modelo = LogisticRegression(C=C, max_iter=1000, solver='liblinear')
            modelo.fit(X_train_selecionado, y_train)

            # Calcular a acurácia e o erro
            y_pred = modelo.predict(X_test_selecionado)
            acuracia = accuracy_score(y_test, y_pred)
            erro = 1 - acuracia

            # Calcular a quantidade de features relevantes e a sensibilidade
            features_relevantes = X_train_selecionado.shape[1]
            sensibilidade = calcular_sensibilidade(y_test, y_pred)

            # Calcular a métrica combinada
            metrica = calcular_metrica_comb(X.shape[1], erro, features_relevantes, sensibilidade, delta_value, p, alpha)

            # Armazenar os resultados
            resultados.append({
                "percentil": p,
                "delta_value": delta_value,
                "acuracia": acuracia,
                "features_relevantes": features_relevantes,
                "sensibilidade": sensibilidade,
                "metrica": metrica
            })
    
    # Converter para DataFrame e ordenar pela métrica
    df = pd.DataFrame(resultados)
    df_ordenado = df.sort_values(by="metrica", ascending=True)
    
    return df_ordenado.head(5)

# Exemplo de uso
if __name__ == "__main__":
    # Selecionar dataset e classe usando a função do load_datasets.py
    nome_dataset, classe_0_nome, X, y, class_names = selecionar_dataset_e_classe()
    if nome_dataset is not None:
        # Executar a busca de hiperparâmetros
        ranking = param_search(X, y)
        print("🔹 **Ranking das 5 Melhores Combinações:**")
        print(ranking)