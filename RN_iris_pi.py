# 🔹 Importação das bibliotecas necessárias
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np

# 🔹 Carregar o dataset Iris
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target  # Classes do dataset
class_names = data.target_names

# 🔹 Transformar o problema em binário (classe 0 contra todas)
y_binario = [1 if label == 0 else 0 for label in y]

# 🔹 Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y_binario, test_size=0.2, random_state=42)

# 🔹 Treinar a Rede Neural (MLP)
mlp = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, learning_rate_init=0.001, random_state=42)
mlp.fit(X_train, y_train)

print("\n✅ Treinamento concluído com sucesso!")

# 🔹 Escolher uma instância para testar a PI-explicação
idx = 0  # Teste outras instâncias mudando o índice (1, 2, 3, etc.)
instancia_test = X_test.iloc[[idx]]
classe_prevista = mlp.predict(instancia_test)[0]

print(f"\n🔹 Classe prevista para a instância {idx}: {classe_prevista}")

# 🔹 Obter os nomes das features
feature_names = X.columns.tolist()

# 🔹 Obter os valores da instância testada
Vs = X_test.iloc[idx].to_dict()

# 🔹 Obter os coeficientes (pesos das features)
# 🔹 Normalizar os coeficientes da rede
w = mlp.coefs_[0].mean(axis=1)
w = w / np.max(np.abs(w))  # Normaliza os pesos entre -1 e 1

# 🔹 Calcular os valores de influência das features
delta = [(Vs[feature] - X[feature].mean()) * w[i] for i, feature in enumerate(feature_names)]

# 🔹 Ajustar R para evitar valores negativos extremos
R = sum(delta) - mlp.predict_log_proba(instancia_test)[0][1]

# 🔹 Definir um limiar para excluir deltas insignificantes
limiar_delta = np.percentile(np.abs(delta), 25)  # Exclui os 25% menores deltas

Xpl = []  # Lista de features explicativas
delta_sorted = sorted(enumerate(delta), key=lambda x: abs(x[1]), reverse=True)
R_atual = R
Idx = 0

while R_atual >= 0 and Idx < len(delta_sorted):
    sorted_idx, delta_value = delta_sorted[Idx]
    feature = feature_names[sorted_idx]
    feature_value = Vs[feature]

    if abs(delta_value) < limiar_delta:  # Remove valores irrelevantes
        break

    Xpl.append(f"{feature} - {feature_value}")
    R_atual -= delta_value
    Idx += 1

# 🔹 Exibir a PI-explicação
print("\n🔹 PI-Explicação para a Rede Neural:")
for item in Xpl:
    print(f"✅ {item}")
