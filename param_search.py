import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif

# Função para carregar o dataset
def carregar_dataset(nome_dataset):
    if nome_dataset == 'breast_cancer':
        data = load_breast_cancer()
        X = data.data
        y = data.target
        class_names = data.target_names
        return X, y, class_names
    else:
        raise ValueError(f"Dataset '{nome_dataset}' não suportado.")

# Função para selecionar dataset e classe
def selecionar_dataset_e_classe():
    """
    Menu para selecionar o dataset e a classe que será a classe 0.
    Retorna:
        - nome_dataset: Nome do dataset escolhido.
        - classe_0_nome: Nome da classe que será a classe 0.
        - X: Features do dataset.
        - y: Labels do dataset.
        - class_names: Nomes das classes.
    """
    # Menu de seleção de datasets
    menu = '''
    |  ************************* MENU ***************************  |
    |  0 - iris                     |  1 - wine                     |
    |  2 - breast_cancer            |  3 - digits                  |
    |  4 - banknote_authentication  |  5 - wine_quality           |
    |  6 - heart_disease            |  7 - parkinsons             |
    |  8 - car_evaluation           |  9 - diabetes_binary        |
    |  Q - SAIR                                                |
    |-------------------------------------------------------------|
    '''
    print(menu)

    # Lista de datasets disponíveis
    nomes_datasets = [
        'iris', 'wine', 'breast_cancer', 'digits', 'banknote_authentication',
        'wine_quality', 'heart_disease', 'parkinsons', 'car_evaluation', 'diabetes_binary'
    ]

    # Solicitar a escolha do dataset
    while True:
        opcao = input("Digite o número do dataset ou 'Q' para sair: ").upper().strip()
        if opcao == 'Q':
            print("Você escolheu sair.")
            return None, None, None, None, None
        elif opcao.isdigit() and 0 <= int(opcao) <= 9:
            nome_dataset = nomes_datasets[int(opcao)]
            break
        else:
            print("Opção inválida. Por favor, escolha um número do menu ou 'Q' para sair.")

    # Carregar o dataset
    X, y, class_names = carregar_dataset(nome_dataset)

    # Exibir as classes disponíveis
    print("\nClasses disponíveis:")
    for i, class_name in enumerate(class_names):
        print(f"   [{i}] - {class_name}")

    # Solicitar a escolha da classe 0
    while True:
        escolha_classe_0 = input("\nDigite o número da classe que será `0`: ")
        if escolha_classe_0.isdigit() and 0 <= int(escolha_classe_0) < len(class_names):
            classe_0_nome = class_names[int(escolha_classe_0)]
            break
        else:
            print("Número inválido! Escolha um número da lista acima.")

    print(f"\n🔹 **Definição do problema binário:**")
    print(f"    Classe `{classe_0_nome}` será a classe `0`")
    print(f"    Classes `{[c for i, c in enumerate(class_names) if i != int(escolha_classe_0)]}` serão agrupadas na classe `1`\n")

    return nome_dataset, classe_0_nome, X, y, class_names

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
    Calcula a sensibilidade (recall da classe positiva).
    """
    return recall_score(y_true, y_pred, pos_label=1)

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
    # Selecionar dataset e classe
    nome_dataset, classe_0_nome, X, y, class_names = selecionar_dataset_e_classe()
    if nome_dataset is not None:
        # Executar a busca de hiperparâmetros
        ranking = param_search(X, y)
        print("🔹 **Ranking das 5 Melhores Combinações:**")
        print(ranking)