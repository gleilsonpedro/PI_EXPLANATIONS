import pandas as pd
import numpy as np
import os
from data.load_datasets import carregar_dataset
from models.train_model import treinar_modelo
from explanations.pi_explanation import analisar_instancias

# 🔹 Função para limpar o terminal
def limpar_terminal():
    if os.name == 'nt':  # Windows
        os.system('cls')
    else:  # Linux/Mac
        os.system('clear')

# 🔹 Menu de seleção de datasets
menu = '''
|  ************************* MENU *************************** |
|  0 - iris                     |  1 - wine                   |
|  2 - breast_cancer            |  3 - digits                 |
|  4 - banknote_authentication  |  5 - wine_quality           |
|  6 - heart_disease            |  7 - parkinsons             |
|  8 - car_evaluation           |  9 - diabetes_binary        |
|  Q - SAIR                                                   |
|-------------------------------------------------------------|
'''

def busca_hiperparametros(dataset, classe_0, num_features, acuracia, alpha=0.5, max_features=10):
    """
    busca os hiperparâmetros (percentil e delta_value) e calcula uma métrica combinada
    para o dataset e à classe 0 escolhida
    -> A métrica combinada é usada para rankear as melhores combinações de percentil e delta_value:
        - O número de features (normalizado por max_features).
        - O erro do modelo (1 - acurácia).
        - O percentil (p / 100).
        - O delta_value (delta_value / 10).
    """
    # Valores de percentil e delta_value para simulação
    percentis = [10, 25, 50, 75]
    valores_delta = [0.5, 1.0, 1.5]
    
    # Lista para armazenar os resultados simulados
    resultados = []
    
    # Calcula o erro (1 - acurácia)
    erro = 1 - acuracia
    
    # Simula os resultados para cada combinação de percentil e delta_value
    for p in percentis:
        for delta_value in valores_delta:
            # Calcula a métrica combinada
            metrica = (alpha * (num_features / max_features)) + ((1 - alpha) * erro) + (p / 100) + (delta_value / 10)
            
            # Adiciona o resultado simulado à lista
            resultados.append({
                "dataset": dataset,
                "classe_0": classe_0,
                "percentil": p,
                "delta_value": delta_value,
                "metrica": metrica
            })
    
    return resultados

def calcular_ranking(resultados):
    """
    Calcula o ranking das 5 melhores combinações de percentil e delta_value.
    """
    # Converte os resultados para um DataFrame
    df = pd.DataFrame(resultados)
    
    # Agrupa por percentil e delta_value, calculando a média da métrica
    df_agrupado = df.groupby(["percentil", "delta_value"])["metrica"].mean().reset_index()
    
    # Ordena pelo valor da métrica (menor é melhor)
    df_agrupado = df_agrupado.sort_values(by="metrica", ascending=True)
    
    # Retorna as 5 melhores combinações
    return df_agrupado.head(5)

def exibir_ranking(ranking, nome_dataset, classe_0_nome, acuracia, desvio_padrao=None):
    """
    Exibe o ranking das 5 melhores combinações com informações adicionais.
    """
    print("\n🔹 **Informações Gerais:**")
    print(f"  - Acurácia: {acuracia:.4f}")
    if desvio_padrao is not None:
        print(f"  - Desvio Padrão: {desvio_padrao:.4f}")
    print(f"  - Nome do Dataset: {nome_dataset}")
    print(f"  - Feature 0 vs Outras Features: Classe `{classe_0_nome}` vs outras classes\n")

    print("🔹 **Ranking das 5 Melhores Combinações:**")
    print("| Percentil | Delta Value | Métrica Combinada |")
    print("|-----------|-------------|-------------------|")
    for _, row in ranking.iterrows():
        print(f"| {row['percentil']:^9} | {row['delta_value']:^11} | {row['metrica']:^17.4f} |")

def main():
    # Exibe o menu e solicita uma escolha
    limpar_terminal()
    print(menu)
    opcao = input("Digite o número do dataset ou 'Q' para sair: ").upper().strip()

    if opcao == 'Q':
        print("Você escolheu sair.")
        return  # Encerra o programa

    if opcao.isdigit() and 0 <= int(opcao) <= 9:
        nomes_datasets = [
            'iris', 'wine', 'breast_cancer', 'digits', 'banknote_authentication',
            'wine_quality', 'heart_disease', 'parkinsons', 'car_evaluation', 'diabetes_binary'
        ]
        nome_dataset = nomes_datasets[int(opcao)]
        
        # Limpa o terminal após a escolha do dataset
        limpar_terminal()
        print(f"**Dataset '{nome_dataset}' escolhido.**\n")
        
        try:
            # Carrega o dataset
            X, y, class_names = carregar_dataset(nome_dataset)
            num_features = X.shape[1]  # Número de features
            
            # Exibe as classes disponíveis
            print("Classes disponíveis:")
            for i, class_name in enumerate(class_names):
                print(f"   [{i}] - {class_name}")
            
            # Solicita a escolha da classe 0
            escolha_classe_0 = int(input("\nDigite o número da classe que será `0`: "))
            if not 0 <= escolha_classe_0 < len(class_names):
                print("Número inválido! Escolha um número da lista acima.")
                return  # Encerra o programa em caso de erro

            classe_0_nome = class_names[escolha_classe_0]
            
            print(f"\n🔹 **Definição do problema binário:**")
            print(f"    Classe `{classe_0_nome}` será a classe `0`")
            print(f"    Classes `{[c for i, c in enumerate(class_names) if i != escolha_classe_0]}` serão agrupadas na classe `1`\n")
            
            # Ajusta o y para o problema binário
            y_binario = [0 if label == escolha_classe_0 else 1 for label in y]
            
            # Treina o modelo e calcula a acurácia
            modelo, X_test, y_test = treinar_modelo(X, y_binario, classe_0=0)
            y_pred = modelo.predict(X_test)
            from sklearn.metrics import accuracy_score
            acuracia = accuracy_score(y_test, y_pred)
            
            # Gera dados simulados para análise
            resultados = busca_hiperparametros(nome_dataset, classe_0_nome, num_features, acuracia)
            
            # Calcula o ranking das 5 melhores combinações
            ranking = calcular_ranking(resultados)
            
            # Exibe o ranking no terminal
            exibir_ranking(ranking, nome_dataset, classe_0_nome, acuracia)
            
        except Exception as e:
            print(f"Erro ao processar o dataset: {e}")
            return  # Encerra o programa em caso de erro
    else:
        print("Opção inválida. Por favor, escolha um número do menu ou 'Q' para sair.")
        return  # Encerra o programa em caso de erro

if __name__ == "__main__":
    main()