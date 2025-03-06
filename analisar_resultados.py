import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def carregar_resultados(json_file):
    """
    Carrega os resultados do arquivo JSON.
    """
    with open(json_file, "r") as f:
        resultados = json.load(f)
    return resultados

def listar_datasets_disponiveis(resultados):
    """
    Lista os datasets disponíveis no JSON.
    """
    datasets = set()
    for resultado in resultados:
        datasets.add(resultado["dataset"])
    return list(datasets)

def escolher_dataset(datasets):
    """
    Exibe um menu para escolher qual dataset analisar.
    """
    print("\nDatasets disponíveis para análise:")
    for i, dataset in enumerate(datasets):
        print(f"{i + 1}. {dataset}")
    
    while True:
        try:
            escolha = int(input("\nDigite o número do dataset que deseja analisar: ")) - 1
            if 0 <= escolha < len(datasets):
                return datasets[escolha]
            print("Número inválido! Escolha um número da lista.")
        except ValueError:
            print("Entrada inválida! Digite um número.")

def filtrar_resultados_por_dataset(resultados, dataset):
    """
    Filtra os resultados para um dataset específico.
    """
    return [r for r in resultados if r["dataset"] == dataset]

def analisar_metricas(df):
    """
    Analisa as métricas (tamanho médio das explicações) por percentil e delta_value.
    """
    print("\nAnálise das Métricas:")
    
    # Agrupa por percentil e delta_value, calculando a média da métrica
    df_agrupado = df.groupby(["percentil", "delta_value"])["metrica"].mean().reset_index()
    
    # Plota um gráfico de calor
    pivot_table = df_agrupado.pivot("percentil", "delta_value", "metrica")
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Métrica Média por Percentil e Delta Value")
    plt.show()

def analisar_features_relevantes(resultados):
    """
    Analisa as features mais relevantes nas explicações.
    """
    print("\nAnálise das Features Mais Relevantes:")
    
    # Conta a frequência de cada feature nas explicações
    contagem_features = Counter()
    for resultado in resultados:
        for perturbacao in resultado["perturbacoes"]:
            feature = perturbacao["feature"]
            contagem_features[feature] += 1
    
    # Converte para DataFrame e plota
    df_features = pd.DataFrame(contagem_features.items(), columns=["Feature", "Contagem"])
    df_features = df_features.sort_values(by="Contagem", ascending=False)
    
    sns.barplot(data=df_features, x="Contagem", y="Feature")
    plt.title("Features Mais Relevantes nas Explicações")
    plt.show()

def analisar_perturbacoes(resultados):
    """
    Analisa os intervalos de perturbação das features.
    """
    print("\nAnálise das Perturbações:")
    
    # Calcula a média dos intervalos de perturbação para cada feature
    perturbacoes_agrupadas = {}
    for resultado in resultados:
        for perturbacao in resultado["perturbacoes"]:
            feature = perturbacao["feature"]
            if feature not in perturbacoes_agrupadas:
                perturbacoes_agrupadas[feature] = {"min_val": [], "max_val": []}
            perturbacoes_agrupadas[feature]["min_val"].append(perturbacao["min_val"])
            perturbacoes_agrupadas[feature]["max_val"].append(perturbacao["max_val"])
    
    # Calcula a média dos intervalos
    for feature, valores in perturbacoes_agrupadas.items():
        valores["min_val_medio"] = np.mean(valores["min_val"])
        valores["max_val_medio"] = np.mean(valores["max_val"])
    
    # Converte para DataFrame e plota
    df_perturbacoes = pd.DataFrame(perturbacoes_agrupadas).T.reset_index()
    df_perturbacoes.columns = ["Feature", "min_val", "max_val", "min_val_medio", "max_val_medio"]
    
    sns.barplot(data=df_perturbacoes, x="min_val_medio", y="Feature", color="blue", label="Min Val")
    sns.barplot(data=df_perturbacoes, x="max_val_medio", y="Feature", color="red", label="Max Val")
    plt.title("Intervalos de Perturbação por Feature")
    plt.legend()
    plt.show()

def analisar_impacto_classe_0(df):
    """
    Analisa o impacto da escolha da classe 0 nas métricas.
    """
    print("\nAnálise do Impacto da Classe 0:")
    
    # Agrupa por classe_0 e calcula a média da métrica
    df_classe_0 = df.groupby("classe_0")["metrica"].mean().reset_index()
    
    # Plota a métrica média por classe_0
    sns.barplot(data=df_classe_0, x="classe_0", y="metrica")
    plt.title("Métrica Média por Classe 0")
    plt.show()

def analisar_distribuicao_explicacoes(resultados):
    """
    Analisa a distribuição do tamanho das explicações.
    """
    print("\nAnálise da Distribuição do Tamanho das Explicações:")
    
    # Calcula o tamanho das explicações
    tamanhos_explicacoes = [len(resultado["perturbacoes"]) for resultado in resultados]
    
    # Plota a distribuição
    sns.histplot(tamanhos_explicacoes, kde=True)
    plt.title("Distribuição do Tamanho das Explicações")
    plt.xlabel("Tamanho da Explicação")
    plt.ylabel("Frequência")
    plt.show()

def main():
    # Carrega os resultados do JSON
    json_file = "resultados_gridsearch.json"
    resultados = carregar_resultados(json_file)
    
    # Lista os datasets disponíveis
    datasets = listar_datasets_disponiveis(resultados)
    
    # Escolhe o dataset para análise
    dataset_escolhido = escolher_dataset(datasets)
    
    # Filtra os resultados para o dataset escolhido
    resultados_filtrados = filtrar_resultados_por_dataset(resultados, dataset_escolhido)
    
    # Converte para DataFrame
    df = pd.DataFrame(resultados_filtrados)
    
    # Executa todas as análises
    analisar_metricas(df)
    analisar_features_relevantes(resultados_filtrados)
    analisar_perturbacoes(resultados_filtrados)
    analisar_impacto_classe_0(df)
    analisar_distribuicao_explicacoes(resultados_filtrados)

if __name__ == "__main__":
    main()