from data.load_datasets import selecionar_dataset_e_classe  
from models.train_model import treinar_modelo
from explanations.pi_explanation import analisar_instancias, contar_features_relevantes, calcular_estatisticas_explicacoes
import time
import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

def limpar_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    # Selecionar dataset e classes
    nome_dataset, classe_0_nome, X, y, class_names = selecionar_dataset_e_classe()
    
    if nome_dataset is None:
        print("Você escolheu sair.")
        return

    limpar_terminal()

    classe_1_nome = class_names[1]  # A classe 1 é o segundo elemento da lista

    print(f"**Dataset '{nome_dataset}'**\n")
    print(f"Classes selecionadas:")
    print(f"Classe 0: '{classe_0_nome}'")
    print(f"Classe 1: '{classe_1_nome}'\n")
    print(f"Total de amostras: {X.shape[0]}")
    print(f"Número de atributos: {X.shape[1]}\n")

    # Converter X para DataFrame com nomes consistentes
    feature_names = getattr(X, 'columns', [f"feature_{i}" for i in range(X.shape[1])])
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Treinar modelo
    modelo, X_test, y_test = treinar_modelo(X_df, y)
    

    print("\nCoeficientes do modelo:")
    for name, coef in zip(feature_names, modelo.coef_[0]):
        print(f"{name}: {coef:.4f}")

    # --- PREPARAÇÃO DOS DADOS PARA EXPLICAÇÃO ---
    # Converter X_test para DataFrame
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    # Garantir que y_test seja numpy array
    if isinstance(y_test, pd.Series):
        y_test = y_test.values

    # --- GERAR EXPLICAÇÕES ---
    explicacoes, contagem = analisar_instancias(X_test_df, y_test, classe_0_nome, classe_1_nome, modelo, X_df)
    # Estatísticas das explicações
    estatisticas = calcular_estatisticas_explicacoes(explicacoes)

    print("\n Estatísticas das explicações:")
    print(f"- Média de features por explicação: {estatisticas['media_tamanho']:.2f}")
    print(f"- Desvio padrão: {estatisticas['desvio_padrao_tamanho']:.2f}")

    # --- ESTATÍSTICAS ---
    print(f"\nTotal de instâncias de teste: {len(y_test)}")
    print(f"Distribuição das classes no teste: {pd.Series(y_test).value_counts()}")
    print(f"Instâncias com explicações geradas: {len(explicacoes)}")

    print("\n**Features mais relevantes:**")
    for classe, features in contagem.items():
        print(f"\nPara '{classe_0_nome if '0' in classe else classe_1_nome}':")
        for f, cnt in sorted(features.items(), key=lambda x: -x[1]):
            print(f"  {f}: {cnt} ocorrências")


if __name__ == "__main__":
    main()