from data.load_datasets import selecionar_dataset_e_classe, carregar_dataset
from models.train_model import treinar_modelo
from explanations.pi_explanation import analisar_instancias, contar_features_relevantes, calcular_estatisticas_explicacoes
import time
import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

def limpar_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    # Selecionar dataset e classes
    nome_dataset, classe_0_nome, X, y, class_names = selecionar_dataset_e_classe()
        # Carrega dataset (usará cache se disponível)
    X, y, class_names = carregar_dataset('heart_disease')
    
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
    
    # Treinar modelo # SETANDO O NUMERO DE FEATURES PARA A EXPLICAÇÃO
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
    print(f"Distribuição das classes no teste:")
    print(f"\nTotal de instâncias de teste: {len(y_test)}")
    valores, contagens = np.unique(y_test, return_counts=True)
    for i, c in zip(valores, contagens):
        nome = class_names[i]
        print(f"  {nome} (Classe {i}): {c} instâncias")

    print("\n**Features mais relevantes:**")
    for classe, features in contagem.items():
        print(f"\nPara '{classe_0_nome if '0' in classe else classe_1_nome}':")
        for f, cnt in sorted(features.items(), key=lambda x: -x[1]):
            print(f"  {f}: {cnt} ocorrências")


    import joblib

    # Empacotar as features realmente usadas por instância
    # (Você já tem a lista `explicacoes` → vamos extrair as features dela)
    features_usadas_lista = []
    for exp in explicacoes:
        if "Nenhuma" in exp:
            features_usadas_lista.append([])
        else:
            partes = exp.split(": ")[1] if ": " in exp else ""
            features = [f.split(" = ")[0].strip() for f in partes.split(", ")]
            features_usadas_lista.append(features)

    # Salvar tudo para uso posterior no teste de robustez
    joblib.dump((modelo, X_test_df, explicacoes, features_usadas_lista, X_df, y_test, class_names), 'artefatos_para_teste_pi.pkl')

    #joblib.dump((modelo, X_test_df, explicacoes, features_usadas_lista, X_df), 'artefatos_para_teste_pi.pkl')
    print("\n Dados salvos em 'artefatos_para_teste_pi.pkl' para validação das PI-explicações.")


if __name__ == "__main__":
    main()