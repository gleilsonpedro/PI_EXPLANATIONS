from data.load_datasets import selecionar_dataset_e_classe  
from models.train_model import treinar_modelo
from explanations.pi_explanation import analisar_instancias, contar_features_relevantes
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

    # Treinar modelo
    inicio_treinamento = time.time()
    
    # Converter X para DataFrame com nomes consistentes
    feature_names = getattr(X, 'columns', [f"feature_{i}" for i in range(X.shape[1])])
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Treinar modelo
    modelo, X_test, y_test = treinar_modelo(X_df, y)
    
    # Converter X_test para DataFrame
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    fim_treinamento = time.time()
    tempo_treinamento = fim_treinamento - inicio_treinamento

    print("\nCoeficientes do modelo:")
    for name, coef in zip(feature_names, modelo.coef_[0]):
        print(f"{name}: {coef:.4f}")

    # Calcular PI-explicações
    inicio_pi = time.time()
    explicacoes = analisar_instancias(X_test_df, y_test, classe_0_nome, classe_1_nome, modelo, X_df)
    fim_pi = time.time()
    tempo_pi = fim_pi - inicio_pi

    # Contar features relevantes
    print("\n**Contagem de features relevantes:**")
    contagem = contar_features_relevantes(explicacoes, classe_0_nome, classe_1_nome)

    # Tempo total
    tempo_total = tempo_treinamento + tempo_pi

    print(f"\n**Tempo de treinamento do modelo:**      {tempo_treinamento:.4f} segundos")
    print(f"**Tempo de cálculo das PI-explicações:** {tempo_pi:.4f} segundos")
    print(f"**Tempo total de execução:**             {tempo_total:.4f} segundos")

if __name__ == "__main__":
    main()