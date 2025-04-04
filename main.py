from data.load_datasets import selecionar_dataset_e_classe  
from models.train_model import treinar_modelo
from explanations.pi_explanation import analisar_instancias, contar_features_relevantes
import time
import os

def limpar_terminal():
    if os.name == 'nt':  # Windows
        os.system('cls')
    else:  # Linux/Mac
        os.system('clear')

def main():
    # Selecionar dataset e classe
    nome_dataset, classe_0_nome, X, y, class_names = selecionar_dataset_e_classe()
    
    if nome_dataset is None:
        print("Você escolheu sair.")
        return

    limpar_terminal()

    print(f"**Dataset '{nome_dataset}'**\n")
    print(f"Classes disponíveis: {class_names}")
    print(f"Classe 0: '{classe_0_nome}'\n")
    print(f"Total de amostras: {X.shape[0]}")
    print(f"Número de atributos: {X.shape[1]}\n")

    # Treinar modelo
    inicio_treinamento = time.time()
    modelo, X_test, y_test = treinar_modelo(X, y, classe_0=0)
    fim_treinamento = time.time()
    tempo_treinamento = fim_treinamento - inicio_treinamento

    print("Coeficientes do modelo:")
    for name, coef in zip(X.columns, modelo.coef_[0]):
        print(f"{name}: {coef:.4f}")

    # Calcular PI-explicações
    inicio_pi = time.time()
    explicacoes = analisar_instancias(X_test, y_test, class_names, modelo, X)
    fim_pi = time.time()
    tempo_pi = fim_pi - inicio_pi

    # Tempo total
    tempo_total = tempo_treinamento + tempo_pi

    print()
    print(f"**Tempo de treinamento do modelo:**      {tempo_treinamento:.4f} segundos")
    print(f"**Tempo de cálculo das PI-explicações:** {tempo_pi:.4f} segundos")
    print(f"**Tempo total de execução:**             {tempo_total:.4f} segundos\n")

    # Contar features relevantes
    print("**Contagem de features relevantes:**")
    contar_features_relevantes(explicacoes, class_names)

if __name__ == "__main__":
    main()