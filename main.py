from data.load_datasets import selecionar_dataset_e_classe
from models.train_model import treinar_modelo
from explanations.pi_explanation import gerar_relatorio
import os

def limpar_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    limpar_terminal()
    
    # Carrega dados
    nome_dataset, _, X, y, class_names = selecionar_dataset_e_classe()
    if nome_dataset is None:
        return

    print(f"\nDataset selecionado: {nome_dataset}")
    print(f"Classes: {class_names[0]} (0) vs {class_names[1]} (1)")

    # Treina modelo
    print("\nTreinando modelo...")
    modelo, X_test, y_test = treinar_modelo(X, y)
    X_train = X.drop(X_test.index)

    # Gera relatório completo
    print("\nGerando explicações...")
    relatorio = gerar_relatorio(modelo, X_test, y_test, X_train, class_names)
    
    # Mostra relatório paginado
    print(relatorio)

if __name__ == "__main__":
    main()