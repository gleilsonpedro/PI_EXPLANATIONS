import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, fetch_openml, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import urllib.request
import os
from io import StringIO

# Timeout para requisições web (segundos)
REQUEST_TIMEOUT = 20

def carregar_dataset(nome_dataset, reduzir_mnist=True, tamanho_mnist=1000):
    """
    Carrega os datasets com tratamento robusto.
    Parâmetros:
        - reduzir_mnist: Se True, reduz MNIST para tamanho_mnist amostras
        - tamanho_mnist: Número de amostras para manter no MNIST
    """
    try:
        if nome_dataset == 'iris':
            data = load_iris()
            X = pd.DataFrame(data.data, columns=data.feature_names)
            y = data.target
            class_names = list(data.target_names)
            
        elif nome_dataset == 'pima_indians_diabetes':
            url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
            col_names = ['num_gravidezes', 'glicose', 'pressao_sangue', 'espessura_pele',
                        'insulina', 'imc', 'diabetes_pedigree', 'idade', 'target']
            data = pd.read_csv(url, header=None, names=col_names)
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            class_names = ['Não Diabético', 'Diabético']
            
        elif nome_dataset == 'breast_cancer':
            data = load_breast_cancer()
            X = pd.DataFrame(data.data, columns=data.feature_names)
            y = data.target
            class_names = list(data.target_names)
            
        elif nome_dataset == 'mnist':
            data = fetch_openml('mnist_784', version=1, as_frame=False)
            X, y = data.data, data.target.astype(int)
            mask = (y == 0) | (y == 1)
            X, y = X[mask], y[mask]
            if reduzir_mnist and len(X) > tamanho_mnist:
                X, _, y, _ = train_test_split(
                    X, y,
                    train_size=tamanho_mnist,
                    stratify=y,
                    random_state=42
                )
            X = pd.DataFrame(X)
            class_names = ['Dígito 0', 'Dígito 1']
            
        elif nome_dataset == 'creditcard':
            data = fetch_openml('creditcard', version=1, as_frame=True)
            X, y = data.data, data.target.astype(int)
            X, _, y, _ = train_test_split(
                X, y,
                train_size=0.05,
                stratify=y,
                random_state=42
            )
            class_names = ['Normal', 'Fraude']
            
        else:
            raise ValueError("Dataset não suportado.")
            
        return X, y, class_names
        
    except Exception as e:
        print(f"\nErro ao carregar {nome_dataset}: {str(e)}")
        return None, None, None

def selecionar_dataset_e_classe():
    """
    Menu interativo com datasets testados e funcionais
    """
    menu = '''
    | **************** MENU DE DATASETS CONFIÁVEIS **************** |
    | [0] Iris (150×4×3)             | [1] Pima Diabetes (768×8×2)  |
    | [2] Breast Cancer (569×30×2)   | [3] MNIST - Dígitos 0/1      |
    | [4] Creditcard Fraud (284×30×2)| [Q] SAIR                     |
    |---------------------------------------------------------------|
    '''
    print(menu)

    nomes_datasets = [
        'iris', 'pima_indians_diabetes', 'breast_cancer',
        'mnist', 'creditcard'
    ]

    while True:
        opcao = input("\nDigite o número do dataset ou 'Q' para sair: ").upper().strip()
        
        if opcao == 'Q':
            return None, None, None, None, None
        
        if opcao.isdigit() and 0 <= int(opcao) < len(nomes_datasets):
            nome_dataset = nomes_datasets[int(opcao)]
            print(f"\nCarregando {nome_dataset}...")
            
            if nome_dataset == 'mnist':
                X, y, class_names = carregar_dataset(nome_dataset, reduzir_mnist=True, tamanho_mnist=1000)
            elif nome_dataset == 'creditcard':
                print("Aviso: Dataset grande, carregamento pode demorar...")
                X, y, class_names = carregar_dataset(nome_dataset)
            else:
                X, y, class_names = carregar_dataset(nome_dataset)
            
            if X is None:
                print("Falha ao carregar dataset. Tente novamente.")
                continue
                
            print(f"Dataset carregado! (Amostras: {X.shape[0]}, Features: {X.shape[1]})")
            
            # Mostrar classes disponíveis
            print("\nClasses disponíveis:")
            for i, nome in enumerate(class_names):
                print(f"   [{i}] - {nome}")
            
            # Selecionar classe 0
            while True:
                escolha_0 = input("\nDigite o número da classe que será a classe 0: ")
                if escolha_0.isdigit() and 0 <= int(escolha_0) < len(class_names):
                    classe_0_nome = class_names[int(escolha_0)]
                    break
                else:
                    print("Número inválido!")
            
            # Selecionar classe 1 (se houver mais de 2 classes)
            if len(class_names) > 2:
                classes_restantes = [i for i in range(len(class_names)) if i != int(escolha_0)]
                print("\nSelecione a classe 1:")
                for i in classes_restantes:
                    print(f"   [{i}] - {class_names[i]}")
                
                while True:
                    escolha_1 = input("\nDigite o número da classe que será a classe 1: ")
                    if escolha_1.isdigit() and int(escolha_1) in classes_restantes:
                        classe_1_nome = class_names[int(escolha_1)]
                        break
                    else:
                        print("Número inválido!")
            else:
                classe_1_nome = class_names[1 - int(escolha_0)]
            
            # Filtrar apenas as duas classes selecionadas
            mask = (y == int(escolha_0)) | (y == int(escolha_1)) if len(class_names) > 2 else (y == 0) | (y == 1)
            X = X[mask]
            y = y[mask]
            
            # Converter para binário (0 e 1)
            y_binario = np.where(y == int(escolha_0), 0, 1)
            class_names_bin = [classe_0_nome, classe_1_nome]
            
            print(f"\n🔹 Configuração binária:")
            print(f"   Classe 0: {classe_0_nome}")
            print(f"   Classe 1: {classe_1_nome}\n")
            
            return nome_dataset, classe_0_nome, X, y_binario, class_names_bin
            
        else:
            print("Opção inválida. Tente novamente.")