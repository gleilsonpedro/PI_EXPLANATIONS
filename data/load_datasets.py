import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, fetch_openml, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import urllib.request
import os
from io import StringIO

# Timeout para requisições web (segundos)
REQUEST_TIMEOUT = 15

def carregar_dataset(nome_dataset, reduzir_mnist=True, tamanho_mnist=1000):
    """
    Carrega os 5 datasets focados com tratamento robusto.
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
            data = pd.read_csv(url)
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            class_names = ['Não Diabético', 'Diabético']
            
        elif nome_dataset == 'breast_cancer':
            # Usando dataset direto do sklearn
            data = load_breast_cancer()
            X = pd.DataFrame(data.data, columns=data.feature_names)
            y = data.target
            class_names = list(data.target_names)
            
        elif nome_dataset == 'heart_disease':
            # Dataset corrigido do Heart Disease
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
            try:
                data = pd.read_csv(url, header=None, na_values="?")
            except:
                # Fallback local se necessário
                print("Usando fallback local para Heart Disease")
                data = pd.read_csv("heart_disease.csv", header=None, na_values="?")
            
            data = data.dropna()
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1].apply(lambda x: 1 if x > 0 else 0)
            class_names = ['Sem Doença Cardíaca', 'Com Doença Cardíaca']
            
        elif nome_dataset == 'mnist':
            # Carrega MNIST e reduz automaticamente
            data = fetch_openml('mnist_784', version=1, as_frame=False)
            X, y = data.data, data.target.astype(int)
            
            # Filtra apenas dígitos 0 e 1
            mask = (y == 0) | (y == 1)
            X, y = X[mask], y[mask]
            
            # Reduz o tamanho se solicitado
            if reduzir_mnist and len(X) > tamanho_mnist:
                X, _, y, _ = train_test_split(
                    X, y,
                    train_size=tamanho_mnist,
                    stratify=y,
                    random_state=42
                )
            
            X = pd.DataFrame(X)
            class_names = ['Dígito 0', 'Dígito 1']
            
        else:
            raise ValueError("Dataset não suportado.")
            
        return X, y, class_names
        
    except Exception as e:
        print(f"\n Erro ao carregar {nome_dataset}: {str(e)}")
        return None, None, None

def selecionar_dataset_e_classe():
    """
    Menu interativo com todas as correções implementadas.
    """
    menu = '''
    |  *************** MENU DE DATASETS (OTIMIZADO) ***************  |
    | [0] Iris (150×4×3)            | [1] Pima Diabetes (768×8×2)    |
    | [2] Breast Cancer (569×30×2)  | [ ] Heart Disease  (303×13×2)  |
    | [4] MNIST - Dígitos 0/1 (reduzido) | [Q] SAIR                  |
    |----------------------------------------------------------------|
    '''
    print(menu)

    nomes_datasets = [
        'iris', 'pima_indians_diabetes', 'breast_cancer',
        'heart_disease', 'mnist'
    ]

    while True:
        opcao = input("\nDigite o número do dataset ou 'Q' para sair: ").upper().strip()
        
        if opcao == 'Q':
            return None, None, None, None, None
        
        if opcao.isdigit() and 0 <= int(opcao) < len(nomes_datasets):
            nome_dataset = nomes_datasets[int(opcao)]
            print(f"\nCarregando {nome_dataset}...")
            
            # Configurações especiais para MNIST
            if nome_dataset == 'mnist':
                X, y, class_names = carregar_dataset(nome_dataset, reduzir_mnist=True, tamanho_mnist=1000)
            else:
                X, y, class_names = carregar_dataset(nome_dataset)
            
            if X is None:
                print("Falha ao carregar dataset. Tente novamente.")
                continue
                
            print(f"Dataset carregado! (Amostras: {X.shape[0]}, Features: {X.shape[1]})")
            
            # MNIST já vem pré-configurado como binário (0 vs 1)
            if nome_dataset == 'mnist':
                return nome_dataset, class_names[0], X, y, class_names
                
            # Para outros datasets, selecionar classe alvo
            print("\nClasses disponíveis:")
            for i, nome in enumerate(class_names):
                print(f"   [{i}] - {nome}")
                
            while True:
                escolha = input("\nDigite o número da classe que será a classe 0: ")
                if escolha.isdigit() and 0 <= int(escolha) < len(class_names):
                    classe_0_nome = class_names[int(escolha)]
                    y_binario = np.where(y == int(escolha), 0, 1)
                    break
                else:
                    print("Número inválido!")
                    
            print(f"\n🔹 Configuração binária:")
            print(f"   Classe 0: {classe_0_nome}")
            print(f"   Classe 1: Outras classes\n")
            
            return nome_dataset, classe_0_nome, X, y_binario, class_names
            
        else:
            print("Opção inválida. Tente novamente.")