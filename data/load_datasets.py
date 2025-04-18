# data/load_datasets.py
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, fetch_openml, load_breast_cancer
from sklearn.model_selection import train_test_split

def carregar_dataset(nome_dataset):
    """Carrega o dataset especificado"""
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
            data = load_breast_cancer()
            X = pd.DataFrame(data.data, columns=data.feature_names)
            y = data.target
            class_names = list(data.target_names)
            
        elif nome_dataset == 'heart_disease':
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
            data = pd.read_csv(url, header=None, na_values="?")
            data = data.dropna()
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1].apply(lambda x: 1 if x > 0 else 0)
            class_names = ['Sem Doença Cardíaca', 'Com Doença Cardíaca']
            
        elif nome_dataset == 'mnist':
            data = fetch_openml('mnist_784', version=1, as_frame=False)
            X, y = data.data, data.target.astype(int)
            mask = (y == 0) | (y == 1)
            X, y = X[mask], y[mask]
            X = pd.DataFrame(X)
            class_names = ['Dígito 0', 'Dígito 1']
            
        return X, y, class_names
        
    except Exception as e:
        print(f"Erro ao carregar {nome_dataset}: {str(e)}")
        return None, None, None

def selecionar_dataset_e_classe():
    """Interface para seleção de dataset e classes"""
    menu = '''
    |  *************** MENU DE DATASETS ***************  |
    | [0] Iris (150×4×3)            | [1] Pima Diabetes (768×8×2)    |
    | [2] Breast Cancer (569×30×2)  | [3] Heart Disease  (303×13×2)  |
    | [4] MNIST - Dígitos 0/1 (reduzido) | [Q] SAIR                  |
    '''
    print(menu)

    datasets = ['iris', 'pima_indians_diabetes', 'breast_cancer', 'heart_disease', 'mnist']
    
    while True:
        opcao = input("\nDigite o número do dataset ou 'Q' para sair: ").upper()
        
        if opcao == 'Q':
            return None, None, None, None, None
        
        if opcao.isdigit() and 0 <= int(opcao) < len(datasets):
            nome_dataset = datasets[int(opcao)]
            print(f"\nCarregando {nome_dataset}...")
            X, y, class_names = carregar_dataset(nome_dataset)
            
            if X is None:
                continue
                
            print(f"Dataset carregado! (Amostras: {X.shape[0]}, Features: {X.shape[1]})")
            
            # Para datasets com mais de 2 classes
            if len(class_names) > 2:
                print("\nClasses disponíveis:")
                for i, nome in enumerate(class_names):
                    print(f"   [{i}] - {nome}")
                    
                while True:
                    try:
                        classe0 = int(input("\nDigite o número da classe 0 (negativa): "))
                        if classe0 not in range(len(class_names)):
                            raise ValueError
                            
                        remaining = [i for i in range(len(class_names)) if i != classe0]
                        if len(remaining) == 1:
                            classe1 = remaining[0]
                            print(f"Classe 1 (positiva) automática: {class_names[classe1]}")
                        else:
                            print("\nEscolha a classe 1 (positiva):")
                            for i in remaining:
                                print(f"[{i}] {class_names[i]}")
                            classe1 = int(input("Opção: "))
                            if classe1 not in remaining:
                                raise ValueError
                        
                        # Filtra as classes selecionadas
                        mask = (y == classe0) | (y == classe1)
                        X = X[mask]
                        y = y[mask]
                        y = np.where(y == classe0, 0, 1)
                        class_names = [class_names[classe0], class_names[classe1]]
                        break
                        
                    except ValueError:
                        print("Opção inválida! Tente novamente.")
            
            return nome_dataset, class_names[0], X, y, class_names
        
        print("Opção inválida. Tente novamente.")