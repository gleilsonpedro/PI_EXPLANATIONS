import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, fetch_openml
from sklearn.model_selection import train_test_split

def carregar_dataset(nome_dataset):
    """
    Carrega o dataset escolhido.
    Retorna:
        - X: Features do dataset.
        - y: Labels do dataset.
        - class_names: Nomes das classes.
    """
    if nome_dataset == 'iris':
        data = load_iris()
        X, y = pd.DataFrame(data.data, columns=data.feature_names), data.target
        class_names = data.target_names
    
    elif nome_dataset == 'pima_indians_diabetes':
        data = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv")
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        class_names = ['No Diabetes', 'Diabetes']
    
    elif nome_dataset == 'heart_disease':
        data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", header=None, na_values="?")
        data = data.dropna()  # Remove valores ausentes
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        class_names = ['No Heart Disease', 'Heart Disease']
    
    elif nome_dataset == 'breast_cancer':
        data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data", header=None)
        X = data.iloc[:, 2:]
        y = data.iloc[:, 1].apply(lambda x: 0 if x == 'B' else 1)  # Benign (0), Malignant (1)
        class_names = ['Benign', 'Malignant']
    
    elif nome_dataset == 'parkinsons':
        data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data")
        X = data.drop(columns=['status', 'name'])
        y = data['status']
        class_names = ['Healthy', 'Parkinsons']
    
    elif nome_dataset == 'mnist':
        data = fetch_openml('mnist_784', version=1)
        X, y = data.data, data.target.astype(int)
        class_names = [str(i) for i in range(10)]  # Dígitos de 0 a 9
    
    elif nome_dataset == 'fashion_mnist':
        data = fetch_openml('Fashion-MNIST', version=1)
        X, y = data.data, data.target.astype(int)
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    elif nome_dataset == 'cifar10':
        data = fetch_openml('CIFAR_10', version=1)
        X, y = data.data, data.target.astype(int)
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    
    elif nome_dataset == 'letter_recognition':
        data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data", header=None)
        X = data.iloc[:, 1:]
        y = data.iloc[:, 0].apply(lambda x: ord(x) - ord('A'))  # Convertendo letras para números (0-25)
        class_names = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # Letras de A a Z
    
    elif nome_dataset == 'covertype':
        data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz", header=None)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1] - 1  # Classes começam em 1, então subtraímos 1 para começar em 0
        class_names = [f"Covertype {i+1}" for i in range(7)]  # 7 tipos de cobertura do solo
    
    else:
        raise ValueError("Nome do dataset não reconhecido. Escolha um dataset válido.")
    
    return X, y, class_names

def selecionar_dataset_e_classe():
    """
    Menu para selecionar o dataset e a classe que será a classe 0.
    Retorna:
        - nome_dataset: Nome do dataset escolhido.
        - classe_0_nome: Nome da classe que será a classe 0.
        - X: Features do dataset.
        - y: Labels do dataset (binário: 0 para a classe escolhida, 1 para as outras).
        - class_names: Nomes das classes.
    """
    # Menu de seleção de datasets
    menu = '''
    |  ************************* MENU ***************************  |
    |  0 - iris                     |  1 - pima_indians_diabetes   |
    |  2 - heart_disease            |  3 - breast_cancer          |
    |  4 - parkinsons               |  5 - mnist                  |
    |  6 - fashion_mnist            |  7 - cifar10                |
    |  8 - letter_recognition       |  9 - covertype              |
    |  Q - SAIR                                                   |
    |-------------------------------------------------------------|
    '''
    print(menu)

    # Lista de datasets disponíveis
    nomes_datasets = [
        'iris', 'pima_indians_diabetes', 'heart_disease', 'breast_cancer',
        'parkinsons', 'mnist', 'fashion_mnist', 'cifar10', 'letter_recognition', 'covertype'
    ]

    # Solicitar a escolha do dataset
    while True:
        opcao = input("Digite o número do dataset ou 'Q' para sair: ").upper().strip()
        if opcao == 'Q':
            print("Você escolheu sair.")
            return None, None, None, None, None
        elif opcao.isdigit() and 0 <= int(opcao) < len(nomes_datasets):
            nome_dataset = nomes_datasets[int(opcao)]
            break
        else:
            print("Opção inválida. Por favor, escolha um número do menu ou 'Q' para sair.")

    # Carregar o dataset
    X, y, class_names = carregar_dataset(nome_dataset)

    # Exibir as classes disponíveis
    print("\nClasses disponíveis:")
    for i, class_name in enumerate(class_names):
        print(f"   [{i}] - {class_name}")

    # Solicitar a escolha da classe 0
    while True:
        escolha_classe_0 = input("\nDigite o número da classe que será `0`: ")
        if escolha_classe_0.isdigit() and 0 <= int(escolha_classe_0) < len(class_names):
            classe_0_nome = class_names[int(escolha_classe_0)]
            break
        else:
            print("Número inválido! Escolha um número da lista acima.")

    # Transformar o problema em binário
    y = np.where(y == int(escolha_classe_0), 0, 1)  # Classe 0 é a escolhida, o resto é 1

    print(f"\n🔹 **Definição do problema binário:**")
    print(f"    Classe `{classe_0_nome}` será a classe `0`")
    print(f"    Classes `{[c for i, c in enumerate(class_names) if i != int(escolha_classe_0)]}` serão agrupadas na classe `1`\n")

    return nome_dataset, classe_0_nome, X, y, class_names