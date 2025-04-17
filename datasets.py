import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from pathlib import Path
from sklearn.model_selection import train_test_split

CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)

def load_dataset(name):
    """Carrega dataset com cache"""
    cache_file = CACHE_DIR / f"{name}.pkl"
    if cache_file.exists():
        return pd.read_pickle(cache_file)

    try:
        if name == 'iris':
            data = load_iris()
            X = pd.DataFrame(data.data, columns=data.feature_names)
            y = data.target
            classes = list(data.target_names)
            
        elif name == 'diabetes':
            url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
            cols = ['pregnancies', 'glucose', 'pressure', 'skin_thickness',
                    'insulin', 'bmi', 'pedigree', 'age', 'outcome']
            df = pd.read_csv(url, header=None, names=cols)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            classes = ['no_diabetes', 'diabetes']
            
        elif name == 'cancer':
            data = load_breast_cancer()
            X = pd.DataFrame(data.data, columns=data.feature_names)
            y = data.target
            classes = list(data.target_names)
            
        elif name == 'wine':
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
            df = pd.read_csv(url, sep=";")
            df['target'] = df['quality'].apply(lambda x: 1 if x >= 6 else 0)
            X = df.drop(['quality', 'target'], axis=1)
            y = df['target']
            classes = ['bad', 'good']
            
        elif name == 'heart':
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
            cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                   'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
            df = pd.read_csv(url, names=cols, na_values="?")
            df = df.dropna()
            df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
            X = df.drop('target', axis=1)
            y = df['target']
            classes = ['no_disease', 'disease']
            
        pd.to_pickle((X, y, classes), cache_file)
        return X, y, classes
        
    except Exception as e:
        print(f"Error loading {name}: {str(e)}")
        return None, None, None

def prepare_data(X, y, class0, class1):
    """Prepara dados para classificação binária"""
    mask = (y == class0) | (y == class1)
    X = X[mask]
    y = np.where(y[mask] == class0, 0, 1)
    return X, y