import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import pandas as pd
import tempfile
import os
from datetime import datetime
from data.load_datasets import selecionar_dataset_e_classe
from models.train_model import treinar_modelo
from explanations.pi_explanation import analisar_instancias, calcular_estatisticas_explicacoes

class ExperimentLogger:
    def __init__(self):
        self.results_dir = "experiment_results"
        os.makedirs(self.results_dir, exist_ok=True)
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = []
        self.current_info = {}  # Adicionado para armazenar informações do dataset
        
    def log_dataset_info(self, nome_dataset, class_names, X):
        """Registra informações básicas do dataset"""
        self.current_info = {
            "dataset": nome_dataset,
            "classe_0": class_names[0],
            "classe_1": class_names[1],
            "amostras": X.shape[0],
            "atributos": X.shape[1]
        }
        
    def log_experiment(self, C, penalty, max_iter, stats, y_test):
        """Registra os resultados de uma configuração específica"""
        valores, contagens = np.unique(y_test, return_counts=True)
        
        self.results.append({
            **self.current_info,  # Inclui as informações do dataset
            "C": C,
            "penalty": penalty,
            "max_iter": max_iter,
            "media_features": stats['media_tamanho'],
            "desvio_padrao": stats['desvio_padrao_tamanho'],
            "test_0": contagens[0] if len(contagens) > 0 else 0,
            "test_1": contagens[1] if len(contagens) > 1 else 0
        })
        
    def save_results(self):
        """Salva os resultados em CSV e JSON"""
        df = pd.DataFrame(self.results)
        
        csv_path = os.path.join(self.results_dir, f"results_{self.experiment_id}.csv")
        json_path = os.path.join(self.results_dir, f"results_{self.experiment_id}.json")
        
        df.to_csv(csv_path, index=False)
        df.to_json(json_path, orient='records', indent=2)
        
        return csv_path

def experimento_completo(valores_C, penalties, max_iters, nome_dataset=None):
    logger = ExperimentLogger()
    
    # Seleção do dataset
    if nome_dataset is None:
        nome_dataset, classe_0_nome, X, y, class_names = selecionar_dataset_e_classe()
    else:
        from data.load_datasets import carregar_dataset
        X, y, class_names = carregar_dataset(nome_dataset)
        classe_0_nome = class_names[0]
        y = np.where(y == 0, 0, 1)
    
    logger.log_dataset_info(nome_dataset, class_names, X)
    
    for C in valores_C:
        for penalty in penalties:
            for max_iter in max_iters:
                print(f"\n🔧 Config: C={C}, penalty={penalty}, max_iter={max_iter}")
                
                modelo, X_test, y_test = treinar_modelo(
                    pd.DataFrame(X), y, 
                    regularizacao=C,
                    penalty=penalty,
                    max_iter=max_iter
                )
                
                X_test_df = pd.DataFrame(X_test, columns=X.columns if hasattr(X, 'columns') else None)
                explicacoes, _ = analisar_instancias(X_test_df, y_test, class_names[0], class_names[1], modelo, pd.DataFrame(X))
                
                stats = calcular_estatisticas_explicacoes(explicacoes)
                logger.log_experiment(C, penalty, max_iter, stats, y_test)
                
                print(f"➡️ Features: {stats['media_tamanho']:.2f} ± {stats['desvio_padrao_tamanho']:.2f}")
                logger.log_experiment(C, penalty, max_iter, stats, y_test)
    return logger.save_results()

def mostrar_resultados_tabulados(csv_path):
    df = pd.read_csv(csv_path)
    print("\n" + "="*80)
    print("RESULTADOS COMPLETOS DOS EXPERIMENTOS")
    print("="*80)
    
    # Agrupar por dataset para melhor visualização
    for dataset, group in df.groupby('dataset'):
        print(f"\n📊 Dataset: {dataset} ({group['amostras'].iloc[0]} amostras, {group['atributos'].iloc[0]} atributos)")
        print(f"Classes: {group['classe_0'].iloc[0]} vs {group['classe_1'].iloc[0]}")
        
        # Reorganizar os dados para tabulate
        table_data = []
        for _, row in group.iterrows():
            table_data.append([
                row['C'],
                row['penalty'],
                row['max_iter'],
                f"{row['media_features']:.2f} ± {row['desvio_padrao']:.2f}",
                f"{row['test_0']} vs {row['test_1']}"
            ])
        
        print(tabulate(
            table_data,
            headers=['C', 'Penalty', 'Max Iter', 'Média Features', 'Test (0 vs 1)'],
            tablefmt='grid',
            floatfmt=".2f"
        ))

if __name__ == "__main__":
    # Configuração dos experimentos
    valores_C = [0.01, 0.1, 1.0, 10.0]
    penalties = ['l1', 'l2']
    max_iters = [200, 600, 1000]
    
    # Para testar com um dataset específico:
    # resultados_path = experimento_completo(valores_C, penalties, max_iters, 'heart_disease')
    
    # Para seleção interativa:
    resultados_path = experimento_completo(valores_C, penalties, max_iters)
    
    mostrar_resultados_tabulados(resultados_path)
    print(f"\n📁 Dados salvos em: {resultados_path}")