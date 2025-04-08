import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from load_datasets import selecionar_dataset_e_classe
import shap
import sys
import warnings
warnings.filterwarnings("ignore")

def main():
    # 1. Configuração inicial
    print("=== ANÁLISE COMPARATIVA SHAP ===")
    
    # 2. Carregar dados
    nome_dataset, classe_0_nome, X, y, class_names = selecionar_dataset_e_classe()
    if nome_dataset is None:
        print("Operação cancelada pelo usuário.")
        return
    
    # 3. Preparar dados
    X = pd.DataFrame(X)
    feature_names = X.columns.tolist()
    
    # 4. Treinar modelo
    print("\nTreinando modelo...")
    model = LogisticRegression(max_iter=1000, solver='liblinear')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Acurácia do modelo: {accuracy:.2%}")
    
    # 5. Configurar explainer SHAP
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_test)
    
    # Corrigindo o formato dos valores esperados para classificação binária
    if not isinstance(explainer.expected_value, list):
        explainer.expected_value = [explainer.expected_value]
    
    # 6. Plot de resumo global
    print("\nGerando visualizações SHAP...")
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title(f"Features Mais Relevantes para '{classe_0_nome}'", fontsize=14)
    plt.tight_layout()
    global_summary_file = f"shap_{nome_dataset}_global_summary.png"
    plt.savefig(global_summary_file, dpi=300)
    plt.close()
    print(f"-> Gráfico global salvo como '{global_summary_file}'")
    
    # 7. Análise por instância
    while True:
        print(f"\nDataset: {nome_dataset} | Classe 0: {classe_0_nome}")
        print(f"Instâncias disponíveis (0 a {len(X_test)-1})")
        idx = input("Digite o índice da instância (ou 's' para sair): ").strip().lower()
        
        if idx == 's':
            break
            
        if not idx.isdigit() or not (0 <= int(idx) < len(X_test)):
            print("Índice inválido!")
            continue
            
        idx = int(idx)
        instance = X_test.iloc[idx]
        true_class = y_test[idx]
        
        # 8. Plot de decisão para a instância
        plt.figure(figsize=(12, 6))
        shap.decision_plot(
            explainer.expected_value[0], 
            shap_values[idx], 
            feature_names=feature_names,
            show=False,
            highlight=int(true_class)
        )
        plt.title(f"Contribuição das Features - Instância {idx}\nClasse Real: {class_names[true_class]}", fontsize=14)
        plt.tight_layout()
        decision_file = f"shap_{nome_dataset}_decision_{idx}.png"
        plt.savefig(decision_file, dpi=300)
        plt.close()
        print(f"-> Gráfico de decisão salvo como '{decision_file}'")
        
        # 9. Métricas e explicação textual
        print("\n" + "="*60)
        print(f"ANÁLISE DA INSTÂNCIA {idx}")
        print(f"Classe Real: {class_names[true_class]}")
        
        # Probabilidades
        proba = model.predict_proba(instance.values.reshape(1, -1))[0]
        print(f"\nProbabilidades: {class_names[0]}: {proba[0]:.2%} | {class_names[1]}: {proba[1]:.2%}")
        
        # Top features
        print("\nTOP 3 FEATURES INFLUENCIADORAS:")
        sorted_idx = np.argsort(-np.abs(shap_values[idx]))[:3]  # Top 3 features
        for i in sorted_idx:
            feat = feature_names[i]
            val = instance.iloc[i]
            sh = shap_values[idx][i]
            direction = "a favor" if (sh > 0 and true_class == 0) or (sh < 0 and true_class == 1) else "contra"
            print(f"- {feat}: {val:.2f} (Contribuição: {sh:.4f} | {direction} da classe)")
        
        print("="*60)

if __name__ == "__main__":
    try:
        import shap
        main()
    except ImportError:
        print("\nERRO: Pacote SHAP não instalado. Execute no terminal:")
        print("pip install shap")
        sys.exit(1)