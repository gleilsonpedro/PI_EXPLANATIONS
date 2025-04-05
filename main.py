from data.load_datasets import selecionar_dataset_e_classe  
from models.train_model import treinar_modelo
from explanations.pi_explanation import analisar_instancias, contar_features_relevantes
import time
import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# suprimir warnings dos plots
import warnings
warnings.filterwarnings("ignore")  # Suprime todos os warnings

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

    # Treinar modelo com garantia de nomes de features
    inicio_treinamento = time.time()
    
    # Converter X para DataFrame com nomes consistentes
    feature_names = getattr(X, 'columns', [f"feature_{i}" for i in range(X.shape[1])])
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Treinar modelo
    modelo, X_test, y_test = treinar_modelo(X_df, y, classe_0=0)
    
    # Converter X_test para DataFrame mantendo nomes e estrutura
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    # Garantir que os tipos de dados são consistentes
    for col in X_test_df.columns:
        X_test_df[col] = X_test_df[col].astype(X_df[col].dtype)
    fim_treinamento = time.time()
    tempo_treinamento = fim_treinamento - inicio_treinamento

    print("Coeficientes do modelo:")
    for name, coef in zip(X.columns, modelo.coef_[0]):
        print(f"{name}: {coef:.4f}")

    # Calcular PI-explicações
    inicio_pi = time.time()
    explicacoes = analisar_instancias(X_test_df, y_test, class_names, modelo, X_df)
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
    contagem = contar_features_relevantes(explicacoes, class_names)
    
    # Plotar visualizações
    plt.figure(figsize=(12, 5))
    
    # 1. Gráfico de importância de features
    plt.subplot(1, 3, 1)
    features = list(contagem.keys())
    counts = list(contagem.values())
    sns.barplot(x=counts, y=features, hue=features, palette='viridis', legend=False, dodge=False)
    plt.title('Frequência de Features nas Explicações')
    plt.xlabel('Número de Ocorrências')
    plt.ylabel('Features')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # 2. Boxplot das features por classe
    plt.subplot(1, 3, 2)
    df = X.copy()
    y_series = pd.Series(y)
    df['Classe'] = y_series.map({0: classe_0_nome, 1: 'Outras'})
    df_melted = df.melt(id_vars='Classe', var_name='Feature', value_name='Valor')
    sns.boxplot(data=df_melted, x='Feature', y='Valor', hue='Classe', 
               palette={classe_0_nome: 'green', 'Outras': 'orange'})
    plt.title(f'Distribuição por Classe\n(Classe 0: {classe_0_nome})')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 3. Histograma de Gamma_A
    plt.subplot(1, 3, 3)
    gamma_values = [modelo.decision_function([X_test.iloc[i]])[0] for i in range(len(X_test))]
    sns.histplot(gamma_values, bins=20, kde=True, color='purple')
    plt.axvline(0, color='red', linestyle='--', label='Limiar de Decisão')
    plt.title('Distribuição dos Valores de Decisão')
    plt.xlabel('Gamma_A (Valor de Decisão)')
    plt.ylabel('Número de Instâncias')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()