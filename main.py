from data.load_datasets import selecionar_dataset_e_classe  
from models.train_model import treinar_modelo
from explanations.pi_explanation import analisar_instancias, contar_features_relevantes
import time
import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Suprimir warnings dos plots
warnings.filterwarnings("ignore")

def limpar_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

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
    
    # Converter X para DataFrame com nomes consistentes
    feature_names = getattr(X, 'columns', [f"feature_{i}" for i in range(X.shape[1])])
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Treinar modelo
    modelo, X_test, y_test = treinar_modelo(X_df, y, classe_0=0)
    
    # Converter X_test para DataFrame
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    fim_treinamento = time.time()
    tempo_treinamento = fim_treinamento - inicio_treinamento

    print("\nCoeficientes do modelo:")
    for name, coef in zip(feature_names, modelo.coef_[0]):
        print(f"{name}: {coef:.4f}")

    # Calcular PI-explicações com limiar de rejeição
    gamma_reject = 0.5  # Pode ser ajustado conforme necessidade
    inicio_pi = time.time()
    explicacoes = analisar_instancias(X_test_df, y_test, classe_0_nome, class_names, modelo, X_df, gamma_reject)
    fim_pi = time.time()
    tempo_pi = fim_pi - inicio_pi

    # Contar features relevantes
    print("\n**Contagem de features relevantes:**")
    contagem = contar_features_relevantes(explicacoes, classe_0_nome)

    # ========== NOVO GRÁFICO DE DISTRIBUIÇÃO ==========
    plt.figure(figsize=(14, 6))
    
    # Coletar todos os valores gamma_A
    gammas = [modelo.decision_function(X_test_df.iloc[[i]])[0] for i in range(len(X_test_df))]
    
    # Histograma
    n, bins, patches = plt.hist(gammas, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    
    # Destacar região de rejeição
    reject_mask = (bins > -gamma_reject) & (bins < gamma_reject)
    for patch, is_reject in zip(patches, reject_mask[:-1]):
        if is_reject:
            patch.set_facecolor('salmon')
    
    # Linhas de referência
    plt.axvline(-gamma_reject, color='red', linestyle='--', label=f'Limiar inferior (-{gamma_reject})')
    plt.axvline(gamma_reject, color='red', linestyle='--', label=f'Limiar superior ({gamma_reject})')
    plt.axvline(0, color='green', linestyle='-', label='Fronteira de decisão')
    
    # Estatísticas
    total = len(gammas)
    rejeitadas = sum(1 for g in gammas if abs(g) < gamma_reject)
    
    plt.title(f'Distribuição dos Valores de Decisão (γ_A)\nDataset: {nome_dataset} | Classe 0: {classe_0_nome}', pad=15)
    plt.xlabel('Valor de γ_A (Função de Decisão)')
    plt.ylabel('Número de Instâncias')
    plt.grid(axis='y', alpha=0.3)
    
    # Legenda com estatísticas
    stats_text = f"""Total: {total} instâncias
Classificadas: {total-rejeitadas} ({(total-rejeitadas)/total:.1%})
Rejeitadas: {rejeitadas} ({rejeitadas/total:.1%})"""
    plt.annotate(stats_text, xy=(0.98, 0.95), xycoords='axes fraction', 
                 ha='right', va='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    plt.legend()
    plt.tight_layout()
    
    # Salvar automaticamente
    plt.savefig(f'gamma_distribution_{nome_dataset}.png', dpi=300, bbox_inches='tight')
    plt.show()
    # ========== FIM DO NOVO GRÁFICO ==========

    # Plotar visualizações originais
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
    
    # 3. Histograma de Gamma_A (versão simplificada)
    plt.subplot(1, 3, 3)
    gamma_values = [modelo.decision_function([X_test_df.iloc[i]])[0] for i in range(len(X_test_df))]
    sns.histplot(gamma_values, bins=20, kde=True, color='purple')
    plt.axvline(0, color='red', linestyle='--', label='Limiar de Decisão')
    plt.title('Distribuição dos Valores de Decisão')
    plt.xlabel('Gamma_A (Valor de Decisão)')
    plt.ylabel('Número de Instâncias')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'analysis_{nome_dataset}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Tempo total
    tempo_total = tempo_treinamento + tempo_pi

    print(f"\n**Tempo de treinamento do modelo:**      {tempo_treinamento:.4f} segundos")
    print(f"**Tempo de cálculo das PI-explicações:** {tempo_pi:.4f} segundos")
    print(f"**Tempo total de execução:**             {tempo_total:.4f} segundos")

if __name__ == "__main__":
    main()