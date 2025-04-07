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

# Configurações de estilo - CORREÇÃO AQUI
sns.set_style("whitegrid")  # Usando estilo do Seaborn diretamente
plt.style.use('ggplot')     # Estilo alternativo válido
warnings.filterwarnings("ignore")

def limpar_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def plot_gamma_distribution(gamma_values, gamma_reject, nome_dataset, classe_0_nome):
    """Plot aprimorado da distribuição de gamma"""
    plt.figure(figsize=(12, 6))
    
    # Histograma com KDE
    ax = sns.histplot(gamma_values, bins=20, kde=True, color='skyblue', 
                     edgecolor='navy', alpha=0.7)
    
    # Região de rejeição
    reject_min, reject_max = -gamma_reject, gamma_reject
    plt.axvspan(reject_min, reject_max, color='salmon', alpha=0.3, 
               label='Região de Rejeição')
    
    # Linhas de referência
    plt.axvline(reject_min, color='red', linestyle='--', linewidth=1.5, 
               label=f'Limiar inferior (-{gamma_reject})')
    plt.axvline(reject_max, color='red', linestyle='--', linewidth=1.5,
               label=f'Limiar superior ({gamma_reject})')
    plt.axvline(0, color='green', linestyle='-', linewidth=2, 
               label='Fronteira de Decisão')
    
    # Estatísticas
    total = len(gamma_values)
    rejeitadas = sum(1 for g in gamma_values if abs(g) < gamma_reject)
    
    # Configurações do gráfico
    plt.title(f'Distribuição dos Valores de Decisão (γ_A)\nDataset: {nome_dataset} | Classe 0: {classe_0_nome}', 
             pad=20, fontsize=14)
    plt.xlabel('Valor de γ_A (Função de Decisão)', fontsize=12)
    plt.ylabel('Número de Instâncias', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Legenda com estatísticas
    stats_text = f"""Total: {total} instâncias
Classificadas: {total-rejeitadas} ({(total-rejeitadas)/total:.1%})
Rejeitadas: {rejeitadas} ({rejeitadas/total:.1%})"""
    plt.annotate(stats_text, xy=(0.98, 0.95), xycoords='axes fraction', 
                 ha='right', va='top', fontsize=10,
                 bbox=dict(boxstyle='round', alpha=0.2))
    
    # Anotação explicativa
    plt.annotate('γ_A = w·x + b\n(valor da função de decisão)', 
                xy=(0.02, 0.95), xycoords='axes fraction',
                ha='left', va='top', fontsize=10)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    return plt

def plot_analysis(X, y, contagem, nome_dataset, classe_0_nome, gamma_values, y_test):
    plt.figure(figsize=(16, 8))  # Tamanho ajustado para 2 plots
    sns.set(font_scale=1.0)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # Layout otimizado para 2 plots lado a lado
    gs = plt.GridSpec(1, 2, width_ratios=[1, 1.5], wspace=0.4)
    
    # 1. Gráfico de features (esquerda)
    ax1 = plt.subplot(gs[0, 0])
    features = [f.replace("sepel", "sepal").replace("pedal", "petal") 
               for f in sorted(contagem.keys(), key=lambda x: contagem[x], reverse=True)]
    counts = [contagem[f] for f in contagem.keys()]
    
    bars = ax1.barh(features[:4], counts[:4], color=sns.color_palette("husl", 4))
    ax1.bar_label(bars, fmt='%d', padding=3, fontsize=10)
    ax1.set_title('Features Mais Relevantes', fontsize=12, pad=10, fontweight='bold')
    ax1.set_xlabel('Número de Ocorrências', fontsize=10)
    ax1.set_ylabel('')
    ax1.grid(axis='x', alpha=0.2)
    
    # 2. Boxplot (direita)
    ax2 = plt.subplot(gs[0, 1])
    df = X.rename(columns=lambda x: x.replace("sepel", "sepal").replace("pedal", "petal")).copy()
    df['Classe'] = pd.Series(y).map({0: classe_0_nome, 1: 'Outras'})
    
    sns.boxplot(data=df.melt(id_vars='Classe', value_vars=features[:4]), 
               x='variable', y='value', hue='Classe',
               palette={classe_0_nome: '#3498db', 'Outras': '#e74c3c'},
               ax=ax2, linewidth=1.2, width=0.7)
    
    ax2.set_title('Distribuição por Classe', fontsize=12, pad=10, fontweight='bold')
    ax2.set_xlabel('')
    ax2.set_ylabel('Valor (cm)', fontsize=10)
    ax2.legend(title='Classe', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    ax2.tick_params(axis='x', rotation=30, labelsize=10)
    ax2.grid(axis='y', alpha=0.2)
    
    plt.suptitle(f'Análise Explicativa: {nome_dataset.capitalize()} - Classe {classe_0_nome}', 
                y=0.98, fontsize=14, fontweight='bold')
    plt.tight_layout(pad=2.0)
    
    return plt

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

    # Coletar todos os valores gamma_A para visualização
    gamma_values = [modelo.decision_function(X_test_df.iloc[[i]])[0] for i in range(len(X_test_df))]

    # Plotar distribuição de gamma
    gamma_plot = plot_gamma_distribution(gamma_values, gamma_reject, nome_dataset, classe_0_nome)
    gamma_plot.savefig(f'gamma_distribution_{nome_dataset}.png', dpi=300, bbox_inches='tight')
    gamma_plot.show()

    # Plotar análise composta
    analysis_plot = plot_analysis(X_df, y, contagem, nome_dataset, classe_0_nome, gamma_values, y_test)
    analysis_plot.savefig(f'analysis_{nome_dataset}.png', dpi=300, bbox_inches='tight')
    analysis_plot.show()

    # Tempo total
    tempo_total = tempo_treinamento + tempo_pi

    print(f"\n**Tempo de treinamento do modelo:**      {tempo_treinamento:.4f} segundos")
    print(f"**Tempo de cálculo das PI-explicações:** {tempo_pi:.4f} segundos")
    print(f"**Tempo total de execução:**             {tempo_total:.4f} segundos")

if __name__ == "__main__":
    main()