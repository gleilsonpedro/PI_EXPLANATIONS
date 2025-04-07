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

def plot_analysis(X, y, contagem, nome_dataset, classe_0_nome, gamma_values, y_test):  # Adicionei y_test nos parâmetros
    """Plot composto aprimorado"""
    plt.figure(figsize=(18, 6))
    
    # 1. Gráfico de importância de features - ordenado e com valores
    plt.subplot(1, 3, 1)
    features = sorted(contagem.keys(), key=lambda x: contagem[x], reverse=True)
    counts = [contagem[f] for f in features]
    
    bars = plt.barh(features, counts, color=sns.color_palette("husl", len(features)))
    plt.bar_label(bars, padding=3, labels=[f'{v}' for v in counts], fontsize=10)
    
    plt.title(f'Features Mais Relevantes para\n{classe_0_nome}', fontsize=12)
    plt.xlabel('Número de Ocorrências nas Explicações', fontsize=10)
    plt.ylabel('Features', fontsize=10)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.xlim(0, max(counts)*1.1)
    
    # 2. Boxplot das features por classe
    plt.subplot(1, 3, 2)
    df = X.copy()
    y_series = pd.Series(y)
    df['Classe'] = y_series.map({0: classe_0_nome, 1: 'Outras Classes'})
    
    # Selecionar apenas as features mais relevantes
    top_features = features[:3]  # Mostrar apenas as 3 mais importantes
    df_melted = df.melt(id_vars='Classe', value_vars=top_features, 
                       var_name='Feature', value_name='Valor')
    
    sns.boxplot(data=df_melted, x='Feature', y='Valor', hue='Classe',
               palette={classe_0_nome: 'green', 'Outras Classes': 'orange'})
    
    plt.title(f'Distribuição das Top 3 Features\npor Classe', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.xlabel('')
    plt.ylabel('Valor Normalizado', fontsize=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 3. Histograma de Gamma_A com distribuição por classe
    plt.subplot(1, 3, 3)
    df_test = pd.DataFrame({'gamma_A': gamma_values, 'Classe': y_test})  # Agora y_test está definido
    
    sns.histplot(data=df_test, x='gamma_A', hue='Classe', 
                bins=20, kde=True, 
                palette={0: 'green', 1: 'orange'},
                hue_order=[0, 1],
                element='step', stat='density', common_norm=False)
    
    plt.axvline(0, color='red', linestyle='--', linewidth=1.5, 
               label='Fronteira de Decisão')
    plt.title('Distribuição de γ_A por Classe Real', fontsize=12)
    plt.xlabel('Valor da Função de Decisão (γ_A)', fontsize=10)
    plt.ylabel('Densidade', fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.suptitle(f'Análise Explicativa para {nome_dataset.capitalize()}\nClasse Alvo: {classe_0_nome}', 
                fontsize=14, y=1.05)
    plt.tight_layout()
    
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