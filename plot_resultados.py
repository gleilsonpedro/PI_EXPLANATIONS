import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analisar_e_plotar_resultados(json_file):
    """
    Analisa os resultados do grid search e plota os melhores parâmetros.
    """
    # Carrega os resultados do JSON
    with open(json_file, "r") as f:
        resultados = json.load(f)
    
    # Converte para DataFrame
    df = pd.DataFrame(resultados)
    
    # Exibe os primeiros registros para verificação
    print("Primeiros registros do DataFrame:")
    print(df.head())
    
    # Encontra o melhor resultado com base na métrica
    melhor_resultado = df.loc[df['metrica'].idxmin()]
    print("\nMelhor resultado:")
    print(melhor_resultado)
    
    # Plota os resultados
    plt.figure(figsize=(10, 6))
    
    # Gráfico de dispersão: percentil vs. delta_value, colorido pela métrica
    sns.scatterplot(data=df, x='percentil', y='delta_value', hue='metrica', palette='viridis', size='metrica', sizes=(50, 200))
    plt.title("Grid Search: Percentil vs. Delta Value (Colorido pela Métrica)")
    plt.xlabel("Percentil")
    plt.ylabel("Delta Value")
    plt.legend(title="Métrica", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Destaca o melhor resultado no gráfico
    plt.scatter(melhor_resultado['percentil'], melhor_resultado['delta_value'], color='red', s=200, label='Melhor Resultado')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Exemplo de execução
analisar_e_plotar_resultados("resultados_gridsearch.json")