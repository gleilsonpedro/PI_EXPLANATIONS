import matplotlib.pyplot as plt
import numpy as np
from data.load_datasets import selecionar_dataset_e_classe
from models.train_model import treinar_modelo
from explanations.pi_explanation import analisar_instancias, calcular_estatisticas_explicacoes
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def experimento_c_regularizacao(valores_C, nome_dataset=None):
    resultados = []

    # Seleção manual (ou forçada) do dataset
    if nome_dataset is None:
        nome_dataset, classe_0_nome, X, y, class_names = selecionar_dataset_e_classe()
    else:
        from data.load_datasets import carregar_dataset
        X, y, class_names = carregar_dataset(nome_dataset)
        classe_0_nome = class_names[0]
        y = np.where(y == 0, 0, 1)  # binariza se necessário

    for C in valores_C:
        print(f"\n🔧 Treinando com C = {C} ...")
        modelo, X_test, y_test = treinar_modelo(pd.DataFrame(X), y, regularizacao=C)

        X_test_df = pd.DataFrame(X_test, columns=X.columns if hasattr(X, 'columns') else [f"f{i}" for i in range(X.shape[1])])
        explicacoes, _ = analisar_instancias(X_test_df, y_test, class_names[0], class_names[1], modelo, pd.DataFrame(X))

        stats = calcular_estatisticas_explicacoes(explicacoes)
        print(f"➡️ Média de features: {stats['media_tamanho']:.2f} | Desvio: {stats['desvio_padrao_tamanho']:.2f}")

        resultados.append((C, stats['media_tamanho'], stats['desvio_padrao_tamanho']))

    return resultados


def plotar_resultados(resultados, nome_dataset):
    Cs, medias, desvios = zip(*resultados)
    plt.figure(figsize=(8,5))
    plt.errorbar(Cs, medias, yerr=desvios, fmt='-o', capsize=5)
    plt.xscale('log')
    plt.xlabel('Valor de C (log)')
    plt.ylabel('Média de features por PI-explicação')
    plt.title(f'Impacto da regularização nas PI-explicações ({nome_dataset})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    valores_C = [0.01, 0.1, 1.0, 10.0, 100.0]

    # Você pode trocar por: nome_dataset = 'pima_indians_diabetes'
    resultados = experimento_c_regularizacao(valores_C)

    if resultados:
        nome_dataset = "Dataset customizado"
        plotar_resultados(resultados, nome_dataset)
