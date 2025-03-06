import numpy as np
import json
import os
from explanations.pi_explanation import analisar_instancias, encontrar_intervalo_perturbacao
from models.train_model import treinar_modelo
from data.load_datasets import carregar_dataset

def salvar_resultados(resultados, dataset_nome, classe_0):
    """
    Salva os resultados em um arquivo JSON, evitando duplicações.
    """
    arquivo_json = f"resultados_gridsearch.json"
    
    # Verifica se o arquivo já existe
    if os.path.exists(arquivo_json):
        with open(arquivo_json, "r") as f:
            dados_existentes = json.load(f)
    else:
        dados_existentes = []
    
    # Verifica se os novos resultados já existem
    for resultado in resultados:
        # Cria uma chave única para cada resultado (dataset + classe_0 + parâmetros)
        chave_resultado = (
            resultado["dataset"],
            resultado["classe_0"],
            resultado["percentil"],
            resultado["delta_value"]
        )
        
        # Verifica se essa chave já existe nos dados existentes
        existe = False
        for dado_existente in dados_existentes:
            chave_existente = (
                dado_existente["dataset"],
                dado_existente["classe_0"],
                dado_existente["percentil"],
                dado_existente["delta_value"]
            )
            if chave_resultado == chave_existente:
                existe = True
                break
        
        # Se não existir, adiciona o novo resultado
        if not existe:
            dados_existentes.append(resultado)
    
    # Salva os dados atualizados no arquivo JSON
    with open(arquivo_json, "w") as f:
        json.dump(dados_existentes, f, indent=4)
    
    print(f"Resultados salvos em '{arquivo_json}'.")

def busca_melhores_parametros(dataset_nome, classe_0_idx, percentis=[10, 25, 50, 75], valores_delta=[0.5, 1.0, 1.5]):
    """
    Realiza um grid search para encontrar os melhores parâmetros para o limiar_delta e delta_value.
    Inclui perturbações positivas e negativas para melhorar a análise.
    """
    # Carrega o dataset
    X, y, class_names = carregar_dataset(dataset_nome)
    
    if classe_0_idx >= len(class_names):
        raise ValueError("Índice da classe 0 inválido.")
    
    # Ajusta o y para o problema binário
    y_binario = np.where(y == classe_0_idx, 0, 1)
    
    # Treina o modelo
    modelo, X_test, y_test = treinar_modelo(X, y_binario, classe_0=0)
    
    resultados = []
    
    # Loop sobre os percentis e valores de delta
    for p in percentis:
        for delta_value in valores_delta:
            # Converte X_test para um array NumPy e calcula o delta
            delta = np.abs(X_test.to_numpy()).flatten()  # Corrigido: usa to_numpy() e flatten()
            limiar_delta = np.percentile(delta, p)
            
            # Analisa as instâncias com os parâmetros atuais
            TUDO = analisar_instancias(X_test, y_test, class_names, modelo, X)
            
            # Calcula uma métrica de qualidade das explicações (exemplo: tamanho médio das explicações)
            metrica = np.mean([len(exp) for exp in TUDO])
            
            # Inclui perturbações positivas e negativas
            perturbacoes = []
            for idx in range(len(X_test)):
                instancia = X_test.iloc[[idx]]  # Corrigido: usa iloc[[idx]] para manter o DataFrame 2D
                for feature in X_test.columns:
                    valor_original = instancia[feature].values[0]
                    min_val, max_val = encontrar_intervalo_perturbacao(modelo, instancia, feature, valor_original, y_test[idx], X)
                    perturbacoes.append({
                        "instancia": idx,
                        "feature": feature,
                        "min_val": min_val,
                        "max_val": max_val
                    })
            
            # Salva os resultados
            resultados.append({
                "dataset": dataset_nome,
                "classe_0": class_names[classe_0_idx],
                "percentil": p,
                "limiar_delta": limiar_delta,
                "delta_value": delta_value,
                "metrica": metrica,
                "perturbacoes": perturbacoes
            })
    
    # Salva os resultados no arquivo JSON
    salvar_resultados(resultados, dataset_nome, class_names[classe_0_idx])

# Executa o grid search quando o script é chamado
if __name__ == "__main__":
    dataset_nome = "iris"  # Exemplo de dataset
    classe_0_idx = 0  # Índice da classe 0
    busca_melhores_parametros(dataset_nome, classe_0_idx)