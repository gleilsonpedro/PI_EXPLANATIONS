# explanations/pi_explanation.py

import numpy as np
import pandas as pd

def one_explanation(Vs, delta, R, feature_names, modelo, instancia_test, X):
    """
    Calcula uma PI-explicação para uma instância específica.
    """
    limiar_delta = np.percentile(np.abs(delta), 25)  # Pega o percentil 25 dos deltas
    Xpl = []
    delta_sorted = sorted(enumerate(delta), key=lambda x: abs(x[1]), reverse=True)
    R_atual = R
    Idx = 0
    
    while R_atual >= 0 and Idx < len(delta_sorted):
        sorted_idx, delta_value = delta_sorted[Idx]
        feature = feature_names[sorted_idx]
        feature_value = Vs[feature]

        if abs(delta_value) < limiar_delta:  # Descarta deltas muito pequenos
           break

        Xpl.append(f"{feature} - {feature_value}")
        R_atual -= delta_value
        Idx += 1
    
    return Xpl

def encontrar_intervalo_perturbacao(modelo, instancia, feature, valor_original, classe_desejada, X, passo=0.1, max_iter=50):
    """
    Encontra o intervalo de valores para uma feature que mantém a classe desejada.
    """
    min_val_data = X[feature].min()
    max_val_data = X[feature].max()
    min_val, max_val = valor_original, valor_original
    
    # Perturba negativamente
    for _ in range(max_iter):
        min_val -= passo
        if min_val < min_val_data:
            min_val = min_val_data
            break
        instancia_perturbada = instancia.copy()
        instancia_perturbada[feature] = min_val
        predicao = modelo.predict(instancia_perturbada)
        if predicao[0] != classe_desejada:
            min_val += passo
            break

    # Perturba positivamente
    for _ in range(max_iter):
        max_val += passo
        if max_val > max_val_data:
            max_val = max_val_data
            break
        instancia_perturbada = instancia.copy()
        instancia_perturbada[feature] = max_val
        predicao = modelo.predict(instancia_perturbada)
        if predicao[0] != classe_desejada:
            max_val -= passo
            break

    return min_val, max_val

def analisar_instancias(X_test, y_test, class_names, modelo, X, instancia_para_analisar=None):
    """
    Analisa as instâncias do conjunto de teste e calcula as PI-explicações.
    Retorna a lista de todas as PI-explicações (TUDO).
    """
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])

    # Obtém os nomes das features
    feature_names = X_test.columns.tolist()  
    
    # Seleciona as instâncias para análise
    num_instancias = len(X_test)
    instancias_para_analisar = range(num_instancias) if instancia_para_analisar is None else [instancia_para_analisar]
    
    TUDO = []
    # Loop para analisar cada instância selecionada
    for idx in instancias_para_analisar:
        Vs = X_test.iloc[idx].to_dict()
        instancia_test = X_test.iloc[[idx]]

        # Calcula `gamma_A` usando `decision_function`
        gamma_A = modelo.decision_function(instancia_test)[0]
        
        # Cálculo do valor delta para cada feature
        delta = []
        w = modelo.coef_[0]
        for i, feature in enumerate(feature_names):
            if w[i] < 0:
                delta.append((Vs[feature] - X[feature].max()) * w[i])
            else:
                delta.append((Vs[feature] - X[feature].min()) * w[i])

        # Calcula R
        R = sum(delta) - gamma_A
        
        # Computa a PI-explicação para a instância atual usando nomes das features
        Xpl = one_explanation(Vs, delta, R, feature_names, modelo, instancia_test, X)
        
        # Imprime os resultados
        classe_verdadeira = y_test[idx]
        print(f"\nInstância {idx}:")
        print(f"Classe verdadeira (binária): {classe_verdadeira}")
        print(f"PI-Explicação: ")
        
        TUDO.append(Xpl)

        for item in Xpl:
            print(f"- {item}")
        
        if not Xpl:
            print('_No-PI-explanation_' * 3)

    # Retorna a lista de todas as PI-explicações
    return TUDO

def contar_features_relevantes(TUDO):
    """
    Conta quantas vezes cada feature aparece nas PI-explicações.
    """
    contagem_features = {}

    # Itera sobre cada item da lista TUDO
    for item in TUDO:
        # Verifica se o item é uma lista
        if isinstance(item, list):
            # Itera sobre cada item da lista
            for feature in item:
                # Extrai o nome da feature
                nome_feature = feature.split(" - ")[0]

                # Verifica se a feature já está no dicionário
                if nome_feature in contagem_features:
                    # Incrementa a contagem
                    contagem_features[nome_feature] += 1
                else:
                    # Adiciona a feature ao dicionário com contagem 1
                    contagem_features[nome_feature] = 1

    # Imprime a contagem de features
    print("\nContagem de features relevantes:")
    for nome_feature, contagem in contagem_features.items():
        print(f"Feature: {nome_feature} | Contagem: {contagem}")