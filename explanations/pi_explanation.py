import numpy as np
import pandas as pd

def one_explanation(Vs, delta, R, feature_names, modelo, instancia_test, X):
    """
    Calcula uma PI-explicação para uma instância específica.
    * HIPERPARAMETROS:
        aumentando ou dmnuindo o percentil e multiplicando o valor do delta por uma constante 
            se > 1 a sensibilidade aumenta ( mais features serão incluidas)
            se 1 > a sensibilidade diminui ( menos features serão incluidas)
    """
   # limiar_delta = np.percentile(np.abs(delta), 10)  # Pega o percentil 25 dos deltas
    Xpl = []
    delta_sorted = sorted(enumerate(delta), key=lambda x: abs(x[1]), reverse=True)
    R_atual = R
    Idx = 0
    
    while R_atual >= 0 and Idx < len(delta_sorted):
        sorted_idx, delta_value = delta_sorted[Idx]
        feature = feature_names[sorted_idx]
        feature_value = Vs[feature]
##### verificar se esta ok
       # if abs(delta_value) < limiar_delta:  # Descarta deltas muito pequenos
       #    break

        Xpl.append(f"{feature} - {feature_value}")
       # R_atual -= delta_value * 0.5 # Diminuir a sensibilidade
        R_atual -= delta_value #* 1.5 # Aumentar a sensibilidade
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
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])

    feature_names = X_test.columns.tolist()
    TUDO = []
    
    for idx in range(len(X_test)):
        Vs = X_test.iloc[idx].to_dict()
        instancia_test = X_test.iloc[[idx]]
        gamma_A = modelo.decision_function(instancia_test)[0]
        classe_verdadeira = y_test[idx]

        # Cálculo ajustado dos deltas
        delta = []
        w = modelo.coef_[0]
        
        for i, feature in enumerate(feature_names):
            if classe_verdadeira == 0:  # Classe positiva (virginica)
                if w[i] < 0:
                    delta_value = (Vs[feature] - X[feature].min()) * abs(w[i])  # Queremos aumentar o valor
                else:
                    delta_value = (Vs[feature] - X[feature].max()) * abs(w[i])  # Queremos diminuir o valor
            else:  # Classe negativa (não virginica)
                if w[i] < 0:
                    delta_value = (Vs[feature] - X[feature].max()) * abs(w[i])  # Inverso do caso positivo
                else:
                    delta_value = (Vs[feature] - X[feature].min()) * abs(w[i])
            delta.append(delta_value)

        # Cálculo de R ajustado
        R = abs(sum(delta) - gamma_A)  # Usamos valor absoluto
        
        # Limiar adaptativo (20% do percentil)
        #limiar_delta = np.percentile(np.abs(delta), 20)
        
        Xpl = []
        delta_sorted = sorted(enumerate(delta), key=lambda x: abs(x[1]), reverse=True)
        
        for i, (feature_idx, delta_val) in enumerate(delta_sorted):
            #if abs(delta_val) < limiar_delta:
            #    break
            feature = feature_names[feature_idx]
            Xpl.append(f"{feature} - {Vs[feature]}")
            R -= abs(delta_val)
            if R <= 0:
                break

        TUDO.append(Xpl)
        
        # Exibição dos resultados
        print(f"\nInstância {idx}:")
        print(f"Classe verdadeira (binária): {classe_verdadeira}")
        print("PI-Explicação:" + (" " + ", ".join(Xpl) if Xpl else " _No-PI-explanation_"))

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