import numpy as np
import pandas as pd

def calcular_gamma_omega(modelo, X, y, classe_verdadeira):
    """
    Calcula gamma_omega (pior cenário) baseado apenas nas amostras da classe verdadeira.
    """
    w = modelo.coef_[0]
    b = modelo.intercept_[0]
    gamma_omega = b

    X_classe = X[y == classe_verdadeira]  # ← Filtra só instâncias da classe-alvo

    for i, w_i in enumerate(w):
        if classe_verdadeira == 0:
            if w_i > 0:
                gamma_omega += w_i * X_classe.iloc[:, i].min()
            else:
                gamma_omega += w_i * X_classe.iloc[:, i].max()
        else:
            if w_i > 0:
                gamma_omega += w_i * X_classe.iloc[:, i].max()
            else:
                gamma_omega += w_i * X_classe.iloc[:, i].min()
    
    return gamma_omega


def calcular_deltas(modelo, X, Vs, w, classe_verdadeira):
    """
    Calcula os deltas conforme definido no artigo
    
    Args:
        Vs: Valores da instância (dict {feature: valor})
        X: DataFrame completo (para calcular min/max)
        w: Pesos do modelo (modelo.coef_[0])
        classe_verdadeira: 0 (classe alvo) ou 1 (outra classe)
    
    Returns:
        Lista de deltas ordenados por magnitude absoluta
    """
    
    assert isinstance(Vs, dict), f"Esperado dict, mas recebi {type(Vs)}"
    ...

    deltas = []
    for i, feature in enumerate(Vs.keys()):
        if classe_verdadeira == 0:  # Classe alvo (minimizar pontuação)
            if w[i] > 0:
                delta = (Vs[feature] - X[feature].min()) * w[i]
            else:
                delta = (Vs[feature] - X[feature].max()) * w[i]
        else:  # Classe não-alvo (maximizar pontuação)
            if w[i] > 0:
                delta = (Vs[feature] - X[feature].max()) * w[i]
            else:
                delta = (Vs[feature] - X[feature].min()) * w[i]
        deltas.append(delta)
    return deltas

def one_explanation(Vs, delta, R, feature_names, classe_0_nome, classe_1_nome, classe_verdadeira):
    """
    Gera uma PI-explicação baseada no Algoritmo 1 do artigo NEURIPS20.
    
    Vs: dicionário de valores da instância
    delta: lista de deltas (δ_j)
    R: valor do limiar Φ = gamma_A - gamma_omega
    feature_names: lista dos nomes das features
    classe_0_nome: nome da classe 0
    classe_1_nome: nome da classe 1
    classe_verdadeira: 0 ou 1 (classe prevista para a instância)
    
    Retorna: string formatada com a explicação
    """
    delta_sorted = sorted(enumerate(delta), key=lambda x: -abs(x[1]))  # ordenar por |δ_j| desc
    Xpl = []
    Idx = 0
    PhiR = R

    while PhiR > 0 and Idx < len(delta_sorted):
        idx_feature, delta_val = delta_sorted[Idx]
        feature_name = feature_names[idx_feature]
        valor = Vs[feature_name]
        Xpl.append(f"{feature_name} = {valor:.3f} (Δ={delta_val:.3f})")
        PhiR -= delta_val
        Idx += 1

    if not Xpl or PhiR > 0:
        Xpl = ["Nenhuma feature significativa identificada"]

    if classe_verdadeira == 0:
        return f"PI-Explicação - {classe_0_nome}: " + ", ".join(Xpl)
    else:
        return f"PI-Explicação - {classe_1_nome}: " + ", ".join(Xpl)


def analisar_instancias(X_test, y_test, classe_0_nome, classe_1_nome, modelo, X, y):
    """
    Analisa todas as instâncias e gera PI-explicações conforme o artigo NeurIPS20.
    """
    if not isinstance(X_test, pd.DataFrame):
        feature_names = X.columns if hasattr(X, 'columns') else [f"feature_{i}" for i in range(X.shape[1])]
        X_test = pd.DataFrame(X_test, columns=feature_names)
    
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=X_test.columns)
    
    feature_names = X_test.columns.tolist()
    explicacoes = []

    for idx in range(len(X_test)):
        instancia_df = X_test.iloc[idx]
        instancia = instancia_df.values
        Vs_dict = instancia_df.to_dict()

        gamma_A = modelo.decision_function([instancia])[0]
        classe_predita = int(modelo.predict([instancia])[0])
        gamma_omega = calcular_gamma_omega(modelo, X, y, classe_predita)
        R = gamma_A - gamma_omega

        # Aqui está a correção!
delta = calcular_deltas(modelo, X, Vs_dict, classe_predita, classe_predita)

        explicacao = one_explanation(Vs_dict, delta, R, feature_names, classe_0_nome, classe_1_nome, classe_predita)

        explicacoes.append(explicacao)

        # Prints para debug e validação
        print(f"\nInstância {idx} (Classe real: {y_test[idx]}):")
        print(f"  Gamma_A: {gamma_A:.4f}, Gamma_omega: {gamma_omega:.4f}, R: {R:.4f}")
        print(f"  {explicacao}")

        if classe_predita == 1:
            print(f"[Predição: Classe 1] Instância {idx} | γ_A: {gamma_A:.4f} | γ_ω: {gamma_omega:.4f} | R: {R:.4f}")

        print("\n[Debug] Verificando gamma_omega para classe 1:")
        for i, col in enumerate(X.columns):
            w_i = modelo.coef_[0][i]
            min_val = X[col].min()
            max_val = X[col].max()
            contrib = w_i * (max_val if w_i > 0 else min_val)
            print(f"{col}: w = {w_i:.4f}, contribuição = {contrib:.4f}")

    return explicacoes


def calcular_estatisticas_explicacoes(explicacoes):
    """
    Calcula estatísticas sobre o tamanho das explicações
    
    Args:
        explicacoes: Lista de todas as explicações geradas
    
    Returns:
        Dicionário com média e desvio padrão do número de features por explicação
    """
    tamanhos = []
    
    for exp in explicacoes:
        # Conta quantas features tem na explicação
        if "Nenhuma feature" in exp:
            tamanhos.append(0)
        else:
            # Divide a explicação pelos separadores para contar as features
            partes = exp.split(": ")[1] if ": " in exp else ""
            num_features = len(partes.split(", ")) if partes else 0
            tamanhos.append(num_features)
    
    media = np.mean(tamanhos)
    desvio_padrao = np.std(tamanhos)
    
    return {
        'media_tamanho': media,
        'desvio_padrao_tamanho': desvio_padrao,
        'tamanhos': tamanhos
    }

def contar_features_relevantes(explicacoes, classe_0_nome, classe_1_nome):
    """
    Conta features relevantes para ambas as classes
    
    Args:
        explicacoes: Lista de explicações
        classe_0_nome: Nome da classe 0
        classe_1_nome: Nome da classe 1
    
    Returns:
        Dicionário {feature: contagem}
    """
    contagem = {'Classe 0': {}, 'Classe 1': {}}
    
    for exp in explicacoes:
        if classe_0_nome in exp:
            partes = exp.split(": ")
            if len(partes) > 1:
                features = partes[1].split(", ")
                for f in features:
                    nome = f.split(" = ")[0].strip()
                    contagem['Classe 0'][nome] = contagem['Classe 0'].get(nome, 0) + 1
        elif classe_1_nome in exp:
            partes = exp.split(": ")
            if len(partes) > 1:
                features = partes[1].split(", ")
                for f in features:
                    nome = f.split(" = ")[0].strip()
                    contagem['Classe 1'][nome] = contagem['Classe 1'].get(nome, 0) + 1
    
    print(f"\nContagem de Features Relevantes:")
    print(f"\nPara '{classe_0_nome}':")
    for f, cnt in sorted(contagem['Classe 0'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {f}: {cnt} ocorrências")
    
    print(f"\nPara '{classe_1_nome}':")
    for f, cnt in sorted(contagem['Classe 1'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {f}: {cnt} ocorrências")
    
    return contagem