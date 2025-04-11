import numpy as np
import pandas as pd

def calcular_gamma_omega(modelo, X, y, classe_predita):
    """
    Calcula gamma_omega para a classe predita conforme o artigo NeurIPS20.
    """
    coef = modelo.coef_[0]
    intercept = modelo.intercept_[0]
    
    gamma = intercept
    for i, col in enumerate(X.columns):
        w_i = coef[i]
        min_val = X[col].min()
        max_val = X[col].max()
        
        # Para classe 1, queremos o pior caso que faria a predição mudar para 0
        if classe_predita == 1:
            contrib = w_i * (min_val if w_i > 0 else max_val)
        # Para classe 0, queremos o pior caso que faria a predição mudar para 1
        else:
            contrib = w_i * (max_val if w_i > 0 else min_val)
            
        gamma += contrib
    
    return gamma

def calcular_deltas(modelo, X, Vs, classe_predita):
    """
    Calcula os deltas conforme definido no artigo NeurIPS20.
    Delta é a diferença entre o valor atual e o pior caso possível.
    """
    coef = modelo.coef_[0]
    delta = []
    
    for i, feature in enumerate(X.columns):
        w_i = coef[i]
        x_i = Vs[feature]
        min_val = X[feature].min()
        max_val = X[feature].max()
        
        # Valor atual
        valor_atual = w_i * x_i
        
        # Pior caso possível
        if classe_predita == 1:
            pior_caso = w_i * (min_val if w_i > 0 else max_val)
        else:
            pior_caso = w_i * (max_val if w_i > 0 else min_val)
            
        # Delta é a diferença entre valor atual e pior caso
        delta_val = valor_atual - pior_caso
        delta.append((feature, delta_val))
    
    # Ordenar por magnitude descendente
    delta.sort(key=lambda x: -abs(x[1]))
    return delta

def one_explanation(Vs, delta, R, feature_names, classe_0_nome, classe_1_nome, classe_predita):
    """
    Gera uma PI-explicação mínima conforme o algoritmo do artigo NeurIPS20.
    """
    Xpl = []
    i = 0
    R_restante = R
    
    # Para classe 1, queremos R > 0
    # Para classe 0, queremos R <= 0
    while (classe_predita == 1 and R_restante > 0) or (classe_predita == 0 and R_restante <= 0):
        if i >= len(delta):
            break
            
        feature, delta_val = delta[i]
        R_restante -= delta_val
        Xpl.append((feature, Vs[feature], delta_val))
        i += 1
    
    if not Xpl:
        return f"PI-Explicação - {classe_1_nome if classe_predita == 1 else classe_0_nome}: Nenhuma feature significativa identificada"
    
    explic_str = ", ".join(
        f"{f} = {v:.3f} (Δ={d:.3f})" for f, v, d in Xpl
    )
    return f"PI-Explicação - {classe_1_nome if classe_predita == 1 else classe_0_nome}: {explic_str}"

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
        delta = calcular_deltas(modelo, X, Vs_dict, classe_predita)

        explicacao = one_explanation(Vs_dict, delta, R, feature_names, classe_0_nome, classe_1_nome, classe_predita)

        print("\n[DEBUG FINAL ANTES DA EXPLICAÇÃO]")
        print(f"Classe predita: {classe_predita}")
        print(f"Gamma_A: {gamma_A:.4f}")
        print(f"Gamma_omega: {gamma_omega:.4f}")
        print(f"R (gamma_A - gamma_omega): {R:.4f}")
        print("Delta (valores ordenados):")
        print(sorted([(feature, round(delta_val, 4)) for feature, delta_val in delta], key=lambda x: -abs(x[1])))
        

        explicacoes.append(explicacao)

        # Prints para debug e validação
        print(f"\nInstância {idx} (Classe real: {y_test[idx]}):")
        print(f"  Gamma_A: {gamma_A:.4f}, Gamma_omega: {gamma_omega:.4f}, R: {R:.4f}")
        print(f"  {explicacao}")

        if classe_predita == 1:
            print(f"[Predição: Classe 1] Instância {idx} | γ_A: {gamma_A:.4f} | γ_ω: {gamma_omega:.4f} | R: {R:.4f}")
            print(f"[DEBUG EXPLICAÇÃO] Classe predita: {classe_predita}")
            print(f"[DEBUG EXPLICAÇÃO] R inicial = {R:.4f}")
            print(f"[DEBUG EXPLICAÇÃO] Início do laço de explicação...")


        print(f"[Debug] Verificando gamma_omega para classe {classe_predita}:")

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