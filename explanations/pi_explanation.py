import numpy as np
import pandas as pd

def calcular_deltas(Vs, X, w, classe_verdadeira):
    """
    Args:
        Vs: Valores da instância (dict {feature: valor})
        X: DataFrame completo (para calcular min/max)
        w: Pesos do modelo (modelo.coef_[0])
        classe_verdadeira: 0 (classe alvo) ou 1 (outra classe)
    
    Returns:
        Lista de deltas ordenados por magnitude absoluta
    """
    deltas = []
    for i, feature in enumerate(Vs.keys()):
        if classe_verdadeira == 0:  # Classe alvo
            extremo = X[feature].max() if w[i] < 0 else X[feature].min()
            delta = (Vs[feature] - extremo) * w[i]
        else:  # Classe não-alvo 
            extremo = X[feature].max() if w[i] < 0 else X[feature].min()
            delta = (Vs[feature] - extremo) * w[i]
        deltas.append(delta)
    return deltas

def one_explanation(Vs, delta, R, feature_names, classe_0_nome, classe_1_nome, classe_verdadeira):
    """
    Gera uma PI-explicação
    
    Args:
        Vs: Valores da instância
        delta: Lista de deltas calculados
        R: Valor residual (Σδ - γ_A)
        feature_names: Nomes das features
        classe_0_nome: Nome da classe 0
        classe_1_nome: Nome da classe 1
        classe_verdadeira: 0 ou 1
    
    Returns:
        String formatada com a explicação
    """
    Xpl = []
    delta_sorted = sorted(enumerate(delta), key=lambda x: abs(x[1]), reverse=True)
    
    for feature_idx, delta_val in delta_sorted:
        if R >= 0:
            feature = feature_names[feature_idx]
            Xpl.append(f"{feature} = {Vs[feature]:.1f} (Δ={delta_val:.2f})")
            R -= delta_val
        else:
            break
    
    if not Xpl:  # Se nenhuma feature foi selecionada
        Xpl.append("Nenhuma feature significativa identificada")
    
    if classe_verdadeira == 0:
        return f"PI-Explicação - {classe_0_nome}: " + ", ".join(Xpl)
    else:
        return f"PI-Explicação - {classe_1_nome}: " + ", ".join(Xpl)

def analisar_instancias(X_test, y_test, classe_0_nome, classe_1_nome, modelo, X):
    """
    Analisa todas as instâncias
    
    Args:
        X_test: Dados de teste (DataFrame ou array)
        y_test: Classes verdadeiras (binárias)
        classe_0_nome: Nome da classe 0
        classe_1_nome: Nome da classe 1
        modelo: Modelo treinado (deve ter decision_function)
        X: DataFrame completo (para cálculo de min/max)
    
    Returns:
        Lista de explicações para todas as instâncias
    """
    # Garantir que X_test é um DataFrame com nomes consistentes
    if not isinstance(X_test, pd.DataFrame):
        feature_names = X.columns if hasattr(X, 'columns') else [f"feature_{i}" for i in range(X.shape[1])]
        X_test = pd.DataFrame(X_test, columns=feature_names)
    
    # Garantir que X também é DataFrame com mesmos nomes
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=X_test.columns)
    
    feature_names = X_test.columns.tolist()
    explicacoes = []
    
    for idx in range(len(X_test)):
        Vs = X_test.iloc[idx].to_dict()
        instancia_test = X_test.iloc[[idx]].copy()
        gamma_A = modelo.decision_function(instancia_test)[0]
        
        classe_verdadeira = y_test[idx]
        
        # 1. Cálculo dos deltas
        w = modelo.coef_[0]
        delta = calcular_deltas(Vs, X, w, classe_verdadeira)
        
        # 2. Cálculo de R
        R = sum(delta) - gamma_A
        
        # 3. Gerar explicação
        explicacao = one_explanation(Vs, delta, R, feature_names, classe_0_nome, classe_1_nome, classe_verdadeira)
        explicacoes.append(explicacao)
        
        # Debug opcional
        print(f"\nInstância {idx} (Classe {classe_verdadeira}):")
        print(f"  Gamma_A: {gamma_A:.4f}, R: {R:.4f}")
        print(f"  {explicacao}")

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