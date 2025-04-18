import numpy as np
import pandas as pd

def calcular_deltas(Vs, X, w, classe_verdadeira):
    """
    Args:
        Vs: Valores da instância (dict {feature: valor})
        X: DataFrame completo (para calcular min/max)
        w: Pesos do modelo (modelo.coef_[0])
        classe_verdadeira: 0 (classe alvo) ou 1 (outras classes)
    
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
            delta = (extremo - Vs[feature]) * w[i]
        deltas.append(delta)
    return deltas

def one_explanation(Vs, delta, R, feature_names, class_names, classe_verdadeira):
    """
    Gera uma PI-explicação conforme Algoritmo 1 do artigo
    
    Args:
        Vs: Valores da instância
        delta: Lista de deltas calculados
        R: Valor residual (Σδ - γ_A)
        feature_names: Nomes das features
        class_names: Nomes das classes
        classe_verdadeira: 0 ou 1
    
    Returns:
        String formatada com a explicação
    """
    Xpl = []
    delta_sorted = sorted(enumerate(delta), key=lambda x: abs(x[1]), reverse=True)
    
    # Calcular threshold para considerar features relevantes
    threshold = 0.1 * abs(sum(delta))  # Considera features que contribuem com pelo menos 10% do total
    
    for feature_idx, delta_val in delta_sorted:
        if abs(delta_val) < threshold:
            continue  # Pula features com contribuição insignificante
            
        feature = feature_names[feature_idx]
        Xpl.append(f"{feature} - {Vs[feature]:.1f} (Δ={delta_val:.2f})")
        
        if classe_verdadeira == 0:
            R -= delta_val
            if R <= 0: 
                break
        else:
            R += delta_val
            if R >= 0: 
                break
    
    if not Xpl:  # Se nenhuma feature foi selecionada
        Xpl.append("Nenhuma feature significativa identificada")
    
    if classe_verdadeira == 0:
        return f"PI-Explicação - {class_names[0]}: " + ", ".join(Xpl)
    else:
        return f"PI-Explicação NÃO-{class_names[0]}: " + ", ".join(Xpl)

def analisar_instancias(X_test, y_test, class_names, modelo, X):
    """
    Analisa todas as instâncias conforme artigo (Seção 3.2)
    
    Args:
        X_test: Dados de teste
        y_test: Classes verdadeiras (binárias)
        class_names: Nomes das classes
        modelo: Modelo treinado
        X: DataFrame completo (para cálculo de min/max)
    
    Returns:
        Lista de explicações para todas as instâncias
    """
    # DEBUG
    print("\nDEBUG - Valores do Modelo:")
    print(f"Coeficientes (w): {modelo.coef_[0]}")
    print(f"Intercept: {modelo.intercept_[0]}")
    
    
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
        # Criar DataFrame mantendo nomes e estrutura
        instancia_test = X_test.iloc[[idx]].copy()
        gamma_A = modelo.decision_function(instancia_test)[0]
        classe_verdadeira = y_test[idx]
        
        # 1. Cálculo dos deltas (Artigo Eq. 14)
        w = modelo.coef_[0]
        delta = calcular_deltas(Vs, X, w, classe_verdadeira)
        
        # 2. Cálculo de R (Artigo Eq. 15)
        R = sum(delta) - gamma_A
        
        # 3. Gerar explicação (Algoritmo 1)
        explicacao = one_explanation(Vs, delta, R, feature_names, class_names, classe_verdadeira)
        explicacoes.append(explicacao)
        
        # Debug opcional (não presente no artigo)
        print(f"\nInstância {idx} (Classe {classe_verdadeira}):")
        print(f"  Gamma_A: {gamma_A:.4f}, R: {R:.4f}")
        print(f"  {explicacao}")
    
    return explicacoes

def contar_features_relevantes(explicacoes, class_names):
    """
    Conta features relevantes para a classe alvo (apenas para análise)
    
    Args:
        explicacoes: Lista de explicações
        class_names: Nomes das classes
    
    Returns:
        Dicionário {feature: contagem}
    """
    contagem = {}
    target_class = class_names[0]
    
    for exp in explicacoes:
        if target_class in exp:  # Apenas explicações da classe alvo
            # Extrai a parte após "PI-Explicação para ...: "
            partes = exp.split(": ")
            if len(partes) > 1:
                features = partes[1].split(", ")
                for f in features:
                    nome = f.split(" - ")[0].strip()
                    contagem[nome] = contagem.get(nome, 0) + 1
    
    print(f"\n Contagem de Features Relevantes para '{target_class}':")
    for f, cnt in sorted(contagem.items(), key=lambda x: x[1], reverse=True):
        print(f"  {f}: {cnt} ocorrências")
    
    return contagem