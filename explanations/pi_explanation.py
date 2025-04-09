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
            delta =  (Vs[feature] - extremo) * w[i]
        deltas.append(delta)
    return deltas

def one_explanation(Vs, delta, R, feature_names, classe_0_nome, classe_verdadeira):
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
    #estranho tem que rever
    ## threshold = 0.1 * abs(sum(delta))  # Considera features que contribuem com pelo menos 10% do total
    
    for feature_idx, delta_val in delta_sorted:
       # if abs(delta_val) < threshold:
       #     continue  # Pula features com contribuição insignificante
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
        return f"PI-Explicação NÃO-{classe_0_nome}: " + ", ".join(Xpl)

def analisar_instancias(X_test, y_test, classe_0_nome, class_names, modelo, X, gamma_reject = 0.5):
    """
    Analisa todas as instâncias com opção de rejeição
    
    Args:
        X_test: Dados de teste (DataFrame ou array)
        y_test: Classes verdadeiras (binárias)
        class_names: Nomes das classes
        modelo: Modelo treinado (deve ter decision_function)
        X: DataFrame completo (para cálculo de min/max)
        gamma_reject: Limiar para rejeição (default: 0.5) quanto mais alto mais rejeições
    
    Returns:
        Lista de explicações para todas as instâncias
    """
    # DEBUG - Mostrar parâmetros do modelo
    print("\nDEBUG - Valores do Modelo:")
    print(f"Coeficientes (w): {modelo.coef_[0]}")
    print(f"Intercept: {modelo.intercept_[0]}")
    print(f"Limiar de rejeição: |γ_A| < {gamma_reject}")

    # Garantir que X_test é um DataFrame com nomes consistentes
    if not isinstance(X_test, pd.DataFrame):
        feature_names = X.columns if hasattr(X, 'columns') else [f"feature_{i}" for i in range(X.shape[1])]
        X_test = pd.DataFrame(X_test, columns=feature_names)
    
    # Garantir que X também é DataFrame com mesmos nomes
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=X_test.columns)
    
    feature_names = X_test.columns.tolist()
    explicacoes = []
    rejeicoes = []  # Lista para armazenar índices das rejeições
    
    for idx in range(len(X_test)):
        Vs = X_test.iloc[idx].to_dict()
        instancia_test = X_test.iloc[[idx]].copy()
        gamma_A = modelo.decision_function(instancia_test)[0]
        
        # Verificar rejeição
        if abs(gamma_A) < gamma_reject:
            if gamma_A >= 0:
                status = f"PRÓXIMA DA CLASSE 1 (NÃO-{classe_0_nome})"
            else:
                status = f"PRÓXIMA DA CLASSE 0 ({classe_0_nome})"
                
            msg = (f"Instância {idx} REJEITADA | {status} | "
                   f"Classe real: {class_names[y_test[idx]]} | "
                   f"|γ_A|={abs(gamma_A):.2f} < {gamma_reject} | "
                   f"Valores: {Vs}")
            explicacoes.append(msg)
            rejeicoes.append(idx)
            continue

        # estranho também    
        classe_verdadeira = y_test[idx]
        
        # 1. Cálculo dos deltas (Artigo Eq. 14)
        w = modelo.coef_[0]
        delta = calcular_deltas(Vs, X, w, classe_verdadeira)
        
        # 2. Cálculo de R (Artigo Eq. 15)
        R = sum(delta) - gamma_A
        
        # 3. Gerar explicação (Algoritmo 1)
        explicacao = one_explanation(Vs, delta, R, feature_names, classe_0_nome, classe_verdadeira)
        explicacoes.append(explicacao)
        
        # Debug opcional
        print(f"\nInstância {idx} (Classe {classe_verdadeira}):")
        print(f"  Gamma_A: {gamma_A:.4f}, R: {R:.4f}")
        print(f"  {explicacao}")

    # Relatório consolidado
    print(f"\nResumo - Total de instâncias: {len(X_test)}")
    print(f"Instâncias rejeitadas: {len(rejeicoes)} ({len(rejeicoes)/len(X_test):.1%})")
    print(f"Instâncias classificadas: {len(X_test)-len(rejeicoes)}")
    
    # Detalhes das rejeições (se houver)
    if rejeicoes:
        print("\n=== DETALHES DAS REJEIÇÕES ===")
        for idx in rejeicoes:
            print(explicacoes[idx])
    else:
        print("\nNenhuma instância rejeitada com o limiar atual.")
    
    return explicacoes

def contar_features_relevantes(explicacoes, classe_0_nome):
    """
    Conta features relevantes para a classe alvo (apenas para análise)
    Ignora instâncias rejeitadas na contagem
    
    Args:
        explicacoes: Lista de explicações (pode conter rejeições)
        class_names: Nomes das classes
    
    Returns:
        Dicionário {feature: contagem}
    """
    contagem = {}
    
    for exp in explicacoes:
        if "REJEITADA" in exp:
            continue
            
        if classe_0_nome in exp:  # Agora usando classe_0_nome diretamente
            partes = exp.split(": ")
            if len(partes) > 1:
                features = partes[1].split(", ")
                for f in features:
                    nome = f.split(" - ")[0].strip()
                    contagem[nome] = contagem.get(nome, 0) + 1
    
    print(f"\nContagem de Features Relevantes para '{classe_0_nome}' (excluindo rejeições):")
    for f, cnt in sorted(contagem.items(), key=lambda x: x[1], reverse=True):
        print(f"  {f}: {cnt} ocorrências")
    
    return contagem