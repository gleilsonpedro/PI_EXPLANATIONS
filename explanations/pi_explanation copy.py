import numpy as np
import pandas as pd

def one_explanation(Vs, delta, R, feature_names, class_names, classe_verdadeira):
    """
    Versão final alinhada com o artigo:
    - Para classe 0: features que garantem a classe
    - Para classe 1: features que excluem a classe 0
    - Mantém sinais originais dos deltas conforme artigo
    """
    Xpl = []
    delta_sorted = sorted(enumerate(delta), key=lambda x: abs(x[1]), reverse=True)
    
    for feature_idx, delta_val in delta_sorted:
        feature = feature_names[feature_idx]
        
        # Apenas features com contribuição significativa
        if abs(delta_val) < 1e-5:  # Limiar pequeno para evitar ruído
            continue
            
        Xpl.append(f"{feature} - {Vs[feature]}")
        R -= delta_val
        
        if R <= 0:
            break
    
    # Formatação clara do output
    if classe_verdadeira == 0:
        return f"🔷 PI-Explicação para {class_names[0]}: " + ", ".join(Xpl)
    else:
        return f"🔶 PI-Explicação para NÃO-{class_names[0]}: " + ", ".join(Xpl)

def analisar_instancias(X_test, y_test, class_names, modelo, X, instancia_para_analisar=None):
    """
    Analisa todas as instâncias do conjunto de teste e calcula as PI-explicações.
    
    Parâmetros:
        X_test: Conjunto de teste
        y_test: Classes verdadeiras (binárias)
        class_names: Nomes das classes
        modelo: Modelo treinado
        X: DataFrame completo (para calcular min/max)
        instancia_para_analisar: Índice específico para analisar (opcional)
    
    Retorna:
        Lista com todas as PI-explicações
    """
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])

    feature_names = X_test.columns.tolist()
    TUDO = []
    
    # Define o range de instâncias para analisar
    if instancia_para_analisar is not None:
        indices = [instancia_para_analisar]
    else:
        indices = range(len(X_test))
    
    print("\n🔎 Análise Detalhada por Instância:")
    for idx in indices:
        Vs = X_test.iloc[idx].to_dict()
        instancia_test = X_test.iloc[[idx]]
        gamma_A = modelo.decision_function(instancia_test)[0]
        classe_verdadeira = y_test[idx]

        # Cálculo dos deltas conforme artigo
        delta = []
        w = modelo.coef_[0]
        
        for i, feature in enumerate(feature_names):
            if classe_verdadeira == 0:  # Classe alvo
                if w[i] < 0:
                    delta_val = (Vs[feature] - X[feature].min()) * w[i]
                else:
                    delta_val = (Vs[feature] - X[feature].max()) * w[i]
            else:  # Classe 1 (não-alvo)
                if w[i] < 0:
                    delta_val = (X[feature].max() - Vs[feature]) * w[i]
                else:
                    delta_val = (X[feature].min() - Vs[feature]) * w[i]
            delta.append(delta_val)

        # Cálculo de R conforme artigo (Σδ - γ)
        R = sum(delta) - gamma_A
        
        # Garantir R mínimo para evitar explicações vazias
        R = max(R, 0.1 * abs(gamma_A)) if classe_verdadeira == 0 else R
        
        # Obter a explicação
        Xpl = one_explanation(Vs, delta, R, feature_names, class_names, classe_verdadeira)
        TUDO.append(Xpl)
        
        # Exibição detalhada
        print(f"\nInstância {idx} (Classe {classe_verdadeira}):")
        print(f"  Valores: {Vs}")
        print(f"  Gamma_A: {gamma_A:.4f}")
        print(f"  R calculado: {R:.4f}")
        print(f"  Explicação: {Xpl}")
        
        # Debug dos deltas (opcional)
        print("  Deltas calculados:")
        for i, (name, val) in enumerate(zip(feature_names, delta)):
            print(f"    {name}: {val:.4f} (w={w[i]:.4f})")

    return TUDO

def contar_features_relevantes(resultados):
    contagem = {}
    for exp in resultados:
        if "NÃO-" in exp:  # Ignora features de exclusão
            continue
        features = exp.split(":")[1].split(",")
        for f in features:
            nome = f.split("-")[0].strip()
            contagem[nome] = contagem.get(nome, 0) + 1
    
    print("\n🔍 Contagem de Features Relevantes para a Classe Alvo:")
    for f, cnt in sorted(contagem.items(), key=lambda x: x[1], reverse=True):
        print(f"{f}: {cnt} ocorrências")