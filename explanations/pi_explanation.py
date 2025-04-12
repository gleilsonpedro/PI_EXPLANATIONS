import numpy as np
import pandas as pd

def calcular_gamma_omega(modelo, X, classe_predita):
    """
    Calcula gamma_omega (pior caso) conforme artigo NeurIPS20
    Args:
        modelo: Modelo de regressão logística treinado
        X: DataFrame com os dados de treino
        classe_predita: Classe predita (0 ou 1)
    """
    coef = modelo.coef_[0]
    intercept = modelo.intercept_[0]
    
    gamma = intercept
    for i, col in enumerate(X.columns):
        w_i = coef[i]
        min_val = X[col].min()
        max_val = X[col].max()
        
        if w_i > 0:
            contrib = w_i * X[col].min()
        else:
            contrib = w_i * X[col].max()

            
        gamma += contrib
    
    return gamma

def calcular_deltas(modelo, X, instancia, classe_predita):
    """
    Calcula deltas (diferença entre valor atual e pior caso)
    Args:
        modelo: Modelo treinado
        X: DataFrame com dados de treino
        instancia: Dicionário com valores da instância a explicar
        classe_predita: Classe predita (0 ou 1)
    """
    coef = modelo.coef_[0]
    deltas = []
    
    for feature in X.columns:
        w_i = coef[X.columns.get_loc(feature)]
        x_i = instancia[feature]
        min_val = X[feature].min()
        max_val = X[feature].max()
        
        valor_atual = w_i * x_i
        
        if classe_predita == 1:
            pior_caso = w_i * (min_val if w_i > 0 else max_val)
        else:
            pior_caso = w_i * (max_val if w_i > 0 else min_val)
            
        delta = valor_atual - pior_caso
        deltas.append((feature, delta))
    
    # Ordena por magnitude absoluta descendente
    deltas.sort(key=lambda x: -abs(x[1]))
    return deltas

def one_explanation(Vs, delta, R, classe_nomes, classe_predita):
    """
    Gera explicação PI mínima conforme artigo NeurIPS20
    Retorna: (explicacao_str, features_usadas)
    """
    Xpl = []
    R_restante = float(R)  # Garante escalar

    print(f"\n>> [DEBUG] Classe: {classe_nomes[classe_predita]} | R inicial: {R_restante:.4f}")

    for feature, delta_val in delta:
        # Verifica se já pode parar
        if (classe_predita == 1 and R_restante <= 0) or (classe_predita == 0 and R_restante > 0):
            print(f"[DEBUG] Parou após {len(Xpl)} features — R_restante: {R_restante:.4f}")
            break

        Xpl.append((feature, Vs[feature], delta_val))
        R_restante -= delta_val
        print(f"  [DEBUG] Adicionou: {feature}, delta={delta_val:.4f}, R_restante={R_restante:.4f}")

    if not Xpl:
        return (f"PI-Explicação - {classe_nomes[classe_predita]}: Predição baseada no caso geral", [])

    explic_str = ", ".join(f"{f} = {v:.3f}" for f, v, _ in Xpl)
    features_usadas = [f for f, _, _ in Xpl]

    return (f"PI-Explicação - {classe_nomes[classe_predita]}: {explic_str}", features_usadas)


def analisar_instancias(X_test, y_test, classe_0_nome, classe_1_nome, modelo, X):
    """
    Versão robusta que funciona com DataFrames ou arrays NumPy
    """
    # Convertendo para DataFrame se necessário
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=X.columns if hasattr(X, 'columns') else None)
    
    if isinstance(y_test, pd.Series):
        y_test = y_test.values
    
    classe_nomes = {0: classe_0_nome, 1: classe_1_nome}
    explicacoes = []
    contagem_features = {'Classe 0': {}, 'Classe 1': {}}
    
    for idx, (_, instancia) in enumerate(X_test.iterrows()):
        Vs = instancia.to_dict()
        gamma_A = modelo.decision_function([instancia.values])[0]
        classe_predita = int(modelo.predict([instancia.values])[0])
        gamma_omega = calcular_gamma_omega(modelo, X, classe_predita)
        R = gamma_A - gamma_omega
        delta = calcular_deltas(modelo, X, Vs, classe_predita)
        
        print(f"\nInstância {idx} - Classe predita: {classe_nomes[classe_predita]}")
        print(f"R = {R:.4f}")
        print("Deltas:")
        for f, d in delta:
            print(f"  {f}: {d:.4f}")

        
        
        explicacao, features = one_explanation(Vs, delta, R, classe_nomes, classe_predita)
        explicacoes.append(explicacao)
        
        # Atualiza contagem de features
        for f in features:
            contagem_features[f'Classe {classe_predita}'][f] = contagem_features[f'Classe {classe_predita}'].get(f, 0) + 1
        
        # Saída formatada
        classe_real = y_test[idx]
        print(f"\nInstância {idx} (Classe real: {classe_nomes.get(classe_real, classe_real)}):")
        print(f"  {explicacao}")
        print(f"  Features usadas: {', '.join(features) if features else 'Nenhuma'}")
    
    return explicacoes, contagem_features


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