import numpy as np
import pandas as pd

def calcular_gamma_omega(modelo, X, classe_predita):
    """
    Calcula γ_ω — o valor mínimo possível de γ (score linear) para mudar a predição. representa o pior caso
    """
    coef = modelo.coef_[0]           # peso das features gerado pela RLogistica
    intercept = modelo.intercept_[0] # bias do modelo
    
    gamma_omega = intercept
    # itera sobre o datset calculando a contribuição "pior possivel" da feature p/ escore
    for i, col in enumerate(X.columns):
        w_i = coef[i]
        min_val = X[col].min()
        max_val = X[col].max()

        # Sempre tentar reduzir a influência da feature (pior caso)
        contrib = w_i * (min_val if w_i > 0 else max_val) 
        gamma_omega += contrib

    return gamma_omega


def calcular_deltas(modelo, X, instancia, classe_predita):
    coef = modelo.coef_[0]
    deltas = []

    for feature in X.columns:
        w_i = coef[X.columns.get_loc(feature)]
        x_i = instancia[feature]
        min_val = X[feature].min()
        max_val = X[feature].max()

        valor_atual = w_i * x_i
        pior_caso = w_i * (min_val if w_i > 0 else max_val)
        delta = valor_atual - pior_caso
        deltas.append((feature, delta))
    # Ordena as features pela magnitude da contribuição (mais importantes primeiro)
    deltas.sort(key=lambda x: -abs(x[1]))
    return deltas


def one_explanation(Vs, delta, R, classe_nomes, classe_predita):
    """
    Gera explicação PI mínima conforme artigo NeurIPS20
    Retorna: (explicacao_str, features_usadas)
    """
    Xpl = []
    R_restante = float(R)  # Garante escalar

    for feature, delta_val in delta:
        if R_restante <= 0:
            break
            
        Xpl.append((feature, Vs[feature], delta_val))
        R_restante -= delta_val

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
        
        explicacao, features = one_explanation(Vs, delta, R, classe_nomes, classe_predita)

        #print(f"\nInstância {idx} - Classe predita: {classe_nomes[classe_predita]}")
        #print(f"R = {R:.4f}")
        #print("Deltas:")
        #for f, d in delta:
            #print(f"  {f}: {d:.4f}")

        explicacao, features = one_explanation(Vs, delta, R, classe_nomes, classe_predita)
        explicacoes.append(explicacao)
        
        # Atualiza contagem de features
        for f in features:
            contagem_features[f'Classe {classe_predita}'][f] = contagem_features[f'Classe {classe_predita}'].get(f, 0) + 1
        
        # Saída formatada
        classe_real = y_test[idx]
        print(f"\nInstância {idx} | Real: {classe_nomes.get(classe_real)} | Predita: {classe_nomes[classe_predita]}")
        print(f"  {explicacao}")
        print(f"  Features usadas ({len(features)}): {', '.join(features) if features else 'Nenhuma'}")


    
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