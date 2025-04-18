import numpy as np
import pandas as pd
from collections import Counter

import numpy as np
import pandas as pd
from collections import Counter

import numpy as np
import pandas as pd
from collections import Counter

def calculate_deltas(model, instance, X_train):
    """Calcula os deltas conforme artigo NEURIPS20"""
    coef = model.coef_[0]
    pred_class = model.predict(instance)[0]
    instance_val = instance.iloc[0]
    
    deltas = []
    for i, feature in enumerate(X_train.columns):
        val = instance_val[feature]
        if pred_class == 1:  # Classe positiva
            worst = X_train[feature].min() if coef[i] > 0 else X_train[feature].max()
        else:  # Classe negativa
            worst = X_train[feature].max() if coef[i] > 0 else X_train[feature].min()
        
        deltas.append((val - worst) * coef[i])
    
    return np.array(deltas)

def one_explanation(model, instance, X_train):
    """Implementação exata do algoritmo do artigo"""
    pred_class = model.predict(instance)[0]
    score = model.decision_function(instance)[0]
    deltas = calculate_deltas(model, instance, X_train)
    
    # Ordena features por importância
    order = np.argsort(-np.abs(deltas))
    sorted_deltas = deltas[order]
    features = X_train.columns[order]
    
    explanation = []
    cum_sum = 0
    phi = (sum(deltas) - score) if pred_class == 1 else (score - sum(deltas))
    
    for i, delta in enumerate(sorted_deltas):
        if (pred_class == 1 and cum_sum <= phi) or (pred_class == 0 and cum_sum < phi):
            explanation.append(f"{features[i]} = {instance.iloc[0][features[i]]}")
            cum_sum += abs(delta)
        else:
            break
    
    return explanation

def gerar_relatorio(model, X_test, y_test, X_train, class_names):
    """Gera relatório completo"""
    # Convertendo y_test para Series se for numpy array
    y_test_series = pd.Series(y_test) if isinstance(y_test, np.ndarray) else y_test
    
    explanations = [one_explanation(model, X_test.iloc[[i]], X_train) 
                   for i in range(len(X_test))]
    
    # Verifica validade das explicações
    validacoes = [verificar_explicacao(model, X_test.iloc[[i]], exp, X_train) 
                 for i, exp in enumerate(explanations)]

    # Cálculo de estatísticas completo
    stats = calcular_estatisticas(explanations)
    
    # Construção do relatório completo
    report = f"""
==================================================
{'ANÁLISE COMPLETA'.center(50)}
==================================================

Dataset: {X_train.shape[1]} features, {len(X_train)} amostras de treino
Classes: {class_names[0]} (0) vs {class_names[1]} (1)

Métricas do Modelo:
- Acurácia: {model.score(X_test, y_test):.2%}
- Explicações válidas: {sum(validacoes)}/{len(validacoes)}

Estatísticas das Explicações:
- Média de features por explicação: {stats['media_features']:.2f} ± {stats['desvio_features']:.2f}
- Número mínimo de features: {min([len(e) for e in explanations])}
- Número máximo de features: {max([len(e) for e in explanations])}

Features mais relevantes (frequência):
"""
    for feat, count in stats['feature_frequentes']:
        report += f"  - {feat}: {count} ocorrências\n"
    
    # Distribuição por classe
    report += "\nDistribuição por classe:\n"
    for cls in [0, 1]:
        cls_exps = [e for i, e in enumerate(explanations) if y_test_series.iloc[i] == cls]
        report += f"- {class_names[cls]}: {len(cls_exps)} explicações "
        report += f"(média {np.mean([len(e) for e in cls_exps]):.1f} features)\n"
    
    # Todas as explicações
    report += "\nExplicações Detalhadas:\n"
    for i, exp in enumerate(explanations):
        report += f"\nInstância {i} ({class_names[y_test_series.iloc[i]]}):\n"
        report += f"Features usadas: {len(exp)}\n"
        report += "\n".join(f"  - {f}" for f in exp)
        report += f"\nValidação: {'✓' if validacoes[i] else '✗'}"
    
    return report

def verificar_explicacao(model, instance, explanation, X_train):
    """Verifica se a explicação realmente garante a predição"""
    original_pred = model.predict(instance)[0]
    
    # Cria uma instância perturbada com valores de pior caso
    perturbed = instance.copy()
    for feature in X_train.columns:
        if feature not in [x.split(' = ')[0] for x in explanation]:
            coef_idx = X_train.columns.get_loc(feature)
            if model.coef_[0][coef_idx] > 0:
                perturbed[feature] = X_train[feature].min()
            else:
                perturbed[feature] = X_train[feature].max()
    
    return original_pred == model.predict(perturbed)[0]

def calcular_estatisticas(explanations):
    """Calcula estatísticas sobre as explicações"""
    num_features = [len(exp) for exp in explanations]
    avg_features = np.mean(num_features)
    std_features = np.std(num_features)
    
    # Conta frequência de features
    feature_counts = Counter()
    for exp in explanations:
        for item in exp:
            feature = item.split(' = ')[0]
            feature_counts[feature] += 1
    
    return {
        'media_features': avg_features,
        'desvio_features': std_features,
        'feature_frequentes': feature_counts.most_common(5)
    }
