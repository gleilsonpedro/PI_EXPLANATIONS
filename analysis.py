import numpy as np
from collections import defaultdict

def calculate_statistics(X, y, explanations, feature_names):
    """Calcula estatísticas avançadas"""
    stats = {
        'global': defaultdict(dict),
        'class_0': defaultdict(dict),
        'class_1': defaultdict(dict)
    }
    
    # Contagem de features nas explicações
    feature_counts = defaultdict(int)
    for exp in explanations:
        for feat in exp:
            feature_counts[feat] += 1
    
    # Cálculo de estatísticas
    for i, feat in enumerate(feature_names):
        values = X[:, i]
        
        # Global
        stats['global'][feat]['mean'] = np.mean(values)
        stats['global'][feat]['std'] = np.std(values)
        stats['global'][feat]['count'] = feature_counts.get(i, 0)
        stats['global'][feat]['frequency'] = feature_counts.get(i, 0) / len(explanations)
        
        # Por classe
        stats['class_0'][feat]['mean'] = np.mean(values[y == 0])
        stats['class_0'][feat]['std'] = np.std(values[y == 0])
        
        stats['class_1'][feat]['mean'] = np.mean(values[y == 1])
        stats['class_1'][feat]['std'] = np.std(values[y == 1])
    
    # Estatísticas agregadas
    stats['summary'] = {
        'mean_features_per_exp': np.mean([len(e) for e in explanations]),
        'std_features_per_exp': np.std([len(e) for e in explanations]),
        'total_explanations': len(explanations)
    }
    
    return stats

def print_statistics(stats, class_names):
    """Exibe estatísticas formatadas"""
    print("\n=== ESTATÍSTICAS AVANÇADAS ===")
    print(f"Média de features por explicação: {stats['summary']['mean_features_per_exp']:.2f} ± {stats['summary']['std_features_per_exp']:.2f}")
    
    print("\nTOP FEATURES (global):")
    print("Feature | Média | DP | Frequência")
    print("--------------------------------")
    for feat in sorted(stats['global'].items(), key=lambda x: x[1]['count'], reverse=True)[:10]:
        print(f"{feat[0]:15} | {feat[1]['mean']:.2f} | {feat[1]['std']:.2f} | {feat[1]['frequency']:.1%}")
    
    print(f"\nCOMPARAÇÃO ENTRE CLASSES ({class_names[0]} vs {class_names[1]}):")
    print("Feature | Média(0) | DP(0) | Média(1) | DP(1)")
    print("--------------------------------------------")
    for feat in sorted(stats['global'].items(), key=lambda x: x[1]['count'], reverse=True)[:5]:
        name = feat[0]
        print(f"{name:15} | {stats['class_0'][name]['mean']:.2f} | {stats['class_0'][name]['std']:.2f} | " +
              f"{stats['class_1'][name]['mean']:.2f} | {stats['class_1'][name]['std']:.2f}")