import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

def generate_pi_explanation_with_details(instance, model, feature_names, predicted_class):
    """Gera explicação PI detalhada para uma instância"""
    weights = model.coef_[0]
    bias = model.intercept_[0]
    
    deltas = []
    worst_cases = []
    for i, w in enumerate(weights):
        if predicted_class == 1:
            worst_case = 0 if w > 0 else 1
        else:
            worst_case = 1 if w > 0 else 0
        delta = w * (instance[i] - worst_case)
        deltas.append(delta)
        worst_cases.append(worst_case)
    
    threshold = np.sum(deltas) - (np.dot(weights, instance) + bias)
    sorted_indices = np.argsort(-np.abs(deltas))
    
    explanation = []
    cumulative = 0
    for i in sorted_indices:
        cumulative += deltas[i]
        explanation.append({
            'feature': feature_names[i],
            'weight': weights[i],
            'value': instance[i],
            'worst_case': worst_cases[i],
            'delta': deltas[i],
            'cumulative': cumulative
        })
        if cumulative > threshold:
            break
    
    return {
        'predicted_class': predicted_class,
        'features': explanation,
        'threshold': threshold,
        'bias': bias
    }


def train_model(X_train, y_train):
    """Treina modelo de regressão logística"""
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    return model, scaler

def evaluate_model(model, X_test, y_test):
    """Avalia o modelo"""
    y_pred = model.predict(X_test)
    return {
        'accuracy': metrics.accuracy_score(y_test, y_pred),
        'report': metrics.classification_report(y_test, y_pred)
    }