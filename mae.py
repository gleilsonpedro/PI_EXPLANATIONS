from datasets import load_dataset, prepare_data
from pi_explanation import train_model, evaluate_model, generate_pi_explanation_with_details
from analysis import calculate_statistics, print_statistics
from sklearn.model_selection import train_test_split

def show_menu():
    """Menu interativo"""
    print("\n=== MENU DE DATASETS ===")
    datasets = {
        '1': ('iris', 3),
        '2': ('diabetes', 2),
        '3': ('cancer', 2),
        '4': ('wine', 2),
        '5': ('heart', 2)
    }
    
    for key, (name, classes) in datasets.items():
        print(f"{key}. {name.capitalize()} ({classes} classes)")
    print("6. Sair")
    
    choice = input("Escolha: ")
    return datasets.get(choice, (None, None))[0]

def select_classes(class_names):
    """Seleção de classes"""
    print("\nClasses disponíveis:")
    for i, name in enumerate(class_names):
        print(f"{i}. {name}")
    
    class0 = int(input("\nClasse negativa (0): "))
    class1 = int(input("Classe positiva (1): "))
    return class0, class1

def print_instance_explanations(explanations, class_names):
    """Imprime explicações detalhadas para todas as instâncias"""
    print("\n=== EXPLICAÇÕES DETALHADAS POR INSTÂNCIA ===")
    for idx, exp in enumerate(explanations):
        print(f"\nInstância {idx + 1}:")
        print(f"Classe predita: {class_names[exp['predicted_class']]}")
        print(f"Número de features na explicação: {len(exp['features'])}")
        print(f"Threshold: {exp['threshold']:.4f}")
        print(f"Bias (intercept): {exp['bias']:.4f}")
        
        print("\nFeatures explicativas:")
        print("Feature | Peso | Valor | Pior caso | Delta | Acumulado")
        print("-----------------------------------------------------")
        for feat in exp['features']:
            print(f"{feat['feature']:15} | {feat['weight']:+.4f} | {feat['value']:.4f} | "
                  f"{feat['worst_case']:9} | {feat['delta']:.4f} | {feat['cumulative']:.4f}")

def main():
    """Fluxo principal"""
    dataset_name = show_menu()
    if not dataset_name:
        return
    
    X, y, class_names = load_dataset(dataset_name)
    class0, class1 = select_classes(class_names)
    X, y = prepare_data(X, y, class0, class1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model, scaler = train_model(X_train, y_train)
    evaluation = evaluate_model(model, scaler.transform(X_test), y_test)
    
    print(f"\nAcurácia: {evaluation['accuracy']:.2f}")
    print("\nRelatório:\n", evaluation['report'])
    
    explanations = []
    X_test_scaled = scaler.transform(X_test)
    
    for i in range(len(X_test)):
        explanation = generate_pi_explanation_with_details(
            X_test_scaled[i],
            model,
            X.columns.tolist(),
            model.predict([X_test_scaled[i]])[0]
        )
        explanations.append(explanation)
    
    print_instance_explanations(explanations, [class_names[class0], class_names[class1]])
    
    stats = calculate_statistics(X_test_scaled, y_test, [[f['feature'] for f in exp['features']] for exp in explanations], X.columns.tolist())
    print_statistics(stats, [class_names[class0]], [class_names[class1]])

if __name__ == "__main__":
    main()