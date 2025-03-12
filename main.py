from data.load_datasets import carregar_dataset
from models.train_model import treinar_modelo
from explanations.pi_explanation import analisar_instancias, contar_features_relevantes
import time
import os

#  Função para limpar o terminal
def limpar_terminal():
    if os.name == 'nt':  # Windows
        os.system('cls')
    else:  # Linux/Mac
        os.system('clear')

#  Menu de seleção de datasets
menu = '''
|  ************************* MENU *************************** |
|  0 - iris                     |  1 - wine                   |
|  2 - breast_cancer            |  3 - digits                 |
|  4 - banknote_authentication  |  5 - wine_quality           |
|  6 - heart_disease            |  7 - parkinsons             |
|  8 - car_evaluation           |  9 - diabetes_binary        |
|  Q - SAIR                                                   |
|-------------------------------------------------------------|
'''

#  Exibe o menu e solicita uma escolha
print(menu)
opcao = input("Digite o número do dataset ou 'Q' para sair: ").upper().strip()

#  Processa a opção selecionada
while True:
    if opcao == 'Q':
        print("Você escolheu sair.")
        break
    elif opcao.isdigit() and 0 <= int(opcao) <= 9:
        nomes_datasets = [
            'iris', 'wine', 'breast_cancer', 'digits', 'banknote_authentication',
            'wine_quality', 'heart_disease', 'parkinsons', 'car_evaluation', 'diabetes_binary'
        ]
        nome_dataset = nomes_datasets[int(opcao)]
        
        limpar_terminal()
        
        print(f"**Dataset '{nome_dataset}' escolhido.**\n")
        try:
            #  Carrega o dataset
            X, y, class_names = carregar_dataset(nome_dataset)
            print(f"Dataset {nome_dataset} carregado com sucesso!")
            print(f"Classes disponíveis: {class_names}")
            print(f"Total de amostras: {X.shape[0]}")
            print(f"Número de atributos: {X.shape[1]}\n")

            while True:
                try:
                    escolha_classe_0 = int(input("\nDigite o número para classe `0`: "))
                    if 0 <= escolha_classe_0 < len(class_names):
                        break
                    else:
                        print("Número inválido! Escolha um número da lista acima.")
                except ValueError:
                    print("Entrada inválida! Digite um número correspondente a uma classe.")

            classe_0_nome = class_names[escolha_classe_0]
            outras_classes = [c for i, c in enumerate(class_names) if i != escolha_classe_0]

            print(f"\n **Definição do problema binário:**")
            print(f"    Classe `{classe_0_nome}` será a classe `0`")
            print(f"    Classes `{outras_classes}` serão agrupadas na classe `1`\n")

            # Ajustar `y` para refletir a nova classe `0`
            y_binario = [0 if label == escolha_classe_0 else 1 for label in y]

            #  Medindo o tempo de treinamento do modelo
            inicio_treinamento = time.time()
            modelo, X_test, y_test = treinar_modelo(X, y_binario, classe_0=0)
            fim_treinamento = time.time()
            tempo_treinamento = fim_treinamento - inicio_treinamento

            #  Medindo o tempo de cálculo das PI-explicações
            inicio_pi = time.time()
            TUDO = analisar_instancias(X_test, y_test, class_names, modelo, X)
            fim_pi = time.time()
            tempo_pi = fim_pi - inicio_pi

            #  Tempo total
            tempo_total = tempo_treinamento + tempo_pi

            #  Exibe os tempos de execução
            print()
            print(f"**Tempo de treinamento do modelo:**      {tempo_treinamento:.4f} segundos")
            print(f"**Tempo de cálculo das PI-explicações:** {tempo_pi:.4f} segundos")
            print(f"**Tempo total de execução:**             {tempo_total:.4f} segundos\n")

            #  Conta as features relevantes
            print("**Contagem de features relevantes:**")
            contar_features_relevantes(TUDO)
            break
        except Exception as e:
            print(f"Erro ao processar o dataset: {e}")
    else:
        #  Limpa o terminal antes de exibir o menu novamente
        limpar_terminal()
        print(menu)
        print("Opção inválida. Por favor, escolha um número do menu ou 'Q' para sair.")
        opcao = input("Digite o número do dataset ou 'Q' para sair: ").upper().strip()
