# PI-Explicação (Prediction Interval Explanation)

Este repositório contém um projeto para realizar **PI-Explicação** (Prediction Interval Explanation) em diversos datasets. O objetivo é fornecer explicações interpretáveis para as predições de modelos de machine learning, identificando as features mais relevantes e analisando o impacto de perturbações nas predições.

O projeto utiliza **Regressão Logística** com a estratégia **One-vs-Rest** para transformar problemas multiclasse em binários, permitindo a geração de explicações claras e eficientes.

---

## Funcionalidades Principais

- **Carregamento de Datasets**: Suporte para múltiplos datasets populares, como `iris`, `wine`, `breast_cancer`, entre outros.
- **Transformação Binária**: Implementa a estratégia **One-vs-Rest** para transformar problemas multiclasse em problemas de classificação binária, adaptando-os para a Regressão Logística.
- **Treinamento de Modelos**: Treina modelos de **Regressão Logística** para classificação binária, utilizando os datasets transformados.
- **PI-Explicação**: Gera explicações para as predições do modelo, identificando as features mais relevantes que influenciaram a decisão.
- **Análise de Sensibilidade**:  Avalia o impacto de cada feature na predição, calculando a variação que cada uma pode ter sem alterar a classe prevista.
- **Otimização e Eficiência**: Design otimizado para eficiência computacional, com complexidade claramente documentada para cada etapa do processo.

---

## Como Funciona?

O projeto segue os seguintes passos:

1. **Escolha do Dataset**: O usuário seleciona um dataset disponível através de um menu interativo.
2. **Definição do Problema Binário**: O usuário escolhe qual classe do dataset será considerada como classe `0` (negativa). As demais classes são agrupadas como classe `1` (positiva) utilizando a estratégia **One-vs-Rest**.
3. **Treinamento do Modelo**: Um modelo de **Regressão Logística** é treinado no dataset transformado em binário.
4. **PI-Explicação**: O modelo gera explicações para cada instância, destacando as features mais relevantes que contribuíram para a predição.
5. **Análise de Sensibilidade (Perturbações)**: Para cada feature identificada na PI-Explicação, o projeto calcula um intervalo de perturbação, indicando a faixa de valores que a feature pode assumir sem alterar a predição do modelo.

---

## Complexidade Computacional

O projeto foi desenvolvido com foco em eficiência. Abaixo está a complexidade computacional das principais etapas:

- **Carregamento do Dataset**: `O(1)` (depende principalmente da velocidade de leitura do arquivo).
- **Transformação Binária (One-vs-Rest)**: `O(n)`, onde `n` é o número de amostras no dataset.
- **Treinamento do Modelo (Regressão Logística)**: `O(n * m)`, onde `n` é o número de amostras e `m` é o número de features no dataset.
- **PI-Explicação**: `O(k * m)`, onde `k` é o número de instâncias analisadas e `m` é o número de features.
- **Análise de Sensibilidade (Perturbações)**: `O(i * p)`, onde `i` é o número de instâncias analisadas e `p` é o número de perturbações calculadas por feature.

---

## Como Usar

### Pré-requisitos

- Python 3.8 ou superior.
- Bibliotecas Python necessárias:
    - `numpy`
    - `pandas`
    - `scikit-learn`
    - `matplotlib` (opcional, para visualizações)
    - `seaborn` (opcional, para visualizações)

Instale as dependências utilizando o gerenciador de pacotes `pip`:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn

Executando o Projeto

Clone o repositório para o seu ambiente local:

git clone <URL_DO_REPOSITORIO>
cd <NOME_DO_REPOSITORIO>

Execute o script principal do projeto:

python main.py


Siga as instruções apresentadas no terminal para selecionar o dataset desejado e definir a classe que será utilizada como classe 0 para a transformação binária.

Estrutura do Projeto

A estrutura do projeto é organizada da seguinte forma:

pi-explicacao/
├── data/
│   └── load_datasets.py  # Script para carregar os datasets
│
├── models/
│   └── train_model.py   # Script para treinar o modelo de Regressão Logística
│
├── explanations/
│   └── pi_explanation.py # Script para gerar PI-Explicações e realizar a análise de perturbações
│
├── main.py              # Script principal para executar o projeto
│
└── README.md            # Documentação do projeto

Datasets Suportados

O projeto oferece suporte aos seguintes datasets:

iris

wine

breast_cancer

digits

banknote_authentication

wine_quality

heart_disease

parkinsons

car_evaluation

diabetes_binary

Contribuindo

Este projeto está em constante evolução, e contribuições são muito bem-vindas! Se você deseja adicionar novos datasets, aprimorar as técnicas de explicação, otimizar o código existente ou corrigir bugs, siga os passos abaixo:

Faça um fork do repositório para a sua conta do GitHub.

Crie uma branch para a sua feature ou correção de bug:

git checkout -b feature/nome-da-feature

Realize as modificações e adicione commits com mensagens descritivas:

git commit -m 'Adicionando nova feature ou corrigindo bug'

Envie as alterações para o seu repositório fork:

git push origin feature/nome-da-feature
Abra um pull request (PR) para o repositório original.

Licença

Este projeto é licenciado sob a licença MIT. Consulte o arquivo LICENSE para obter mais detalhes sobre os termos e condições de uso.

Contato

Se você tiver alguma dúvida, sugestão ou feedback sobre o projeto, entre em contato:

Nome: [Gleilson_Pedro]

Email: [gleilsonsvo@gmail.com]

GitHub: [https://github.com/gleilsonpedro]



