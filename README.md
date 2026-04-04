# Sistema de Análise de Estresse em Estudantes

<p align="center">
  <b>Aplicativo interativo para análise e previsão do estresse em estudantes</b><br>
  Desenvolvido com Streamlit, Python e Machine Learning<br>
</p>

---

## Descrição

Este projeto consiste em um **sistema de análise e previsão do estresse em estudantes** com base em diversas métricas de saúde física, mental, acadêmica e social. Ele permite explorar dados, visualizar correlações, avaliar modelos de Machine Learning e realizar previsões personalizadas.

O modelo utilizado é **Logistic Regression**, integrado a um pipeline que inclui normalização dos dados.

Você pode acessar a versão online do aplicativo aqui:  
[https://student-stress-monitoring-cqg9rtjlskwuyc9cjtpdxk.streamlit.app/](https://student-stress-monitoring-cqg9rtjlskwuyc9cjtpdxk.streamlit.app/)

---

## Funcionalidades

O aplicativo possui cinco seções principais:

1. **Visão Geral**  
   - Visualiza o dataset completo com tradução das colunas para português.  
   - Mostra estatísticas descritivas do conjunto de dados.

2. **Modelo**  
   - Treina o modelo de regressão logística.  
   - Exibe métricas de desempenho: acurácia, relatório de classificação e matriz de confusão.

3. **Gráficos**  
   - Distribuição do estresse.  
   - Correlação de variáveis com o nível de estresse.  
   - Visualizações detalhadas de relações entre variáveis específicas (ex: ansiedade e pressão arterial).

4. **Grupos**  
   - Agrupa variáveis em categorias: Saúde Mental, Saúde Física, Ambiente, Acadêmico e Social.  
   - Calcula correlação média de cada grupo com o nível de estresse e gera gráficos de comparação.

5. **Previsão**  
   - Permite ao usuário inserir valores personalizados para as variáveis.  
   - Realiza previsão do nível de estresse (Baixo, Médio, Alto) com probabilidades associadas.

---

## Tecnologias Utilizadas

- **Streamlit** – Interface web interativa  
- **Pandas** – Manipulação de dados  
- **KaggleHub** – Fonte/armazenamento de datasets  
- **Matplotlib** – Visualização de dados  
- **Seaborn** – Visualização de dados  
- **Scikit-learn** – Machine Learning (Logistic Regression, Pipeline, StandardScaler)  
- **SHAP** – Explicação de modelos de Machine Learning  

---

## Estrutura do Projeto

    ├── app.py # Código principal do Streamlit
    ├── experimento.py # Experimentos e testes adicionais
    ├── data.csv # Dataset utilizado
    ├── README.md # Documentação do projeto
    └── requirements.txt # Dependências do projeto