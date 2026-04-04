# =====================================
# IMPORTAÇÕES
# =====================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



# =====================================
# CONFIG
# =====================================
st.set_page_config(page_title="Análise de Estresse", layout="wide")
st.title("Sistema de Análise de Estresse em Estudantes")

st.sidebar.write("Modelo utilizado: Logistic Regression")



# =====================================
# TRADUÇÃO
# =====================================
traducao = {
    "anxiety_level": "Ansiedade",
    "self_esteem": "Autoestima",
    "mental_health_history": "Histórico de Saúde Mental",
    "depression": "Depressão",
    "headache": "Dor de Cabeça",
    "blood_pressure": "Pressão Arterial",
    "sleep_quality": "Qualidade do Sono",
    "breathing_problem": "Problemas Respiratórios",
    "noise_level": "Nível de Ruído",
    "living_conditions": "Condições de Moradia",
    "safety": "Segurança",
    "basic_needs": "Necessidades Básicas",
    "academic_performance": "Desempenho Acadêmico",
    "study_load": "Carga de Estudo",
    "teacher_student_relationship": "Relação Professor-Aluno",
    "future_career_concerns": "Preocupação com a Carreira",
    "social_support": "Apoio Social",
    "peer_pressure": "Pressão dos Colegas",
    "extracurricular_activities": "Atividades Extracurriculares",
    "bullying": "Bullying",
    "stress_level": "Nível de Estresse"
}



# =====================================
# CARREGAR DADOS
# =====================================
@st.cache_data
def carregar():
    return pd.read_csv("data.csv")

df = carregar()



# =====================================
# MODELO
# =====================================
@st.cache_resource
def carregar_modelo():
    X = df.drop("stress_level", axis=1)
    y = df["stress_level"]

    modelo = Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", LogisticRegression(max_iter=1000))
    ])

    modelo.fit(X, y)
    return modelo

modelo = carregar_modelo()



# =====================================
# MENU
# =====================================
st.sidebar.title("Navegação")
opcao = st.sidebar.radio(
    "Selecione:",
    ["Visão Geral", "Modelo", "Gráficos", "Grupos"]
)



# =====================================
# VISÃO GERAL
# =====================================
if opcao == "Visão Geral":

    st.subheader("Dataset")
    st.dataframe(df.rename(columns=traducao))

    st.subheader("Estatísticas")
    st.dataframe(df.describe())



# =====================================
# MODELO
# =====================================
elif opcao == "Modelo":

    X = df.drop("stress_level", axis=1)
    y = df["stress_level"]

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo_temp = Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", LogisticRegression(max_iter=1000))
    ])

    modelo_temp.fit(X_treino, y_treino)

    y_pred = modelo_temp.predict(X_teste)

    st.subheader("Desempenho do Modelo")
    st.metric("Acurácia", f"{accuracy_score(y_teste, y_pred):.2f}")

    st.text("Relatório:")
    st.text(classification_report(y_teste, y_pred))

    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_teste, y_pred), annot=True, fmt="d", ax=ax)
    st.pyplot(fig)



# =====================================
# GRÁFICOS
# =====================================
elif opcao == "Gráficos":

    st.subheader("Distribuição do Estresse")

    fig1, ax1 = plt.subplots()
    sns.countplot(x=df["stress_level"], ax=ax1)
    st.pyplot(fig1)

    st.subheader("Correlação com Estresse")

    corr = df.corr()["stress_level"].sort_values(ascending=False)

    fig2, ax2 = plt.subplots()
    sns.barplot(x=corr.values[1:10], y=corr.index[1:10], ax=ax2)
    st.pyplot(fig2)

    st.subheader("Ansiedade vs Estresse")

    fig3, ax3 = plt.subplots()
    sns.scatterplot(x=df["anxiety_level"], y=df["stress_level"], ax=ax3)
    st.pyplot(fig3)

    st.subheader("Pressão Arterial vs Estresse")

    fig4, ax4 = plt.subplots()
    sns.boxplot(x=df["stress_level"], y=df["blood_pressure"], ax=ax4)
    st.pyplot(fig4)



# =====================================
# GRUPOS
# =====================================
elif opcao == "Grupos":

    grupos = {
        "Saúde Mental": ["anxiety_level", "depression", "self_esteem"],
        "Saúde Física": ["headache", "blood_pressure", "sleep_quality"],
        "Ambiente": ["noise_level", "living_conditions"],
        "Acadêmico": ["academic_performance", "study_load"],
        "Social": ["social_support", "bullying"]
    }

    df_grupos = df.copy()

    for nome, cols in grupos.items():
        df_grupos[nome] = df_grupos[cols].mean(axis=1)

    resultado = {
        g: df_grupos[[g, "stress_level"]].corr().iloc[0,1]
        for g in grupos
    }

    df_resultado = pd.DataFrame({
        "Grupo": resultado.keys(),
        "Correlação": resultado.values()
    }).sort_values(by="Correlação", ascending=False)

    st.dataframe(df_resultado)

    fig, ax = plt.subplots()
    sns.barplot(data=df_resultado, x="Correlação", y="Grupo", ax=ax)
    st.pyplot(fig)