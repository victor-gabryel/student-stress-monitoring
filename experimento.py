# =====================================
# IMPORTAÇÕES
# =====================================
import pandas as pd

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



# =====================================
# CARREGAR DADOS
# =====================================
df = pd.read_csv("data.csv")

X = df.drop("stress_level", axis=1)
y = df["stress_level"]



# =====================================
# DIVISÃO
# =====================================
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42
)



# =====================================
# MODELOS
# =====================================
modelos = {
    "Random Forest": Pipeline([
        ("modelo", RandomForestClassifier(n_estimators=200, random_state=42))
    ]),

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", LogisticRegression(max_iter=1000))
    ]),

    "Decision Tree": Pipeline([
        ("modelo", DecisionTreeClassifier(random_state=42))
    ])
}



# =====================================
# RESULTADOS
# =====================================
print("\n===== COMPARAÇÃO DE MODELOS =====")

melhor_modelo_nome = None
melhor_modelo = None
melhor_score = 0

for nome, pipeline in modelos.items():

    print(f"\n--- {nome} ---")

    # Validação cruzada
    scores = cross_val_score(pipeline, X, y, cv=5)

    media = scores.mean()
    desvio = scores.std()

    print(f"Acurácia média: {media:.3f}")
    print(f"Desvio padrão: {desvio:.3f}")

    # Seleção do melhor modelo
    if media > melhor_score:
        melhor_score = media
        melhor_modelo_nome = nome
        melhor_modelo = pipeline

    # Treino e teste
    pipeline.fit(X_treino, y_treino)
    y_pred = pipeline.predict(X_teste)

    print("\nAcurácia no teste:", round(accuracy_score(y_teste, y_pred), 3))
    print("Relatório:")
    print(classification_report(y_teste, y_pred))


# =====================================
# RESULTADO FINAL
# =====================================
print("\n=================================")
print(f"Melhor modelo: {melhor_modelo_nome}")
print(f"Acurácia média: {melhor_score:.3f}")
print("=================================")