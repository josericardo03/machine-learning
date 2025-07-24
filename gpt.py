import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression

# 🔐 Credenciais
usuario = "postgres.roziechzdpxxdtzlkaep"
senha = "Jj20134849%40%40%40"  # cuidado ao compartilhar publicamente
host = "aws-0-sa-east-1.pooler.supabase.com"
porta = 5432
banco = "postgres"

# 🔌 Conectando ao banco
conn_str = f"postgresql+psycopg2://{usuario}:{senha}@{host}:{porta}/{banco}"
engine = create_engine(conn_str)

# 🔎 Consulta à view
query = "SELECT ano, id_municipio_nome, desmatado, valor_agropecuaria, pib_per_capita FROM real"
df = pd.read_sql(query, engine)

# 🔍 Filtra Barão de Melgaço
df_barao = df[df["id_municipio_nome"] == "Barão de Melgaço"].dropna()

# 📊 Regressão linear múltipla
X = df_barao[["valor_agropecuaria", "pib_per_capita"]]
y = df_barao["desmatado"]

modelo = LinearRegression()
modelo.fit(X, y)

# 📈 Coeficientes
print(f"Intercepto (β₀): {modelo.intercept_}")
print(f"Coef. valor_agropecuaria (β₁): {modelo.coef_[0]}")
print(f"Coef. pib_per_capita (β₂): {modelo.coef_[1]}")

# 📅 Previsões para 2022–2024
futuro = pd.DataFrame({
    "ano": [2022, 2023, 2024],
    "valor_agropecuaria": [45000, 46000, 47000],
    "pib_per_capita": [16000, 17000, 18000]
})
futuro["desmatado_previsto"] = modelo.predict(futuro[["valor_agropecuaria", "pib_per_capita"]])
futuro["impacto_agro"] = modelo.coef_[0] * futuro["valor_agropecuaria"]
futuro["impacto_pib"] = modelo.coef_[1] * futuro["pib_per_capita"]

# 📋 Exibe resultado
print("\nPrevisões 2022–2024 com impacto das variáveis:")
print(futuro[["ano", "desmatado_previsto", "impacto_agro", "impacto_pib"]])