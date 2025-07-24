import pandas as pd
import psycopg2
from sklearn.linear_model import LinearRegression

# 1. Conecta ao banco
conn = psycopg2.connect(
    dbname="postgres",
    user="postgres.roziechzdpxxdtzlkaep",
    password="Jj20134849@@@",
    host="aws-0-sa-east-1.pooler.supabase.com",
    port="5432"
)

# 2. Consulta a view
query = "SELECT ano, id_municipio_nome, desmatado, valor_agropecuaria, pib_per_capita FROM real"
df = pd.read_sql_query(query, conn)
conn.close()

# 3. Filtra Barão de Melgaço
df_barao = df[df["id_municipio_nome"] == "Aquidauana"].dropna()

# 4. Regressão múltipla para prever desmatamento
X = df_barao[["valor_agropecuaria", "pib_per_capita"]]
y = df_barao["desmatado"]
modelo_desmatamento = LinearRegression().fit(X, y)

# 5. Regressões individuais para prever agro e pib com base no ano
modelo_agro = LinearRegression().fit(df_barao[["ano"]], df_barao["valor_agropecuaria"])
modelo_pib = LinearRegression().fit(df_barao[["ano"]], df_barao["pib_per_capita"])

# 6. Gera anos futuros e projeta agro e pib
anos_futuros = pd.DataFrame({"ano": [2022, 2023, 2024]})
anos_futuros["valor_agropecuaria"] = modelo_agro.predict(anos_futuros[["ano"]])
anos_futuros["pib_per_capita"] = modelo_pib.predict(anos_futuros[["ano"]])

# 7. Aplica o modelo de desmatamento nas projeções futuras
anos_futuros["desmatado_previsto"] = modelo_desmatamento.predict(
    anos_futuros[["valor_agropecuaria", "pib_per_capita"]]
)

# 8. Impacto individual
anos_futuros["impacto_agro"] = modelo_desmatamento.coef_[0] * anos_futuros["valor_agropecuaria"]
anos_futuros["impacto_pib"] = modelo_desmatamento.coef_[1] * anos_futuros["pib_per_capita"]

# 9. Mostra os resultados
print(anos_futuros[["ano", "valor_agropecuaria", "pib_per_capita", "desmatado_previsto", "impacto_agro", "impacto_pib"]])
