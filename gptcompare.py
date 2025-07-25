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

# 2. Consulta a base
query = "SELECT ano, id_municipio_nome, desmatado, valor_agropecuaria, pib_per_capita FROM real"
df = pd.read_sql_query(query, conn)
conn.close()

# 3. Inicializa resultados
resultados = []

# 4. Loop por município
for municipio in df["id_municipio_nome"].unique():
    df_mun = df[df["id_municipio_nome"] == municipio].dropna()
    if len(df_mun) < 5:
        continue

    try:
        # Modelos de regressão múltipla e temporal
        X = df_mun[["valor_agropecuaria", "pib_per_capita"]]
        y = df_mun["desmatado"]
        modelo_desmatamento = LinearRegression().fit(X, y)

        modelo_agro = LinearRegression().fit(df_mun[["ano"]], df_mun["valor_agropecuaria"])
        modelo_pib = LinearRegression().fit(df_mun[["ano"]], df_mun["pib_per_capita"])

        # Projeções para 2022–2024
        anos_futuros = pd.DataFrame({"ano": [2022, 2023, 2024]})
        anos_futuros["valor_agropecuaria"] = modelo_agro.predict(anos_futuros[["ano"]])
        anos_futuros["pib_per_capita"] = modelo_pib.predict(anos_futuros[["ano"]])
        anos_futuros["desmatado_previsto"] = modelo_desmatamento.predict(
            anos_futuros[["valor_agropecuaria", "pib_per_capita"]]
        )

        # Coeficientes de impacto
        coef_agro = modelo_desmatamento.coef_[0]
        coef_pib = modelo_desmatamento.coef_[1]

        # Impacto isolado de cada variável
        anos_futuros["impacto_agro"] = coef_agro * anos_futuros["valor_agropecuaria"]
        anos_futuros["impacto_pib"] = coef_pib * anos_futuros["pib_per_capita"]

        # Diferença de desmatamento entre 2022 e 2024
        crescimento = anos_futuros["desmatado_previsto"].iloc[-1] - anos_futuros["desmatado_previsto"].iloc[0]

        # Armazena resultados
        resultados.append({
            "municipio": municipio,
            "desmatamento_2022": anos_futuros["desmatado_previsto"].iloc[0],
            "desmatamento_2024": anos_futuros["desmatado_previsto"].iloc[-1],
            "crescimento_desmatamento": crescimento,
            "agro_2022": anos_futuros["valor_agropecuaria"].iloc[0],
            "agro_2024": anos_futuros["valor_agropecuaria"].iloc[-1],
            "pib_2022": anos_futuros["pib_per_capita"].iloc[0],
            "pib_2024": anos_futuros["pib_per_capita"].iloc[-1],
            "impacto_agro_2022": anos_futuros["impacto_agro"].iloc[0],
            "impacto_agro_2024": anos_futuros["impacto_agro"].iloc[-1],
            "impacto_pib_2022": anos_futuros["impacto_pib"].iloc[0],
            "impacto_pib_2024": anos_futuros["impacto_pib"].iloc[-1]
        })

    except Exception as e:
        print(f"Erro no município {municipio}: {e}")
        continue

# 5. Organiza e exibe o DataFrame final
df_resultados = pd.DataFrame(resultados).sort_values(by="crescimento_desmatamento", ascending=False)
pd.set_option("display.float_format", "{:.2f}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

print(df_resultados.to_string(index=False))