import pandas as pd
import numpy as np
import psycopg2
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error

# 1. Conecta ao banco de dados
conn = psycopg2.connect(
    dbname="postgres",
    user="postgres.roziechzdpxxdtzlkaep",
    password="Jj20134849@@@",
    host="aws-0-sa-east-1.pooler.supabase.com",
    port="5432"
)

# 2. Consulta os dados da view
query = """
SELECT ano, id_municipio_nome, desmatado, valor_agropecuaria, pib_per_capita, 
       valor_industria, valor_administracao_publica, amazonia_legal_bin
FROM view_dados_completos
WHERE ano >= 2010 AND ano <= 2021
"""
df = pd.read_sql_query(query, conn)
conn.close()

# 3. Lista para armazenar os resultados
resultados = []

# 4. Loop por município
for municipio in df["id_municipio_nome"].unique():
    df_mun = df[df["id_municipio_nome"] == municipio].dropna()
    if len(df_mun) < 5:
        continue

    try:
        # Variáveis explicativas
        X = df_mun[["ano", "valor_agropecuaria", "pib_per_capita", 
                    "valor_industria", "valor_administracao_publica", "amazonia_legal_bin"]]
        y = df_mun["desmatado"]

        # Modelo principal com RidgeCV
        modelo = RidgeCV(alphas=np.logspace(-2, 4, 100), cv=3).fit(X, y)
        y_pred = modelo.predict(X)

        # Modelos auxiliares para prever crescimento das variáveis
        modelo_agro = RidgeCV().fit(df_mun[["ano"]], df_mun["valor_agropecuaria"])
        modelo_pib = RidgeCV().fit(df_mun[["ano"]], df_mun["pib_per_capita"])
        modelo_ind = RidgeCV().fit(df_mun[["ano"]], df_mun["valor_industria"])
        modelo_adm = RidgeCV().fit(df_mun[["ano"]], df_mun["valor_administracao_publica"])
        amazonia_bin = df_mun["amazonia_legal_bin"].iloc[-1]

        # Últimos valores
        desmatado_2020 = df_mun[df_mun["ano"] == 2020]["desmatado"].values[0] if 2020 in df_mun["ano"].values else None
        desmatado_2021 = df_mun[df_mun["ano"] == 2021]["desmatado"].values[0]

        # Previsões para 2022–2024
        anos_futuros = pd.DataFrame({"ano": [2022, 2023, 2024]})
        anos_futuros["valor_agropecuaria"] = modelo_agro.predict(anos_futuros[["ano"]])
        anos_futuros["pib_per_capita"] = modelo_pib.predict(anos_futuros[["ano"]])
        anos_futuros["valor_industria"] = modelo_ind.predict(anos_futuros[["ano"]])
        anos_futuros["valor_administracao_publica"] = modelo_adm.predict(anos_futuros[["ano"]])
        anos_futuros["amazonia_legal_bin"] = amazonia_bin

        # Previsão de desmatamento
        X_futuro = anos_futuros[["ano", "valor_agropecuaria", "pib_per_capita",
                                 "valor_industria", "valor_administracao_publica", "amazonia_legal_bin"]]
        anos_futuros["desmatado_previsto"] = modelo.predict(X_futuro)

        # Correção de tendência crescente (se 2020 < 2021)
        if desmatado_2020 is not None and desmatado_2021 > desmatado_2020:
            for i in range(len(anos_futuros)):
                if anos_futuros["desmatado_previsto"].iloc[i] < desmatado_2021:
                    anos_futuros.loc[i, "desmatado_previsto"] = desmatado_2021 + (i + 1) * 50  # crescimento mínimo

        # Impactos das variáveis
        coef = modelo.coef_
        anos_futuros["impacto_ano"] = coef[0] * anos_futuros["ano"]
        anos_futuros["impacto_agro"] = coef[1] * anos_futuros["valor_agropecuaria"]
        anos_futuros["impacto_pib"] = coef[2] * anos_futuros["pib_per_capita"]
        anos_futuros["impacto_ind"] = coef[3] * anos_futuros["valor_industria"]
        anos_futuros["impacto_adm"] = coef[4] * anos_futuros["valor_administracao_publica"]
        anos_futuros["impacto_amazonia"] = coef[5] * anos_futuros["amazonia_legal_bin"]

        # Avaliação do modelo
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)

        # Armazena os resultados
        resultados.append({
            "municipio": municipio,
            "desmatamento_2022": anos_futuros["desmatado_previsto"].iloc[0],
            "desmatamento_2024": anos_futuros["desmatado_previsto"].iloc[2],
            "crescimento_desmatamento": anos_futuros["desmatado_previsto"].iloc[2] - anos_futuros["desmatado_previsto"].iloc[0],
            "r2_score": round(r2, 2),
            "mse": round(mse, 2),
            "impacto_ano_2024": anos_futuros["impacto_ano"].iloc[2],
            "impacto_agro_2024": anos_futuros["impacto_agro"].iloc[2],
            "impacto_pib_2024": anos_futuros["impacto_pib"].iloc[2],
            "impacto_ind_2024": anos_futuros["impacto_ind"].iloc[2],
            "impacto_adm_2024": anos_futuros["impacto_adm"].iloc[2],
            "impacto_amazonia": anos_futuros["impacto_amazonia"].iloc[2],
        })

    except Exception as e:
        print(f"Erro em {municipio}: {e}")
        continue

# 5. Resultado final
df_resultados = pd.DataFrame(resultados).sort_values(by="desmatamento_2024", ascending=False)
pd.set_option("display.float_format", "{:.2f}".format)
pd.set_option("display.max_columns", None)
print(df_resultados.to_string(index=False))
df_resultados.to_excel("previsao_desmatamento_2022_2024.xlsx", index=False)
print("Arquivo Excel salvo como 'previsao_desmatamento_2022_2024.xlsx'")
df_resultados.to_csv("previsao_desmatamento_2022__2024.csv", index=False, encoding="utf-8-sig")
print("Arquivo CSV salvo como 'previsao_desmatamento_2022_2024.csv'")