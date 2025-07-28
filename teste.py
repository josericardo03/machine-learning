import pandas as pd
import numpy as np
import psycopg2
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Configura estética dos gráficos
sns.set(style="whitegrid")

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
       valor_industria, valor_administracao_publica
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
        # Variáveis explicativas (sem ano)
        X = df_mun[["valor_agropecuaria", "pib_per_capita", 
                    "valor_industria", "valor_administracao_publica"]]
        y = df_mun["desmatado"]

        # Modelo principal
        modelo = RidgeCV(alphas=np.logspace(-2, 4, 100), cv=3).fit(X, y)
        y_pred = modelo.predict(X)

        # Modelos auxiliares para projeção das variáveis
        modelo_agro = RidgeCV().fit(df_mun[["ano"]], df_mun["valor_agropecuaria"])
        modelo_pib  = RidgeCV().fit(df_mun[["ano"]], df_mun["pib_per_capita"])
        modelo_ind  = RidgeCV().fit(df_mun[["ano"]], df_mun["valor_industria"])
        modelo_adm  = RidgeCV().fit(df_mun[["ano"]], df_mun["valor_administracao_publica"])

        # Últimos valores reais
        desmatado_2020 = df_mun[df_mun["ano"] == 2020]["desmatado"].values[0] if 2020 in df_mun["ano"].values else None
        desmatado_2021 = df_mun[df_mun["ano"] == 2021]["desmatado"].values[0]

        # 5. Previsões futuras (2022–2024)
        anos_futuros = pd.DataFrame({"ano": [2022, 2023, 2024]})
        anos_futuros["valor_agropecuaria"]          = modelo_agro.predict(anos_futuros[["ano"]])
        anos_futuros["pib_per_capita"]              = modelo_pib.predict(anos_futuros[["ano"]])
        anos_futuros["valor_industria"]             = modelo_ind.predict(anos_futuros[["ano"]])
        anos_futuros["valor_administracao_publica"] = modelo_adm.predict(anos_futuros[["ano"]])

        # Previsão de desmatamento
        X_futuro = anos_futuros[[
            "valor_agropecuaria",
            "pib_per_capita",
            "valor_industria",
            "valor_administracao_publica"
        ]]
        anos_futuros["desmatado_previsto"] = modelo.predict(X_futuro)

        # Correção de tendência crescente
        if desmatado_2020 is not None and desmatado_2021 > desmatado_2020:
            for i in range(len(anos_futuros)):
                if anos_futuros.loc[i, "desmatado_previsto"] < desmatado_2021:
                    anos_futuros.loc[i, "desmatado_previsto"] = desmatado_2021 + (i + 1) * 50

        # Impactos das variáveis (coeficiente × valor previsto)
        coef = modelo.coef_
        anos_futuros["impacto_agro"] = coef[0] * anos_futuros["valor_agropecuaria"]
        anos_futuros["impacto_pib"]  = coef[1] * anos_futuros["pib_per_capita"]
        anos_futuros["impacto_ind"]  = coef[2] * anos_futuros["valor_industria"]
        anos_futuros["impacto_adm"]  = coef[3] * anos_futuros["valor_administracao_publica"]

        # Métricas de ajuste
        r2  = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)

        # Armazena resultados
        resultados.append({
            "municipio": municipio,
            "desmatamento_2022":         anos_futuros.loc[anos_futuros.ano==2022, "desmatado_previsto"].values[0],
            "desmatamento_2024":         anos_futuros.loc[anos_futuros.ano==2024, "desmatado_previsto"].values[0],
            "crescimento_desmatamento":  anos_futuros.loc[anos_futuros.ano==2024, "desmatado_previsto"].values[0] - 
                                          anos_futuros.loc[anos_futuros.ano==2022, "desmatado_previsto"].values[0],
            "r2_score":                  round(r2, 2),
            "mse":                       round(mse, 2),
            "impacto_agro_2024":         anos_futuros.loc[anos_futuros.ano==2024, "impacto_agro"].values[0],
            "impacto_pib_2024":          anos_futuros.loc[anos_futuros.ano==2024, "impacto_pib"].values[0],
            "impacto_ind_2024":          anos_futuros.loc[anos_futuros.ano==2024, "impacto_ind"].values[0],
            "impacto_adm_2024":          anos_futuros.loc[anos_futuros.ano==2024, "impacto_adm"].values[0],
        })

    except Exception as e:
        print(f"Erro em {municipio}: {e}")
        continue

# 6. Resultado final em DataFrame
df_resultados = pd.DataFrame(resultados).sort_values(by="desmatamento_2024", ascending=False)

# Ajustes de exibição e salvamento
pd.set_option("display.float_format", "{:.2f}".format)
pd.set_option("display.max_columns", None)
print(df_resultados.to_string(index=False))
df_resultados.to_excel("previsao_desmatamento_sem_ano.xlsx", index=False)
df_resultados.to_csv("previsao_desmatamento_sem_ano.csv", index=False, encoding="utf-8-sig")

# --- 7. Gráficos para o município de Cáceres ---

# Filtra histórico de Cáceres
df_cac = df[df["id_municipio_nome"] == "Aquidauana"].dropna().sort_values("ano")

# Ajusta modelo principal para histórico de Cáceres
X_cac_hist = df_cac[["valor_agropecuaria", "pib_per_capita",  
                     "valor_industria", "valor_administracao_publica"]]
y_cac_hist = df_cac["desmatado"]
modelo_cac = RidgeCV(alphas=np.logspace(-2, 4, 100), cv=3).fit(X_cac_hist, y_cac_hist)

# Projeção para 2022–2024 em Cáceres
anos_fut_cac = pd.DataFrame({"ano": [2022, 2023, 2024]})
for var, mod in [
    ("valor_agropecuaria", RidgeCV().fit(df_cac[["ano"]], df_cac["valor_agropecuaria"])),
    ("pib_per_capita",        RidgeCV().fit(df_cac[["ano"]], df_cac["pib_per_capita"])),
    ("valor_industria",       RidgeCV().fit(df_cac[["ano"]], df_cac["valor_industria"])),
    ("valor_administracao_publica",
                              RidgeCV().fit(df_cac[["ano"]], df_cac["valor_administracao_publica"]))
]:
    anos_fut_cac[var] = mod.predict(anos_fut_cac[["ano"]])

# Previsão de desmatamento em Cáceres
X_cac_fut = anos_fut_cac[[
    "valor_agropecuaria", "pib_per_capita",
    "valor_industria", "valor_administracao_publica"
]]
anos_fut_cac["desmatado_previsto"] = modelo_cac.predict(X_cac_fut)

# 7.1 Série Temporal: Histórico vs Previsto
ts_cac = pd.concat([
    df_cac[["ano", "desmatado"]].rename(columns={"desmatado": "valor"}),
    anos_fut_cac[["ano", "desmatado_previsto"]].rename(columns={"desmatado_previsto": "valor"})
], ignore_index=True)
ts_cac["tipo"] = ["Histórico"] * len(df_cac) + ["Previsto"] * len(anos_fut_cac)

plt.figure(figsize=(10, 6))
for tipo, grp in ts_cac.groupby("tipo"):
    plt.plot(grp["ano"], grp["valor"], marker="o", label=tipo)
plt.title("Desmatamento em Cáceres: Histórico (2010–2021) e Previsão (2022–2024)")
plt.xlabel("Ano")
plt.ylabel("Área desmatada (ha)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("grafico_aquidauana_tempo.png")
plt.close()

# 7.2 Impactos das variáveis em 2024
coef_cac = modelo_cac.coef_
impactos_cac = {
    "Agropecuária": coef_cac[0] * anos_fut_cac.loc[anos_fut_cac.ano==2024, "valor_agropecuaria"].values[0],
    "PIB per cap.":  coef_cac[1] * anos_fut_cac.loc[anos_fut_cac.ano==2024, "pib_per_capita"].values[0],
    "Indústria":    coef_cac[2] * anos_fut_cac.loc[anos_fut_cac.ano==2024, "valor_industria"].values[0],
    "Admin. Públ.": coef_cac[3] * anos_fut_cac.loc[anos_fut_cac.ano==2024, "valor_administracao_publica"].values[0],
}

plt.figure(figsize=(8, 6))
plt.bar(impactos_cac.keys(), impactos_cac.values())
plt.title("Impacto das Variáveis no Desmatamento de Cáceres em 2024")
plt.ylabel("Impacto Estimado (ha)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("grafico_aquidauana_impactos.png")
plt.close()

print("Gráficos gerados para Aquidauana:\n"
      " - grafico_aquidauana_tempo.png\n"
      " - grafico_aquidauana_impactos.png")
