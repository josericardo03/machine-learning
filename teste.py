import pandas as pd
import numpy as np
import psycopg2
import cvxpy as cp
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
        previsao_bruta = modelo.predict(X_futuro)

        # PROGRAMAÇÃO CONVEXA: Ajusta previsões com restrições monótonas
        z = cp.Variable(3)  # Variáveis para 2022, 2023, 2024
        
        # Calcula tendência histórica para definir crescimento mínimo
        if len(df_mun) >= 3:
            # Calcula taxa de crescimento médio dos últimos 3 anos
            anos_recentes = df_mun.sort_values("ano").tail(3)
            if len(anos_recentes) >= 2:
                crescimento_medio = np.mean(np.diff(anos_recentes["desmatado"]))
                # Define crescimento mínimo como 20% da tendência histórica ou 1.5% do valor atual (muito suave)
                crescimento_minimo = max(crescimento_medio * 0.2, desmatado_2021 * 0.015)
            else:
                crescimento_minimo = desmatado_2021 * 0.015  # 1.5% mínimo (muito suave)
        else:
            crescimento_minimo = desmatado_2021 * 0.015  # 1.5% mínimo (muito suave)
        
        # Restrições de monotonicidade com crescimento muito suave
        restricoes = [
            z[0] >= desmatado_2021 + crescimento_minimo,  # 2022 >= 2021 + crescimento_mínimo
            z[1] >= z[0] + crescimento_minimo * 0.75,     # 2023 >= 2022 + crescimento_mínimo * 0.75 (cresce muito devagar)
            z[2] >= z[1] + crescimento_minimo * 0.5       # 2024 >= 2023 + crescimento_mínimo * 0.5 (cresce ainda mais devagar)
        ]
        
        # Função objetivo: minimizar distância das previsões originais + penalização por desvio
        # Peso maior para manter proximidade com previsão original
        peso_fidelidade = 1.0
        peso_suavizacao = 0.1
        
        objetivo = cp.Minimize(
            peso_fidelidade * cp.sum_squares(z - previsao_bruta) + 
            peso_suavizacao * cp.sum_squares(cp.diff(z))  # Suavização da trajetória
        )
        
        # Resolve o problema de otimização
        problema = cp.Problem(objetivo, restricoes)
        problema.solve()
        
        # Aplica as previsões ajustadas
        anos_futuros["desmatado_previsto"] = z.value

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
            "crescimento_minimo_anual":  round(crescimento_minimo, 2),
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
df_resultados.to_excel("previsao_desmatamento_convexo.xlsx", index=False)
df_resultados.to_csv("previsao_desmatamento_convexo.csv", index=False, encoding="utf-8-sig")

# --- 7. Gráficos para o município de Aquidauana ---

# Filtra histórico de Aquidauana
df_cac = df[df["id_municipio_nome"] == "Aquidauana"].dropna().sort_values("ano")

# Ajusta modelo principal para histórico de Aquidauana
X_cac_hist = df_cac[["valor_agropecuaria", "pib_per_capita",  
                     "valor_industria", "valor_administracao_publica"]]
y_cac_hist = df_cac["desmatado"]
modelo_cac = RidgeCV(alphas=np.logspace(-2, 4, 100), cv=3).fit(X_cac_hist, y_cac_hist)

# Projeção para 2022–2024 em Aquidauana
anos_fut_cac = pd.DataFrame({"ano": [2022, 2023, 2024]})
for var, mod in [
    ("valor_agropecuaria", RidgeCV().fit(df_cac[["ano"]], df_cac["valor_agropecuaria"])),
    ("pib_per_capita",        RidgeCV().fit(df_cac[["ano"]], df_cac["pib_per_capita"])),
    ("valor_industria",       RidgeCV().fit(df_cac[["ano"]], df_cac["valor_industria"])),
    ("valor_administracao_publica",
                              RidgeCV().fit(df_cac[["ano"]], df_cac["valor_administracao_publica"]))
]:
    anos_fut_cac[var] = mod.predict(anos_fut_cac[["ano"]])

# Previsão de desmatamento em Aquidauana
X_cac_fut = anos_fut_cac[[
    "valor_agropecuaria", "pib_per_capita",
    "valor_industria", "valor_administracao_publica"
]]
previsao_bruta_cac = modelo_cac.predict(X_cac_fut)

# PROGRAMAÇÃO CONVEXA para Aquidauana
desmatado_2021_cac = df_cac[df_cac["ano"] == 2021]["desmatado"].values[0]

# Calcula tendência histórica para definir crescimento mínimo
if len(df_cac) >= 3:
    # Calcula taxa de crescimento médio dos últimos 3 anos
    anos_recentes_cac = df_cac.sort_values("ano").tail(3)
    if len(anos_recentes_cac) >= 2:
        crescimento_medio_cac = np.mean(np.diff(anos_recentes_cac["desmatado"]))
        # Define crescimento mínimo como 20% da tendência histórica ou 1.5% do valor atual (muito suave)
        crescimento_minimo_cac = max(crescimento_medio_cac * 0.2, desmatado_2021_cac * 0.015)
    else:
        crescimento_minimo_cac = desmatado_2021_cac * 0.015  # 1.5% mínimo (muito suave)
else:
    crescimento_minimo_cac = desmatado_2021_cac * 0.015  # 1.5% mínimo (muito suave)

z_cac = cp.Variable(3)
restricoes_cac = [
    z_cac[0] >= desmatado_2021_cac + crescimento_minimo_cac,  # 2022 >= 2021 + crescimento_mínimo
    z_cac[1] >= z_cac[0] + crescimento_minimo_cac * 0.75,     # 2023 >= 2022 + crescimento_mínimo * 0.75
    z_cac[2] >= z_cac[1] + crescimento_minimo_cac * 0.5       # 2024 >= 2023 + crescimento_mínimo * 0.5
]

# Função objetivo com fidelidade e suavização
peso_fidelidade_cac = 1.0
peso_suavizacao_cac = 0.1

objetivo_cac = cp.Minimize(
    peso_fidelidade_cac * cp.sum_squares(z_cac - previsao_bruta_cac) + 
    peso_suavizacao_cac * cp.sum_squares(cp.diff(z_cac))  # Suavização da trajetória
)

problema_cac = cp.Problem(objetivo_cac, restricoes_cac)
problema_cac.solve()

anos_fut_cac["desmatado_previsto"] = z_cac.value

# 7.1 Série Temporal: Histórico vs Previsto
ts_cac = pd.concat([
    df_cac[["ano", "desmatado"]].rename(columns={"desmatado": "valor"}),
    anos_fut_cac[["ano", "desmatado_previsto"]].rename(columns={"desmatado_previsto": "valor"})
], ignore_index=True)
ts_cac["tipo"] = ["Histórico"] * len(df_cac) + ["Previsto"] * len(anos_fut_cac)

plt.figure(figsize=(10, 6))
for tipo, grp in ts_cac.groupby("tipo"):
    plt.plot(grp["ano"], grp["valor"], marker="o", label=tipo)
plt.title("Desmatamento em Aquidauana: Histórico (2010–2021) e Previsão (2022–2024)\nProgramação Convexa com Crescimento Muito Suave")
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
plt.title("Impacto das Variáveis no Desmatamento de Aquidauana em 2024\nProgramação Convexa com Crescimento Muito Suave")
plt.ylabel("Impacto Estimado (ha)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("grafico_aquidauana_impactos.png")
plt.close()

print("Gráficos gerados para Aquidauana:\n"
      " - grafico_aquidauana_tempo.png\n"
      " - grafico_aquidauana_impactos.png")
