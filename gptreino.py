import pandas as pd
import psycopg2
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

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

# 3. Filtra o município desejado
df_barao = df[df["id_municipio_nome"] == "Barão de Melgaço"].dropna()

# 4. Regressão múltipla para prever desmatamento
X = df_barao[["valor_agropecuaria", "pib_per_capita"]]
y = df_barao["desmatado"]
modelo_desmatamento = LinearRegression().fit(X, y)

# 5. Regressões individuais para projetar agro e PIB
modelo_agro = LinearRegression().fit(df_barao[["ano"]], df_barao["valor_agropecuaria"])
modelo_pib = LinearRegression().fit(df_barao[["ano"]], df_barao["pib_per_capita"])

# 6. Gera anos futuros com projeções
anos_futuros = pd.DataFrame({"ano": [2022, 2023, 2024]})
anos_futuros["valor_agropecuaria"] = modelo_agro.predict(anos_futuros[["ano"]])
anos_futuros["pib_per_capita"] = modelo_pib.predict(anos_futuros[["ano"]])

# 7. Previsões de desmatamento
anos_futuros["desmatado_previsto"] = modelo_desmatamento.predict(
    anos_futuros[["valor_agropecuaria", "pib_per_capita"]]
)
anos_futuros["impacto_agro"] = modelo_desmatamento.coef_[0] * anos_futuros["valor_agropecuaria"]
anos_futuros["impacto_pib"] = modelo_desmatamento.coef_[1] * anos_futuros["pib_per_capita"]

# 8. Mostra os resultados no console
print("\n📊 Previsões de Desmatamento (Bodoquena) para 2022–2024:")
print(anos_futuros[["ano", "valor_agropecuaria", "pib_per_capita", "desmatado_previsto", "impacto_agro", "impacto_pib"]])

# -------------------------------
# 🔽 Gráficos
sns.set(style="whitegrid")

# 🔹 Junta dados reais e previstos
df_real = df_barao[["ano", "desmatado"]].copy().rename(columns={"desmatado": "desmatado_previsto"})
df_real["tipo"] = "Real"
df_prev = anos_futuros[["ano", "desmatado_previsto"]].copy()
df_prev["tipo"] = "Previsto"
df_plot = pd.concat([df_real, df_prev])

# 1️⃣ Gráfico de Linha – Desmatamento Real vs. Previsto
plt.figure(figsize=(10, 5))
sns.lineplot(data=df_plot, x="ano", y="desmatado_previsto", hue="tipo", marker="o")
plt.title("Desmatamento Real e Previsto (Bodoquena)")
plt.ylabel("Área Desmatada (ha)")
plt.xlabel("Ano")
plt.legend(title="Tipo de dado")
plt.tight_layout()
plt.show()

# 2️⃣ Gráfico de Tendência – PIB e Agropecuária
plt.figure(figsize=(10, 5))
sns.lineplot(x=df_barao["ano"], y=df_barao["pib_per_capita"], label="PIB per capita", marker="o")
sns.lineplot(x=df_barao["ano"], y=df_barao["valor_agropecuaria"], label="Valor Agropecuário", marker="o")
sns.lineplot(x=anos_futuros["ano"], y=anos_futuros["pib_per_capita"], label="PIB (Projeção)", linestyle="--")
sns.lineplot(x=anos_futuros["ano"], y=anos_futuros["valor_agropecuaria"], label="Agro (Projeção)", linestyle="--")
plt.title("Evolução de PIB per capita e Valor Agropecuário")
plt.xlabel("Ano")
plt.ylabel("R$")
plt.legend()
plt.tight_layout()
plt.show()

# 3️⃣ Gráfico de Barras – Impacto individual das variáveis
df_imp = anos_futuros[["ano", "impacto_agro", "impacto_pib"]].melt(id_vars="ano", 
    var_name="variavel", value_name="impacto")
plt.figure(figsize=(8, 5))
sns.barplot(data=df_imp, x="ano", y="impacto", hue="variavel")
plt.title("Impacto de cada variável no Desmatamento Previsto")
plt.ylabel("Impacto (ha)")
plt.xlabel("Ano")
plt.legend(title="Variável")
plt.tight_layout()
plt.show()

# 4️⃣ Gráfico 3D – Dispersão
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_barao["valor_agropecuaria"], df_barao["pib_per_capita"], df_barao["desmatado"],
           label="Histórico", color='blue')
ax.scatter(anos_futuros["valor_agropecuaria"], anos_futuros["pib_per_capita"], anos_futuros["desmatado_previsto"],
           label="Previsto", color='red')
ax.set_xlabel("Valor Agropecuário")
ax.set_ylabel("PIB per capita")
ax.set_zlabel("Desmatamento")
ax.set_title("Desmatamento x PIB x Agropecuária")
ax.legend()
plt.tight_layout()
plt.show()
