# -*- coding: utf-8 -*-
"""
Regressão linear simples por município
--------------------------------------
Modelo: desmatado = β0 + β1 * X + ε

Escolha o índice X em INDEX_VAR (uma string) ou use "ALL" para rodar para todos:
    INDEX_VAR = "valor_agropecuaria"
    # ou
    INDEX_VAR = "ALL"

Saídas:
- regressao_simples_por_municipio.csv
- regressao_simples_por_municipio.xlsx
- (se INDEX_VAR == "ALL"): regressao_simples_pivot.xlsx  (β1 por variável em colunas)

Requisitos: pandas, numpy, psycopg2, statsmodels (opcional; se não houver, cai para numpy.polyfit sem p-valor/erro).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import psycopg2

# statsmodels para métricas completas; se não tiver, usamos fallback
try:
    import statsmodels.api as sm
    HAS_SM = True
except Exception:
    HAS_SM = False

# ============== CONFIG ==============
HOST     = "aws-0-sa-east-1.pooler.supabase.com"
PORT     = "5432"
DBNAME   = "postgres"
USER     = "postgres.roziechzdpxxdtzlkaep"
PASSWORD = "Jj20134849@@@"

TABELA_VIEW = "view_dados_completos"
ANOS_INI, ANOS_FIM = 2010, 2021

VAR_Y = "desmatado"
VAR_XS = [
    "valor_agropecuaria",
    "pib_per_capita",
    "valor_industria",
    "valor_administracao_publica",
]

# Escolha aqui: uma variável específica ou "ALL"
INDEX_VAR = "valor_agropecuaria"   # ex.: "pib_per_capita" | "ALL"

SAIDA_BASE = "regressao_simples_por_municipio"
SAIDA_PIVOT = "regressao_simples_pivot.xlsx"  # usado quando INDEX_VAR == "ALL"


# ============== DADOS ==============
def carrega():
    conn = psycopg2.connect(
        dbname=DBNAME, user=USER, password=PASSWORD, host=HOST, port=PORT
    )
    try:
        query = f"""
        SELECT
            ano,
            id_municipio_nome AS municipio,
            {VAR_Y} AS desmatado,
            {", ".join(VAR_XS)}
        FROM {TABELA_VIEW}
        WHERE ano >= {ANOS_INI} AND ano <= {ANOS_FIM}
        """
        df = pd.read_sql_query(query, conn)
        df["ano"] = df["ano"].astype(int)
        return df
    finally:
        conn.close()


# ============== REGRESSÃO ==============
def regressao_simples_xy(y, x):
    """
    Executa regressão linear simples y = b0 + b1*x + e.
    Retorna dict: slope (b1), intercept (b0), r2, p_value, std_err, n
    - Se statsmodels não estiver disponível, p_value e std_err serão NaN.
    """
    # Remove NaNs
    d = pd.DataFrame({"y": y, "x": x}).dropna()
    n = len(d)
    if n < 2 or d["x"].std() == 0:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan,
                "p_value": np.nan, "std_err": np.nan, "n": int(n)}

    if HAS_SM:
        X = sm.add_constant(d["x"].values)  # intercepto
        model = sm.OLS(d["y"].values, X).fit()
        # Parâmetros: [constante, slope]
        intercept = float(model.params[0])
        slope = float(model.params[1])
        r2 = float(model.rsquared)
        # p-valor e erro-padrão do slope:
        p_value = float(model.pvalues[1]) if model.pvalues.shape[0] > 1 else np.nan
        std_err = float(model.bse[1]) if model.bse.shape[0] > 1 else np.nan
        return {"slope": slope, "intercept": intercept, "r2": r2,
                "p_value": p_value, "std_err": std_err, "n": int(n)}
    else:
        # Fallback: numpy.polyfit (sem p-valor/erro)
        b1, b0 = np.polyfit(d["x"].values, d["y"].values, 1)
        y_hat = b1 * d["x"].values + b0
        ss_res = np.sum((d["y"].values - y_hat) ** 2)
        ss_tot = np.sum((d["y"].values - np.mean(d["y"].values)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {"slope": float(b1), "intercept": float(b0), "r2": float(r2),
                "p_value": np.nan, "std_err": np.nan, "n": int(n)}


def roda_para_variavel(df, varx):
    linhas = []
    for mun, dmun in df.groupby("municipio"):
        res = regressao_simples_xy(dmun[VAR_Y], dmun[varx])
        row = {"municipio": mun, "variavel": varx}
        row.update(res)
        linhas.append(row)
    out = pd.DataFrame(linhas).sort_values("municipio").reset_index(drop=True)
    return out


def main():
    df = carrega()

    # sanity check
    needed = {"ano", "municipio", VAR_Y, *VAR_XS}
    miss = needed.difference(df.columns)
    if miss:
        raise RuntimeError(f"Faltam colunas na base: {miss}")

    if INDEX_VAR == "ALL":
        # Empilha resultados para todas as variáveis
        frames = []
        for v in VAR_XS:
            frames.append(roda_para_variavel(df, v))
        df_full = pd.concat(frames, ignore_index=True)

        # Exporta empilhado
        df_full.to_csv(f"{SAIDA_BASE}.csv", index=False, encoding="utf-8-sig")
        with pd.ExcelWriter(f"{SAIDA_BASE}.xlsx", engine="openpyxl") as xw:
            df_full.to_excel(xw, index=False, sheet_name="empilhado")

        # Cria versão pivoteada com β1 (slope) por variável em colunas
        piv = df_full.pivot(index="municipio", columns="variavel", values="slope").reset_index()
        with pd.ExcelWriter(SAIDA_PIVOT, engine="openpyxl") as xw:
            piv.to_excel(xw, index=False, sheet_name="slope_por_variavel")

        # Print rápido
        print("\n=== Amostra (empilhado) ===")
        print(df_full.head().to_string(index=False))
        print("\n=== Slope por variável (pivoteado) ===")
        print(piv.head().to_string(index=False))

    else:
        if INDEX_VAR not in VAR_XS:
            raise ValueError(f"INDEX_VAR inválida: {INDEX_VAR}. Use uma de {VAR_XS} ou 'ALL'.")

        df_res = roda_para_variavel(df, INDEX_VAR)

        # Exporta
        df_res.to_csv(f"{SAIDA_BASE}.csv", index=False, encoding="utf-8-sig")
        with pd.ExcelWriter(f"{SAIDA_BASE}.xlsx", engine="openpyxl") as xw:
            df_res.to_excel(xw, index=False, sheet_name=INDEX_VAR)

        print(f"\n=== Regressão simples por município | X = {INDEX_VAR} ===")
        print(df_res.head().to_string(index=False))


if __name__ == "__main__":
    main()
