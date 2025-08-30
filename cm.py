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
import matplotlib.pyplot as plt  # ADIÇÃO

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

# Pastas para salvar os gráficos
PLOT_DIR = "plots_regressao_invertido"      # X=desmatado, Y=variável (mantido)
PLOT_DIR_TABELA = "plots_coerentes_com_tabela"  # X=variável, Y=desmatado (NOVO)


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
    d = pd.DataFrame({"y": y, "x": x}).dropna()
    n = len(d)
    if n < 2 or d["x"].std() == 0:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan,
                "p_value": np.nan, "std_err": np.nan, "n": int(n)}

    if HAS_SM:
        X = sm.add_constant(d["x"].values)  # intercepto
        model = sm.OLS(d["y"].values, X).fit()
        intercept = float(model.params[0])
        slope = float(model.params[1])
        r2 = float(model.rsquared)
        p_value = float(model.pvalues[1]) if model.pvalues.shape[0] > 1 else np.nan
        std_err = float(model.bse[1]) if model.bse.shape[0] > 1 else np.nan
        return {"slope": slope, "intercept": intercept, "r2": r2,
                "p_value": p_value, "std_err": std_err, "n": int(n)}
    else:
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
        # >>> ESTA É A MESMA ORIENTAÇÃO DA SUA TABELA: y = desmatado, x = varx
        res = regressao_simples_xy(dmun[VAR_Y], dmun[varx])
        row = {"municipio": mun, "variavel": varx}
        row.update(res)
        linhas.append(row)
    out = pd.DataFrame(linhas).sort_values("municipio").reset_index(drop=True)
    return out


# ===================== PLOTS (mantido) =====================
def regressao_var_em_desmatado(y_var, x_desmatado):
    """ Ajusta: y_var = b0 + b1 * x_desmatado + e (mantido) """
    return regressao_simples_xy(y=y_var, x=x_desmatado)


def plot_invertido_por_municipio(df, varx, outdir=PLOT_DIR):
    """
    Mantido: X=desmatado, Y=varx (reta de varx ~ desmatado).
    Não é o gráfico coerente com a tabela, mas deixamos para comparação.
    """
    import os
    subdir = os.path.join(outdir, varx)
    os.makedirs(subdir, exist_ok=True)

    for mun, dmun in df.groupby("municipio"):
        d = dmun[["desmatado", varx]].dropna().copy()
        if len(d) < 2 or d["desmatado"].std() == 0 or d[varx].std() == 0:
            continue

        fit = regressao_var_em_desmatado(d[varx], d["desmatado"])
        b1, b0 = fit["slope"], fit["intercept"]

        x = d["desmatado"].to_numpy()
        y = d[varx].to_numpy()
        y_hat = b0 + b1 * x

        plt.figure(figsize=(6,4))
        plt.scatter(x, y, alpha=0.85)
        plt.plot(x, y_hat, linewidth=2)
        plt.xlabel("desmatado (ha)")
        plt.ylabel(varx.replace("_"," "))
        plt.title(f"{mun} — {varx} = {b0:.2f} + {b1:.6f}·desmatado\nR²={fit['r2']:.2f}, n={fit['n']}")
        plt.tight_layout()
        fname = f"{mun.replace('/','-').replace(' ','_')}.png"
        plt.savefig(os.path.join(subdir, fname), dpi=150)
        plt.close()


# ===================== PLOT NOVO (COERENTE COM A TABELA) =====================
def plot_coerente_com_tabela(df, varx, outdir=PLOT_DIR_TABELA):
    """
    NOVO: gráfico coerente com a TABELA (y = desmatado, x = varx).
    Usa a MESMA regressão da sua tabela: desmatado ~ varx.
    Desenha: pontos (varx, desmatado) + reta: desmatado = b0 + b1*varx.
    """
    import os
    subdir = os.path.join(outdir, varx)
    os.makedirs(subdir, exist_ok=True)

    for mun, dmun in df.groupby("municipio"):
        d = dmun[[varx, "desmatado"]].dropna().copy()
        if len(d) < 2 or d[varx].std() == 0 or d["desmatado"].std() == 0:
            continue

        # Coeficientes na MESMA orientação da sua tabela
        fit = regressao_simples_xy(d["desmatado"], d[varx])  # y=desmatado, x=varx
        b1, b0 = fit["slope"], fit["intercept"]

        x = d[varx].to_numpy()
        y = d["desmatado"].to_numpy()
        y_hat = b0 + b1 * x

        plt.figure(figsize=(6,4))
        plt.scatter(x, y, alpha=0.85)
        plt.plot(x, y_hat, linewidth=2)
        plt.xlabel(varx.replace("_"," "))
        plt.ylabel("desmatado (ha)")
        plt.title(f"{mun} — desmatado = {b0:.2f} + {b1:.6f}·{varx}\nR²={fit['r2']:.2f}, n={fit['n']}")
        plt.tight_layout()
        fname = f"{mun.replace('/','-').replace(' ','_')}.png"
        plt.savefig(os.path.join(subdir, fname), dpi=150)
        plt.close()


def main():
    df = carrega()

    # sanity check
    needed = {"ano", "municipio", VAR_Y, *VAR_XS}
    miss = needed.difference(df.columns)
    if miss:
        raise RuntimeError(f"Faltam colunas na base: {miss}")

    if INDEX_VAR == "ALL":
        # Empilha resultados para todas as variáveis (mesma orientação da TABELA)
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

        # >>> GRÁFICO COERENTE COM A TABELA (X=var, Y=desmatado)
        for v in VAR_XS:
            plot_coerente_com_tabela(df, v, outdir=PLOT_DIR_TABELA)

        # (Opcional) Mantém também o gráfico invertido para comparação
        # for v in VAR_XS:
        #     plot_invertido_por_municipio(df, v, outdir=PLOT_DIR)

        import os
        print("Gráficos (coerentes com a TABELA) em:", os.path.abspath(PLOT_DIR_TABELA))

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

        # >>> GRÁFICO COERENTE COM A TABELA para a variável escolhida
        plot_coerente_com_tabela(df, INDEX_VAR, outdir=PLOT_DIR_TABELA)

        # (Opcional) Mantém também o invertido para comparação
        # plot_invertido_por_municipio(df, INDEX_VAR, outdir=PLOT_DIR)

        import os
        print("Gráficos (coerentes com a TABELA) em:",
              os.path.join(os.path.abspath(PLOT_DIR_TABELA), INDEX_VAR))


if __name__ == "__main__":
    main()
