# -*- coding: utf-8 -*-
"""
Regressão linear simples por município
--------------------------------------
Modelo: desmatado = β0 + β1 * X + ε

Saídas (sempre):
- regressao_simples_por_municipio.csv        (TODAS as variáveis, empilhado)
- regressao_simples_por_municipio.xlsx       (aba 'empilhado' com TODAS)
- regressao_simples_pivot.xlsx               (β1 por variável em colunas)

Saídas por variável (sempre, 1 arquivo por variável):
- regressao_simples_por_municipio__<variavel>.csv

Gráficos:
- plots_coerentes_com_tabela/<variavel>/<municipio>.png  (dispersão + reta, y=desmatado, x=variável)
- plots_slopes_por_variavel/slope_<variavel>_por_municipio.csv/.png (barras β1)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt

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

# Se quiser focar num gráfico específico durante os prints, pode usar:
INDEX_VAR = "valor_agropecuaria"   # "ALL" também funciona, mas as saídas agora SEMPRE cobrem todas as variáveis

SAIDA_BASE  = "regressao_simples_por_municipio"
SAIDA_PIVOT = "regressao_simples_pivot.xlsx"

# Pastas para salvar os gráficos
PLOT_DIR_TABELA = "plots_coerentes_com_tabela"  # X=variável, Y=desmatado (reta)
PLOT_DIR_SLOPE  = "plots_slopes_por_variavel"   # barras de β1 por município


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
    y = desmatado, x = variável X
    Retorna: slope (b1), intercept (b0), r2, p_value, std_err, n
    """
    d = pd.DataFrame({"y": y, "x": x}).dropna()
    n = len(d)
    if n < 2 or d["x"].std() == 0:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan,
                "p_value": np.nan, "std_err": np.nan, "n": int(n)}

    if HAS_SM:
        X = sm.add_constant(d["x"].values)
        model = sm.OLS(d["y"].values, X).fit()
        intercept = float(model.params[0])
        slope     = float(model.params[1])
        r2        = float(model.rsquared)
        p_value   = float(model.pvalues[1]) if model.pvalues.shape[0] > 1 else np.nan
        std_err   = float(model.bse[1])     if model.bse.shape[0] > 1 else np.nan
        return {"slope": slope, "intercept": intercept, "r2": r2,
                "p_value": p_value, "std_err": std_err, "n": int(n)}
    else:
        b1, b0 = np.polyfit(d["x"].values, d["y"].values, 1)
        y_hat  = b1 * d["x"].values + b0
        ss_res = np.sum((d["y"].values - y_hat) ** 2)
        ss_tot = np.sum((d["y"].values - np.mean(d["y"].values)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {"slope": float(b1), "intercept": float(b0), "r2": float(r2),
                "p_value": np.nan, "std_err": np.nan, "n": int(n)}


def roda_para_variavel(df, varx):
    linhas = []
    for mun, dmun in df.groupby("municipio"):
        # orientação da sua tabela: y = desmatado, x = varx
        res = regressao_simples_xy(dmun[VAR_Y], dmun[varx])
        row = {"municipio": mun, "variavel": varx}
        row.update(res)
        linhas.append(row)
    out = pd.DataFrame(linhas).sort_values("municipio").reset_index(drop=True)
    return out


# ===================== GRÁFICOS =====================
def plot_coerente_com_tabela(df, varx, outdir=PLOT_DIR_TABELA):
    """
    y = desmatado, x = varx. Desenha pontos (x, y) + reta ajustada.
    """
    import os
    subdir = os.path.join(outdir, varx)
    os.makedirs(subdir, exist_ok=True)

    for mun, dmun in df.groupby("municipio"):
        d = dmun[[varx, "desmatado"]].dropna().copy()
        if len(d) < 2 or d[varx].std() == 0 or d["desmatado"].std() == 0:
            continue

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


def grafico_barras_slope(df_resultados, varx, outdir=PLOT_DIR_SLOPE):
    """
    df_resultados: colunas ['municipio','variavel','slope',...]
    plota barras de β1 por município e salva CSV/PNG.
    """
    import os
    os.makedirs(outdir, exist_ok=True)

    d = df_resultados[df_resultados["variavel"] == varx][["municipio", "slope"]].copy()
    d = d.dropna().sort_values("slope", ascending=False).reset_index(drop=True)

    # CSV por variável (barras)
    csv_path = os.path.join(outdir, f"slope_{varx}_por_municipio.csv")
    d.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Plot
    plt.figure(figsize=(10, max(4, 0.35 * len(d))))
    plt.barh(d["municipio"], d["slope"])
    plt.gca().invert_yaxis()
    plt.xlabel("Coeficiente angular (β1)")
    plt.title(f"β1 por município — modelo: desmatado ~ {varx}")

    for i, v in enumerate(d["slope"].values):
        plt.text(v + (abs(v)*0.02 if v != 0 else 0.01), i, f"{v:.4g}",
                 va="center", ha="left" if v >= 0 else "right")

    plt.tight_layout()
    png_path = os.path.join(outdir, f"slope_{varx}_por_municipio.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"[β1] Gráfico salvo: {png_path}")
    print(f"[β1] CSV salvo:     {csv_path}")


# ===================== MAIN =====================
def main():
    df = carrega()

    # sanity check
    esperadas = {"ano", "municipio", VAR_Y, *VAR_XS}
    faltantes = esperadas.difference(df.columns)
    if faltantes:
        raise RuntimeError(f"Faltam colunas na base: {faltantes}")

    # ---------- SEMPRE calcular TODAS as variáveis ----------
    frames_all = [roda_para_variavel(df, v) for v in VAR_XS]
    df_full_all = pd.concat(frames_all, ignore_index=True)

    # 1) Arquivo principal (TODAS empilhadas)
    df_full_all.to_csv(f"{SAIDA_BASE}.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(f"{SAIDA_BASE}.xlsx", engine="openpyxl") as xw:
        df_full_all.to_excel(xw, index=False, sheet_name="empilhado")

    # 2) Pivot com β1 por variável
    piv = df_full_all.pivot(index="municipio", columns="variavel", values="slope").reset_index()
    with pd.ExcelWriter(SAIDA_PIVOT, engine="openpyxl") as xw:
        piv.to_excel(xw, index=False, sheet_name="slope_por_variavel")

    # 3) CSV separado para CADA variável (pedido novo)
    #    Ex.: regressao_simples_por_municipio__valor_agropecuaria.csv
    for v in VAR_XS:
        df_v = df_full_all[df_full_all["variavel"] == v].copy()
        df_v.to_csv(f"{SAIDA_BASE}__{v}.csv", index=False, encoding="utf-8-sig")

    # 4) Gráficos (reta) e barras β1 — para todas as variáveis
    for v in VAR_XS:
        plot_coerente_com_tabela(df, v, outdir=PLOT_DIR_TABELA)
        grafico_barras_slope(df_full_all, v, outdir=PLOT_DIR_SLOPE)

    # Prints de amostra
    pd.set_option("display.float_format", "{:.6f}".format)
    print("\n=== Amostra (empilhado de TODAS as variáveis) ===")
    print(df_full_all.head().to_string(index=False))
    print("\n=== Slope por variável (pivoteado) ===")
    print(piv.head().to_string(index=False))

    print("\nArquivos individuais por variável gerados para:")
    for v in VAR_XS:
        print(f" - {SAIDA_BASE}__{v}.csv")


if __name__ == "__main__":
    main()
