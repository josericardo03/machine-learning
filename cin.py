# -*- coding: utf-8 -*-
"""
Regressão linear simples por município (modo normal e invertido)
----------------------------------------------------------------
Modelos por município para cada variável X em:
    X ∈ {valor_agropecuaria, pib_per_capita, valor_industria, valor_administracao_publica}

A) Modo "normal"  : desmatado = b0 + b1 * X + e
B) Modo "invertido": X         = b0 + b1 * desmatado + e  <<< útil p/ "quanto o valor cresce por 1.000 ha"

Saídas (sempre):
- regressao_NORMAL_por_municipio_all.csv/.xlsx       (todas as variáveis empilhadas)
- regressao_INVERTIDA_por_municipio_all.csv/.xlsx    (todas as variáveis empilhadas, com delta_y_por_1000ha)
- regressao_NORMAL__<variavel>.csv
- regressao_INVERTIDA__<variavel>.csv
- regressao_slopes_NORMAL_pivot.xlsx                 (β1 normal em colunas)
- regressao_slopes_INVERTIDA_pivot.xlsx              (β1 invertido em colunas)

Gráficos (sempre, por variável):
- plots_invertida/<variavel>/<municipio>.png         (X ~ desmatado, dispersão + reta)
- plots_slopes_invertida/slope_<variavel>_por_municipio.png (barras β1 invertido)
- plots_slopes_invertida/slope_per_1000ha_<variavel>_por_municipio.png (barras β1*1000)

Requisitos: pandas, numpy, psycopg2, matplotlib, (opcional) statsmodels
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt

# statsmodels para métricas completas; se não tiver, cai no fallback
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

# Pastas para salvar os gráficos
PLOT_DIR_INVERTIDA = "plots_invertida"                 # dispersão+reta: y=X, x=desmatado
PLOT_DIR_BARS_INV  = "plots_slopes_invertida"          # barras β1 invertido (e por 1000 ha)

# Nomes de arquivos principais
SAIDA_NORMAL_ALL_CSV  = "regressao_NORMAL_por_municipio_all.csv"
SAIDA_NORMAL_ALL_XLSX = "regressao_NORMAL_por_municipio_all.xlsx"
SAIDA_INV_ALL_CSV     = "regressao_INVERTIDA_por_municipio_all.csv"
SAIDA_INV_ALL_XLSX    = "regressao_INVERTIDA_por_municipio_all.xlsx"
SAIDA_PIVOT_NORMAL    = "regressao_slopes_NORMAL_pivot.xlsx"
SAIDA_PIVOT_INV       = "regressao_slopes_INVERTIDA_pivot.xlsx"


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
def regressao_y_on_x(y, x):
    """
    Regressão linear simples y = b0 + b1*x + e.
    Retorna dict: slope (b1), intercept (b0), r2, p_value, std_err, n
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


def roda_NORMAL_por_variavel(df, varx):
    """
    Modelo "normal": desmatado = b0 + b1*varx + e
    """
    linhas = []
    for mun, dmun in df.groupby("municipio"):
        res = regressao_y_on_x(dmun["desmatado"], dmun[varx])
        row = {"municipio": mun, "variavel": varx}
        row.update(res)
        linhas.append(row)
    return pd.DataFrame(linhas).sort_values("municipio").reset_index(drop=True)


def roda_INVERTIDA_por_variavel(df, varx):
    """
    Modelo "invertido": varx = b0 + b1*desmatado + e
    Adiciona delta_y_por_1000ha = b1 * 1000 (interpretação direta)
    """
    linhas = []
    for mun, dmun in df.groupby("municipio"):
        res = regressao_y_on_x(dmun[varx], dmun["desmatado"])
        row = {"municipio": mun, "variavel": varx}
        row.update(res)
        # interpretação: quanto o valor (y) varia por +1000 ha de desmatado (x)
        row["delta_y_por_1000ha"] = res["slope"] * 1000 if pd.notna(res["slope"]) else np.nan
        linhas.append(row)
    return pd.DataFrame(linhas).sort_values("municipio").reset_index(drop=True)


# ============== GRÁFICOS (INVERTIDO) ==============
def plot_invertida_por_municipio(df, varx, outdir=PLOT_DIR_INVERTIDA):
    """
    Dispersão + reta para o modelo INVERTIDO: y=varx, x=desmatado.
    """
    subdir = os.path.join(outdir, varx)
    os.makedirs(subdir, exist_ok=True)

    for mun, dmun in df.groupby("municipio"):
        d = dmun[["desmatado", varx]].dropna().copy()
        if len(d) < 2 or d["desmatado"].std() == 0 or d[varx].std() == 0:
            continue

        fit = regressao_y_on_x(d[varx], d["desmatado"])  # y=varx, x=desmatado
        b1, b0 = fit["slope"], fit["intercept"]

        x = d["desmatado"].to_numpy()
        y = d[varx].to_numpy()
        y_hat = b0 + b1 * x

        plt.figure(figsize=(6, 4))
        plt.scatter(x, y, alpha=0.85)
        plt.plot(x, y_hat, linewidth=2)
        plt.xlabel("desmatado (ha)")
        plt.ylabel(varx.replace("_", " "))
        plt.title(
            f"{mun} — {varx} = {b0:.2f} + {b1:.6f}·desmatado\n"
            f"Δy/1000ha = {b1*1000:.2f} | R²={fit['r2']:.2f}, n={fit['n']}"
        )
        plt.tight_layout()
        fname = f"{mun.replace('/','-').replace(' ','_')}.png"
        plt.savefig(os.path.join(subdir, fname), dpi=150)
        plt.close()


def barras_slope_invertido(df_inv_all, varx, outdir=PLOT_DIR_BARS_INV):
    """
    Barras do slope (β1) do modelo invertido: varx ~ desmatado.
    Também salva versão por 1000 ha.
    """
    os.makedirs(outdir, exist_ok=True)

    d = df_inv_all[df_inv_all["variavel"] == varx][["municipio", "slope", "delta_y_por_1000ha"]].dropna()
    d = d.sort_values("slope", ascending=False).reset_index(drop=True)

    # CSV
    csv_path = os.path.join(outdir, f"slope_INVERTIDA_{varx}_por_municipio.csv")
    d.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Barras do slope
    plt.figure(figsize=(10, max(4, 0.35 * len(d))))
    plt.barh(d["municipio"], d["slope"])
    plt.gca().invert_yaxis()
    plt.xlabel("Coeficiente angular (β1) — invertido (var ~ desmatado)")
    plt.title(f"β1 (INVERTIDA) por município — {varx}")
    for i, v in enumerate(d["slope"].values):
        plt.text(v + (abs(v)*0.02 if v != 0 else 0.01), i, f"{v:.4g}",
                 va="center", ha="left" if v >= 0 else "right")
    plt.tight_layout()
    png_path = os.path.join(outdir, f"slope_INVERTIDA_{varx}_por_municipio.png")
    plt.savefig(png_path, dpi=150)
    plt.close()

    # Barras por 1000 ha
    d2 = d.sort_values("delta_y_por_1000ha", ascending=False).reset_index(drop=True)
    plt.figure(figsize=(10, max(4, 0.35 * len(d2))))
    plt.barh(d2["municipio"], d2["delta_y_por_1000ha"])
    plt.gca().invert_yaxis()
    plt.xlabel("Variação do valor por 1.000 ha (β1 × 1000)")
    plt.title(f"Δ{varx.replace('_',' ')} por 1.000 ha desmatados")
    for i, v in enumerate(d2["delta_y_por_1000ha"].values):
        plt.text(v + (abs(v)*0.02 if v != 0 else 0.01), i, f"{v:.4g}",
                 va="center", ha="left" if v >= 0 else "right")
    plt.tight_layout()
    png_path2 = os.path.join(outdir, f"slope_per_1000ha_INVERTIDA_{varx}_por_municipio.png")
    plt.savefig(png_path2, dpi=150)
    plt.close()

    print(f"[INV β1] Gráfico salvo: {png_path}")
    print(f"[INV 1k] Gráfico salvo: {png_path2}")
    print(f"[INV CSV] Salvo:       {csv_path}")


# ============== MAIN ==============
def main():
    df = carrega()

    # sanity check
    esperadas = {"ano", "municipio", VAR_Y, *VAR_XS}
    faltantes = esperadas.difference(df.columns)
    if faltantes:
        raise RuntimeError(f"Faltam colunas na base: {faltantes}")

    # ---------- MODELO NORMAL: desmatado ~ X ----------
    frames_norm = [roda_NORMAL_por_variavel(df, v) for v in VAR_XS]
    df_NORMAL_all = pd.concat(frames_norm, ignore_index=True)

    df_NORMAL_all.to_csv(SAIDA_NORMAL_ALL_CSV, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(SAIDA_NORMAL_ALL_XLSX, engine="openpyxl") as xw:
        df_NORMAL_all.to_excel(xw, index=False, sheet_name="empilhado")

    # também um CSV por variável (normal)
    for v in VAR_XS:
        df_NORMAL_all[df_NORMAL_all["variavel"] == v].to_csv(
            f"regressao_NORMAL__{v}.csv", index=False, encoding="utf-8-sig"
        )

    # pivot de slopes (normal)
    piv_norm = df_NORMAL_all.pivot(index="municipio", columns="variavel", values="slope").reset_index()
    with pd.ExcelWriter(SAIDA_PIVOT_NORMAL, engine="openpyxl") as xw:
        piv_norm.to_excel(xw, index=False, sheet_name="slope_NORMAL_por_variavel")

    # ---------- MODELO INVERTIDO: X ~ desmatado ----------
    frames_inv = [roda_INVERTIDA_por_variavel(df, v) for v in VAR_XS]
    df_INV_all = pd.concat(frames_inv, ignore_index=True)

    df_INV_all.to_csv(SAIDA_INV_ALL_CSV, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(SAIDA_INV_ALL_XLSX, engine="openpyxl") as xw:
        df_INV_all.to_excel(xw, index=False, sheet_name="empilhado")

    # também um CSV por variável (invertido)
    for v in VAR_XS:
        df_INV_all[df_INV_all["variavel"] == v].to_csv(
            f"regressao_INVERTIDA__{v}.csv", index=False, encoding="utf-8-sig"
        )

    # pivot de slopes (invertido)
    piv_inv = df_INV_all.pivot(index="municipio", columns="variavel", values="slope").reset_index()
    with pd.ExcelWriter(SAIDA_PIVOT_INV, engine="openpyxl") as xw:
        piv_inv.to_excel(xw, index=False, sheet_name="slope_INVERTIDA_por_variavel")

    # ---------- GRÁFICOS (INVERTIDOS) ----------
    for v in VAR_XS:
        plot_invertida_por_municipio(df, v, outdir=PLOT_DIR_INVERTIDA)
        barras_slope_invertido(df_INV_all, v, outdir=PLOT_DIR_BARS_INV)

    # Prints de amostra
    pd.set_option("display.float_format", "{:.6f}".format)
    print("\n=== Amostra NORMAL (empilhado) ===")
    print(df_NORMAL_all.head().to_string(index=False))
    print("\n=== Amostra INVERTIDA (empilhado) ===")
    print(df_INV_all.head().to_string(index=False))

    print("\nArquivos gerados (principais):")
    print(" -", SAIDA_NORMAL_ALL_CSV)
    print(" -", SAIDA_NORMAL_ALL_XLSX)
    print(" -", SAIDA_INV_ALL_CSV)
    print(" -", SAIDA_INV_ALL_XLSX)
    print(" -", SAIDA_PIVOT_NORMAL)
    print(" -", SAIDA_PIVOT_INV)
    for v in VAR_XS:
        print(f" - regressao_NORMAL__{v}.csv")
        print(f" - regressao_INVERTIDA__{v}.csv")


if __name__ == "__main__":
    main()
