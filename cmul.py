# -*- coding: utf-8 -*-
"""
Regressão linear MÚLTIPLA por município
---------------------------------------
Modelo: desmatado = β0 + β1*valor_agropecuaria + β2*pib_per_capita
                    + β3*valor_industria + β4*valor_administracao_publica + ε

Para cada município, o script calcula:
- Coeficientes (β), erro-padrão, p-valor
- R² e R² ajustado
- Betas padronizados (comparáveis)
- VIF por preditor

Saídas:
- regressao_multipla_por_municipio.csv  (empilhado: uma linha por município x variável)
- regressao_multipla_por_municipio.xlsx (aba 'empilhado' + abas de pivôs)
- vif_por_municipio.csv                 (diagnóstico de multicolinearidade)
- plots_betas_padronizados/<municipio>.png  (barras com betas padronizados)

Requisitos: pandas, numpy, psycopg2, statsmodels, openpyxl, matplotlib
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt

try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_SM = True
except Exception:
    HAS_SM = False

# ============== CONFIG ==============
HOST     = "aws-0-sa-east-1.pooler.supabase.com"
PORT     = "5432"
DBNAME   = "postgres"
USER     = "postgres.roziechzdpxxdtzlkaep"
PASSWORD = "Jj20134849@@@"  # em produção, use variável de ambiente

TABELA_VIEW = "view_dados_completos"
ANOS_INI, ANOS_FIM = 2010, 2021

VAR_Y = "desmatado"
VAR_XS = [
    "valor_agropecuaria",
    "pib_per_capita",
    "valor_industria",
    "valor_administracao_publica",
]

SAIDA_CSV   = "regressao_multipla_por_municipio.csv"
SAIDA_XLSX  = "regressao_multipla_por_municipio.xlsx"
SAIDA_VIF   = "vif_por_municipio.csv"
PLOT_DIR    = "plots_betas_padronizados"


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


# ============== UTIL ==============
def zscore(s):
    """Padroniza uma série (média 0, desvio 1) preservando NaNs."""
    m = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.nan, index=s.index)
    return (s - m) / sd

def compute_vif(X_design, columns):
    """Calcula VIF dado um design matrix com constante na primeira coluna."""
    # VIF é calculado SEM a constante; statsmodels espera só os preditores
    if X_design.shape[1] <= 2:  # const + 1 preditor -> VIF não faz sentido
        return {columns[0]: np.nan}
    X_no_const = X_design[:, 1:]  # remove a constante
    vifs = {}
    for i, col in enumerate(columns):
        try:
            vifs[col] = float(variance_inflation_factor(X_no_const, i))
        except Exception:
            vifs[col] = np.nan
    return vifs

def betas_padronizados_from_raw(betas_raw, y, X):
    """
    Converte coeficientes brutos em betas padronizados:
    β_std_j = β_raw_j * (sd(X_j) / sd(Y))
    - betas_raw: array-like (inclui intercepto em [0])
    - retorna dict {var: beta_std} (sem intercepto)
    """
    sd_y = y.std(ddof=0)
    betas_std = {}
    for j, var in enumerate(X.columns):
        sd_xj = X[var].std(ddof=0)
        if sd_y == 0 or sd_xj == 0 or np.isnan(sd_y) or np.isnan(sd_xj):
            betas_std[var] = np.nan
        else:
            betas_std[var] = float(betas_raw[j+1] * (sd_xj / sd_y))  # +1 por conta do intercepto
    return betas_std


# ============== REG MULTIPLA ==============
def regressao_multipla_municipio(df_mun, mun_name):
    """
    Ajusta OLS múltipla para um município:
      y = desmatado, X = VAR_XS (todas)
    Retorna:
      - linhas: lista de dicts (1 por variável) com coef, erro, p, beta_std, R², R²adj, n
      - vif_dict: {var: VIF}
    """
    d = df_mun[["ano", "municipio", VAR_Y] + VAR_XS].dropna().copy()
    # precisa ter pelo menos k+2 pontos (k preditores + intercepto) e variância > 0
    if len(d) < (len(VAR_XS) + 2):
        return [], {}

    # checagem de variância
    if d[VAR_Y].std() == 0 or any(d[x].std() == 0 for x in VAR_XS):
        return [], {}

    if not HAS_SM:
        raise RuntimeError("statsmodels não disponível. Instale 'statsmodels' para OLS múltipla.")

    y = d[VAR_Y].astype(float)
    X = d[VAR_XS].astype(float)

    # Design matrix com constante
    Xc = sm.add_constant(X.values)
    model = sm.OLS(y.values, Xc).fit()

    # VIF
    vif_dict = compute_vif(Xc, VAR_XS)

    # Betas padronizados (a partir dos coeficientes brutos)
    betas_std = betas_padronizados_from_raw(model.params, y, X)

    linhas = []
    for j, var in enumerate(VAR_XS, start=1):  # pular intercepto (índice 0)
        linhas.append({
            "municipio": mun_name,
            "variavel": var,
            "coeficiente": float(model.params[j]),
            "std_err": float(model.bse[j]),
            "p_value": float(model.pvalues[j]),
            "beta_padronizado": betas_std[var],
            "r2": float(model.rsquared),
            "r2_ajustado": float(model.rsquared_adj),
            "n": int(len(d)),
            "intercepto": float(model.params[0]),
        })

    return linhas, vif_dict


# ============== GRÁFICO (betas padronizados) ==============
def plot_barras_betas_padronizados(df_coef, municipio, outdir=PLOT_DIR):
    os.makedirs(outdir, exist_ok=True)
    d = df_coef[df_coef["municipio"] == municipio][["variavel", "beta_padronizado"]].copy()
    d = d.dropna().sort_values("beta_padronizado", ascending=True)

    if d.empty:
        return

    plt.figure(figsize=(8, 4.5))
    plt.barh(d["variavel"].str.replace("_", " "), d["beta_padronizado"])
    for i, v in enumerate(d["beta_padronizado"].values):
        plt.text(v + (0.02 if v >= 0 else -0.02), i, f"{v:.3f}",
                 va="center", ha="left" if v >= 0 else "right")
    plt.axvline(0, linewidth=1)
    plt.title(f"Betas padronizados — {municipio}")
    plt.xlabel("Beta padronizado (comparável entre variáveis)")
    plt.tight_layout()
    fpath = os.path.join(outdir, f"{municipio.replace('/','-').replace(' ','_')}.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"[plot] {fpath}")


# ============== MAIN ==============
def main():
    df = carrega()

    # sanity check
    esperadas = {"ano", "municipio", VAR_Y, *VAR_XS}
    faltantes = esperadas.difference(df.columns)
    if faltantes:
        raise RuntimeError(f"Faltam colunas na base: {faltantes}")
    if not HAS_SM:
        raise RuntimeError("Este script requer 'statsmodels'. Instale e rode novamente.")

    linhas_all = []
    vif_rows = []

    for mun, dmun in df.groupby("municipio"):
        linhas, vif = regressao_multipla_municipio(dmun, mun)
        if not linhas:
            # registro vazio para saber que não foi possível ajustar
            vif_rows.append({"municipio": mun, **{v: np.nan for v in VAR_XS}})
            continue

        linhas_all.extend(linhas)
        vif_rows.append({"municipio": mun, **vif})

    if not linhas_all:
        raise RuntimeError("Nenhuma regressão pôde ser ajustada (dados insuficientes/variância zero).")

    df_out = pd.DataFrame(linhas_all).sort_values(["municipio", "variavel"]).reset_index(drop=True)
    df_vif = pd.DataFrame(vif_rows).sort_values("municipio").reset_index(drop=True)

    # Salvas principais
    df_out.to_csv(SAIDA_CSV, index=False, encoding="utf-8-sig")
    df_vif.to_csv(SAIDA_VIF, index=False, encoding="utf-8-sig")

    # Pivôs úteis
    piv_coef   = df_out.pivot(index="municipio", columns="variavel", values="coeficiente").reset_index()
    piv_beta   = df_out.pivot(index="municipio", columns="variavel", values="beta_padronizado").reset_index()
    piv_pvalor = df_out.pivot(index="municipio", columns="variavel", values="p_value").reset_index()
    piv_r2     = df_out[["municipio", "r2", "r2_ajustado"]].drop_duplicates()

    # XLSX com abas
    with pd.ExcelWriter(SAIDA_XLSX, engine="openpyxl") as xw:
        df_out.to_excel(xw, index=False, sheet_name="empilhado")
        piv_coef.to_excel(xw, index=False, sheet_name="coeficientes")
        piv_beta.to_excel(xw, index=False, sheet_name="betas_padronizados")
        piv_pvalor.to_excel(xw, index=False, sheet_name="p_values")
        piv_r2.to_excel(xw, index=False, sheet_name="r2_por_municipio")
        df_vif.to_excel(xw, index=False, sheet_name="vif")

    # Gráficos por município (betas padronizados)
    for mun in df_out["municipio"].unique():
        plot_barras_betas_padronizados(df_out, mun, outdir=PLOT_DIR)

    # Prints rápidos
    pd.set_option("display.float_format", "{:.6f}".format)
    print("\n=== Amostra (empilhado) ===")
    print(df_out.head().to_string(index=False))
    print("\n=== VIF (amostra) ===")
    print(df_vif.head().to_string(index=False))
    print(f"\nArquivos salvos:\n- {SAIDA_CSV}\n- {SAIDA_XLSX}\n- {SAIDA_VIF}\n- {PLOT_DIR}/<municipio>.png")


if __name__ == "__main__":
    main()
