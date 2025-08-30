# -*- coding: utf-8 -*-
"""
Tabela de correlação de Pearson (desmatamento vs variáveis econômicas) + GRÁFICO
-------------------------------------------------------------------------------
- Fonte: view_dados_completos (2010–2021)
- Saídas:
    1) correlacao_pearson_desmatamento.xlsx  (abas: por_municipio, geral)
    2) correlacao_pearson_por_municipio.csv  (tabela por município)
    3) correlacao_pearson_geral.csv          (tabela geral)
    4) correlacao_pearson_<variavel>_por_municipio.csv  (dados do gráfico)
    5) correlacao_pearson_<variavel>_por_municipio.png  (gráfico de barras)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt

# =============== CONFIGURAÇÕES ===============
HOST     = "aws-0-sa-east-1.pooler.supabase.com"
PORT     = "5432"
DBNAME   = "postgres"
USER     = "postgres.roziechzdpxxdtzlkaep"
PASSWORD = "Jj20134849@@@"   # mova p/ variável de ambiente em produção

TABELA_VIEW = "view_dados_completos"
ANOS_INI, ANOS_FIM = 2010, 2021

VAR_Y   = "desmatado"
VAR_XS  = ["valor_agropecuaria", "pib_per_capita", "valor_industria", "valor_administracao_publica"]

# Ligue se quiser gerar correlações adicionais:
DO_DELTA   = False   # Pearson em primeira diferença
DO_LOGDIFF = False   # Pearson em log-diff

# >>> VARIÁVEL PARA O GRÁFICO (mude aqui)
VAR_TO_PLOT = "valor_agropecuaria"   # exemplos: "pib_per_capita", "valor_industria", "valor_administracao_publica"

SAIDA_XLSX = "correlacao_pearson_desmatamento.xlsx"
SAIDA_CSV_MUN = "correlacao_pearson_por_municipio.csv"
SAIDA_CSV_GERAL = "correlacao_pearson_geral.csv"


# =============== CONSULTA ===============
def carrega_dados():
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
        cols_keep = ["ano", "municipio", VAR_Y] + VAR_XS
        df = df[cols_keep].dropna(how="all")
        return df
    finally:
        conn.close()


# =============== CORRELAÇÕES ===============
def corr_pearson_pairwise(df, y_col, x_cols):
    out = {}
    for x in x_cols:
        sub = df[[y_col, x]].dropna()
        if len(sub) >= 2 and sub[y_col].std() > 0 and sub[x].std() > 0:
            coef = float(sub[y_col].corr(sub[x], method="pearson"))
            out[f"pearson_{x}"] = coef
            out[f"n_{x}"] = int(len(sub))
        else:
            out[f"pearson_{x}"] = np.nan
            out[f"n_{x}"] = int(len(sub))
    return out

def corr_delta(df, y_col, x_cols):
    d = df.sort_values("ano")[[y_col] + x_cols].copy().diff().dropna()
    return corr_pearson_pairwise(d, y_col, x_cols)

def corr_logdiff(df, y_col, x_cols):
    g = np.log1p(df.sort_values("ano")[[y_col] + x_cols]).diff().dropna()
    return corr_pearson_pairwise(g, y_col, x_cols)

def gera_tabelas_correlacao(df):
    linhas = []
    for mun, dmun in df.groupby("municipio"):
        base = {"municipio": mun}
        base.update(corr_pearson_pairwise(dmun, VAR_Y, VAR_XS))

        if DO_DELTA:
            delta_vals = corr_delta(dmun, VAR_Y, VAR_XS)
            base.update({k.replace("pearson_", "pearson_delta_"): v for k, v in delta_vals.items() if not k.startswith("n_")})
            base.update({k.replace("n_", "n_delta_"): v for k, v in delta_vals.items() if k.startswith("n_")})

        if DO_LOGDIFF:
            g_vals = corr_logdiff(dmun, VAR_Y, VAR_XS)
            base.update({k.replace("pearson_", "pearson_logdiff_"): v for k, v in g_vals.items() if not k.startswith("n_")})
            base.update({k.replace("n_", "n_logdiff_"): v for k, v in g_vals.items() if k.startswith("n_")})

        linhas.append(base)

    df_por_mun = pd.DataFrame(linhas).sort_values("municipio").reset_index(drop=True)

    # Geral (todos municípios juntos; correlação em níveis)
    df_agg = df.groupby("ano", as_index=False).mean(numeric_only=True)
    geral_vals = corr_pearson_pairwise(df_agg, VAR_Y, VAR_XS)
    df_geral_tab = pd.DataFrame([geral_vals])
    df_geral_tab.insert(0, "escopo", "Geral (média por ano)")

    return df_por_mun, df_geral_tab


def exporta_tabelas(df_por_mun, df_geral):
    df_por_mun.to_csv(SAIDA_CSV_MUN, index=False, encoding="utf-8-sig")
    df_geral.to_csv(SAIDA_CSV_GERAL, index=False, encoding="utf-8-sig")

    from openpyxl import Workbook  # garante engine disponível
    with pd.ExcelWriter(SAIDA_XLSX, engine="openpyxl") as xw:
        df_por_mun.to_excel(xw, index=False, sheet_name="por_municipio")
        df_geral.to_excel(xw, index=False, sheet_name="geral")


# =============== GRÁFICO ===============
def grafico_barras_pearson(df_por_mun, var_to_plot):
    col = f"pearson_{var_to_plot}"
    if col not in df_por_mun.columns:
        raise ValueError(f"Coluna '{col}' não encontrada. Verifique VAR_TO_PLOT.")

    df_plot = df_por_mun[["municipio", col]].copy()
    df_plot = df_plot.dropna().sort_values(col, ascending=False).reset_index(drop=True)

    # salva CSV do gráfico
    csv_out = f"correlacao_pearson_{var_to_plot}_por_municipio.csv"
    df_plot.to_csv(csv_out, index=False, encoding="utf-8-sig")

    # gráfico
    plt.figure(figsize=(10, max(4, 0.35 * len(df_plot))))  # altura ajusta c/ nº de cidades
    plt.barh(df_plot["municipio"], df_plot[col])
    plt.gca().invert_yaxis()  # maior no topo
    plt.xlabel(f"Correlação de Pearson com {var_to_plot.replace('_',' ')}")
    plt.title(f"Correlação (Pearson) por município — variável: {var_to_plot}")
    # rótulos de valor
    for i, v in enumerate(df_plot[col].values):
        plt.text(v + (0.01 if v >= 0 else -0.01), i, f"{v:.2f}", va="center",
                 ha="left" if v >= 0 else "right")
    plt.tight_layout()
    png_out = f"correlacao_pearson_{var_to_plot}_por_municipio.png"
    plt.savefig(png_out, dpi=150)
    plt.close()
    print(f"Gráfico salvo em: {png_out}")
    print(f"Tabela do gráfico salva em: {csv_out}")


def main():
    # 1) Carrega e saneia
    df = carrega_dados()
    esperadas = {"ano", "municipio", VAR_Y, *VAR_XS}
    faltantes = esperadas.difference(df.columns)
    if faltantes:
        raise RuntimeError(f"Colunas faltantes na consulta: {faltantes}")

    # 2) Correlações
    df_por_mun, df_geral = gera_tabelas_correlacao(df)
    exporta_tabelas(df_por_mun, df_geral)

    # 3) Resumos no console
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n=== Correlações por município (amostra) ===")
    print(df_por_mun.head().to_string(index=False))
    print("\n=== Correlações gerais (média por ano) ===")
    print(df_geral.to_string(index=False))

    # 4) Gráfico de barras para a variável escolhida
    if VAR_TO_PLOT not in VAR_XS:
        raise ValueError(f"VAR_TO_PLOT inválida: {VAR_TO_PLOT}. Use uma de {VAR_XS}.")
    grafico_barras_pearson(df_por_mun, VAR_TO_PLOT)


if __name__ == "__main__":
    main()
