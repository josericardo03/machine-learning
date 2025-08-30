# -*- coding: utf-8 -*-
"""
Tabela de correlação de Pearson (desmatamento vs variáveis econômicas)
----------------------------------------------------------------------
- Fonte: view_dados_completos (2010–2021)
- Saída:
    1) correlacao_pearson_por_municipio.xlsx  (aba: por_municipio, geral)
    2) correlacao_pearson_por_municipio.csv   (por_municipio)
    3) correlacao_pearson_geral.csv           (geral)
- O coeficiente é calculado em NÍVEIS (ano a ano), com exclusão de NaNs (pairwise).
- Opcional: também calcula em Δ (primeira diferença) e log-diff; basta ligar os FLAGS.

Requisitos: pandas, numpy, psycopg2
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import psycopg2

# =============== CONFIGURAÇÕES ===============
HOST     = "aws-0-sa-east-1.pooler.supabase.com"
PORT     = "5432"
DBNAME   = "postgres"
USER     = "postgres.roziechzdpxxdtzlkaep"
PASSWORD = "Jj20134849@@@"   # ***Sugestão: mover para variável de ambiente em produção.***

TABELA_VIEW = "view_dados_completos"
ANOS_INI, ANOS_FIM = 2010, 2021

VAR_Y   = "desmatado"
VAR_XS  = ["valor_agropecuaria", "pib_per_capita", "valor_industria", "valor_administracao_publica"]

# Ligue se quiser gerar correlações adicionais:
DO_DELTA   = False   # Pearson em primeira diferença
DO_LOGDIFF = False   # Pearson em log-diff

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
        # Tipos
        df["ano"] = df["ano"].astype(int)
        # Remove linhas completamente vazias nas variáveis de interesse
        cols_keep = ["ano", "municipio", VAR_Y] + VAR_XS
        df = df[cols_keep].dropna(how="all")
        return df
    finally:
        conn.close()


# =============== CORRELAÇÕES ===============
def corr_pearson_pairwise(df, y_col, x_cols):
    """
    Calcula a correlação de Pearson (níveis) para cada X em x_cols contra y_col.
    Usa 'pairwise complete observations' (dropna por par).
    Retorna: dict {f"pearson_{x}": coef, "n_{x}": n_pares}
    """
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
    """
    Pearson em primeira diferença (Δ).
    """
    d = df.sort_values("ano")[[y_col] + x_cols].copy().diff().dropna()
    return corr_pearson_pairwise(d, y_col, x_cols)


def corr_logdiff(df, y_col, x_cols):
    """
    Pearson em log-diff: diff(log(1+x)).
    """
    g = np.log1p(df.sort_values("ano")[[y_col] + x_cols]).diff().dropna()
    return corr_pearson_pairwise(g, y_col, x_cols)


def gera_tabelas_correlacao(df):
    """
    Gera:
      - Tabela por município (níveis + opcional Δ, log-diff)
      - Tabela geral (todos municípios juntos)
    """
    linhas = []
    for mun, dmun in df.groupby("municipio"):
        base = {"municipio": mun}

        # Pearson níveis
        base.update(corr_pearson_pairwise(dmun, VAR_Y, VAR_XS))

        # Opcional: Δ e log-diff (colunas com sufixo)
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
    df_geral = df[["ano", VAR_Y] + VAR_XS].copy()
    # Como há vários municípios por ano, agregamos por ano (média) para um “sinal anual” único:
    df_agg = df.groupby("ano", as_index=False).mean(numeric_only=True)
    geral_vals = corr_pearson_pairwise(df_agg, VAR_Y, VAR_XS)
    df_geral_tab = pd.DataFrame([geral_vals])
    df_geral_tab.insert(0, "escopo", "Geral (média por ano)")

    return df_por_mun, df_geral_tab


def exporta(df_por_mun, df_geral):
    # CSVs
    df_por_mun.to_csv(SAIDA_CSV_MUN, index=False, encoding="utf-8-sig")
    df_geral.to_csv(SAIDA_CSV_GERAL, index=False, encoding="utf-8-sig")

    # Excel com abas
    with pd.ExcelWriter(SAIDA_XLSX, engine="openpyxl") as xw:
        df_por_mun.to_excel(xw, index=False, sheet_name="por_municipio")
        df_geral.to_excel(xw, index=False, sheet_name="geral")


def main():
    df = carrega_dados()

    # Garantir colunas e limpeza básica
    esperadas = {"ano", "municipio", VAR_Y, *VAR_XS}
    faltantes = esperadas.difference(df.columns)
    if faltantes:
        raise RuntimeError(f"Colunas faltantes na consulta: {faltantes}")

    # Remove linhas sem Y (desmatado) ou todas X vazias
    if df[VAR_Y].isna().all():
        raise RuntimeError("A coluna 'desmatado' está completamente vazia após consulta.")
    if df[VAR_Y].isna().any():
        # manteremos pairwise; não é necessário dropar tudo aqui
        pass

    df_por_mun, df_geral = gera_tabelas_correlacao(df)
    exporta(df_por_mun, df_geral)

    # Exibição-resumo
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n=== Correlações por município (amostra) ===")
    print(df_por_mun.head().to_string(index=False))
    print("\n=== Correlações gerais (média por ano) ===")
    print(df_geral.to_string(index=False))

if __name__ == "__main__":
    main()
