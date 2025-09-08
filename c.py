# -*- coding: utf-8 -*-
"""
Regressão linear (Ridge) robusta para poucos dados + projeção 4 variáveis + monotonia estrita
---------------------------------------------------------------------------------------------
- Treino por município com 2011..2020 (2010 vira lag); teste cego = 2021
- Seleção de alpha por validação temporal (expanding 1-ano)
- Peso maior para anos recentes no treino (amostragem ponderada)
- Projeção de X (agro, PIBpc, indústria, adm. pública) por tendência (2018–2021; 2019–2021; senão persistência)
- Âncora no nível de 2021 (corrige viés): previsões futuras recebem delta = (y2021_obs - yhat2021)
- Monotonia estrita com passo dinâmico: eps = max(EPS_INCREASE_ABS, EPS_INCREASE_PCT*y2021)
- Saídas: CSV/XLSX com previsões brutas e ajustadas; métricas; coeficientes padronizados; correlações; gráficos

Autor: você :)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

# ===================== CONFIG =====================
HOST     = "aws-0-sa-east-1.pooler.supabase.com"
PORT     = "5432"
DBNAME   = "postgres"
USER     = "postgres.roziechzdpxxdtzlkaep"
PASSWORD = "Jj20134849@@@"

TABELA_VIEW = "view_dados_completos"
ANOS_INI, ANOS_FIM = 2010, 2021

VAR_Y  = "desmatado"
VAR_XS = ["valor_agropecuaria", "pib_per_capita", "valor_industria", "valor_administracao_publica"]

ANOS_FUTUROS = [2022, 2023, 2024]

# Monotonia estrita (garantir subida mínima dinâmica)
EPS_INCREASE_ABS = 1.0     # mínimo absoluto em ha
EPS_INCREASE_PCT = 0.01    # mínimo percentual vs y2021 (1% por padrão)
CLIP_NONNEG = True         # força y >= 0 (sem desmatamento negativo)

# Peso maior aos anos mais recentes no treino (1.0 -> FACTOR ao longo do período de treino)
RECENT_WEIGHT_FACTOR = 2.0

# Grid de alphas do Ridge para seleção por validação temporal
ALPHAS = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

OUT_DIR   = "saida_previsao_linear4vars_small_data"
PLOTS_DIR = os.path.join(OUT_DIR, "plots_real_vs_prev")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ===================== I/O =====================
def carrega_dados():
    conn = psycopg2.connect(dbname=DBNAME, user=USER, password=PASSWORD, host=HOST, port=PORT)
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

# ===================== UTIL =====================
def build_lagged_frame(df_mun: pd.DataFrame) -> pd.DataFrame:
    """Cria quadro com y_t, y_{t-1}, X_t para anos em que tudo existe (após o lag)."""
    d = df_mun.sort_values("ano").copy()
    d["y"] = d[VAR_Y]
    d["y_lag1"] = d["y"].shift(1)
    cols = ["ano", "municipio", "y", "y_lag1"] + VAR_XS
    d = d[cols].dropna().reset_index(drop=True)
    return d

def linear_trend_forecast(series: pd.Series, anos_futuros: list):
    """
    Projeta valores via tendência linear (mín. 3 pts):
      prioridade 2018..2021; senão 2019..2021; senão persistência (último valor).
    """
    s = series.dropna().copy()
    if s.empty:
        return {t: 0.0 for t in anos_futuros}

    def fit_and_pred(x_years, y_vals, targets):
        a, b = np.polyfit(np.array(x_years, dtype=float), np.array(y_vals, dtype=float), 1)
        out = {}
        for t in targets:
            val = a*float(t) + b
            out[t] = float(max(0.0, val))
        return out

    # tenta 2018..2021
    if all(y in s.index for y in [2018, 2019, 2020, 2021]):
        return fit_and_pred([2018, 2019, 2020, 2021], [s.loc[y] for y in [2018, 2019, 2020, 2021]], anos_futuros)
    # tenta 2019..2021
    if all(y in s.index for y in [2019, 2020, 2021]):
        return fit_and_pred([2019, 2020, 2021], [s.loc[y] for y in [2019, 2020, 2021]], anos_futuros)

    # persistência
    last_val = float(s.loc[int(s.index.max())])
    return {t: float(max(0.0, last_val)) for t in anos_futuros}

def make_recent_weights(n: int, factor: float = 2.0):
    """Gera pesos crescentes do passado (1.0) ao mais recente (~factor). Normaliza média=1."""
    if n <= 1:
        return np.ones(n, dtype=float)
    w = np.linspace(1.0, max(1.0, factor), n)
    return (w / w.mean()).astype(float)

def gen_time_splits(n: int, min_train: int = 5, max_folds: int = 3):
    """
    Gera splits temporais (expanding window; validação de 1 ano).
    Ex.: n=10, min_train=5 -> folds: (train 0..4,val 5), (0..5,val 6), (0..6,val 7)
    Deixa os últimos pontos para proximidade de 2021.
    """
    splits = []
    tr_end = min_train
    while tr_end < n - 1 and len(splits) < max_folds:
        train_idx = np.arange(0, tr_end)
        val_idx   = np.array([tr_end])
        splits.append((train_idx, val_idx))
        tr_end += 1
    if not splits and n > 1:
        splits.append((np.arange(0, n-1), np.array([n-1])))
    return splits

def select_alpha_timecv(X: np.ndarray, y: np.ndarray, years: np.ndarray, alphas=ALPHAS):
    """
    Seleciona alpha por validação temporal com pesos crescentes (anos recentes mais pesados).
    """
    n = X.shape[0]
    splits = gen_time_splits(n, min_train=min(5, n-1), max_folds=3)
    if not splits:
        return 1.0

    best_alpha, best_mse = None, np.inf
    for alpha in alphas:
        mses = []
        for tr_idx, va_idx in splits:
            # pesos no treino
            w = make_recent_weights(len(tr_idx), factor=RECENT_WEIGHT_FACTOR)
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha, random_state=0))
            ])
            pipe.fit(X[tr_idx], y[tr_idx], ridge__sample_weight=w)
            y_hat = pipe.predict(X[va_idx])
            mses.append(np.mean((y_hat - y[va_idx])**2))
        avg = float(np.mean(mses)) if mses else np.inf
        if avg < best_mse:
            best_mse = avg
            best_alpha = alpha
    return best_alpha if best_alpha is not None else 1.0

def enforce_strict_monotone_dynamic(y2021_obs: float, raw_preds: dict, clip_nonneg=True) -> (dict, float):
    """
    eps dinâmico = max(EPS_INCREASE_ABS, EPS_INCREASE_PCT * y2021_obs)
    Garante: 2022>=2021+eps; 2023>=2022+eps; 2024>=2023+eps; y>=0
    Retorna (ajustadas, eps_usado)
    """
    eps = max(EPS_INCREASE_ABS, EPS_INCREASE_PCT * float(y2021_obs))
    v2022 = max(raw_preds.get(2022, 0.0), y2021_obs + eps)
    v2023 = max(raw_preds.get(2023, 0.0), v2022 + eps)
    v2024 = max(raw_preds.get(2024, 0.0), v2023 + eps)
    adj = {2022: v2022, 2023: v2023, 2024: v2024}
    if clip_nonneg:
        adj = {k: max(0.0, v) for k, v in adj.items()}
    return adj, eps

def pearson_by_municipio(df: pd.DataFrame) -> pd.DataFrame:
    """Correlação de Pearson (níveis) entre desmatado e cada X, por município (2010–2021)."""
    linhas = []
    for mun, dmun in df.groupby("municipio"):
        d = dmun[["ano", VAR_Y] + VAR_XS].dropna()
        row = {"municipio": mun}
        for x in VAR_XS:
            sub = d[[VAR_Y, x]].dropna()
            if len(sub) >= 2 and sub[VAR_Y].std() > 0 and sub[x].std() > 0:
                row[f"pearson_{x}"] = float(sub[VAR_Y].corr(sub[x], method="pearson"))
            else:
                row[f"pearson_{x}"] = np.nan
        linhas.append(row)
    return pd.DataFrame(linhas).sort_values("municipio").reset_index(drop=True)

# ===================== TREINO + PREVISÃO =====================
def train_and_predict_municipio(df_mun: pd.DataFrame):
    """
    - Seleciona alpha (Ridge) com validação temporal
    - Treina com pesos que favorecem anos recentes
    - Âncora no nível de 2021 (corrige viés de 1 ano à frente)
    - Projeta X e prevê 2022–2024 (iterativo), com monotonia estrita dinâmica
    """
    dlag = build_lagged_frame(df_mun)  # anos >= 2011
    if dlag.empty or 2021 not in dlag["ano"].values:
        return None

    # Treino/teste
    dtrain = dlag[dlag["ano"] <= 2020].copy()
    dtest  = dlag[dlag["ano"] == 2021].copy()
    if len(dtrain) < 2:
        return None

    feat_cols = ["y_lag1"] + VAR_XS
    X_tr = dtrain[feat_cols].values.astype(float)
    y_tr = dtrain["y"].values.astype(float)
    years_tr = dtrain["ano"].values.astype(int)

    # Seleção de alpha por validação temporal
    best_alpha = select_alpha_timecv(X_tr, y_tr, years_tr, alphas=ALPHAS)

    # Treino final com pesos crescentes (anos recentes mais pesados)
    w_tr = make_recent_weights(len(dtrain), factor=RECENT_WEIGHT_FACTOR)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=best_alpha, random_state=0))
    ])
    pipe.fit(X_tr, y_tr, ridge__sample_weight=w_tr)

    # Métrica no teste 2021 + âncora (bias correction)
    X_te = dtest[feat_cols].values.astype(float)
    y_te = dtest["y"].values.astype(float)
    y_hat_2021 = float(pipe.predict(X_te)[0])
    y_true_2021 = float(y_te[0])
    bias_delta = y_true_2021 - y_hat_2021  # correção a aplicar nas previsões futuras

    err = y_hat_2021 - y_true_2021
    rmse_test = float(np.sqrt(err**2))
    r2_test = 1.0 - (err**2) / np.var(y_tr) if np.var(y_tr) > 0 else np.nan

    # Coeficientes padronizados (interpretação relativa)
    ridge = pipe.named_steps["ridge"]
    coefs_std = {"municipio": dlag["municipio"].iloc[0], "alpha": float(best_alpha), "intercept_std": float(ridge.intercept_)}
    for name, coef in zip(feat_cols, ridge.coef_):
        coefs_std[f"coef_{name}_std"] = float(coef)

    # ---------- Projeção de X (2022–2024) ----------
    hist = df_mun.set_index("ano").sort_index()
    proj_X = {x: linear_trend_forecast(hist[x], ANOS_FUTUROS) for x in VAR_XS}

    # ---------- Previsão iterativa com âncora + monotonia ----------
    y_last = float(hist.loc[2021, VAR_Y])  # começa em 2021 observado
    raw_preds = {}
    for ano in ANOS_FUTUROS:
        feats = [y_last] + [proj_X[x][ano] for x in VAR_XS]
        X_new = np.array(feats, dtype=float).reshape(1, -1)
        y_pred = float(pipe.predict(X_new)[0] + bias_delta)  # aplica âncora
        if CLIP_NONNEG:
            y_pred = max(0.0, y_pred)
        raw_preds[ano] = y_pred
        y_last = y_pred

    # Monotonia estrita dinâmica
    adj_preds, eps_used = enforce_strict_monotone_dynamic(float(hist.loc[2021, VAR_Y]), raw_preds, clip_nonneg=CLIP_NONNEG)

    return {
        "municipio": dlag["municipio"].iloc[0],
        "rmse_teste_2021": rmse_test,
        "r2_teste_2021": r2_test,
        "n_treinamento": int(len(dtrain)),
        "alpha_selecionado": float(best_alpha),
        "bias_delta_2021": float(bias_delta),
        "coefs_std": coefs_std,
        "raw_preds": raw_preds,
        "adj_preds": adj_preds,
        "eps_usado": float(eps_used),
        "proj_X": proj_X,
        "y2021_obs": float(hist.loc[2021, VAR_Y])
    }

# ===================== CORRELAÇÕES =====================
def pearson_by_municipio(df: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for mun, dmun in df.groupby("municipio"):
        d = dmun[["ano", VAR_Y] + VAR_XS].dropna()
        row = {"municipio": mun}
        for x in VAR_XS:
            sub = d[[VAR_Y, x]].dropna()
            if len(sub) >= 2 and sub[VAR_Y].std() > 0 and sub[x].std() > 0:
                row[f"pearson_{x}"] = float(sub[VAR_Y].corr(sub[x], method="pearson"))
            else:
                row[f"pearson_{x}"] = np.nan
        linhas.append(row)
    return pd.DataFrame(linhas).sort_values("municipio").reset_index(drop=True)

# ===================== MAIN =====================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = carrega_dados()

    metrics_rows = []
    coef_rows = []
    prev_rows = []

    # Correlações (para discussão no TCC)
    df_pearson = pearson_by_municipio(df)
    df_pearson.to_csv(os.path.join(OUT_DIR, "correlacao_pearson_por_municipio.csv"),
                      index=False, encoding="utf-8-sig")

    for mun, dmun in df.groupby("municipio"):
        dmun = dmun.dropna(subset=[VAR_Y]).copy()
        if dmun.empty:
            continue

        res = train_and_predict_municipio(dmun)
        if res is None:
            continue

        # métricas
        metrics_rows.append({
            "municipio": res["municipio"],
            "rmse_teste_2021": res["rmse_teste_2021"],
            "r2_teste_2021": res["r2_teste_2021"],
            "n_treinamento": res["n_treinamento"],
            "alpha_selecionado": res["alpha_selecionado"],
            "bias_delta_2021": res["bias_delta_2021"],
            "eps_monotonia_usado": res["eps_usado"]
        })

        # coeficientes padronizados
        coef_rows.append(res["coefs_std"].copy())

        # previsões
        for ano in ANOS_FUTUROS:
            prev_rows.append({
                "municipio": res["municipio"],
                "ano": ano,
                "y_prev_bruto": res["raw_preds"][ano],
                "y_prev_ajustado_monotono": res["adj_preds"][ano],
                "regra_monotonia": f"2022>=2021+max({EPS_INCREASE_ABS} ha, {EPS_INCREASE_PCT*100:.1f}%); 2023>=2022+...; 2024>=2023+...; y>=0"
            })

        # plot
        try:
            hist = dmun.set_index("ano").sort_index()
            anos_hist = hist.index.tolist()
            y_hist = hist[VAR_Y].values
            y_prev_raw = [res["raw_preds"][a] for a in ANOS_FUTUROS]
            y_prev_adj = [res["adj_preds"][a] for a in ANOS_FUTUROS]

            plt.figure(figsize=(7,4))
            plt.plot(anos_hist, y_hist, marker="o", label="Observado (2010–2021)")
            plt.plot(ANOS_FUTUROS, y_prev_raw, marker="o", linestyle="--", label="Previsto bruto (22–24)")
            plt.plot(ANOS_FUTUROS, y_prev_adj, marker="o", label="Previsto ajustado (estrito dinâmico)")
            plt.axhline(res["y2021_obs"], linewidth=1, alpha=0.4)
            plt.title(res["municipio"])
            plt.xlabel("Ano"); plt.ylabel("Desmatado (ha)")
            plt.legend()
            plt.tight_layout()
            fig_path = os.path.join(PLOTS_DIR, f"{res['municipio'].replace('/','-').replace(' ','_')}.png")
            plt.savefig(fig_path, dpi=150); plt.close()
        except Exception:
            pass

    if not prev_rows:
        print("Sem previsões geradas. Verifique a base de dados.")
        return

    # DataFrames finais
    df_metrics = pd.DataFrame(metrics_rows).sort_values("municipio").reset_index(drop=True)
    df_coefs   = pd.DataFrame(coef_rows).sort_values("municipio").reset_index(drop=True)
    df_prev    = pd.DataFrame(prev_rows).sort_values(["municipio","ano"]).reset_index(drop=True)

    # Salvar CSVs
    df_metrics.to_csv(os.path.join(OUT_DIR, "metrics_por_municipio.csv"), index=False, encoding="utf-8-sig")
    df_coefs.to_csv(os.path.join(OUT_DIR, "coeficientes_ridge_por_municipio.csv"), index=False, encoding="utf-8-sig")
    df_prev.to_csv(os.path.join(OUT_DIR, "previsoes_2022_2024.csv"), index=False, encoding="utf-8-sig")

    # XLSX consolidado
    xlsx_path = os.path.join(OUT_DIR, "previsoes_2022_2024.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        df_prev.to_excel(xw, index=False, sheet_name="previsoes")
        df_metrics.to_excel(xw, index=False, sheet_name="metrics")
        df_coefs.to_excel(xw, index=False, sheet_name="coeficientes")
        df_pearson.to_excel(xw, index=False, sheet_name="correlacoes")

    print("\nArquivos gerados em:", os.path.abspath(OUT_DIR))
    print(" - metrics_por_municipio.csv")
    print(" - coeficientes_ridge_por_municipio.csv")
    print(" - previsoes_2022_2024.csv")
    print(" - previsoes_2022_2024.xlsx")
    print(" - correlacao_pearson_por_municipio.csv")
    print(" - Pasta de plots:", PLOTS_DIR)

if __name__ == "__main__":
    main()
