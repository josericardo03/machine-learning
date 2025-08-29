# -*- coding: utf-8 -*-
"""
Previsão de desmatamento municipal no Pantanal — v3 (rápido + honesto + seleção de modelo)
+ Geração de 12 gráficos (3 por município) para estudo de caso
+ Cálculo e export de correlações (Pearson: nível/Δ/logΔ; Spearman; Parcial controlando lags do Y)
"""

import warnings, os, re
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="whitegrid")
import cvxpy as cp

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# -------------------------- FLAGS --------------------------
FAST_MODE   = False       # True = busca/tuning mais curta
N_JOBS      = -1          # usa todos os núcleos
RANDOM_STATE= 42

# MUNICÍPIOS DO ESTUDO DE CASO (gerar 3 gráficos cada)
MUNICIPIOS_CASOS = ["Corumbá", "Porto Murtinho", "Cáceres", "Sonora"]

# -------------------------- 1. Conexão --------------------------
conn = psycopg2.connect(
    dbname="postgres",
    user="postgres.roziechzdpxxdtzlkaep",
    password="Jj20134849@@@",
    host="aws-0-sa-east-1.pooler.supabase.com",
    port="5432"
)
query_hist = """
SELECT ano, id_municipio_nome, desmatado,
       valor_agropecuaria, pib_per_capita,
       valor_industria, valor_administracao_publica
FROM view_dados_completos
WHERE ano >= 2010 AND ano <= 2021
"""
df = pd.read_sql_query(query_hist, conn)
df = df.dropna().copy()
df["ano"] = df["ano"].astype(int)

# Também já trazemos os valores OBSERVADOS 2022–2023 para comparação
query_obs = """
SELECT ano,
       id_municipio_nome AS municipio,
       desmatado
FROM dados_municipios
WHERE ano >= 2022
ORDER BY municipio, ano
"""
obs_22_23 = pd.read_sql_query(query_obs, conn)
conn.close()

# -------------------------- 2. Features --------------------------
def make_features(df_mun):
    d = df_mun.sort_values("ano").copy()
    for col in ["valor_agropecuaria", "pib_per_capita", "valor_industria", "valor_administracao_publica"]:
        d[f"d_{col}"] = d[col].diff()
        d[f"g_{col}"] = np.log1p(d[col]).diff()

    d["y_lag1"] = d["desmatado"].shift(1)
    d["y_lag2"] = d["desmatado"].shift(2)

    d = d.dropna().copy()
    d["y_log"] = np.log1p(d["desmatado"])

    feat_cols = [
        "valor_agropecuaria","pib_per_capita","valor_industria","valor_administracao_publica",
        "d_valor_agropecuaria","d_pib_per_capita","d_valor_industria","d_valor_administracao_publica",
        "g_valor_agropecuaria","g_pib_per_capita","g_valor_industria","g_valor_administracao_publica",
        "y_lag1","y_lag2"
    ]
    X = d[feat_cols].to_numpy(dtype=float)
    y_log = d["y_log"].to_numpy(dtype=float)
    anos = d["ano"].to_numpy(int)
    return X, y_log, anos, d, feat_cols

def project_features_linear(df_mun):
    anos_fut = pd.DataFrame({"ano": [2022, 2023, 2024]})
    base_cols = ["valor_agropecuaria","pib_per_capita","valor_industria","valor_administracao_publica"]
    for var in base_cols:
        Xy = df_mun[["ano", var]].dropna().copy()
        X = Xy[["ano"]].to_numpy(float); y = Xy[var].to_numpy(float)
        mdl = Ridge(alpha=1.0, random_state=RANDOM_STATE).fit(X, y)
        anos_fut[var] = mdl.predict(anos_fut[["ano"]].to_numpy())
    anos_fut = anos_fut.sort_values("ano")
    for col in base_cols:
        anos_fut[f"d_{col}"] = anos_fut[col].diff().fillna(0.0)
        anos_fut[f"g_{col}"] = np.log1p(anos_fut[col]).diff().fillna(0.0)
    return anos_fut

# -------------------------- 3. Modelos e Tuning --------------------------
def fit_rf_tuned(X, y_log, n_iter=12, cv_splits=3):
    if FAST_MODE:
        n_iter = min(n_iter, 8); cv_splits = min(cv_splits, 3)
    tscv = TimeSeriesSplit(n_splits=min(cv_splits, max(2, len(y_log)-2)))
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS)
    param_dist = {
        "n_estimators":[100,140,180],
        "max_depth":[3,4,5,6,8],
        "min_samples_leaf":[2,3,4],
        "max_features":[0.5,0.7,0.9],
        "bootstrap":[True],
        "max_samples":[0.7,0.9],
        "ccp_alpha":[0.0,1e-4,5e-4,1e-3]
    }
    search = RandomizedSearchCV(
        rf, param_distributions=param_dist, n_iter=n_iter, cv=tscv,
        scoring="neg_mean_squared_error", random_state=RANDOM_STATE,
        n_jobs=N_JOBS, verbose=0
    )
    search.fit(X, y_log)
    return search.best_estimator_, search.best_params_

def select_alpha_ridge_1se(X, y_log, alphas=None, cv_splits=3):
    if alphas is None:
        alphas = np.logspace(1, 4, 40)
    tscv = TimeSeriesSplit(n_splits=min(cv_splits, max(2, len(y_log)-2)))
    mse_means, mse_stds = [], []
    for a in alphas:
        fold_mse = []
        for tr, te in tscv.split(X):
            pipe = Pipeline([("scaler", StandardScaler()),
                             ("ridge", Ridge(alpha=a, random_state=RANDOM_STATE))])
            pipe.fit(X[tr], y_log[tr])
            yhat = pipe.predict(X[te])
            fold_mse.append(mean_squared_error(y_log[te], yhat))
        mse_means.append(np.mean(fold_mse)); mse_stds.append(np.std(fold_mse))
    mse_means = np.array(mse_means); mse_stds = np.array(mse_stds)
    idx_min = int(np.argmin(mse_means))
    thr = mse_means[idx_min] + mse_stds[idx_min]
    idx_1se = np.max(np.where(mse_means <= thr)[0])
    return float(alphas[idx_1se])

def fit_ridge_tuned(X, y_log, cv_splits=3):
    a = select_alpha_ridge_1se(X, y_log, cv_splits=cv_splits)
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("ridge", Ridge(alpha=a, random_state=RANDOM_STATE))])
    pipe.fit(X, y_log)
    return pipe, {"alpha_1se": a}

# -------------------------- 4. Avaliação honesta --------------------------
def r2_mse_original(y_log_true, y_log_pred):
    y_true = np.expm1(y_log_true); y_pred = np.expm1(y_log_pred)
    return r2_score(y_true, y_pred), mean_squared_error(y_true, y_pred)

def evaluate_honest(X, y_log, anos, fit_fn):
    mdl_full, params_full = fit_fn(X, y_log)
    yhat_in = mdl_full.predict(X)
    r2_in_log = r2_score(y_log, yhat_in); mse_in_log = mean_squared_error(y_log, yhat_in)
    r2_in_raw, mse_in_raw = r2_mse_original(y_log, yhat_in)

    r2_hold_log = mse_hold_log = r2_hold_raw = mse_hold_raw = np.nan
    mask_tr = (anos <= 2019); mask_te = (anos >= 2020) & (anos <= 2021)
    if mask_tr.sum() >= 4 and mask_te.sum() >= 1:
        Xtr, ytr = X[mask_tr], y_log[mask_tr]; Xte, yte = X[mask_te], y_log[mask_te]
        mdl_tr, _ = fit_fn(Xtr, ytr)
        y_te_pred = mdl_tr.predict(Xte)
        r2_hold_log = r2_score(yte, y_te_pred); mse_hold_log = mean_squared_error(yte, y_te_pred)
        r2_hold_raw, mse_hold_raw = r2_mse_original(yte, y_te_pred)

    r2_cv = np.nan
    if len(X) >= 6:
        n_splits = min(5 if not FAST_MODE else 3, len(X)-2)
        if n_splits >= 2:
            tscv = TimeSeriesSplit(n_splits=n_splits); scores = []
            for tr, te in tscv.split(X):
                mdl_fold, _ = fit_fn(X[tr], y_log[tr])
                scores.append(r2_score(y_log[te], mdl_fold.predict(X[te])))
            r2_cv = float(np.nanmean(scores))

    return (r2_in_log, mse_in_log, r2_hold_log, mse_hold_log, r2_cv,
            r2_in_raw, mse_in_raw, r2_hold_raw, mse_hold_raw, mdl_full, params_full)

# -------------------------- 5. Pós-processamento convexo --------------------------
def adjust_with_monotony_constraint(previsao_bruta, desmatado_2021, df_hist):
    if len(df_hist) >= 3:
        ult = df_hist.sort_values("ano").tail(3)["desmatado"].to_numpy()
        cresc_medio = float(np.mean(np.diff(ult))) if len(ult) >= 2 else 0.0
    else:
        cresc_medio = 0.0
    g_min = max(0.2*cresc_medio, 0.005*desmatado_2021)

    z = cp.Variable(3)
    cons = [ z[0] >= desmatado_2021 + g_min,
             z[1] >= z[0] + g_min,
             z[2] >= z[1] + g_min ]
    obj = cp.Minimize(cp.sum_squares(z - previsao_bruta))
    cp.Problem(obj, cons).solve()
    return np.array(z.value, dtype=float), g_min

# -------------------------- 6. Previsão iterativa --------------------------
def forecast_iterative(modelo, anos_fut, y_2021, feat_cols):
    prevs = []; y_lag1 = y_2021; y_lag2 = np.nan
    for ano in [2022, 2023, 2024]:
        row = anos_fut[anos_fut["ano"] == ano].iloc[0].to_dict()
        feats = {c: row.get(c, 0.0) for c in feat_cols}
        feats["y_lag1"] = y_lag1
        feats["y_lag2"] = (prevs[-1] if len(prevs)>=1 else (y_lag2 if not np.isnan(y_lag2) else y_lag1))
        x = np.array([feats.get(c,0.0) for c in feat_cols], dtype=float).reshape(1,-1)
        y_log_hat = modelo.predict(x)[0]; y_hat = float(np.expm1(y_log_hat))
        prevs.append(y_hat); y_lag2 = y_lag1; y_lag1 = y_hat
    return np.array(prevs, dtype=float)

# -------------------------- helpers de gráficos --------------------------
def safe_dir(name:str)->str:
    return re.sub(r"[^A-Za-z0-9_\- ]+", "_", name).strip().replace(" ", "_")

def plot_importancias(mun, modelo, modelo_nome, feat_cols, outpath):
    plt.figure(figsize=(8,5))
    if modelo_nome == "RandomForest":
        importances = modelo.feature_importances_
        imp_df = pd.DataFrame({"feature": feat_cols, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=True).tail(10)
        plt.barh(imp_df["feature"], imp_df["importance"])
        plt.xlabel("Importância (RF)"); plt.title(f"{mun} — Importância das variáveis (top 10)")
    else:
        ridge = modelo.named_steps["ridge"]; coefs = np.abs(ridge.coef_)
        imp_df = pd.DataFrame({"feature": feat_cols, "importance": coefs})
        imp_df = imp_df.sort_values("importance", ascending=True).tail(10)
        plt.barh(imp_df["feature"], imp_df["importance"])
        plt.xlabel("|Coeficiente padronizado|"); plt.title(f"{mun} — Importância (Ridge, top 10)")
    plt.tight_layout(); plt.savefig(outpath); plt.close()

def plot_real_22_23(mun, obs_df, outpath):
    d = obs_df[(obs_df["municipio"]==mun) & (obs_df["ano"].isin([2022,2023]))]
    if d.empty:
        return
    plt.figure(figsize=(6,4))
    plt.bar(d["ano"].astype(str), d["desmatado"].astype(float))
    plt.ylabel("Área desmatada (ha)"); plt.title(f"{mun} — Valores reais 2022–2023")
    plt.tight_layout(); plt.savefig(outpath); plt.close()

def plot_prev_vs_real_22_23(mun, pred22, pred23, obs_df, outpath):
    d = obs_df[(obs_df["municipio"]==mun) & (obs_df["ano"].isin([2022,2023]))][["ano","desmatado"]].copy()
    d = d.set_index("ano")["desmatado"].to_dict()
    real22 = float(d.get(2022, np.nan)); real23 = float(d.get(2023, np.nan))
    anos = ["2022","2023"]
    real = [real22, real23]; prev = [pred22, pred23]

    real_arr = np.array([x for x in real if not np.isnan(x)])
    prev_arr = np.array([p for p,x in zip(prev,real) if not np.isnan(x)])
    rmse = np.sqrt(np.mean((prev_arr - real_arr)**2)) if len(real_arr)>0 else np.nan
    mape = np.mean(np.abs((prev_arr - real_arr)/np.maximum(real_arr,1e-9)))*100 if len(real_arr)>0 else np.nan

    x = np.arange(len(anos)); w = 0.35
    plt.figure(figsize=(7,4))
    plt.bar(x - w/2, real, width=w, label="Real")
    plt.bar(x + w/2, prev, width=w, label="Previsto")
    plt.xticks(x, anos); plt.ylabel("Área desmatada (ha)")
    ttl = f"{mun} — Previsto vs Real (2022–2023)"
    if not np.isnan(rmse):
        ttl += f" | RMSE={rmse:,.0f} ha | MAPE={mape:,.1f}%"
    plt.title(ttl); plt.legend()
    plt.tight_layout(); plt.savefig(outpath); plt.close()

# -------------------------- 2.1. Cálculo de correlações --------------------------
BASE_COLS = ["valor_agropecuaria","pib_per_capita","valor_industria","valor_administracao_publica"]

def _partial_corr(y, x, controls_df):
    """Correlação parcial de Pearson entre y e x controlando por 'controls_df' (com intercepto)."""
    C = np.column_stack([np.ones(len(controls_df)), controls_df.values])
    beta_y, *_ = np.linalg.lstsq(C, y, rcond=None)
    res_y = y - C @ beta_y
    beta_x, *_ = np.linalg.lstsq(C, x, rcond=None)
    res_x = x - C @ beta_x
    sy, sx = res_y.std(), res_x.std()
    if sy == 0 or sx == 0:
        return np.nan
    return float(np.corrcoef(res_y, res_x)[0, 1])

def compute_correlacoes(df_mun):
    """Retorna dicionário com correlações + DF de trabalho para heatmap."""
    d = df_mun.sort_values("ano")[["ano","desmatado"] + BASE_COLS].dropna().copy()
    out = {}

    # Pearson em níveis
    corr_level = d.corr(method="pearson")["desmatado"].drop("desmatado")
    for c, v in corr_level.items(): out[f"pearson_nivel_{c}"] = float(v)

    # Pearson em 1ª diferença (Δ)
    dd = d[["desmatado"] + BASE_COLS].diff().dropna()
    corr_delta = dd.corr(method="pearson")["desmatado"].drop("desmatado")
    for c, v in corr_delta.items(): out[f"pearson_delta_{c}"] = float(v)

    # Pearson em crescimento (log-diff)
    gd = np.log1p(d[["desmatado"] + BASE_COLS]).diff().dropna()
    corr_g = gd.corr(method="pearson")["desmatado"].drop("desmatado")
    for c, v in corr_g.items(): out[f"pearson_logdiff_{c}"] = float(v)

    # Spearman em níveis
    corr_spear = d.corr(method="spearman")["desmatado"].drop("desmatado")
    for c, v in corr_spear.items(): out[f"spearman_nivel_{c}"] = float(v)

    # Parcial (níveis) controlando y_lag1 e y_lag2
    d["y_lag1"] = d["desmatado"].shift(1)
    d["y_lag2"] = d["desmatado"].shift(2)
    dp = d.dropna().copy()
    if len(dp) >= 3:
        controls = dp[["y_lag1","y_lag2"]]
        y = dp["desmatado"].values
        for c in BASE_COLS:
            out[f"partial_nivel_{c}_ctl_lags"] = _partial_corr(y, dp[c].values, controls)
    else:
        for c in BASE_COLS:
            out[f"partial_nivel_{c}_ctl_lags"] = np.nan

    return out, d

# -------------------------- 7. Loop por município + armazenamento para gráficos --------------------------
resultados = []
store = {}      # guarda objetos p/ os 4 municípios do estudo
cor_rows = []   # NOVO: correlações por município

municipios = df["id_municipio_nome"].unique()
for municipio in municipios:
    df_mun = df[df["id_municipio_nome"] == municipio].copy()
    if len(df_mun) < 6 or 2021 not in df_mun["ano"].values:
        continue
    try:
        X, y_log, anos_np, dfeat, feat_cols = make_features(df_mun)

        def fit_fn_rf(Xtr, ytr):
            return fit_rf_tuned(Xtr, ytr, n_iter=(8 if FAST_MODE else 12), cv_splits=(3 if FAST_MODE else 4))
        def fit_fn_ridge(Xtr, ytr):
            return fit_ridge_tuned(Xtr, ytr, cv_splits=(3 if FAST_MODE else 4))

        m_rf = evaluate_honest(X, y_log, anos_np, fit_fn_rf)
        m_rg = evaluate_honest(X, y_log, anos_np, fit_fn_ridge)

        def key(m):
            _,_,_,_, r2_cv, r2_in_raw,_, r2_hold_raw,_,_,_ = m
            if not np.isnan(r2_hold_raw): return r2_hold_raw
            if not np.isnan(r2_cv): return r2_cv
            return r2_in_raw

        chosen = m_rf if key(m_rf) >= key(m_rg) else m_rg
        (r2_in_log, mse_in_log, r2_hold_log, mse_hold_log, r2_cv,
         r2_in_raw, mse_in_raw, r2_hold_raw, mse_hold_raw, modelo, params) = chosen
        modelo_nome = "RandomForest" if chosen is m_rf else "Ridge"

        anos_fut = project_features_linear(df_mun)
        y_2021 = float(df_mun.loc[df_mun["ano"]==2021, "desmatado"].values[0])
        prev_bruta = forecast_iterative(modelo, anos_fut, y_2021, feat_cols)
        prev_adj, gmin = adjust_with_monotony_constraint(prev_bruta, y_2021, df_mun)

        resultados.append({
            "municipio": municipio,
            "desmatamento_2022": float(prev_adj[0]),
            "desmatamento_2023": float(prev_adj[1]),
            "desmatamento_2024": float(prev_adj[2]),
            "crescimento_desmatamento": float(prev_adj[2] - prev_adj[0]),
            "r2_in_sample_log": round(float(r2_in_log),4) if not np.isnan(r2_in_log) else None,
            "mse_in_sample_log": round(float(mse_in_log),4) if not np.isnan(mse_in_log) else None,
            "r2_holdout_log": round(float(r2_hold_log),4) if not np.isnan(r2_hold_log) else None,
            "mse_holdout_log": round(float(mse_hold_log),4) if not np.isnan(mse_hold_log) else None,
            "r2_cv_log": round(float(r2_cv),4) if not np.isnan(r2_cv) else None,
            "r2_in_sample_raw": round(float(r2_in_raw),4) if not np.isnan(r2_in_raw) else None,
            "mse_in_sample_raw": round(float(mse_in_raw),4) if not np.isnan(mse_in_raw) else None,
            "r2_holdout_raw": round(float(r2_hold_raw),4) if not np.isnan(r2_hold_raw) else None,
            "mse_holdout_raw": round(float(mse_hold_raw),4) if not np.isnan(mse_hold_raw) else None,
            "modelo_escolhido": modelo_nome,
            "melhores_parametros": str(params),
            "piso_crescimento_aplicado": round(float(gmin),2)
        })

        # ---- NOVO: Correlações para este município ----
        corr_dict, d_for_heatmap = compute_correlacoes(df_mun)
        cor_row = {"municipio": municipio}; cor_row.update(corr_dict)
        cor_rows.append(cor_row)

        # Guarda DF para heatmap se for estudo de caso
        if municipio in MUNICIPIOS_CASOS:
            store.setdefault(municipio, {})
            store[municipio]["corr_df"] = d_for_heatmap[["desmatado"] + BASE_COLS].dropna()
            store[municipio].update({
                "modelo": modelo,
                "modelo_nome": modelo_nome,
                "feat_cols": feat_cols,
                "pred_2022": float(prev_adj[0]),
                "pred_2023": float(prev_adj[1]),
                "df_hist": df_mun.copy()
            })

    except Exception as e:
        print(f"Erro em {municipio}: {e}")
        continue

# -------------------------- 8. Resultado + Export --------------------------
df_resultados = pd.DataFrame(resultados)
if not df_resultados.empty:
    df_resultados = df_resultados.sort_values(by="desmatamento_2024", ascending=False)
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", None)
    print(df_resultados.to_string(index=False))
    df_resultados.to_excel("previsao_desmatamento_v3.xlsx", index=False)
    df_resultados.to_csv("previsao_desmatamento_v3.csv", index=False, encoding="utf-8-sig")
else:
    print("Nenhum município atendeu aos critérios mínimos.")

# ---- NOVO: Exporta correlações por município ----
df_cor = pd.DataFrame(cor_rows)
if not df_cor.empty:
    cols_order = ["municipio"] + sorted([c for c in df_cor.columns if c != "municipio"])
    df_cor = df_cor[cols_order]
    print("\nCorrelações (amostra):")
    print(df_cor.head().to_string(index=False))
    df_cor.to_excel("correlacoes_desmatamento.xlsx", index=False)
    df_cor.to_csv("correlacoes_desmatamento.csv", index=False, encoding="utf-8-sig")

# -------------------------- 9. GERAÇÃO DOS 12 GRÁFICOS --------------------------
base_dir = "graficos_estudo_de_caso"
os.makedirs(base_dir, exist_ok=True)

# garante que temos observados 2022–2023
if obs_22_23.empty:
    print("Aviso: não vieram observações de 2022–2023 em dados_municipios; gráficos de comparação podem ficar vazios.")

for mun, payload in store.items():
    folder = os.path.join(base_dir, safe_dir(mun))
    os.makedirs(folder, exist_ok=True)

    modelo      = payload["modelo"]
    modelo_nome = payload["modelo_nome"]
    feat_cols   = payload["feat_cols"]

    # 0) NOVO — Heatmap de correlação (níveis, Pearson)
    if "corr_df" in payload and not payload["corr_df"].empty:
        hm = payload["corr_df"].corr(method="pearson")
        plt.figure(figsize=(6,5))
        sns.heatmap(hm, annot=True, fmt=".2f", vmin=-1, vmax=1, cmap="coolwarm")
        plt.title(f"{mun} — Correlação (níveis, Pearson)")
        plt.tight_layout(); plt.savefig(os.path.join(folder, "0_correlacao_heatmap.png")); plt.close()

    # 1) Importância das variáveis
    plot_importancias(mun, modelo, modelo_nome, feat_cols,
                      outpath=os.path.join(folder, "1_importancia_variaveis.png"))

    # 2) Valores reais 2022–2023 (barras)
    plot_real_22_23(mun, obs_22_23,
                    outpath=os.path.join(folder, "2_reais_2022_2023.png"))

    # 3) Previsto vs Real 2022–2023 (barras lado a lado)
    plot_prev_vs_real_22_23(mun,
                            pred22=payload["pred_2022"],
                            pred23=payload["pred_2023"],
                            obs_df=obs_22_23,
                            outpath=os.path.join(folder, "3_previsto_vs_real_2022_2023.png"))

print(f"\nGráficos salvos em: {os.path.abspath(base_dir)}")
