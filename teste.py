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
import cvxpy as cp  # pyright: ignore[reportMissingImports]

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# -------------------------- FLAGS --------------------------
FAST_MODE   = False       # True = busca/tuning mais curta
       # usa todos os núcleos
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
    
    # APENAS features econômicas PURAS (sem qualquer informação temporal do desmatamento)
    for col in ["valor_agropecuaria", "pib_per_capita", "valor_industria", "valor_administracao_publica"]:
        d[f"d_{col}"] = d[col].diff()
        d[f"g_{col}"] = np.log1p(d[col]).diff()
    
    # Features econômicas em níveis (mais estáveis)
    d["pib_per_capita_norm"] = d["pib_per_capita"] / d["pib_per_capita"].rolling(window=5, min_periods=1).mean()
    d["valor_agropecuaria_norm"] = d["valor_agropecuaria"] / d["valor_agropecuaria"].rolling(window=5, min_periods=1).mean()
    
    # Interações econômicas simples
    d["agro_pib_ratio"] = d["valor_agropecuaria"] / (d["pib_per_capita"] + 1e-9)
    d["industria_pib_ratio"] = d["valor_industria"] / (d["pib_per_capita"] + 1e-9)
    
    d = d.dropna().copy()
    d["y_log"] = np.log1p(d["desmatado"])

    # APENAS features econômicas (ZERO informação temporal do desmatamento)
    feat_cols = [
        "valor_agropecuaria","pib_per_capita","valor_industria","valor_administracao_publica",
        "d_valor_agropecuaria","d_pib_per_capita","d_valor_industria","d_valor_administracao_publica",
        "g_valor_agropecuaria","g_pib_per_capita","g_valor_industria","g_valor_administracao_publica",
        "pib_per_capita_norm","valor_agropecuaria_norm","agro_pib_ratio","industria_pib_ratio"
    ]
    X = d[feat_cols].to_numpy(dtype=float)
    y_log = d["y_log"].to_numpy(dtype=float)
    anos = d["ano"].to_numpy(int)
    return X, y_log, anos, d, feat_cols

def project_features_linear(df_mun):
    anos_fut = pd.DataFrame({"ano": [2022, 2023, 2024]})
    base_cols = ["valor_agropecuaria","pib_per_capita","valor_industria","valor_administracao_publica"]
    
    # Projetar variáveis econômicas base
    for var in base_cols:
        Xy = df_mun[["ano", var]].dropna().copy()
        if len(Xy) >= 3:  # Mínimo 3 pontos para regressão
            X = Xy[["ano"]].to_numpy(float); y = Xy[var].to_numpy(float)
            mdl = Ridge(alpha=10.0).fit(X, y)  # Alpha mais conservador
            anos_fut[var] = mdl.predict(anos_fut[["ano"]].to_numpy())
        else:
            # Se não tiver dados suficientes, usar último valor conhecido
            anos_fut[var] = df_mun[var].iloc[-1] if not df_mun[var].empty else 0.0
    
    anos_fut = anos_fut.sort_values("ano")
    
    # Calcular diferenças e crescimentos
    for col in base_cols:
        anos_fut[f"d_{col}"] = anos_fut[col].diff().fillna(0.0)
        anos_fut[f"g_{col}"] = np.log1p(anos_fut[col]).diff().fillna(0.0)
    
    # Calcular features normalizadas e ratios
    for ano in anos_fut["ano"]:
        row = anos_fut[anos_fut["ano"] == ano].iloc[0]
        
        # Normalizações (usar valores projetados)
        anos_fut.loc[anos_fut["ano"] == ano, "pib_per_capita_norm"] = row["pib_per_capita"] / (row["pib_per_capita"] + 1e-9)
        anos_fut.loc[anos_fut["ano"] == ano, "valor_agropecuaria_norm"] = row["valor_agropecuaria"] / (row["valor_agropecuaria"] + 1e-9)
        
        # Ratios
        anos_fut.loc[anos_fut["ano"] == ano, "agro_pib_ratio"] = row["valor_agropecuaria"] / (row["pib_per_capita"] + 1e-9)
        anos_fut.loc[anos_fut["ano"] == ano, "industria_pib_ratio"] = row["valor_industria"] / (row["pib_per_capita"] + 1e-9)
    
    return anos_fut

# -------------------------- 3. Modelos e Tuning --------------------------


def select_alpha_ridge_1se(X, y_log, alphas=None, cv_splits=3):
    # SEM VALIDAÇÃO CRUZADA - usar alpha fixo muito conservador
    # Com apenas 11 anos, precisamos ser extremamente conservadores
    
    # Alpha base muito alto para evitar overfitting
    alpha_base = 1000.0
    
    # Penalizar ainda mais se dataset for pequeno
    if len(X) < 10:
        alpha_base *= 5.0
    
    if len(X) < 8:
        alpha_base *= 3.0
    
    # Penalizar se R² in-sample for muito alto (sinal de overfitting)
    try:
        pipe_test = Pipeline([("scaler", StandardScaler()),
                             ("ridge", Ridge(alpha=alpha_base, random_state=RANDOM_STATE))])
        pipe_test.fit(X, y_log)
        yhat_test = pipe_test.predict(X)
        r2_test = r2_score(y_log, yhat_test)
        
        # Se R² for muito alto, aumentar alpha
        if r2_test > 0.8:
            alpha_base *= 2.0
        if r2_test > 0.9:
            alpha_base *= 3.0
            
    except:
        pass
    
    return alpha_base

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

def calculate_model_stability(X, y_log, fit_fn, n_iterations=5):
    """Calcula estabilidade do modelo através de múltiplas execuções."""
    if len(X) < 8:
        return np.nan, np.nan
    
    coefs_list = []
    for _ in range(n_iterations):
        try:
            mdl, _ = fit_fn(X, y_log)
            coefs = mdl.named_steps["ridge"].coef_
            coefs_list.append(coefs)
        except:
            continue
    
    if len(coefs_list) < 2:
        return np.nan, np.nan
    
    coefs_array = np.array(coefs_list)
    coef_std = np.std(coefs_array, axis=0)
    coef_cv = np.mean(coef_std / (np.abs(np.mean(coefs_array, axis=0)) + 1e-9))
    
    return float(np.mean(coef_cv)), float(np.std(coef_cv))

def calculate_model_quality_metrics(X, y_log, modelo):
    """Calcula métricas de qualidade do modelo mais robustas."""
    try:
        # Calcular resíduos
        y_pred = modelo.predict(X)
        residuals = y_log - y_pred
        
        # Variância dos resíduos (deve ser baixa para bom modelo)
        residual_variance = np.var(residuals)
        
        # Coeficiente de variação dos resíduos
        residual_cv = np.std(residuals) / (np.abs(np.mean(y_log)) + 1e-9)
        
        # Teste de normalidade dos resíduos (Jarque-Bera simplificado)
        residual_skew = np.mean(((residuals - np.mean(residuals)) / (np.std(residuals) + 1e-9)) ** 3)
        residual_kurt = np.mean(((residuals - np.mean(residuals)) / (np.std(residuals) + 1e-9)) ** 4)
        
        # NOVA: R² ajustado para penalizar overfitting
        n = len(X)
        p = X.shape[1]  # número de features
        r2_adj = 1 - (1 - r2_score(y_log, y_pred)) * (n - 1) / (n - p - 1)
        
        # NOVA: Critério de informação de Akaike (AIC simplificado)
        aic = n * np.log(residual_variance) + 2 * p
        
        return {
            "residual_variance": float(residual_variance),
            "residual_cv": float(residual_cv),
            "residual_skewness": float(residual_skew),
            "residual_kurtosis": float(residual_kurt),
            "r2_ajustado": float(r2_adj),
            "aic_simplificado": float(aic)
        }
    except:
        return {
            "residual_variance": np.nan,
            "residual_cv": np.nan,
            "residual_skewness": np.nan,
            "residual_kurtosis": np.nan,
            "r2_ajustado": np.nan,
            "aic_simplificado": np.nan
        }

def evaluate_honest(X, y_log, anos, fit_fn):
    # SEM HOLDOUT - usar todos os 11 anos para treino
    mdl_full, params_full = fit_fn(X, y_log)
    yhat_in = mdl_full.predict(X)
    r2_in_log = r2_score(y_log, yhat_in); mse_in_log = mean_squared_error(y_log, yhat_in)
    r2_in_raw, mse_in_raw = r2_mse_original(y_log, yhat_in)

    # SEM VALIDAÇÃO CRUZADA - usar apenas métricas in-sample robustas
    # Com apenas 11 anos, qualquer split temporal será problemático
    r2_cv = np.nan
    
    # Calcular estabilidade e qualidade do modelo
    mdl_full.stability_metrics = calculate_model_stability(X, y_log, fit_fn)
    mdl_full.quality_metrics = calculate_model_quality_metrics(X, y_log, mdl_full)

    # Retornar valores NaN para holdout e CV (não usados mais)
    r2_hold_log = mse_hold_log = r2_hold_raw = mse_hold_raw = np.nan

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
    prevs = []; y_2021_float = float(y_2021)
    
    for ano in [2022, 2023, 2024]:
        row = anos_fut[anos_fut["ano"] == ano].iloc[0].to_dict()
        feats = {c: row.get(c, 0.0) for c in feat_cols}
        
        # Calcular features econômicas normalizadas para o ano atual
        if "pib_per_capita_norm" in feat_cols:
            # Usar valores projetados diretamente (sem dependência de previsões anteriores)
            feats["pib_per_capita_norm"] = feats["pib_per_capita"] / (feats["pib_per_capita"] + 1e-9)
            feats["valor_agropecuaria_norm"] = feats["valor_agropecuaria"] / (feats["valor_agropecuaria"] + 1e-9)
            feats["agro_pib_ratio"] = feats["valor_agropecuaria"] / (feats["pib_per_capita"] + 1e-9)
            feats["industria_pib_ratio"] = feats["valor_industria"] / (feats["pib_per_capita"] + 1e-9)
        
        x = np.array([feats.get(c,0.0) for c in feat_cols], dtype=float).reshape(1,-1)
        y_log_hat = modelo.predict(x)[0]; y_hat = float(np.expm1(y_log_hat))
        prevs.append(y_hat)
    
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

        def fit_fn_ridge(Xtr, ytr):
            return fit_ridge_tuned(Xtr, ytr, cv_splits=(3 if FAST_MODE else 4))

        (r2_in_log, mse_in_log, r2_hold_log, mse_hold_log, r2_cv,
         r2_in_raw, mse_in_raw, r2_hold_raw, mse_hold_raw, modelo, params) = evaluate_honest(
            X, y_log, anos_np, fit_fn_ridge
        )
        modelo_nome = "Ridge"

        anos_fut = project_features_linear(df_mun)
        y_2021 = float(df_mun.loc[df_mun["ano"]==2021, "desmatado"].values[0])
        prev_bruta = forecast_iterative(modelo, anos_fut, y_2021, feat_cols)
        prev_adj, gmin = adjust_with_monotony_constraint(prev_bruta, y_2021, df_mun)

        # Preparar métricas de estabilidade e qualidade
        estabilidade_coef = estabilidade_std = np.nan
        if hasattr(modelo, 'stability_metrics'):
            estabilidade_coef, estabilidade_std = modelo.stability_metrics
        
        quality_metrics = {}
        if hasattr(modelo, 'quality_metrics'):
            quality_metrics = modelo.quality_metrics
        
        resultados.append({
            "municipio": municipio,
            "desmatamento_2022": float(prev_adj[0]),
            "desmatamento_2023": float(prev_adj[1]),
            "desmatamento_2024": float(prev_adj[2]),
            "crescimento_desmatamento": float(prev_adj[2] - prev_adj[0]),
            "r2_in_sample_log": round(float(r2_in_log),4) if not np.isnan(r2_in_log) else None,
            "mse_in_sample_log": round(float(mse_in_log),4) if not np.isnan(mse_in_log) else None,
            "r2_cv_log": round(float(r2_cv),4) if not np.isnan(r2_cv) else None,
            "r2_in_sample_raw": round(float(r2_in_raw),4) if not np.isnan(r2_in_raw) else None,
            "mse_in_sample_raw": round(float(mse_in_raw),4) if not np.isnan(mse_in_raw) else None,
            "modelo_escolhido": modelo_nome,
            "melhores_parametros": str(params),
            "piso_crescimento_aplicado": round(float(gmin),2),
            "estabilidade_coeficientes": round(float(estabilidade_coef),4) if not np.isnan(estabilidade_coef) else None,
            "estabilidade_std": round(float(estabilidade_std),4) if not np.isnan(estabilidade_std) else None,
            "residual_variance": round(float(quality_metrics.get("residual_variance", np.nan)),6) if not np.isnan(quality_metrics.get("residual_variance", np.nan)) else None,
            "residual_cv": round(float(quality_metrics.get("residual_cv", np.nan)),4) if not np.isnan(quality_metrics.get("residual_cv", np.nan)) else None,
            "r2_ajustado": round(float(quality_metrics.get("r2_ajustado", np.nan)),4) if not np.isnan(quality_metrics.get("r2_ajustado", np.nan)) else None,
            "aic_simplificado": round(float(quality_metrics.get("aic_simplificado", np.nan)),2) if not np.isnan(quality_metrics.get("aic_simplificado", np.nan)) else None,
            "observacao": "Modelo treinado com todos os 11 anos (2010-2021) - SEM validação cruzada - APENAS features econômicas - Alpha fixo conservador"
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
