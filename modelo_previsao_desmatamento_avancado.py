import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURAÇÃO DO BANCO ===
DB_USER = 'postgres.roziechzdpxxdtzlkaep'
DB_PASS = 'Jj20134849%40%40%40'
DB_HOST = 'aws-0-sa-east-1.pooler.supabase.com'
DB_PORT = '5432'
DB_NAME = 'postgres'
VIEW_NAME = 'real'
DATABASE_URL = f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

# 1. Carregar os dados da view 'real' do banco
en = create_engine(DATABASE_URL)
df = pd.read_sql(f'SELECT * FROM {VIEW_NAME}', en)

# Seleção das variáveis principais (agora incluindo hidrografia)
variaveis_principais = ['desmatado', 'valor_agropecuaria', 'pib_per_capita', 'hidrografia']
variaveis_extra = [col for col in df.columns if col not in variaveis_principais + ['id_municipio', 'id_municipio_nome', 'ano', 'cluster'] and df[col].dtype in [np.float64, np.int64]]
variaveis_analise = ['valor_agropecuaria', 'pib_per_capita', 'hidrografia'] + variaveis_extra

# Remover linhas com NaN
X = df[variaveis_analise].copy()
y = df['desmatado']
mask = ~X.isnull().any(axis=1) & ~y.isnull()
X = X[mask]
y = y[mask]

# Padronizar variáveis (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. Matriz de correlação
plt.figure(figsize=(7, 6))
sns.heatmap(pd.concat([y.reset_index(drop=True), pd.DataFrame(X_scaled, columns=X.columns)], axis=1).corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de Correlação (variáveis padronizadas)')
plt.tight_layout()
plt.savefig('matriz_correlacao_padronizada.png')
plt.show()

# 2. Diagnóstico de multicolinearidade (VIF)
vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
print('\n=== VIF (Multicolinearidade) ===')
print(vif_data.sort_values('VIF', ascending=False))

# 3. Seleção de features (remover VIF > 10)
features_vif = vif_data[vif_data['VIF'] <= 10]['feature'].tolist()
if len(features_vif) < 2:
    features_vif = X.columns.tolist()  # Se todas têm VIF alto, usa todas
X_sel = X[features_vif]
X_sel_scaled = scaler.fit_transform(X_sel)

# 4. Regressão Linear Múltipla
reg = LinearRegression()
reg.fit(X_sel_scaled, y)
y_pred = reg.predict(X_sel_scaled)
print('\n=== REGRESSÃO LINEAR AVANÇADA ===')
print('Features usadas:', features_vif)
print('Coeficientes:', dict(zip(features_vif, reg.coef_)))
print(f'Intercepto: {reg.intercept_:.2f}')
print(f'R²: {r2_score(y, y_pred):.3f} | MSE: {mean_squared_error(y, y_pred):.1f} | MAE: {mean_absolute_error(y, y_pred):.1f}')

# 5. Gráfico de coeficientes (padronizados)
plt.figure(figsize=(8, 5))
coefs = reg.coef_
plt.barh(features_vif, coefs, color=['red' if c < 0 else 'green' for c in coefs])
plt.xlabel('Coeficiente padronizado')
plt.title('Efeito das Variáveis no Desmatamento (padronizado)')
plt.tight_layout()
plt.savefig('coeficientes_regressao_avancada.png')
plt.show()

# 6. Gráfico de resíduos
residuos = y - y_pred
plt.figure(figsize=(7, 4))
plt.scatter(y_pred, residuos, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Valor Ajustado')
plt.ylabel('Resíduo')
plt.title('Resíduos da Regressão Linear Avançada')
plt.tight_layout()
plt.savefig('residuos_regressao_avancada.png')
plt.show()

# 7. Análise de normalidade dos resíduos
import scipy.stats as stats
plt.figure(figsize=(6, 4))
stats.probplot(residuos, dist="norm", plot=plt)
plt.title('QQ-plot dos resíduos')
plt.tight_layout()
plt.savefig('qqplot_residuos_avancada.png')
plt.show()

# 8. Gráficos de dispersão com linha de tendência para as features selecionadas
for var in features_vif:
    plt.figure(figsize=(7, 4))
    plt.scatter(X_sel[var], y, alpha=0.6)
    z = np.polyfit(X_sel[var], y, 1)
    p = np.poly1d(z)
    plt.plot(X_sel[var], p(X_sel[var]), 'r--')
    plt.xlabel(var)
    plt.ylabel('Desmatado')
    plt.title(f'Desmatamento vs {var} (padronizado)')
    plt.tight_layout()
    plt.savefig(f'desmatamento_vs_{var}_padronizado.png')
    plt.show()

# === PREVISÃO SIMPLES PARA CÁCERES USANDO APENAS O HISTÓRICO DO MUNICÍPIO ===
print('\n=== PREVISÃO LINEAR SIMPLES PARA CÁCERES (APENAS HISTÓRICO LOCAL) ===')
df_caceres = df[df['id_municipio_nome'] == 'Cáceres'].sort_values('ano')
X_ano = df_caceres[['ano']].values
y_desmatado = df_caceres['desmatado'].values
reg_caceres = LinearRegression().fit(X_ano, y_desmatado)
anos_futuros = np.array([[2022], [2023], [2024]])
y_pred_futuro = reg_caceres.predict(anos_futuros)
# Garante que a previsão nunca seja menor que 2021 se a tendência for de crescimento ou estabilidade
if reg_caceres.coef_[0] >= 0:
    y_pred_futuro = np.maximum(y_pred_futuro, y_desmatado[-1])

# Mostrar resultados
print('Coeficiente da regressão:', reg_caceres.coef_[0])
print('Intercepto:', reg_caceres.intercept_)
print('Valor real 2021:', y_desmatado[-1])
print('Previsão para 2022, 2023, 2024:', y_pred_futuro)

plt.figure(figsize=(7, 4))
plt.plot(df_caceres['ano'], df_caceres['desmatado'], 'o-', label='Real')
plt.plot(anos_futuros.flatten(), y_pred_futuro, 's--', label='Previsto (regressão local)')
plt.xlabel('Ano')
plt.ylabel('Desmatado')
plt.title('Previsão Linear Simples para Cáceres (2022-2024)')
plt.legend()
plt.tight_layout()
plt.savefig('previsao_linear_simples_caceres.png')
plt.show()

print('\n=== FIM DO PIPELINE DE REGRESSÃO LINEAR AVANÇADA ===')
