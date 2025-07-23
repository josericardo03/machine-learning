import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIGURAÇÃO DO BANCO ===
# Edite aqui com suas credenciais
DB_USER = 'postgres.roziechzdpxxdtzlkaep'
DB_PASS = 'Jj20134849%40%40%40'
DB_HOST = 'aws-0-sa-east-1.pooler.supabase.com'
DB_PORT = '5432'
DB_NAME = 'postgres'
VIEW_NAME = 'real'

DATABASE_URL = f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}'




# 1. Carregar os dados da view 'real' do banco
engine = create_engine(DATABASE_URL)
df = pd.read_sql(f'SELECT * FROM {VIEW_NAME}', engine)

# 2. Pré-processamento e feature engineering
# Ordenar por município e ano
df = df.sort_values(['id_municipio', 'ano']).reset_index(drop=True)

# Remover linhas com valores faltantes em desmatado
df = df.dropna(subset=['desmatado'])

# Criar variáveis de tendência temporal
df['ano_quadrado'] = (df['ano'] - 2010) ** 2

# Variáveis percentuais
if 'vegetacao_natural' in df.columns and 'area_total' in df.columns:
    df['veg_natural_pct'] = df['vegetacao_natural'] / df['area_total']
if 'valor_agropecuaria' in df.columns and 'area_total' in df.columns:
    df['agropecuaria_pct'] = df['valor_agropecuaria'] / df['area_total']

# Mudança percentual ano a ano
df['mudanca_desmatado'] = df.groupby('id_municipio')['desmatado'].pct_change() * 100
if 'pib_per_capita' in df.columns:
    df['mudanca_pib'] = df.groupby('id_municipio')['pib_per_capita'].pct_change() * 100
if 'vegetacao_natural' in df.columns:
    df['mudanca_veg'] = df.groupby('id_municipio')['vegetacao_natural'].pct_change() * 100

# Preencher NaN resultantes de pct_change com 0
df = df.fillna(0)

# 3. Seleção de features
features = [
    'ano', 'ano_quadrado', 'area_total', 'vegetacao_natural', 'veg_natural_pct',
    'valor_agropecuaria', 'agropecuaria_pct', 'pib_per_capita', 'mudanca_pib', 'mudanca_veg',
    'amazonia_legal_bin'
]

# Se houver variáveis categóricas, transforme em dummies
df = pd.get_dummies(df, columns=['bioma', 'hierarquia_urbana'], drop_first=True)
features += [col for col in df.columns if col.startswith('bioma_') or col.startswith('hierarquia_urbana_')]

X = df[features]
y = df['desmatado']

# 4. Padronização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 6. Modelos a testar
modelos = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

resultados = {}
for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    cv = cross_val_score(modelo, X_train, y_train, cv=5, scoring='r2')
    resultados[nome] = {'r2': r2, 'mse': mse, 'mae': mae, 'cv_mean': cv.mean(), 'cv_std': cv.std(), 'modelo': modelo, 'y_pred': y_pred}
    print(f'{nome}: R²={r2:.3f} | MSE={mse:.3f} | MAE={mae:.3f} | CV={cv.mean():.3f}±{cv.std():.3f}')

# 7. Gráficos de desempenho
plt.figure(figsize=(12, 6))
plt.bar(resultados.keys(), [resultados[n]['r2'] for n in resultados], color='skyblue')
plt.ylabel('R² Teste')
plt.title('Desempenho dos Modelos (R² no Teste)')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('desempenho_modelos.png')
plt.show()

# 8. Gráfico Real vs Predito do melhor modelo
melhor_nome = max(resultados, key=lambda n: resultados[n]['r2'])
melhor_modelo = resultados[melhor_nome]['modelo']
y_pred = resultados[melhor_nome]['y_pred']

plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Desmatado Real')
plt.ylabel('Desmatado Predito')
plt.title(f'Real vs Predito - {melhor_nome}')
plt.tight_layout()
plt.savefig('real_vs_predito.png')
plt.show()

# 9. Importância das features (se disponível)
if hasattr(melhor_modelo, 'coef_'):
    importancias = np.abs(melhor_modelo.coef_)
    nomes_features = X.columns
elif hasattr(melhor_modelo, 'feature_importances_'):
    importancias = melhor_modelo.feature_importances_
    nomes_features = X.columns
else:
    importancias = None
    nomes_features = []

if importancias is not None:
    imp_df = pd.DataFrame({'feature': nomes_features, 'importancia': importancias})
    imp_df = imp_df.sort_values('importancia', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importancia', y='feature', data=imp_df.head(10), palette='viridis')
    plt.title(f'Top 10 Features Mais Importantes - {melhor_nome}')
    plt.tight_layout()
    plt.savefig('importancia_features.png')
    plt.show()
    print('\nTop 10 variáveis mais importantes:')
    print(imp_df.head(10))

# 10. Salvar resultados
resultados_df = pd.DataFrame({
    'real': y_test,
    'predito': y_pred
})
resultados_df.to_csv('resultados_predicao.csv', index=False)

print(f'\nMelhor modelo: {melhor_nome}')
print('Gráficos salvos: desempenho_modelos.png, real_vs_predito.png, importancia_features.png')

# 11. Previsão para os próximos anos de Cáceres usando tendência linear
from sklearn.linear_model import LinearRegression

municipio_nome = 'Cáceres'
df_caceres = df[df['id_municipio_nome'] == municipio_nome].sort_values('ano')
anos_futuros = [2022, 2023, 2024]
entradas = []
for ano_futuro in anos_futuros:
    entrada = {}
    for feat in features:
        if feat == 'ano':
            entrada[feat] = ano_futuro
        elif feat in df_caceres.columns and df_caceres[feat].dtype in [np.float64, np.int64]:
            X = df_caceres[['ano']].values
            y = df_caceres[feat].values
            if len(np.unique(X)) > 1:
                reg = LinearRegression().fit(X, y)
                entrada[feat] = reg.predict([[ano_futuro]])[0]
            else:
                entrada[feat] = y[-1]
        elif feat in df_caceres.columns:
            entrada[feat] = df_caceres[feat].iloc[-1]
        else:
            entrada[feat] = 0
    entradas.append(entrada)

X_futuro = pd.DataFrame(entradas)[features]
print('\nVariáveis projetadas para Cáceres (2022-2024):')
print(X_futuro)
X_futuro_scaled = scaler.transform(X_futuro)
y_pred_futuro = melhor_modelo.predict(X_futuro_scaled)
previsoes_futuro = pd.DataFrame({'ano': anos_futuros, 'previsao_desmatamento': y_pred_futuro})
print('\nPrevisão de desmatamento para Cáceres (2022-2024):')
print(previsoes_futuro) 