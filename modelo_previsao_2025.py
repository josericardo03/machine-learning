import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🚀 Iniciando Modelo de Previsão de Faturamento 2025")
print("=" * 60)

# Configuração da conexão
usuario = "postgres.roziechzdpxxdtzlkaep"
senha = "Jj20134849%40%40%40"
host = "aws-0-sa-east-1.pooler.supabase.com"
porta = 5432
banco = "postgres"

# Conectar ao banco
print("📊 Conectando ao banco de dados...")
engine = create_engine(f"postgresql://{usuario}:{senha}@{host}:{porta}/{banco}")

# Carregar dados da view
query = "SELECT * FROM view_previsao_simples ORDER BY id_barbeiro, ano, mes"
df = pd.read_sql(query, con=engine)
print(f"✅ Dados carregados: {df.shape[0]} registros, {df.shape[1]} variáveis")

# Análise exploratória inicial
print(f"\n📈 Período dos dados: {df['ano'].min()}-{df['mes'].min():.0f} a {df['ano'].max()}-{df['mes'].max():.0f}")
print(f"👥 Barbeiros únicos: {df['id_barbeiro'].nunique()}")
print(f"💰 Faturamento total: R$ {df['faturamento_total'].sum():,.2f}")
print(f"📊 Faturamento médio por mês/barbeiro: R$ {df['faturamento_total'].mean():,.2f}")

# Separar dados de treino (2023-2024) e dados para previsão (2025)
df_treino = df[df['ano'] < 2025].copy()
df_2025 = df[df['ano'] == 2025].copy()

print(f"\n📚 Dados de treino: {df_treino.shape[0]} registros")
print(f"🔮 Dados de 2025: {df_2025.shape[0]} registros")

# Verificar se há dados suficientes
if len(df_treino) < 50:
    print("⚠️  AVISO: Poucos dados para treino. Modelo pode não generalizar bem.")

# Preparar features para o modelo
features = [
    'eh_mes_ferias', 'eh_black_friday', 'eh_mes_promocao',
    'anos_experiencia', 'fator_antiguidade',
    'num_agendamentos', 'valor_medio_agendamento', 'dias_trabalhados', 'num_clientes_unicos',
    'faturamento_mes_anterior', 'variacao_pct_faturamento',
    'agendamentos_por_dia', 'taxa_fidelidade'
]

# Verificar se todas as features existem
features = [f for f in features if f in df_treino.columns]
print(f"\n🎯 Features selecionadas ({len(features)}): {features}")

# Preparar dados de treino
X_train = df_treino[features]
y_train = df_treino['faturamento_total']

# Tratar valores nulos
X_train = X_train.fillna(0)
y_train = y_train.fillna(0)

print(f"\n📊 Shape dos dados de treino: X={X_train.shape}, y={y_train.shape}")

# 1. MODELO DE REGRESSÃO LINEAR BÁSICO
print("\n" + "="*50)
print("🎯 MODELO 1: REGRESSÃO LINEAR")
print("="*50)

modelo_lr = LinearRegression()
modelo_lr.fit(X_train, y_train)

# Avaliação do modelo
y_pred_lr = modelo_lr.predict(X_train)
r2_lr = r2_score(y_train, y_pred_lr)
mae_lr = mean_absolute_error(y_train, y_pred_lr)
cv_r2_lr = cross_val_score(modelo_lr, X_train, y_train, cv=5, scoring='r2').mean()

print(f"✅ R² Score: {r2_lr:.4f}")
print(f"✅ MAE: R$ {mae_lr:.2f}")
print(f"✅ CV R² Score: {cv_r2_lr:.4f}")

# 2. MODELO RIDGE (COM REGULARIZAÇÃO)
print("\n" + "="*50)
print("🎯 MODELO 2: RIDGE REGRESSION")
print("="*50)

# Grid search para encontrar melhor alpha
param_grid_ridge = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
ridge = Ridge()
grid_ridge = GridSearchCV(ridge, param_grid_ridge, cv=5, scoring='r2', n_jobs=-1)
grid_ridge.fit(X_train, y_train)

print(f"✅ Melhor alpha: {grid_ridge.best_params_['alpha']}")
print(f"✅ R² Score: {grid_ridge.best_score_:.4f}")

# 3. MODELO RANDOM FOREST
print("\n" + "="*50)
print("🎯 MODELO 3: RANDOM FOREST")
print("="*50)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_train)
r2_rf = r2_score(y_train, y_pred_rf)
mae_rf = mean_absolute_error(y_train, y_pred_rf)
cv_r2_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2').mean()

print(f"✅ R² Score: {r2_rf:.4f}")
print(f"✅ MAE: R$ {mae_rf:.2f}")
print(f"✅ CV R² Score: {cv_r2_rf:.4f}")

# 4. COMPARAÇÃO DOS MODELOS
print("\n" + "="*50)
print("🏆 COMPARAÇÃO DOS MODELOS")
print("="*50)

resultados = {
    'Regressão Linear': {'R²': r2_lr, 'MAE': mae_lr, 'CV R²': cv_r2_lr},
    'Ridge': {'R²': grid_ridge.best_score_, 'MAE': mean_absolute_error(y_train, grid_ridge.predict(X_train)), 'CV R²': grid_ridge.best_score_},
    'Random Forest': {'R²': r2_rf, 'MAE': mae_rf, 'CV R²': cv_r2_rf}
}

df_resultados = pd.DataFrame(resultados).T
print(df_resultados.round(4))

# Escolher o melhor modelo
melhor_modelo = max(resultados.keys(), key=lambda x: resultados[x]['CV R²'])
print(f"\n🏆 MELHOR MODELO: {melhor_modelo}")

if melhor_modelo == 'Regressão Linear':
    modelo_final = modelo_lr
elif melhor_modelo == 'Ridge':
    modelo_final = grid_ridge.best_estimator_
else:
    modelo_final = rf

# 5. ANÁLISE DE IMPORTÂNCIA DAS VARIÁVEIS
print("\n" + "="*50)
print("📊 ANÁLISE DE IMPORTÂNCIA DAS VARIÁVEIS")
print("="*50)

if hasattr(modelo_final, 'feature_importances_'):
    importancia = modelo_final.feature_importances_
    tipo_importancia = "Feature Importance"
elif hasattr(modelo_final, 'coef_'):
    importancia = abs(modelo_final.coef_)
    tipo_importancia = "Coeficiente Absoluto"
else:
    importancia = None

if importancia is not None:
    df_importancia = pd.DataFrame({
        'Feature': features,
        'Importância': importancia
    }).sort_values('Importância', ascending=False)
    
    print(f"\n{tipo_importancia} das variáveis:")
    for idx, row in df_importancia.head(10).iterrows():
        print(f"  {row['Feature']}: {row['Importância']:.4f}")

# 6. PREVISÃO PARA 2025
print("\n" + "="*50)
print("🔮 PREVISÃO PARA 2025")
print("="*50)

# Preparar dados de 2025 para previsão
if len(df_2025) > 0:
    X_2025 = df_2025[features].fillna(0)
    y_real_2025 = df_2025['faturamento_total']
    
    # Fazer previsões
    y_pred_2025 = modelo_final.predict(X_2025)
    
    # Avaliar previsões (se temos dados reais de 2025)
    if not y_real_2025.isna().all():
        r2_2025 = r2_score(y_real_2025, y_pred_2025)
        mae_2025 = mean_absolute_error(y_real_2025, y_pred_2025)
        print(f"✅ R² Score 2025: {r2_2025:.4f}")
        print(f"✅ MAE 2025: R$ {mae_2025:.2f}")
    
    # Criar DataFrame com previsões
    df_previsoes = df_2025[['id_barbeiro', 'ano', 'mes']].copy()
    df_previsoes['faturamento_previsto'] = y_pred_2025
    df_previsoes['faturamento_real'] = y_real_2025
    
    print(f"\n📊 Previsões para 2025:")
    print(df_previsoes[['id_barbeiro', 'mes', 'faturamento_previsto', 'faturamento_real']].round(2))
    
    # Salvar previsões
    df_previsoes.to_csv('previsoes_faturamento_2025.csv', index=False)
    print(f"\n💾 Previsões salvas em 'previsoes_faturamento_2025.csv'")

# 7. PREVISÃO PARA MESES FUTUROS (se não temos dados de 2025)
print("\n" + "="*50)
print("🔮 PREVISÃO PARA MESES FUTUROS")
print("="*50)

# Gerar dados para previsão de meses futuros
barbeiros = df['id_barbeiro'].unique()
meses_futuros = list(range(1, 13))
previsoes_futuras = []

for barbeiro in barbeiros:
    # Pegar dados históricos do barbeiro
    dados_barbeiro = df_treino[df_treino['id_barbeiro'] == barbeiro].sort_values(['ano', 'mes'])
    
    if len(dados_barbeiro) > 0:
        # Usar último faturamento como base
        ultimo_faturamento = dados_barbeiro['faturamento_total'].iloc[-1]
        
        for mes in meses_futuros:
            # Calcular médias históricas do mês
            dados_mes = dados_barbeiro[dados_barbeiro['mes'] == mes]
            
            if len(dados_mes) > 0:
                # Usar médias históricas do mês
                entrada = {
                    'eh_mes_ferias': int(mes in [1, 7, 12]),
                    'eh_black_friday': int(mes == 11),
                    'eh_mes_promocao': int(mes in [4, 5]),  # 2025
                    'anos_experiencia': 2,  # 2025 - 2023
                    'fator_antiguidade': 1 + (0.1 * (11 - barbeiro) / 9) if barbeiro <= 10 else 1,
                    'num_agendamentos': dados_mes['num_agendamentos'].mean(),
                    'valor_medio_agendamento': dados_mes['valor_medio_agendamento'].mean(),
                    'dias_trabalhados': dados_mes['dias_trabalhados'].mean(),
                    'num_clientes_unicos': dados_mes['num_clientes_unicos'].mean(),
                    'faturamento_mes_anterior': ultimo_faturamento,
                    'variacao_pct_faturamento': dados_mes['variacao_pct_faturamento'].mean(),
                    'agendamentos_por_dia': dados_mes['agendamentos_por_dia'].mean(),
                    'taxa_fidelidade': dados_mes['taxa_fidelidade'].mean()
                }
                
                # Fazer previsão
                X_pred = pd.DataFrame([entrada])[features]
                faturamento_previsto = modelo_final.predict(X_pred)[0]
                
                # Aplicar limites realistas
                faturamento_medio_mes = dados_mes['faturamento_total'].mean()
                faturamento_max_mes = dados_mes['faturamento_total'].max()
                
                # Limitar a 20% acima do máximo histórico
                faturamento_previsto = min(faturamento_previsto, faturamento_max_mes * 1.2)
                faturamento_previsto = max(faturamento_previsto, 0)
                
                previsoes_futuras.append({
                    'id_barbeiro': barbeiro,
                    'ano': 2025,
                    'mes': mes,
                    'faturamento_previsto': faturamento_previsto,
                    'faturamento_medio_historico': faturamento_medio_mes,
                    'faturamento_max_historico': faturamento_max_mes
                })
                
                ultimo_faturamento = faturamento_previsto

# Criar DataFrame de previsões futuras
df_previsoes_futuras = pd.DataFrame(previsoes_futuras)

print(f"\n📊 Previsões para 2025 (meses futuros):")
print(df_previsoes_futuras[['id_barbeiro', 'mes', 'faturamento_previsto', 'faturamento_medio_historico']].round(2))

# Salvar previsões futuras
df_previsoes_futuras.to_csv('previsoes_futuras_2025.csv', index=False)
print(f"\n💾 Previsões futuras salvas em 'previsoes_futuras_2025.csv'")

# 8. RESUMO FINAL
print("\n" + "="*50)
print("📋 RESUMO FINAL")
print("="*50)

print(f"🎯 Modelo escolhido: {melhor_modelo}")
print(f"📊 Performance (CV R²): {resultados[melhor_modelo]['CV R²']:.4f}")
print(f"💰 MAE médio: R$ {resultados[melhor_modelo]['MAE']:.2f}")
print(f"👥 Barbeiros previstos: {len(barbeiros)}")
print(f"📅 Meses previstos: {len(meses_futuros)}")
print(f"📈 Total de previsões: {len(previsoes_futuras)}")

# Estatísticas das previsões
if len(previsoes_futuras) > 0:
    faturamento_total_previsto = sum(p['faturamento_previsto'] for p in previsoes_futuras)
    print(f"💰 Faturamento total previsto 2025: R$ {faturamento_total_previsto:,.2f}")
    print(f"📊 Faturamento médio por mês/barbeiro: R$ {faturamento_total_previsto / len(previsoes_futuras):,.2f}")

print("\n✅ Modelo de previsão concluído com sucesso!") 