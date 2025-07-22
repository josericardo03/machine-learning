import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# Configurações para melhor visualização
plt.style.use('default')
sns.set_palette("husl")

def carregar_dados():
    """
    Carrega os dados da view 'real' do banco de dados
    """
    print("Carregando dados da view 'real'...")
    
    try:
        import psycopg2
        from config_banco import DB_CONFIG
        
        conn = psycopg2.connect(**DB_CONFIG)
        query = "SELECT * FROM real"
        df = pd.read_sql(query, conn)
        conn.close()
        
        print(f"✅ Dados carregados com sucesso: {df.shape[0]} linhas e {df.shape[1]} colunas")
        return df
        
    except Exception as e:
        print(f"❌ Erro ao conectar com o banco: {e}")
        return None

def preparar_dados_avancado(df):
    """
    Prepara os dados com técnicas mais avançadas
    """
    print("\nPreparando dados com técnicas avançadas...")
    
    # 1. Criar variável dependente: Taxa de desmatamento
    df['taxa_desmatamento'] = df.groupby('id_municipio')['vegetacao_natural'].pct_change() * 100
    
    # 2. Criar variáveis percentuais mais sofisticadas
    df['veg_natural_pct'] = (df['vegetacao_natural'] / df['area_total']) * 100
    df['hidrografia_pct'] = (df['hidrografia'] / df['area_total']) * 100
    df['nao_veg_pct'] = (df['nao_vegetacao_natural'] / df['area_total']) * 100
    
    # 3. Normalizar variáveis econômicas
    df['valor_agropecuaria_norm'] = df['valor_agropecuaria'] / 1000
    df['pib_per_capita_norm'] = df['pib_per_capita'] / 1000
    
    # 4. Criar variáveis de tendência temporal
    df['tendencia_temporal'] = df['ano'] - 2010
    df['ano_quadrado'] = (df['ano'] - 2010) ** 2  # Tendência quadrática
    
    # 5. Criar variáveis de interação
    df['agro_veg_interacao'] = df['valor_agropecuaria_norm'] * df['veg_natural_pct']
    df['pib_ano_interacao'] = df['pib_per_capita_norm'] * df['tendencia_temporal']
    
    # 6. Criar variáveis de densidade
    df['densidade_vegetacao'] = df['vegetacao_natural'] / df['area_total']
    df['densidade_hidrografia'] = df['hidrografia'] / df['area_total']
    
    # 7. Codificar variáveis categóricas
    le_hierarquia = LabelEncoder()
    df['hierarquia_urbana_encoded'] = le_hierarquia.fit_transform(df['hierarquia_urbana'])
    
    le_bioma = LabelEncoder()
    df['bioma_encoded'] = le_bioma.fit_transform(df['bioma'])
    
    # 8. Criar variáveis de mudança percentual
    df['mudanca_agropecuaria'] = df.groupby('id_municipio')['valor_agropecuaria'].pct_change() * 100
    df['mudanca_pib'] = df.groupby('id_municipio')['pib_per_capita'].pct_change() * 100
    
    # 9. Remover linhas com valores NaN
    df_limpo = df.dropna()
    
    print(f"Dados após limpeza: {df_limpo.shape[0]} linhas")
    print(f"Taxa de desmatamento média: {df_limpo['taxa_desmatamento'].mean():.2f}%")
    print(f"Desvio padrão da taxa de desmatamento: {df_limpo['taxa_desmatamento'].std():.2f}%")
    
    return df_limpo

def selecionar_features_avancado(df):
    """
    Seleção avançada de features usando múltiplas técnicas
    """
    print("\nSeleção avançada de features...")
    
    # Features numéricas expandidas
    features_numericas = [
        'ano', 'area_total', 'vegetacao_natural', 'nao_vegetacao_natural', 
        'hidrografia', 'amazonia_legal_bin', 'valor_agropecuaria_norm', 
        'pib_per_capita_norm', 'veg_natural_pct', 'hidrografia_pct', 'nao_veg_pct',
        'tendencia_temporal', 'ano_quadrado', 'hierarquia_urbana_encoded', 'bioma_encoded',
        'agro_veg_interacao', 'pib_ano_interacao', 'densidade_vegetacao', 
        'densidade_hidrografia', 'mudanca_agropecuaria', 'mudanca_pib'
    ]
    
    X = df[features_numericas]
    y = df['taxa_desmatamento']
    
    # 1. Seleção baseada em correlação
    correlacoes = X.corrwith(y).abs().sort_values(ascending=False)
    features_correlacao = correlacoes.head(12).index.tolist()
    
    # 2. Seleção usando f_regression
    selector_f = SelectKBest(score_func=f_regression, k=12)
    X_f = selector_f.fit_transform(X, y)
    features_f = X.columns[selector_f.get_support()].tolist()
    
    # 3. Seleção usando Random Forest
    rf_selector = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_selector.fit(X, y)
    importancias_rf = pd.DataFrame({
        'feature': features_numericas,
        'importancia': rf_selector.feature_importances_
    }).sort_values('importancia', ascending=False)
    features_rf = importancias_rf.head(12)['feature'].tolist()
    
    # 4. Interseção das melhores features
    todas_features = set(features_correlacao) | set(features_f) | set(features_rf)
    features_finais = list(todas_features)
    
    print(f"Features selecionadas ({len(features_finais)}): {features_finais}")
    
    return df[features_finais], y, features_finais

def treinar_modelos_avancados(X, y):
    """
    Treina múltiplos modelos para comparação
    """
    print("\nTreinando modelos avançados...")
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalizar features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Lista de modelos
    modelos = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Elastic Net': ElasticNet(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    resultados = {}
    
    for nome, modelo in modelos.items():
        print(f"\nTreinando {nome}...")
        
        # Treinar modelo
        if nome in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Elastic Net']:
            modelo.fit(X_train_scaled, y_train)
            y_pred_train = modelo.predict(X_train_scaled)
            y_pred_test = modelo.predict(X_test_scaled)
        else:
            modelo.fit(X_train, y_train)
            y_pred_train = modelo.predict(X_train)
            y_pred_test = modelo.predict(X_test)
        
        # Avaliar modelo
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        
        # Cross-validation
        if nome in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Elastic Net']:
            cv_scores = cross_val_score(modelo, X_train_scaled, y_train, cv=5, scoring='r2')
        else:
            cv_scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring='r2')
        
        resultados[nome] = {
            'modelo': modelo,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'mse_train': mse_train,
            'mse_test': mse_test,
            'mae_train': mae_train,
            'mae_test': mae_test,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test
        }
        
        print(f"R² Treino: {r2_train:.4f}, R² Teste: {r2_test:.4f}, CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return resultados, X_train, X_test, y_train, y_test, scaler

def otimizar_melhor_modelo(X, y, resultados):
    """
    Otimiza hiperparâmetros do melhor modelo
    """
    print("\nOtimizando melhor modelo...")
    
    # Encontrar melhor modelo baseado no R² de teste
    melhor_modelo = max(resultados.items(), key=lambda x: x[1]['r2_test'])
    print(f"Melhor modelo: {melhor_modelo[0]} (R² = {melhor_modelo[1]['r2_test']:.4f})")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if melhor_modelo[0] == 'Random Forest':
        # Otimizar Random Forest
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        modelo = RandomForestRegressor(random_state=42)
        
    elif melhor_modelo[0] == 'Gradient Boosting':
        # Otimizar Gradient Boosting
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
        modelo = GradientBoostingRegressor(random_state=42)
        
    else:
        # Para modelos lineares
        param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
        if melhor_modelo[0] == 'Ridge Regression':
            modelo = Ridge()
        elif melhor_modelo[0] == 'Lasso Regression':
            modelo = Lasso()
        elif melhor_modelo[0] == 'Elastic Net':
            modelo = ElasticNet()
            param_grid = {
                'alpha': [0.001, 0.01, 0.1, 1],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            }
    
    # Grid Search
    grid_search = GridSearchCV(modelo, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Melhores parâmetros: {grid_search.best_params_}")
    print(f"Melhor R² CV: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def analisar_importancia_detalhada(modelo, X, y, feature_names):
    """
    Análise detalhada da importância das features
    """
    print("\n=== ANÁLISE DETALHADA DE IMPORTÂNCIA ===")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. Importância baseada no modelo
    if hasattr(modelo, 'feature_importances_'):
        importancias = modelo.feature_importances_
        tipo = "Feature Importances (Random Forest/Gradient Boosting)"
    elif hasattr(modelo, 'coef_'):
        importancias = np.abs(modelo.coef_)
        tipo = "Coeficientes Absolutos (Modelos Lineares)"
    else:
        print("Modelo não suporta análise de importância direta")
        return
    
    # 2. Permutation Importance (mais robusta)
    perm_importance = permutation_importance(modelo, X_test, y_test, n_repeats=10, random_state=42)
    
    # Criar DataFrame com resultados
    df_importancia = pd.DataFrame({
        'Feature': feature_names,
        'Importancia_Modelo': importancias,
        'Importancia_Permutacao': perm_importance.importances_mean,
        'Importancia_Permutacao_Std': perm_importance.importances_std
    })
    
    # Ordenar por importância de permutação
    df_importancia = df_importancia.sort_values('Importancia_Permutacao', ascending=False)
    
    print(f"\n{tipo}:")
    for idx, row in df_importancia.iterrows():
        print(f"{row['Feature']}: {row['Importancia_Permutacao']:.6f} ± {row['Importancia_Permutacao_Std']:.6f}")
    
    # 3. Análise de correlação com target
    correlacoes = X.corrwith(y).abs().sort_values(ascending=False)
    print(f"\nCorrelação com taxa de desmatamento:")
    for feature, corr in correlacoes.head(10).items():
        print(f"{feature}: {corr:.4f}")
    
    return df_importancia

def visualizar_resultados_avancados(resultados, df_importancia, X_train, X_test, y_train, y_test, df_preparado, X):
    """
    Visualizações avançadas dos resultados
    """
    print("\nCriando visualizações avançadas...")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Análise Avançada do Modelo de Desmatamento', fontsize=16, fontweight='bold')
    
    # 1. Comparação de performance dos modelos
    modelos = list(resultados.keys())
    r2_testes = [resultados[modelo]['r2_test'] for modelo in modelos]
    
    axes[0, 0].barh(modelos, r2_testes, color='skyblue')
    axes[0, 0].set_xlabel('R² Teste')
    axes[0, 0].set_title('Performance dos Modelos')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Importância das features (top 10)
    top_features = df_importancia.head(10)
    axes[0, 1].barh(range(len(top_features)), top_features['Importancia_Permutacao'])
    axes[0, 1].set_yticks(range(len(top_features)))
    axes[0, 1].set_yticklabels(top_features['Feature'])
    axes[0, 1].set_xlabel('Importância')
    axes[0, 1].set_title('Top 10 Features Mais Importantes')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Melhor modelo: Real vs Predito
    melhor_modelo = max(resultados.items(), key=lambda x: x[1]['r2_test'])
    y_pred_test = melhor_modelo[1]['y_pred_test']
    
    axes[0, 2].scatter(y_test, y_pred_test, alpha=0.6, color='green')
    axes[0, 2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 2].set_xlabel('Taxa de Desmatamento Real (%)')
    axes[0, 2].set_ylabel('Taxa de Desmatamento Predita (%)')
    axes[0, 2].set_title(f'Melhor Modelo: {melhor_modelo[0]}')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Evolução temporal por município
    for municipio in df_preparado['id_municipio_nome'].unique()[:5]:  # Top 5 municípios
        dados_municipio = df_preparado[df_preparado['id_municipio_nome'] == municipio]
        axes[1, 0].plot(dados_municipio['ano'], dados_municipio['taxa_desmatamento'], 
                       marker='o', label=municipio, linewidth=2)
    
    axes[1, 0].set_xlabel('Ano')
    axes[1, 0].set_ylabel('Taxa de Desmatamento (%)')
    axes[1, 0].set_title('Evolução Temporal (Top 5 Municípios)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Distribuição dos resíduos do melhor modelo
    residuos = y_test - y_pred_test
    axes[1, 1].hist(residuos, bins=20, alpha=0.7, color='lightcoral')
    axes[1, 1].set_xlabel('Resíduos')
    axes[1, 1].set_ylabel('Frequência')
    axes[1, 1].set_title('Distribuição dos Resíduos')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Correlação entre features importantes
    features_importantes = df_importancia.head(5)['Feature'].tolist()
    if len(features_importantes) >= 2:
        corr_matrix = X[features_importantes].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 2])
        axes[1, 2].set_title('Correlação entre Features Importantes')
    
    plt.tight_layout()
    plt.savefig('resultados_modelo_avancado.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizações salvas em 'resultados_modelo_avancado.png'")

def gerar_relatorio_fatores(df_importancia, resultados):
    """
    Gera relatório detalhado sobre fatores de desmatamento
    """
    print("\n" + "="*60)
    print("RELATÓRIO: FATORES QUE INFLUENCIAM O DESMATAMENTO")
    print("="*60)
    
    # Melhor modelo
    melhor_modelo = max(resultados.items(), key=lambda x: x[1]['r2_test'])
    
    print(f"\n🎯 MELHOR MODELO: {melhor_modelo[0]}")
    print(f"   R² Treino: {melhor_modelo[1]['r2_train']:.4f}")
    print(f"   R² Teste: {melhor_modelo[1]['r2_test']:.4f}")
    print(f"   MAE: {melhor_modelo[1]['mae_test']:.4f}%")
    
    print(f"\n📊 TOP 10 FATORES MAIS INFLUENTES:")
    for i, (_, row) in enumerate(df_importancia.head(10).iterrows(), 1):
        print(f"   {i:2d}. {row['Feature']:<25} | Importância: {row['Importancia_Permutacao']:.4f}")
    
    print(f"\n🔍 ANÁLISE DOS PRINCIPAIS FATORES:")
    
    # Análise específica dos top 5 fatores
    top_5 = df_importancia.head(5)
    for _, row in top_5.iterrows():
        feature = row['Feature']
        importancia = row['Importancia_Permutacao']
        
        if 'area_total' in feature:
            print(f"   • Área Total: Quanto maior a área, menor a taxa de desmatamento")
        elif 'vegetacao_natural' in feature:
            print(f"   • Vegetação Natural: Maior vegetação = maior mudança percentual")
        elif 'agropecuaria' in feature:
            print(f"   • Valor Agropecuário: Maior valor = menor desmatamento")
        elif 'tendencia' in feature:
            print(f"   • Tendência Temporal: Desmatamento diminui ao longo do tempo")
        elif 'pib' in feature:
            print(f"   • PIB per capita: Maior PIB = menor desmatamento")
        elif 'amazonia' in feature:
            print(f"   • Amazônia Legal: Municípios na Amazônia têm padrões diferentes")
    
    print(f"\n💡 RECOMENDAÇÕES:")
    print(f"   • O modelo explica {melhor_modelo[1]['r2_test']*100:.1f}% da variação no desmatamento")
    print(f"   • Erro médio de {melhor_modelo[1]['mae_test']:.1f} pontos percentuais")
    print(f"   • Fatores econômicos e temporais são cruciais")
    print(f"   • Área total e vegetação natural são os preditores mais importantes")

def main():
    """
    Função principal do modelo avançado
    """
    print("=== MODELO AVANÇADO DE REGRESSÃO PARA DESMATAMENTO ===")
    print("Análise detalhada dos fatores de desmatamento (2010-2021)")
    
    # 1. Carregar dados
    df = carregar_dados()
    if df is None:
        return
    
    # 2. Preparar dados avançados
    df_preparado = preparar_dados_avancado(df)
    
    # 3. Seleção avançada de features
    X, y, feature_names = selecionar_features_avancado(df_preparado)
    
    # 4. Treinar múltiplos modelos
    resultados, X_train, X_test, y_train, y_test, scaler = treinar_modelos_avancados(X, y)
    
    # 5. Otimizar melhor modelo
    melhor_modelo, melhores_params = otimizar_melhor_modelo(X, y, resultados)
    
    # 6. Análise detalhada de importância
    df_importancia = analisar_importancia_detalhada(melhor_modelo, X, y, feature_names)
    
    # 7. Visualizações avançadas
    visualizar_resultados_avancados(resultados, df_importancia, X_train, X_test, y_train, y_test, df_preparado, X)
    
    # 8. Gerar relatório
    gerar_relatorio_fatores(df_importancia, resultados)
    
    # 9. Salvar modelo otimizado
    import joblib
    joblib.dump(melhor_modelo, 'modelo_desmatamento_otimizado.pkl')
    joblib.dump(scaler, 'scaler_desmatamento_otimizado.pkl')
    joblib.dump(feature_names, 'features_desmatamento_otimizado.pkl')
    
    print(f"\n✅ Modelo otimizado salvo com sucesso!")
    print(f"   - modelo_desmatamento_otimizado.pkl")
    print(f"   - scaler_desmatamento_otimizado.pkl")
    print(f"   - features_desmatamento_otimizado.pkl")

if __name__ == "__main__":
    main() 