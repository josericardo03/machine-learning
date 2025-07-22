import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression
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
        
        # Configurações de conexão com o banco
        conn = psycopg2.connect(**DB_CONFIG)
        
        # Query para carregar dados da view 'real'
        query = "SELECT * FROM real"
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        print(f"✅ Dados carregados com sucesso: {df.shape[0]} linhas e {df.shape[1]} colunas")
        print(f"Colunas disponíveis: {list(df.columns)}")
        
        return df
        
    except ImportError:
        print("❌ Erro: psycopg2 não está instalado. Execute: pip install psycopg2-binary")
        return None
    except Exception as e:
        print(f"❌ Erro ao conectar com o banco: {e}")
        print("⚠️  Verifique as configurações de conexão no código")
        return None

def preparar_dados(df):
    """
    Prepara os dados para o modelo de desmatamento
    """
    print("\nPreparando dados para o modelo...")
    
    # 1. Criar variável dependente: Taxa de desmatamento
    df['taxa_desmatamento'] = df.groupby('id_municipio')['vegetacao_natural'].pct_change() * 100
    
    # 2. Criar variáveis percentuais
    df['veg_natural_pct'] = (df['vegetacao_natural'] / df['area_total']) * 100
    df['hidrografia_pct'] = (df['hidrografia'] / df['area_total']) * 100
    
    # 3. Normalizar valor agropecuária (dividir por 1000 para facilitar interpretação)
    df['valor_agropecuaria_norm'] = df['valor_agropecuaria'] / 1000
    
    # 4. Criar variável de tendência temporal
    df['tendencia_temporal'] = df['ano'] - 2010
    
    # 5. Codificar variáveis categóricas
    le_hierarquia = LabelEncoder()
    df['hierarquia_urbana_encoded'] = le_hierarquia.fit_transform(df['hierarquia_urbana'])
    
    le_bioma = LabelEncoder()
    df['bioma_encoded'] = le_bioma.fit_transform(df['bioma'])
    
    # 6. Remover linhas com valores NaN (primeiro ano de cada município)
    df_limpo = df.dropna()
    
    print(f"Dados após limpeza: {df_limpo.shape[0]} linhas")
    print(f"Taxa de desmatamento média: {df_limpo['taxa_desmatamento'].mean():.2f}%")
    print(f"Desvio padrão da taxa de desmatamento: {df_limpo['taxa_desmatamento'].std():.2f}%")
    
    return df_limpo

def selecionar_features(df):
    """
    Seleciona as features mais relevantes para o modelo
    """
    print("\nSelecionando features...")
    
    # Features numéricas para o modelo
    features_numericas = [
        'ano', 'area_total', 'vegetacao_natural', 'nao_vegetacao_natural', 
        'hidrografia', 'amazonia_legal_bin', 'valor_agropecuaria_norm', 
        'pib_per_capita', 'veg_natural_pct', 'hidrografia_pct', 
        'tendencia_temporal', 'hierarquia_urbana_encoded', 'bioma_encoded'
    ]
    
    X = df[features_numericas]
    y = df['taxa_desmatamento']
    
    # Seleção de features usando f_regression
    selector = SelectKBest(score_func=f_regression, k=8)
    X_selected = selector.fit_transform(X, y)
    
    # Obter nomes das features selecionadas
    selected_features = X.columns[selector.get_support()].tolist()
    print(f"Features selecionadas: {selected_features}")
    
    return X_selected, y, selected_features

def treinar_modelo(X, y):
    """
    Treina o modelo de regressão linear
    """
    print("\nTreinando modelo de regressão linear...")
    
    # Dividir dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normalizar features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Treinar modelo
    modelo = LinearRegression()
    modelo.fit(X_train_scaled, y_train)
    
    # Fazer previsões
    y_pred_train = modelo.predict(X_train_scaled)
    y_pred_test = modelo.predict(X_test_scaled)
    
    # Avaliar modelo
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    print(f"\n=== RESULTADOS DO MODELO ===")
    print(f"R² Treino: {r2_train:.4f}")
    print(f"R² Teste: {r2_test:.4f}")
    print(f"MSE Treino: {mse_train:.4f}")
    print(f"MSE Teste: {mse_test:.4f}")
    print(f"MAE Treino: {mae_train:.4f}")
    print(f"MAE Teste: {mae_test:.4f}")
    
    return modelo, scaler, X_train_scaled, X_test_scaled, y_train, y_test, y_pred_train, y_pred_test

def analisar_features(modelo, selected_features):
    """
    Analisa a importância das features no modelo
    """
    print("\n=== ANÁLISE DE FEATURES ===")
    
    # Coeficientes do modelo
    coeficientes = modelo.coef_
    
    # Criar DataFrame com coeficientes
    feature_importance = pd.DataFrame({
        'Feature': selected_features,
        'Coeficiente': coeficientes,
        'Importância_Absoluta': np.abs(coeficientes)
    })
    
    # Ordenar por importância absoluta
    feature_importance = feature_importance.sort_values('Importância_Absoluta', ascending=False)
    
    print("\nImportância das features (coeficientes):")
    for idx, row in feature_importance.iterrows():
        print(f"{row['Feature']}: {row['Coeficiente']:.6f}")
    
    return feature_importance

def visualizar_resultados(y_train, y_test, y_pred_train, y_pred_test, df):
    """
    Cria visualizações dos resultados do modelo
    """
    print("\nCriando visualizações...")
    
    # Configurar subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Análise do Modelo de Desmatamento', fontsize=16, fontweight='bold')
    
    # 1. Gráfico de dispersão: Valores reais vs preditos (treino)
    axes[0, 0].scatter(y_train, y_pred_train, alpha=0.6, color='blue')
    axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Taxa de Desmatamento Real (%)')
    axes[0, 0].set_ylabel('Taxa de Desmatamento Predita (%)')
    axes[0, 0].set_title('Treino: Real vs Predito')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Gráfico de dispersão: Valores reais vs preditos (teste)
    axes[0, 1].scatter(y_test, y_pred_test, alpha=0.6, color='green')
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('Taxa de Desmatamento Real (%)')
    axes[0, 1].set_ylabel('Taxa de Desmatamento Predita (%)')
    axes[0, 1].set_title('Teste: Real vs Predito')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Evolução temporal da taxa de desmatamento por município
    for municipio in df['id_municipio_nome'].unique():
        dados_municipio = df[df['id_municipio_nome'] == municipio]
        axes[1, 0].plot(dados_municipio['ano'], dados_municipio['taxa_desmatamento'], 
                       marker='o', label=municipio, linewidth=2)
    
    axes[1, 0].set_xlabel('Ano')
    axes[1, 0].set_ylabel('Taxa de Desmatamento (%)')
    axes[1, 0].set_title('Evolução Temporal do Desmatamento por Município')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Distribuição dos resíduos
    residuos_train = y_train - y_pred_train
    residuos_test = y_test - y_pred_test
    
    axes[1, 1].hist(residuos_train, bins=15, alpha=0.7, label='Treino', color='blue')
    axes[1, 1].hist(residuos_test, bins=15, alpha=0.7, label='Teste', color='green')
    axes[1, 1].set_xlabel('Resíduos')
    axes[1, 1].set_ylabel('Frequência')
    axes[1, 1].set_title('Distribuição dos Resíduos')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('resultados_modelo_desmatamento.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizações salvas em 'resultados_modelo_desmatamento.png'")

def salvar_modelo(modelo, scaler, selected_features):
    """
    Salva o modelo treinado para uso futuro
    """
    import joblib
    
    # Salvar modelo e scaler
    joblib.dump(modelo, 'modelo_desmatamento.pkl')
    joblib.dump(scaler, 'scaler_desmatamento.pkl')
    joblib.dump(selected_features, 'features_desmatamento.pkl')
    
    print("\nModelo salvo com sucesso!")
    print("- modelo_desmatamento.pkl")
    print("- scaler_desmatamento.pkl") 
    print("- features_desmatamento.pkl")

def main():
    """
    Função principal que executa todo o pipeline
    """
    print("=== MODELO DE REGRESSÃO LINEAR PARA DESMATAMENTO ===")
    print("Análise do avanço do desmatamento por município (2010-2021)")
    
    # 1. Carregar dados
    df = carregar_dados()
    
    if df is None:
        print("❌ Não foi possível carregar os dados. Verifique a conexão com o banco.")
        return
    
    # 2. Preparar dados
    df_preparado = preparar_dados(df)
    
    if df_preparado is None or df_preparado.empty:
        print("❌ Não foi possível preparar os dados. Verifique se a view 'real' contém os dados necessários.")
        return
    
    # 3. Selecionar features
    X, y, selected_features = selecionar_features(df_preparado)
    
    # 4. Treinar modelo
    modelo, scaler, X_train_scaled, X_test_scaled, y_train, y_test, y_pred_train, y_pred_test = treinar_modelo(X, y)
    
    # 5. Analisar features
    feature_importance = analisar_features(modelo, selected_features)
    
    # 6. Visualizar resultados
    visualizar_resultados(y_train, y_test, y_pred_train, y_pred_test, df_preparado)
    
    # 7. Salvar modelo
    salvar_modelo(modelo, scaler, selected_features)
    
    print("\n=== RESUMO FINAL ===")
    print("✅ Modelo de regressão linear treinado com sucesso!")
    print("✅ Variável dependente: Taxa de desmatamento (%)")
    print("✅ Período analisado: 2010-2021")
    print("✅ Features utilizadas: Variáveis econômicas, ambientais e temporais")
    print("✅ Modelo salvo para uso futuro")

if __name__ == "__main__":
    main() 