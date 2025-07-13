import pandas as pd
from sqlalchemy import create_engine, text
import os

# Dados da conexão
usuario = "postgres.roziechzdpxxdtzlkaep"
senha = "Jj20134849%40%40%40"  # senha codificada
host = "aws-0-sa-east-1.pooler.supabase.com"
porta = 5432
banco = "postgres"

print("Tentando conectar ao banco de dados...")

try:
    # Conexão SQLAlchemy
    engine = create_engine(f"postgresql://{usuario}:{senha}@{host}:{porta}/{banco}")
    
    # Testar a conexão
    with engine.connect() as conn:
        print("✅ Conexão estabelecida com sucesso!")
        
        # Verificar se a VIEW existe
        check_view = text("SELECT EXISTS (SELECT 1 FROM information_schema.views WHERE table_name = 'faturamento_mensal_barbeiro')")
        result = conn.execute(check_view).fetchone()
        
        if result is not None and result[0]:
            print("✅ VIEW 'faturamento_mensal_barbeiro' encontrada!")
        else:
            print("❌ VIEW 'faturamento_mensal_barbeiro' não encontrada!")
            # Listar todas as VIEWs disponíveis
            list_views = text("SELECT table_name FROM information_schema.views WHERE table_schema = 'public'")
            views = conn.execute(list_views).fetchall()
            print("VIEWs disponíveis:")
            for view in views:
                print(f"  - {view[0]}")
            exit(1)
    
    # Ler dados da VIEW
    print("Lendo dados da VIEW...")
    query = "SELECT * FROM faturamento_mensal_barbeiro"
    
    df = pd.read_sql(query, con=engine)
    
    print(f"✅ Dados carregados com sucesso!")
    print(f"📊 Total de registros: {len(df)}")
    print(f"📋 Colunas: {list(df.columns)}")
    
    if len(df) > 0:
        # --- Análise exploratória (já existente) ---
        print("\nFormato do DataFrame (linhas, colunas):")
        print(df.shape)
        print("\nPrimeiros 5 registros:")
        print(df.head())
        print("\nEstatísticas básicas:")
        print(df.describe(include='all'))
        print("\nTipos de dados de cada coluna:")
        print(df.dtypes)
        print("\nInformações do DataFrame:")
        print(df.info())
        print("\nVerificando valores nulos por coluna:")
        print(df.isnull().sum())
        print("\nColunas com todos os valores vazios:")
        print(df.columns[df.isnull().all()])
        print("\nColunas que parecem ser numéricas mas vieram como texto:")
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_numeric(df[col])
                    print(f"  - {col}")
                except Exception:
                    pass
        if 'faturamento' in df.columns:
            negativos = df[df['faturamento'] < 0]
            if not negativos.empty:
                print("⚠️  Existem valores negativos na coluna 'faturamento':")
                print(negativos[['faturamento']])
            else:
                print("✅ Todos os valores de 'faturamento' são positivos ou zero.")

        # --- Pipeline de Machine Learning ---
        print("\n--- Pipeline de Machine Learning ---")
        # 1. Separar X e y
        X = df[[
            "num_agendamentos",
            "valor_medio_agendamento",
            "num_clientes_unicos",
            "num_promocoes",
            "dias_trabalhados",
            "mes"
        ]]
        y = df["faturamento_total"]
        print("\nFeatures (X):")
        print(X.head())
        print("\nTarget (y):")
        print(y.head())

        # 2. Separar em treino e teste
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if hasattr(X_train, 'shape') and hasattr(X_test, 'shape'):
            print(f"\nTamanho do treino: {X_train.shape[0]} registros")
            print(f"Tamanho do teste: {X_test.shape[0]} registros")
        else:
            print(f"\nTamanho do treino: {len(X_train)} registros")
            print(f"Tamanho do teste: {len(X_test)} registros")

        # 3. (Opcional) Padronizar os dados
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 4. Treinar modelo de Regressão Linear
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        print("\nModelo treinado com sucesso!")

        # 5. Avaliar o modelo
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        y_pred = model.predict(X_test_scaled)
        print("\n--- Avaliação do Modelo ---")
        print("MAE (Erro Médio Absoluto):", mean_absolute_error(y_test, y_pred))
        print("MSE (Erro Quadrático Médio):", mean_squared_error(y_test, y_pred))
        print("R² (Coeficiente de Determinação):", r2_score(y_test, y_pred))

        # 6. (Opcional) Mostrar os coeficientes do modelo
        print("\nCoeficientes do modelo:")
        for nome, coef in zip(X.columns, model.coef_):
            print(f"{nome}: {coef}")
        print(f"Intercepto: {model.intercept_}")

        # 7. Gráfico Real vs Previsto
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,6))
        plt.scatter(y_test, y_pred, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
        plt.xlabel('Faturamento Real')
        plt.ylabel('Faturamento Previsto')
        plt.title('Real vs Previsto')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 8. Distribuição dos resíduos (erros)
        import seaborn as sns
        residuals = y_test - y_pred
        plt.figure(figsize=(8,6))
        sns.histplot(residuals, kde=True)
        plt.title('Distribuição dos Resíduos')
        plt.xlabel('Resíduo (Erro)')
        plt.ylabel('Frequência')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 9. Gráfico de dispersão dos resíduos
        plt.figure(figsize=(8,6))
        plt.scatter(y_test, residuals)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Faturamento Real')
        plt.ylabel('Resíduo (Real - Previsto)')
        plt.title('Resíduos do Modelo')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    else:
        print("⚠️  A VIEW está vazia (0 registros)")
        
except Exception as e:
    print(f"❌ Erro: {e}")
    print(f"Tipo do erro: {type(e).__name__}")
