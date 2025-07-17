import pandas as pd
from sqlalchemy import create_engine, text
import os
import numpy as np

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
        check_view = text("SELECT EXISTS (SELECT 1 FROM information_schema.views WHERE table_name = 'faturamento_mensal_barbeiro')")
        result = conn.execute(check_view).fetchone()
        if result is not None and result[0]:
            print("✅ VIEW 'faturamento_mensal_barbeiro' encontrada!")
        else:
            print("❌ VIEW 'faturamento_mensal_barbeiro' não encontrada!")
            list_views = text("SELECT table_name FROM information_schema.views WHERE table_schema = 'public'")
            views = conn.execute(list_views).fetchall()
            print("VIEWs disponíveis:")
            for view in views:
                print(f"  - {view[0]}")
            exit(1)
    
    print("Lendo dados da VIEW...")
    query = "SELECT * FROM faturamento_mensal_barbeiro"
    df = pd.read_sql(query, con=engine)
    print(f"✅ Dados carregados com sucesso!")
    print(f"📊 Total de registros: {len(df)}")
    print(f"📋 Colunas: {list(df.columns)}")
    
    if len(df) > 0:
        if len(df) < 100:
            print("⚠️  AVISO: O volume de dados é pequeno. Modelos complexos podem não generalizar bem.")
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

        # --- Diagnóstico de Multicolinearidade (VIF) ---
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        print("\n--- VIF (Variance Inflation Factor) das variáveis ---")
        vif_data = pd.DataFrame()
        vif_data["Variável"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        print(vif_data)
        # Remover só a variável de maior VIF se for MUITO alta (>30)
        if vif_data["VIF"].max() > 30:
            drop_var = vif_data.sort_values("VIF", ascending=False)["Variável"].iloc[0]
            print(f"Removendo variável com VIF muito alto: {drop_var}")
            X = X.drop(columns=[drop_var])
            vif_data = pd.DataFrame()
            vif_data["Variável"] = X.columns
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            print(vif_data)

        # 2. Separar em treino e teste
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"\nTamanho do treino: {X_train.shape[0]} registros")
        print(f"Tamanho do teste: {X_test.shape[0]} registros")

        # 3. Padronizar os dados
        from sklearn.preprocessing import StandardScaler, PolynomialFeatures
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 4. Testar também PolynomialFeatures (grau 2)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X_train_scaled)
        X_test_poly = poly.transform(X_test_scaled)

        # 5. Modelos a serem testados
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        try:
            from xgboost import XGBRegressor
            xgb_available = True
        except ImportError:
            xgb_available = False
        from sklearn.model_selection import GridSearchCV, cross_val_score
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        modelos = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'ElasticNet': ElasticNet(),
            'RandomForest': RandomForestRegressor(random_state=42),
            'GradientBoosting': GradientBoostingRegressor(random_state=42)
        }
        if xgb_available:
            modelos['XGBoost'] = XGBRegressor(random_state=42, verbosity=0)

        # 6. Hiperparâmetros para GridSearch (mais amplo)
        params = {
            'Ridge': {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]},
            'Lasso': {'alpha': [0.001, 0.01, 0.1, 1, 10]},
            'ElasticNet': {'alpha': [0.001, 0.01, 0.1, 1, 10], 'l1_ratio': [0.1, 0.5, 0.9]},
            'RandomForest': {'n_estimators': [50, 100, 200], 'max_depth': [2, 4, 6, 8]},
            'GradientBoosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.05, 0.1, 0.2]},
        }
        if xgb_available:
            params['XGBoost'] = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.05, 0.1, 0.2]}

        def avaliar_modelos(Xtr, Xte, ytr, yte, descricao):
            print(f"\n=== Avaliando modelos: {descricao} ===")
            resultados = {}
            melhores_modelos = {}
            for nome, modelo in modelos.items():
                print(f"\nTreinando e avaliando modelo: {nome}")
                if nome in params:
                    grid = GridSearchCV(modelo, params[nome], cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
                    grid.fit(Xtr, ytr)
                    melhor_modelo = grid.best_estimator_
                    print(f"Melhores parâmetros: {grid.best_params_}")
                else:
                    melhor_modelo = modelo.fit(Xtr, ytr)
                y_pred = melhor_modelo.predict(Xte)
                mae = mean_absolute_error(yte, y_pred)
                mse = mean_squared_error(yte, y_pred)
                r2 = r2_score(yte, y_pred)
                cv_r2 = cross_val_score(melhor_modelo, Xtr, ytr, cv=5, scoring='r2').mean()
                resultados[nome] = {'MAE': mae, 'MSE': mse, 'R2': r2, 'CV_R2': cv_r2}
                melhores_modelos[nome] = melhor_modelo
                print(f"MAE: {mae:.2f} | MSE: {mse:.2f} | R²: {r2:.4f} | CV R²: {cv_r2:.4f}")
                # Feature importance
                if hasattr(melhor_modelo, 'feature_importances_'):
                    print("Importância das variáveis:")
                    for nome_feat, imp in zip(poly.get_feature_names_out(X.columns) if 'poly' in descricao else X.columns, melhor_modelo.feature_importances_):
                        print(f"  {nome_feat}: {imp:.3f}")
                elif hasattr(melhor_modelo, 'coef_'):
                    print("Coeficientes das variáveis:")
                    for nome_feat, coef in zip(poly.get_feature_names_out(X.columns) if 'poly' in descricao else X.columns, melhor_modelo.coef_):
                        print(f"  {nome_feat}: {coef:.3f}")
            return resultados, melhores_modelos

        # Avaliar modelos com features originais
        resultados1, melhores1 = avaliar_modelos(X_train_scaled, X_test_scaled, y_train, y_test, 'originais')
        # Avaliar modelos com features polinomiais
        resultados2, melhores2 = avaliar_modelos(X_train_poly, X_test_poly, y_train, y_test, 'poly')

        # 7. Relatório final
        print("\n=== RELATÓRIO FINAL DE DESEMPENHO ===")
        df_resultados1 = pd.DataFrame(resultados1).T
        df_resultados2 = pd.DataFrame(resultados2).T
        print("\nModelos com features originais:")
        print(df_resultados1.sort_values('R2', ascending=False))
        print("\nModelos com features polinomiais:")
        print(df_resultados2.sort_values('R2', ascending=False))
        # Melhor de todos
        melhor_nome1 = df_resultados1['R2'].idxmax()
        melhor_nome2 = df_resultados2['R2'].idxmax()
        if df_resultados1['R2'].max() >= df_resultados2['R2'].max():
            melhor_nome = melhor_nome1
            melhor_modelo = melhores1[melhor_nome]
            Xte = X_test_scaled
            descricao = 'originais'
        else:
            melhor_nome = melhor_nome2
            melhor_modelo = melhores2[melhor_nome]
            Xte = X_test_poly
            descricao = 'poly'
        print(f"\n🏆 Melhor modelo geral: {melhor_nome} ({descricao})")
        if descricao == 'originais':
            print(df_resultados1.loc[melhor_nome])
        else:
            print(df_resultados2.loc[melhor_nome])

        # 8. Gráficos do melhor modelo
        import matplotlib.pyplot as plt
        import seaborn as sns
        y_pred = melhor_modelo.predict(Xte)
        residuals = y_test - y_pred
        plt.figure(figsize=(8,6))
        sns.histplot(residuals, kde=True)
        plt.title('Distribuição dos Resíduos')
        plt.xlabel('Resíduo (Erro)')
        plt.ylabel('Frequência')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(8,6))
        plt.scatter(y_test, residuals)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Faturamento Real')
        plt.ylabel('Resíduo (Real - Previsto)')
        plt.title('Resíduos do Modelo')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 9. Previsão dos próximos meses para cada barbeiro (usando melhor modelo)
        print("\n--- Previsão dos próximos 3 meses para cada barbeiro ---")
        meses_futuros = [10, 11, 12]
        barbeiros = df['id_barbeiro'].unique()
        medias = df.groupby('id_barbeiro').mean(numeric_only=True).reset_index()
        previsoes = []
        for barbeiro in barbeiros:
            dados_barbeiro = medias[medias['id_barbeiro'] == barbeiro]
            for mes in meses_futuros:
                entrada = {col: dados_barbeiro[col].values[0] for col in X.columns}
                entrada['mes'] = mes
                X_novo = pd.DataFrame([entrada])
                X_novo_scaled = scaler.transform(X_novo[X.columns])
                if descricao == 'poly':
                    X_novo_scaled = poly.transform(X_novo_scaled)
                faturamento_previsto = melhor_modelo.predict(X_novo_scaled)[0]
                previsoes.append({
                    'id_barbeiro': barbeiro,
                    'mes': mes,
                    'faturamento_previsto': faturamento_previsto
                })
        df_previsoes = pd.DataFrame(previsoes)
        print(df_previsoes)

    else:
        print("⚠️  A VIEW está vazia (0 registros)")
        
except Exception as e:
    print(f"❌ Erro: {e}")
    print(f"Tipo do erro: {type(e).__name__}")
