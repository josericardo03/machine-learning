import pandas as pd
import numpy as np

print('🔍 ANÁLISE DAS LINHAS PROBLEMÁTICAS')
print('=' * 50)

# Ler o CSV
df = pd.read_csv('chegadas_2024certo2.csv', sep=',', quotechar='"', encoding='utf-8')

# Remover última coluna vazia
if df.columns[-1] == 'Unnamed: 12':
    df = df.iloc[:, :-1]

# Limpar nomes das colunas
df.columns = [col.strip().replace('"', '') for col in df.columns]

# Converter códigos para numérico
colunas_codigo = ['cod continente', 'cod pais', 'cod uf', 'cod via', 'cod mes']
for col in colunas_codigo:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

# Converter ano e chegadas
df['ano'] = pd.to_numeric(df['ano'], errors='coerce').fillna(0).astype(int)
df['Chegadas'] = pd.to_numeric(df['Chegadas'], errors='coerce').fillna(0).astype(int)

# Aplicar o mesmo filtro do script principal
mask_validas = (
    (df['cod continente'] > 0) &
    (df['cod pais'] > 0) &
    (df['cod uf'] > 0) &
    (df['cod via'].isin([1, 2, 3, 4])) &
    (df['ano'] > 0) &
    (df['cod mes'] >= 1) & (df['cod mes'] <= 12) &
    (df['Chegadas'] >= 0)
)

linhas_invalidas = df[~mask_validas]

print(f'Total de linhas: {len(df)}')
print(f'Linhas válidas: {mask_validas.sum()}')
print(f'Linhas inválidas: {len(linhas_invalidas)}')

print('\n🔍 PRIMEIRAS 10 LINHAS INVÁLIDAS:')
for i, (idx, row) in enumerate(linhas_invalidas.head(10).iterrows()):
    print(f'\n--- Linha {idx} (índice original) ---')
    print(f'Continente: "{row["Continente"]}" (código: {row["cod continente"]})')
    print(f'País: "{row["País"]}" (código: {row["cod pais"]})')
    print(f'UF: "{row["UF"]}" (código: {row["cod uf"]})')
    print(f'Via: "{row["Via"]}" (código: {row["cod via"]})')
    print(f'Ano: {row["ano"]}')
    print(f'Mês: "{row["Mês"]}" (código: {row["cod mes"]})')
    print(f'Chegadas: {row["Chegadas"]}')
    
    # Mostrar valores brutos
    valores_brutos = list(row)
    print(f'Valores brutos: {valores_brutos}')

print('\n🔍 VERIFICAR SE HÁ PADRÕES NAS LINHAS INVÁLIDAS:')
print('Valores únicos de continente nas linhas inválidas:')
print(linhas_invalidas['Continente'].unique())

print('\nValores únicos de país nas linhas inválidas:')
print(linhas_invalidas['País'].unique())

print('\nValores únicos de UF nas linhas inválidas:')
print(linhas_invalidas['UF'].unique()) 