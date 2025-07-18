import pandas as pd
import numpy as np

print('🔍 VALIDAÇÃO DE INTEGRIDADE DOS DADOS')
print('=' * 50)

# Ler dados originais
print('📋 Lendo dados originais...')
df_original = pd.read_csv('chegadas_2024certo2.csv', sep=',', quotechar='"', encoding='utf-8')

# Remover última coluna vazia
if df_original.columns[-1] == 'Unnamed: 12':
    df_original = df_original.iloc[:, :-1]

# Limpar nomes das colunas
df_original.columns = [col.strip().replace('"', '') for col in df_original.columns]

print(f'Dados originais: {len(df_original)} linhas')

# Ler dados processados
print('\n📋 Lendo dados processados...')
df_processado = pd.read_csv('tabela_chegadas.csv', encoding='utf-8')
print(f'Dados processados: {len(df_processado)} linhas')

# Verificar se há perda de dados
print(f'\n📊 COMPARAÇÃO:')
print(f'Linhas originais: {len(df_original)}')
print(f'Linhas processadas: {len(df_processado)}')
print(f'Diferença: {len(df_original) - len(df_processado)} linhas')

if len(df_original) - len(df_processado) == 100:
    print('✅ Diferença esperada: 100 linhas corrompidas da China foram removidas')
else:
    print('❌ Diferença inesperada!')

# Verificar integridade dos dados válidos
print('\n🔍 Verificando integridade dos dados válidos...')

# Converter códigos para numérico no original
colunas_codigo = ['cod continente', 'cod pais', 'cod uf', 'cod via', 'cod mes']
for col in colunas_codigo:
    df_original[col] = pd.to_numeric(df_original[col], errors='coerce').fillna(0).astype(int)

df_original['ano'] = pd.to_numeric(df_original['ano'], errors='coerce').fillna(0).astype(int)
df_original['Chegadas'] = pd.to_numeric(df_original['Chegadas'], errors='coerce').fillna(0).astype(int)

# Aplicar o mesmo filtro do script principal
mask_validas_original = (
    (df_original['cod continente'] > 0) &
    (df_original['cod pais'] > 0) &
    (df_original['cod uf'] > 0) &
    (df_original['cod via'].isin([1, 2, 3, 4])) &
    (df_original['ano'] > 0) &
    (df_original['cod mes'] >= 1) & (df_original['cod mes'] <= 12) &
    (df_original['Chegadas'] >= 0)
)

df_original_validas = df_original[mask_validas_original]

print(f'Linhas válidas no original: {len(df_original_validas)}')
print(f'Linhas válidas no processado: {len(df_processado)}')

# Verificar se os dados são idênticos
print('\n🔍 Comparando dados válidos...')

# Mapear colunas do processado para o original
mapeamento = {
    'id_continente': 'cod continente',
    'id_pais': 'cod pais', 
    'id_uf': 'cod uf',
    'id_via': 'cod via',
    'ano': 'ano',
    'mes': 'cod mes',
    'chegadas': 'Chegadas'
}

# Comparar cada coluna
diferencas_encontradas = False

for col_processado, col_original in mapeamento.items():
    if col_processado in df_processado.columns and col_original in df_original_validas.columns:
        # Comparar valores
        valores_processados = df_processado[col_processado].values
        valores_originais = df_original_validas[col_original].values
        
        if len(valores_processados) == len(valores_originais):
            if np.array_equal(valores_processados, valores_originais):
                print(f'✅ {col_processado}: Dados idênticos')
            else:
                print(f'❌ {col_processado}: DADOS DIFERENTES!')
                diferencas_encontradas = True
                
                # Mostrar primeiras diferenças
                for i in range(min(5, len(valores_processados))):
                    if valores_processados[i] != valores_originais[i]:
                        print(f'  Linha {i}: Original={valores_originais[i]}, Processado={valores_processados[i]}')
        else:
            print(f'❌ {col_processado}: Número diferente de linhas!')
            diferencas_encontradas = True

# Verificar estatísticas
print('\n📊 ESTATÍSTICAS COMPARATIVAS:')

# Estatísticas do original (apenas válidas)
print('\nDados originais (válidos):')
print(f'Total chegadas: {df_original_validas["Chegadas"].sum():,}')
print(f'Média chegadas: {df_original_validas["Chegadas"].mean():.2f}')
print(f'Anos únicos: {sorted(df_original_validas["ano"].unique())}')
print(f'Meses únicos: {sorted(df_original_validas["cod mes"].unique())}')

# Estatísticas do processado
print('\nDados processados:')
print(f'Total chegadas: {df_processado["chegadas"].sum():,}')
print(f'Média chegadas: {df_processado["chegadas"].mean():.2f}')
print(f'Anos únicos: {sorted(df_processado["ano"].unique())}')
print(f'Meses únicos: {sorted(df_processado["mes"].unique())}')

# Verificar se as estatísticas são idênticas
if (df_original_validas["Chegadas"].sum() == df_processado["chegadas"].sum() and
    abs(df_original_validas["Chegadas"].mean() - df_processado["chegadas"].mean()) < 0.01):
    print('\n✅ ESTATÍSTICAS IDÊNTICAS - DADOS NÃO FORAM ADULTERADOS!')
else:
    print('\n❌ ESTATÍSTICAS DIFERENTES - POSSÍVEL ADULTERAÇÃO!')

# Verificar exemplo específico da China
print('\n🔍 Verificando exemplo específico da China (linha 34 do original):')
linha_34_original = df_original.iloc[33]  # índice 33 = linha 34
print(f'Original - Continente: {linha_34_original["Continente"]}')
print(f'Original - País: {linha_34_original["País"]}')
print(f'Original - UF: {linha_34_original["UF"]}')
print(f'Original - Via: {linha_34_original["Via"]}')
print(f'Original - Ano: {linha_34_original["ano"]}')
print(f'Original - Mês: {linha_34_original["Mês"]}')
print(f'Original - Chegadas: {linha_34_original["Chegadas"]}')

# Verificar se essa linha está no processado
china_no_processado = df_processado[
    (df_processado['id_continente'] == 5) & 
    (df_processado['chegadas'] == 8)
]

if len(china_no_processado) > 0:
    print(f'\n✅ Linha da China encontrada no processado: {len(china_no_processado)} ocorrências')
else:
    print('\n❌ Linha da China NÃO encontrada no processado')

print('\n🎯 CONCLUSÃO:')
if not diferencas_encontradas:
    print('✅ DADOS NÃO FORAM ADULTERADOS - Processamento mantém integridade!')
else:
    print('❌ DADOS FORAM ALTERADOS - Verificar processamento!') 