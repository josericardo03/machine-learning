import pandas as pd

print('🔍 DIAGNÓSTICO DO CSV')
print('=' * 50)

# Ler o CSV
df = pd.read_csv('chegadas_2024certo2.csv', sep=',', quotechar='"', encoding='utf-8')

print(f'Shape do DataFrame: {df.shape}')
print(f'Colunas encontradas: {list(df.columns)}')

print('\n📋 Primeiras 5 linhas:')
for i in range(min(5, len(df))):
    print(f'Linha {i}: {list(df.iloc[i])}')

print('\n🔍 Linha 33 (que está dando erro):')
if len(df) > 33:
    linha_33 = list(df.iloc[33])
    print(f'Valores: {linha_33}')
    
    # Mapear manualmente baseado no CSV que você mostrou
    print('\n📊 Mapeamento manual da linha 33:')
    print(f'Continente: "{linha_33[0]}"')
    print(f'cod continente: "{linha_33[1]}"')
    print(f'País: "{linha_33[2]}"')
    print(f'cod pais: "{linha_33[3]}"')
    print(f'UF: "{linha_33[4]}"')
    print(f'cod uf: "{linha_33[5]}"')
    print(f'Via: "{linha_33[6]}"')
    print(f'cod via: "{linha_33[7]}"')
    print(f'ano: "{linha_33[8]}"')
    print(f'Mês: "{linha_33[9]}"')
    print(f'cod mes: "{linha_33[10]}"')
    print(f'Chegadas: "{linha_33[11]}"')

print('\n🔍 Verificar se há cabeçalho duplicado:')
# Verificar se a primeira linha é cabeçalho
primeira_linha = list(df.iloc[0])
print(f'Primeira linha: {primeira_linha}')

# Verificar se há linhas vazias no início
print('\n🔍 Verificar linhas vazias no início:')
for i in range(min(10, len(df))):
    linha = list(df.iloc[i])
    if all(str(x).strip() == '' for x in linha):
        print(f'Linha {i} está vazia')
    elif any(str(x).strip() == '' for x in linha):
        print(f'Linha {i} tem valores vazios: {linha}') 