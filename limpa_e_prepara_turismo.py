import pandas as pd
import numpy as np
from datetime import datetime
import re

def limpar_e_normalizar(csv_path):
    print('🚀 Lendo o CSV bruto...')
    
    # Primeiro, vamos verificar o arquivo CSV
    try:
        # Tentar diferentes codificações
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                print(f'Tentando encoding: {encoding}')
                df = pd.read_csv(csv_path, sep=',', quotechar='"', encoding=encoding)
                print(f'✅ CSV lido com sucesso usando {encoding}! Shape: {df.shape}')
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            print('❌ Não foi possível ler o CSV com nenhuma codificação')
            return
            
        print(f'Colunas encontradas: {list(df.columns)}')
        
        # Mostrar primeiras linhas para diagnóstico
        print('\n📋 Primeiras 3 linhas do DataFrame:')
        print(df.head(3))
        
    except Exception as e:
        print(f'❌ Erro ao ler CSV: {e}')
        return

    # Remover última coluna se for toda vazia
    if df.columns[-1] == '' or df[df.columns[-1]].isnull().all():
        df = df.iloc[:, :-1]
        print('🗑️ Coluna vazia removida')

    # Limpar nomes das colunas
    df.columns = [col.strip().replace('"', '') for col in df.columns]
    print(f'Colunas após limpeza: {list(df.columns)}')

    # Verificar se temos as colunas necessárias
    colunas_necessarias = [
        'Continente', 'cod continente', 'País', 'cod pais', 'UF', 'cod uf', 
        'Via', 'cod via', 'ano', 'Mês', 'cod mes', 'Chegadas'
    ]
    
    colunas_faltando = [col for col in colunas_necessarias if col not in df.columns]
    if colunas_faltando:
        print(f'❌ Colunas faltando: {colunas_faltando}')
        print('Colunas disponíveis:', list(df.columns))
        return
    
    print('✅ Todas as colunas necessárias encontradas!')

    # Limpar dados das colunas string
    colunas_string = ['Continente', 'País', 'UF', 'Via', 'Mês']
    for col in colunas_string:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.replace('"', '')
            # Tratar strings vazias como NaN
            df[col] = df[col].replace(['', 'nan', 'None'], np.nan)

    # Converter códigos para numérico
    colunas_codigo = ['cod continente', 'cod pais', 'cod uf', 'cod via', 'cod mes']
    for col in colunas_codigo:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Tratar valores inválidos como 0
            df[col] = df[col].fillna(0).astype(int)

    # Converter ano e chegadas
    if 'ano' in df.columns:
        df['ano'] = pd.to_numeric(df['ano'], errors='coerce').fillna(0).astype(int)
    if 'Chegadas' in df.columns:
        df['Chegadas'] = pd.to_numeric(df['Chegadas'], errors='coerce').fillna(0).astype(int)

    # Padronizar nomes
    if 'País' in df.columns:
        df['País'] = df['País'].replace({
            'China, Hong Kong': 'Hong Kong',
            'Outros países da América Central e Caribe': 'Outros América Central e Caribe'
        })

    # Criar colunas auxiliares
    if 'ano' in df.columns and 'cod mes' in df.columns:
        def cria_data(row):
            if pd.notnull(row['ano']) and pd.notnull(row['cod mes']):
                if row['ano'] > 0 and 1 <= row['cod mes'] <= 12:
                    return f"{row['ano']}-{row['cod mes']:02d}-01"
            return pd.NaT
        
        df['data_completa'] = df.apply(cria_data, axis=1)
        df['data_completa'] = pd.to_datetime(df['data_completa'], errors='coerce')
        
        # Trimestre
        df['trimestre'] = df['cod mes'].apply(
            lambda x: ((x-1)//3)+1 if pd.notnull(x) and 1 <= x <= 12 else np.nan
        )
        
        # Estação
        def estacao(m):
            if pd.isnull(m) or not (1 <= m <= 12):
                return ''
            if m in [12, 1, 2]: return 'Verão'
            if m in [3, 4, 5]: return 'Outono'
            if m in [6, 7, 8]: return 'Inverno'
            return 'Primavera'
        
        df['estacao'] = df['cod mes'].apply(estacao)

    print(f'\n📊 Dados processados. Shape final: {df.shape}')

    # Criar tabelas normalizadas
    print('\n🔧 Criando tabelas normalizadas...')
    
    # Continentes
    continentes = df[['cod continente', 'Continente']].drop_duplicates()
    continentes = continentes[continentes['cod continente'] > 0]
    continentes = continentes.rename(columns={
        'cod continente': 'id_continente',
        'Continente': 'nome_continente'
    })
    
    # Países
    paises = df[['cod pais', 'País', 'cod continente']].drop_duplicates()
    paises = paises[paises['cod pais'] > 0]
    paises = paises.rename(columns={
        'cod pais': 'id_pais',
        'País': 'nome_pais',
        'cod continente': 'id_continente'
    })
    
    # UFs
    ufs = df[['cod uf', 'UF']].drop_duplicates()
    ufs = ufs[ufs['cod uf'] > 0]
    ufs = ufs.rename(columns={
        'cod uf': 'id_uf',
        'UF': 'nome_uf'
    })
    
    # Vias
    vias = df[['cod via', 'Via']].drop_duplicates()
    vias = vias[vias['cod via'].isin([1, 2, 3, 4])]
    vias = vias.rename(columns={
        'cod via': 'id_via',
        'Via': 'nome_via'
    })

    # DEBUG: Mostrar mapeamento das colunas
    print('\n🔍 DEBUG - Mapeamento das colunas:')
    for i, col in enumerate(df.columns):
        print(f"Coluna {i}: '{col}'")
    
    print('\n📋 Primeiras 3 linhas com valores:')
    for i in range(min(3, len(df))):
        print(f"Linha {i}: {list(df.iloc[i])}")
    
    # Tabela de chegadas - mapeamento CORRETO baseado no diagnóstico
    chegadas = pd.DataFrame({
        'id_continente': df['cod continente'],
        'id_pais': df['cod pais'].astype(str).str.strip(),  # Pode ter espaços
        'id_uf': df['cod uf'].astype(str).str.strip(),      # Pode ter espaços  
        'id_via': df['cod via'],
        'ano': df['ano'],
        'mes': df['cod mes'],
        'nome_mes': df['Mês'],
        'chegadas': df['Chegadas'],
        'data_completa': df['data_completa'],
        'trimestre': df['trimestre'],
        'estacao': df['estacao']
    })
    
    # Converter códigos para numérico (tratando strings)
    chegadas['id_pais'] = pd.to_numeric(chegadas['id_pais'], errors='coerce').fillna(0).astype(int)
    chegadas['id_uf'] = pd.to_numeric(chegadas['id_uf'], errors='coerce').fillna(0).astype(int)

    # Aplicar filtros mais rigorosos
    print('\n🔍 Aplicando filtros de qualidade...')
    
    # Filtro mais rigoroso: remover linhas com dados corrompidos
    mask_validas = (
        (chegadas['id_continente'] > 0) &
        (chegadas['id_pais'] > 0) &
        (chegadas['id_uf'] > 0) &
        (chegadas['id_via'].isin([1, 2, 3, 4])) &
        (chegadas['ano'] > 0) &
        (chegadas['mes'] >= 1) & (chegadas['mes'] <= 12) &
        (chegadas['chegadas'] >= 0)
    )
    
    chegadas_validas = chegadas[mask_validas].copy()
    chegadas_invalidas = chegadas[~mask_validas].copy()

    print(f'✅ Linhas válidas: {len(chegadas_validas)}')
    print(f'❌ Linhas inválidas: {len(chegadas_invalidas)}')
    
    # Verificar se as linhas inválidas são todas da China (dados corrompidos)
    if not chegadas_invalidas.empty:
        paises_invalidos = chegadas_invalidas['nome_mes'].unique()  # Usando nome_mes como proxy para país
        print(f'Países nas linhas inválidas: {paises_invalidos}')
        
        # Se todas as linhas inválidas são da China, vamos removê-las
        if len(chegadas_invalidas) <= 100 and all('China' in str(x) for x in chegadas_invalidas['nome_mes']):
            print('⚠️ Detectadas linhas corrompidas da China. Removendo...')
            chegadas_validas = chegadas[mask_validas].copy()
            print(f'✅ Após remoção: {len(chegadas_validas)} linhas válidas')

    # Mostrar detalhes das linhas inválidas
    if not chegadas_invalidas.empty:
        print('\n⚠️ Detalhes das linhas inválidas:')
        
        # Contar quantas linhas falham em cada critério
        print('\n📊 Análise dos motivos de rejeição:')
        print(f"id_continente <= 0: {(chegadas['id_continente'] <= 0).sum()}")
        print(f"id_pais <= 0: {(chegadas['id_pais'] <= 0).sum()}")
        print(f"id_uf <= 0: {(chegadas['id_uf'] <= 0).sum()}")
        print(f"id_via não é 1,2,3,4: {(~chegadas['id_via'].isin([1,2,3,4])).sum()}")
        print(f"ano <= 0: {(chegadas['ano'] <= 0).sum()}")
        print(f"mes < 1 ou > 12: {((chegadas['mes'] < 1) | (chegadas['mes'] > 12)).sum()}")
        print(f"chegadas < 0: {(chegadas['chegadas'] < 0).sum()}")
        
        # Mostrar exemplos detalhados
        for idx, row in chegadas_invalidas.head(5).iterrows():
            print(f"\n🔍 Linha {idx} - Motivos de rejeição:")
            motivos = []
            if row['id_continente'] <= 0:
                motivos.append(f"id_continente={row['id_continente']} (deve ser > 0)")
            if row['id_pais'] <= 0:
                motivos.append(f"id_pais={row['id_pais']} (deve ser > 0)")
            if row['id_uf'] <= 0:
                motivos.append(f"id_uf={row['id_uf']} (deve ser > 0)")
            if row['id_via'] not in [1,2,3,4]:
                motivos.append(f"id_via={row['id_via']} (deve ser 1,2,3 ou 4)")
            if row['ano'] <= 0:
                motivos.append(f"ano={row['ano']} (deve ser > 0)")
            if row['mes'] < 1 or row['mes'] > 12:
                motivos.append(f"mes={row['mes']} (deve ser entre 1 e 12)")
            if row['chegadas'] < 0:
                motivos.append(f"chegadas={row['chegadas']} (deve ser >= 0)")
            
            print(f"  Valores: continente={row['id_continente']}, pais={row['id_pais']}, uf={row['id_uf']}, via={row['id_via']}, ano={row['ano']}, mes={row['mes']}, chegadas={row['chegadas']}")
            print(f"  Motivos: {', '.join(motivos)}")

    # Salvar arquivos
    print('\n💾 Salvando arquivos...')
    
    continentes.to_csv('tabela_continentes.csv', index=False, encoding='utf-8')
    paises.to_csv('tabela_paises.csv', index=False, encoding='utf-8')
    ufs.to_csv('tabela_ufs.csv', index=False, encoding='utf-8')
    vias.to_csv('tabela_vias.csv', index=False, encoding='utf-8')
    chegadas_validas.to_csv('tabela_chegadas.csv', index=False, encoding='utf-8')
    
    print('✅ CSVs salvos com sucesso!')

    # Gerar scripts SQL
    print('\n📝 Gerando scripts SQL...')
    
    with open('insert_continentes.sql', 'w', encoding='utf-8') as f:
        for _, row in continentes.iterrows():
            f.write(f"INSERT INTO continentes (id_continente, nome_continente) VALUES ({row['id_continente']}, '{row['nome_continente']}');\n")
    
    with open('insert_paises.sql', 'w', encoding='utf-8') as f:
        for _, row in paises.iterrows():
            f.write(f"INSERT INTO paises (id_pais, nome_pais, id_continente) VALUES ({row['id_pais']}, '{row['nome_pais']}', {row['id_continente']});\n")
    
    with open('insert_ufs.sql', 'w', encoding='utf-8') as f:
        for _, row in ufs.iterrows():
            f.write(f"INSERT INTO ufs (id_uf, nome_uf) VALUES ({row['id_uf']}, '{row['nome_uf']}');\n")
    
    with open('insert_vias.sql', 'w', encoding='utf-8') as f:
        for _, row in vias.iterrows():
            f.write(f"INSERT INTO vias (id_via, nome_via) VALUES ({row['id_via']}, '{row['nome_via']}');\n")
    
    with open('insert_chegadas.sql', 'w', encoding='utf-8') as f:
        for _, row in chegadas_validas.iterrows():
            data_str = row['data_completa'].strftime('%Y-%m-%d') if pd.notnull(row['data_completa']) else 'NULL'
            trimestre_str = str(row['trimestre']) if pd.notnull(row['trimestre']) else 'NULL'
            # Usar valores diretamente sem conversão de codificação
            f.write(f"INSERT INTO chegadas_turismo (id_continente, id_pais, id_uf, id_via, ano, mes, nome_mes, chegadas, data_completa, trimestre, estacao) VALUES ({row['id_continente']}, {row['id_pais']}, {row['id_uf']}, {row['id_via']}, {row['ano']}, {row['mes']}, '{row['nome_mes']}', {row['chegadas']}, '{data_str}', {trimestre_str}, '{row['estacao']}');\n")
    
    print('✅ Scripts SQL gerados!')
    
    # Resumo final
    print(f'\n📊 RESUMO FINAL:')
    print(f'Continentes: {len(continentes)}')
    print(f'Países: {len(paises)}')
    print(f'UFs: {len(ufs)}')
    print(f'Vias: {len(vias)}')
    print(f'Chegadas válidas: {len(chegadas_validas)}')
    print(f'Chegadas inválidas: {len(chegadas_invalidas)}')

if __name__ == '__main__':
    limpar_e_normalizar('chegadas_2024certo2.csv')
    print('\n🎉 Processamento concluído!') 