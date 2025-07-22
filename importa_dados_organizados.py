import pandas as pd
from sqlalchemy import create_engine

DATABASE_URL = "postgresql://postgres.roziechzdpxxdtzlkaep:Jj20134849%40%40%40@aws-0-sa-east-1.pooler.supabase.com:5432/postgres"
TABELA = 'dados_organizados'
ARQUIVO_CSV = 'dados_organizados_limp.csv'

# Mapeamento dos nomes do CSV para os nomes da tabela (todos minúsculos)
colunas_mapeamento = {
    'Ano': 'ano',
    'Código da Grande Região': 'codigo_grande_regiao',
    'Nome da Grande Região': 'nome_grande_regiao',
    'Código da Unidade da Federação': 'codigo_unidade_federacao',
    'Sigla da Unidade da Federação': 'sigla_unidade_federacao',
    'Nome da Unidade da Federação': 'nome_unidade_federacao',
    'Código do Município': 'codigo_municipio',
    'Nome do Município': 'nome_municipio',
    'Região Metropolitana': 'regiao_metropolitana',
    'Código da Mesorregião': 'codigo_mesorregiao',
    'Nome da Mesorregião': 'nome_mesorregiao',
    'Código da Microrregião': 'codigo_microrregiao',
    'Nome da Microrregião': 'nome_microrregiao',
    'Código da Região Geográfica Imediata': 'codigo_regiao_geografica_imediata',
    'Nome da Região Geográfica Imediata': 'nome_regiao_geografica_imediata',
    'Município da Região Geográfica Imediata': 'municipio_regiao_geografica_imediata',
    'Código da Região Geográfica Intermediária': 'codigo_regiao_geografica_intermediaria',
    'Nome da Região Geográfica Intermediária': 'nome_regiao_geografica_intermediaria',
    'Município da Região Geográfica Intermediária': 'municipio_regiao_geografica_intermediaria',
    'Código Concentração Urbana': 'codigo_concentracao_urbana',
    'Nome Concentração Urbana': 'nome_concentracao_urbana',
    'Tipo Concentração Urbana': 'tipo_concentracao_urbana',
    'Código Arranjo Populacional': 'codigo_arranjo_populacional',
    'Nome Arranjo Populacional': 'nome_arranjo_populacional',
    'Hierarquia Urbana': 'hierarquia_urbana',
    'Hierarquia Urbana (principais categorias)': 'hierarquia_urbana_principais_categorias',
    'Código da Região Rural': 'codigo_regiao_rural',
    'Nome da Região Rural': 'nome_regiao_rural',
    'Região rural (segundo classificação do núcleo)': 'regiao_rural_segundo_classificacao_nucleo',
    'Amazônia Legal': 'amazonia_legal',
    'Semiárido': 'semiarido',
    'Cidade-Região de São Paulo': 'cidade_regiao_sao_paulo',
    'Valor adicionado bruto da Agropecuária,  a preços correntes (R$ 1.000)': 'valor_agropecuaria',
    'Valor adicionado bruto da Indústria, a preços correntes (R$ 1.000)': 'valor_industria',
    'Valor adicionado bruto dos Serviços, a preços correntes  - exceto Administração, defesa, educação e saúde públicas e seguridade social (R$ 1.000)': 'valor_servicos',
    'Valor adicionado bruto da Administração, defesa, educação e saúde públicas e seguridade social,  a preços correntes (R$ 1.000)': 'valor_administracao_publica',
    'Valor adicionado bruto total,  a preços correntes (R$ 1.000)': 'valor_total',
    'Impostos, líquidos de subsídios, sobre produtos,  a preços correntes (R$ 1.000)': 'impostos_liquidos',
    'Produto Interno Bruto,  a preços correntes (R$ 1.000)': 'pib',
    'Produto Interno Bruto per capita,  a preços correntes (R$ 1,00)': 'pib_per_capita',
    'Atividade com maior valor adicionado bruto': 'atividade_maior_valor',
    'Atividade com segundo maior valor adicionado bruto': 'atividade_segundo_maior_valor',
    'Atividade com terceiro maior valor adicionado bruto': 'atividade_terceiro_maior_valor'
}

# Lê o CSV
df = pd.read_csv(ARQUIVO_CSV)

# Renomeia as colunas conforme o mapeamento
colunas_para_renomear = {k: v for k, v in colunas_mapeamento.items() if k in df.columns}
df = df.rename(columns=colunas_para_renomear)

# Mantém apenas as colunas que existem na tabela
colunas_tabela = list(colunas_mapeamento.values())
df = df[[col for col in colunas_tabela if col in df.columns]]

# Remove espaços extras e converte números para float (corrige vírgula e espaço)
for col in [
    'valor_agropecuaria', 'valor_industria', 'valor_servicos', 'valor_administracao_publica',
    'valor_total', 'impostos_liquidos', 'pib', 'pib_per_capita'
]:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace('.', '', regex=False)  # remove separador de milhar
            .str.replace(',', '.', regex=False)  # troca vírgula por ponto
            .str.replace(' ', '', regex=False)   # remove espaços
            .replace({'nan': None})
            .astype(float)
        )

# Mostra os nomes das colunas para conferência
print('Colunas do DataFrame:', df.columns.tolist())
print(df.head(2))

# Cria engine de conexão
engine = create_engine(DATABASE_URL)

# Insere os dados na tabela em blocos ainda menores, sem usar o método 'multi'
df.to_sql(TABELA, engine, if_exists='append', index=False, chunksize=2)

print('Importação concluída com sucesso!')