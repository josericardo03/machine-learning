import pandas as pd
from sqlalchemy import create_engine

# Dados de conexão
DATABASE_URL = "postgresql://postgres.roziechzdpxxdtzlkaep:Jj20134849%40%40%40@aws-0-sa-east-1.pooler.supabase.com:5432/postgres"
TABELA = 'dados_municipios'
ARQUIVO_CSV = 'bquxjob_457adb50_198323f62b7.csv'

# Lê o CSV
df = pd.read_csv(ARQUIVO_CSV)

# Cria engine de conexão
engine = create_engine(DATABASE_URL)

# Insere os dados na tabela (append = adiciona, não apaga o que já existe)
df.to_sql(TABELA, engine, if_exists='append', index=False)

print('Importação concluída com sucesso!')