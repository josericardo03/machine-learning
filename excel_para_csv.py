import pandas as pd

# Nome do arquivo Excel de entrada
arquivo_excel = "PIB dos Municípios - base de dados 2010-2021.xlsx"

# Nome do arquivo CSV de saída
arquivo_csv = "pib_municipios.csv"

# Lê o arquivo Excel (por padrão, lê a primeira aba)
df = pd.read_excel(arquivo_excel)

# Salva como CSV, com separador vírgula e cabeçalho
df.to_csv(arquivo_csv, index=False, encoding='utf-8')

print("Arquivo CSV criado com sucesso!")