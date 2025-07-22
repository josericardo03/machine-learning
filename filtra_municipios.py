import pandas as pd

# Lista dos municípios (atenção à acentuação e nomes exatos)
municipios = [
    "Cáceres", "Sonora", "Rondonópolis", "Coxim", "Porto Murtinho", "Poconé", "Ladário",
    "Rio Verde de Mato Grosso", "Corumbá", "Nossa Senhora do Livramento", "Miranda",
    "Barão de Melgaço", "Santo Antônio do Leverger", "Porto Esperidião", "Lambari D'Oeste",
    "Curvelândia", "Juscimeira", "Bodoquena", "Mirassol d'Oeste", "Glória D'Oeste",
    "Aquidauana", "Itiquira"
]

# Nome do arquivo Excel
arquivo = "PIB dos Municípios - base de dados 2010-2021.xlsx"

# Lê o arquivo Excel
df = pd.read_excel(arquivo)

# Filtra os anos de 2010 a 2021
df_filtrado = df[df['Ano'].between(2010, 2021)]

# Filtra os municípios (atenção ao nome exato na planilha)
df_final = df_filtrado[df_filtrado['Nome do Município'].str.strip().str.lower().isin([m.lower() for m in municipios])]

# Salva o resultado em um novo arquivo Excel
df_final.to_excel("municipios_filtrados.xlsx", index=False)

print("Arquivo 'municipios_filtrados.xlsx' criado com sucesso!")