# -*- coding: utf-8 -*-
import os
import io
import re
import difflib
import unicodedata
import requests
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import tempfile, zipfile

# tentar usar classificação por quantis
try:
    import mapclassify  # exigido por geopandas.plot(..., scheme="Quantiles")
    HAS_MC = True
except Exception:
    HAS_MC = False

# =========================
# 1) SEUS DADOS (15 municípios)
#    — Se preferir ler de CSV, defina CSV_PATH e comente o bloco "dados_inline"
# =========================
CSV_PATH = None  # ex.: "pearson_municipios.csv" (colunas: municipio, pearson_valor_agropecuaria)

dados_inline = """municipio,pearson_valor_agropecuaria
Rio Verde de Mato Grosso,0.9292273366371547
Coxim,0.9169129825309994
Corumbá,0.895550841604981
Ladário,0.8683283296876749
Miranda,0.846504813733229
Sonora,0.8239480880832701
Aquidauana,0.803800372903488
Porto Murtinho,0.7704969861578651
Cáceres,0.7545391236417517
Nossa Senhora do Livramento,0.7080321693637666
Itiquira,0.6610813010871014
Santo Antônio do Leverger,0.6414291803091261
Porto Esperidião,0.6065352699783582
Poconé,0.552731356552654
Barão de Melgaço,0.44160719422921096
"""

if CSV_PATH and os.path.exists(CSV_PATH):
    df_val = pd.read_csv(CSV_PATH)
else:
    df_val = pd.read_csv(io.StringIO(dados_inline))

# =========================
# 2) Funções auxiliares
# =========================
def noaccent(s: str) -> str:
    if pd.isna(s):
        return s
    txt = "".join(ch for ch in unicodedata.normalize("NFKD", str(s))
                  if not unicodedata.combining(ch)).lower()
    # remove tudo que não é letra/número e comprime espaços
    txt = re.sub(r"[^a-z0-9]+", " ", txt).strip()
    txt = re.sub(r"\s+", " ", txt)
    return txt

def add_norm_cols(df, col="municipio"):
    df = df.copy()
    df[col + "_norm"] = df[col].apply(noaccent)
    return df

# =========================
# 3) Baixar malha municipal IBGE (2022) como ZIP (shapefile) — MT e MS
# =========================
def load_ibge_mt_ms():
    """
    Baixa as malhas municipais 2022 de MT e MS (zip com shapefile),
    extrai para temp e devolve um GeoDataFrame concatenado.
    """
    urls = {
        "MT": "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2022/UFs/MT/MT_Municipios_2022.zip",
        "MS": "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2022/UFs/MS/MS_Municipios_2022.zip",
    }

    gdfs = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for uf, url in urls.items():
            zpath = os.path.join(tmpdir, f"{uf}.zip")
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            with open(zpath, "wb") as f:
                f.write(r.content)

            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(os.path.join(tmpdir, uf))

            # localizar o .shp extraído
            shp_path = None
            for root, _, files in os.walk(os.path.join(tmpdir, uf)):
                for fn in files:
                    if fn.lower().endswith(".shp"):
                        shp_path = os.path.join(root, fn)
                        break
                if shp_path:
                    break
            if not shp_path:
                raise RuntimeError(f"Não encontrei .shp dentro do zip de {uf}")

            gdf = gpd.read_file(shp_path)

            # padronizar nome/UF
            name_col = None
            for cand in ["NM_MUN", "NM_MUNICIP", "NM_MUNICIPIO", "NM_MUN_2022", "NM_MUN"]:
                if cand in gdf.columns:
                    name_col = cand
                    break
            if name_col is None:
                raise RuntimeError(f"Não encontrei coluna de nome do município no shapefile de {uf}.")

            uf_col = None
            for cand in ["SIGLA_UF", "SG_UF", "UF"]:
                if cand in gdf.columns:
                    uf_col = cand
                    break
            if uf_col is None:
                gdf["UF"] = uf
                uf_col = "UF"

            gdf = gdf.rename(columns={name_col: "municipio_ibge", uf_col: "UF"})
            gdf = gdf[["municipio_ibge", "UF", "geometry"]]
            gdfs.append(gdf)

    return pd.concat(gdfs, ignore_index=True)

gdf_ibge = load_ibge_mt_ms()

# =========================
# 4) Normalizar nomes e fazer merge (MT e MS)
# =========================
gdf_ibge = gdf_ibge[gdf_ibge["UF"].isin(["MT", "MS"])].copy()
gdf_ibge = add_norm_cols(gdf_ibge, "municipio_ibge")
df_val = add_norm_cols(df_val, "municipio")

# ---- ALIASES manuais (ajustes de grafia conhecidos) ----
ALIASES = {
    # ajuste fino se necessário (ex.: "do" ↔ "de")
    # "santo antonio do leverger": "santo antonio de leverger",
}

# aplica aliases no df_val_norm
df_val["municipio_norm"] = df_val["municipio_norm"].apply(lambda x: ALIASES.get(x, x))

# ---- MERGE inicial ----
gdf_join = gdf_ibge.merge(
    df_val[["municipio", "municipio_norm", "pearson_valor_agropecuaria"]],
    left_on="municipio_ibge_norm",
    right_on="municipio_norm",
    how="right",
    validate="1:1"
)

# ---- Auto-fix por similaridade para o que sobrou sem geometria ----
missing = gdf_join[gdf_join["geometry"].isna()].copy()
if not missing.empty:
    ibge_names = gdf_ibge["municipio_ibge_norm"].dropna().unique().tolist()
    fixes = {}
    for m in missing["municipio_norm"].tolist():
        best = difflib.get_close_matches(m, ibge_names, n=1, cutoff=0.85)
        if best:
            fixes[m] = best[0]
    if fixes:
        df_val["municipio_norm"] = df_val["municipio_norm"].apply(lambda x: fixes.get(x, x))
        gdf_join = gdf_ibge.merge(
            df_val[["municipio", "municipio_norm", "pearson_valor_agropecuaria"]],
            left_on="municipio_ibge_norm",
            right_on="municipio_norm",
            how="right",
            validate="1:1"
        )

# reporta o que sobrou
missing2 = gdf_join[gdf_join["geometry"].isna()]
if not missing2.empty:
    print("\n[Atenção] Ainda restam municípios sem casar — verifique grafia/UF:")
    print(missing2[["municipio"]])

# =========================
# 5) Plot coroplético (paleta RdYlGn + quantis)
# =========================
out_dir = "mapas_municipios_pearson"
os.makedirs(out_dir, exist_ok=True)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# base leve: toda a malha MT+MS (cinza claro)
gdf_ibge.boundary.plot(ax=ax, linewidth=0.3, color="#999999")
gdf_ibge.plot(ax=ax, color="#f0f0f0", linewidth=0, alpha=0.4)

plot_kwargs = dict(
    column="pearson_valor_agropecuaria",
    cmap="RdYlGn",   # vermelho (baixo) -> verde (alto)
    linewidth=0.6,
    edgecolor="#333333",
    legend=True,
    legend_kwds={"title": "Correlação de Pearson\n(desmatado × valor agropecuário)"},
    ax=ax
)

if HAS_MC:
    # classifica em 5 quantis (mais editorial)
    gdf_join.plot(scheme="Quantiles", k=5, **plot_kwargs)
else:
    # fallback: contínuo
    gdf_join.plot(**plot_kwargs)

ax.set_title("Correlação de Pearson por município (desmatado × valor agropecuário)\nPantanal — MT & MS", fontsize=14)
ax.set_axis_off()

png_path = os.path.join(out_dir, "mapa_pearson_valor_agropecuaria_MT_MS.png")
plt.tight_layout()
plt.savefig(png_path, dpi=300)
plt.close()
print(f"Mapa salvo em: {png_path}")

# =========================
# 6) Exportar GeoJSON (útil para QGIS / webmap)
# =========================
geojson_path = os.path.join(out_dir, "municipios_pearson.geojson")
gdf_join.to_file(geojson_path, driver="GeoJSON")
print(f"GeoJSON com valores salvo em: {geojson_path}")

# =========================
# 7) Tabela final (conferência)
# =========================
out_csv = os.path.join(out_dir, "municipios_pearson_join.csv")
gdf_join.drop(columns="geometry").to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"Tabela unida salva em: {out_csv}")
