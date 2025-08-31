# -*- coding: utf-8 -*-
import os, io, re, difflib, unicodedata, requests, tempfile, zipfile
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# tentar quantis (opcional)
try:
    import mapclassify         # para scheme="Quantiles"
    HAS_MC = True
except Exception:
    HAS_MC = False

# -----------------------
# 1) DADOS INLINE (4 VARS)
# -----------------------
CSV_AGRO = """municipio,pearson_valor_agropecuaria
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

CSV_PIB = """municipio,pearson_pib_per_capita
Aquidauana,0.9896185956705067
Poconé,0.9873006123747472
Nossa Senhora do Livramento,0.9872390265371422
Rio Verde de Mato Grosso,0.9869744047598696
Cáceres,0.9854228220188784
Coxim,0.9714909369191335
Miranda,0.9472905697255545
Porto Murtinho,0.9455728128078261
Porto Esperidião,0.9113010436218121
Sonora,0.8868063154469451
Barão de Melgaço,0.8770845975047801
Santo Antônio do Leverger,0.8501920419103878
Ladário,0.8033876849998736
Corumbá,0.7716250054007053
Itiquira,0.7375830804176859
"""

CSV_ADM = """municipio,pearson_valor_administracao_publica
Cáceres,0.9965363980270525
Coxim,0.9932962584958038
Rio Verde de Mato Grosso,0.9916294650014615
Poconé,0.9891572334885755
Nossa Senhora do Livramento,0.9859042559305328
Porto Esperidião,0.9843360184160226
Aquidauana,0.9827554696282734
Corumbá,0.9820996286794469
Barão de Melgaço,0.9818271234794151
Ladário,0.9802267576804014
Santo Antônio do Leverger,0.9777840640984845
Itiquira,0.9751698835693018
Porto Murtinho,0.9737406502984669
Miranda,0.9731858487607828
Sonora,0.9615505729676427
"""

CSV_IND = """municipio,pearson_valor_industria
Barão de Melgaço,0.9707062419193749
Itiquira,0.9413080015116558
Sonora,0.9191430926474111
Cáceres,0.9034906419757974
Santo Antônio do Leverger,0.8996643752142822
Poconé,0.8990871915107098
Aquidauana,0.8906688063031598
Rio Verde de Mato Grosso,0.8589418941545575
Nossa Senhora do Livramento,0.8576235011267009
Miranda,0.7479025278741974
Porto Esperidião,0.680072658974349
Ladário,0.5309063439387597
Coxim,0.3586177471249791
Corumbá,-0.010673701185278392
Porto Murtinho,-0.4622030153941258
"""

# -----------------------
# 2) UTILIDADES
# -----------------------
def noaccent(s: str) -> str:
    if pd.isna(s): return s
    t = "".join(ch for ch in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(ch)).lower()
    t = re.sub(r"[^a-z0-9]+", " ", t).strip()
    return re.sub(r"\s+", " ", t)

def add_norm(df, col="municipio"):
    df = df.copy()
    df[col + "_norm"] = df[col].apply(noaccent)
    return df

def load_ibge_mt_ms():
    urls = {
        "MT": "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2022/UFs/MT/MT_Municipios_2022.zip",
        "MS": "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2022/UFs/MS/MS_Municipios_2022.zip",
    }
    gdfs = []
    import tempfile, zipfile
    with tempfile.TemporaryDirectory() as tmp:
        for uf, url in urls.items():
            zf = os.path.join(tmp, f"{uf}.zip")
            r = requests.get(url, timeout=120); r.raise_for_status()
            open(zf, "wb").write(r.content)
            with zipfile.ZipFile(zf) as z:
                z.extractall(os.path.join(tmp, uf))
            shp = None
            for root, _, files in os.walk(os.path.join(tmp, uf)):
                for f in files:
                    if f.lower().endswith(".shp"):
                        shp = os.path.join(root, f); break
                if shp: break
            gdf = gpd.read_file(shp)
            name = next(c for c in ["NM_MUN","NM_MUNICIP","NM_MUNICIPIO","NM_MUN_2022","NM_MUN"] if c in gdf.columns)
            ufcol = next((c for c in ["SIGLA_UF","SG_UF","UF"] if c in gdf.columns), None)
            if ufcol is None: gdf["UF"]=uf; ufcol="UF"
            gdf = gdf.rename(columns={name:"municipio_ibge", ufcol:"UF"})[["municipio_ibge","UF","geometry"]]
            gdfs.append(gdf)
    return pd.concat(gdfs, ignore_index=True)

def df_from_csv(csv_str: str, value_col: str):
    df = pd.read_csv(io.StringIO(csv_str))
    df = add_norm(df, "municipio")
    return df[["municipio","municipio_norm",value_col]]

def merge_ibge(gdf_ibge, df_vals, value_col):
    df_vals = df_vals.copy()
    gdf_join = gdf_ibge.merge(df_vals, left_on="municipio_ibge_norm", right_on="municipio_norm",
                              how="right", validate="1:1")
    miss = gdf_join[gdf_join.geometry.isna()]
    if not miss.empty:
        ibge_names = gdf_ibge["municipio_ibge_norm"].unique().tolist()
        fixes = {}
        for m in miss["municipio_norm"]:
            best = difflib.get_close_matches(m, ibge_names, n=1, cutoff=0.85)
            if best: fixes[m]=best[0]
        if fixes:
            df_vals["municipio_norm"] = df_vals["municipio_norm"].map(lambda x: fixes.get(x,x))
            gdf_join = gdf_ibge.merge(df_vals, left_on="municipio_ibge_norm", right_on="municipio_norm",
                                      how="right", validate="1:1")
    still = gdf_join[gdf_join.geometry.isna()]
    if not still.empty:
        print(f"\n[Atenção] Sem casar ({value_col}):"); print(still[["municipio"]])
    return gdf_join

def plot_export(gdf_base, gdf_join, value_col, leg_title, map_title, prefix, out_dir="mapas_municipios_pearson"):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(1,1,figsize=(10,8))
    gdf_base.boundary.plot(ax=ax, linewidth=0.3, color="#999")
    gdf_base.plot(ax=ax, color="#f0f0f0", linewidth=0, alpha=0.4)

    kwargs = dict(column=value_col, cmap="RdYlGn_r", linewidth=0.6, edgecolor="#333",
                  legend=True, legend_kwds={"title": leg_title}, ax=ax)
    if HAS_MC: gdf_join.plot(scheme="Quantiles", k=5, **kwargs)
    else:      gdf_join.plot(**kwargs)

    ax.set_title(map_title, fontsize=14); ax.set_axis_off()
    png = os.path.join(out_dir, f"{prefix}.png"); plt.tight_layout(); plt.savefig(png, dpi=300); plt.close()
    print("Mapa salvo em:", png)
    gj = os.path.join(out_dir, f"{prefix}.geojson"); gdf_join.to_file(gj, driver="GeoJSON"); print("GeoJSON salvo em:", gj)
    csv = os.path.join(out_dir, f"{prefix}.csv"); gdf_join.drop(columns="geometry").to_csv(csv, index=False, encoding="utf-8-sig"); print("CSV salvo em:", csv)

# -----------------------
# 3) PIPELINE
# -----------------------
gdf_ibge = load_ibge_mt_ms()
gdf_ibge = gdf_ibge[gdf_ibge["UF"].isin(["MT","MS"])].copy()
gdf_ibge = add_norm(gdf_ibge, "municipio_ibge")

# AGRO
df_agro = df_from_csv(CSV_AGRO, "pearson_valor_agropecuaria")
gdf_agro = merge_ibge(gdf_ibge, df_agro, "pearson_valor_agropecuaria")
plot_export(
    gdf_ibge, gdf_agro, "pearson_valor_agropecuaria",
    "Correlação de Pearson\n(desmatado × valor agropecuário)",
    "Correlação de Pearson por município (desmatado × valor agropecuário)\nPantanal — MT & MS",
    "mapa_pearson_valor_agropecuaria_MT_MS"
)

# PIB per capita
df_pib = df_from_csv(CSV_PIB, "pearson_pib_per_capita")
gdf_pib = merge_ibge(gdf_ibge, df_pib, "pearson_pib_per_capita")
plot_export(
    gdf_ibge, gdf_pib, "pearson_pib_per_capita",
    "Correlação de Pearson\n(desmatado × PIB per capita)",
    "Correlação de Pearson por município (desmatado × PIB per capita)\nPantanal — MT & MS",
    "mapa_pearson_pib_per_capita_MT_MS"
)

# Administração pública
df_adm = df_from_csv(CSV_ADM, "pearson_valor_administracao_publica")
gdf_adm = merge_ibge(gdf_ibge, df_adm, "pearson_valor_administracao_publica")
plot_export(
    gdf_ibge, gdf_adm, "pearson_valor_administracao_publica",
    "Correlação de Pearson\n(desmatado × valor administração pública)",
    "Correlação de Pearson por município (desmatado × valor administração pública)\nPantanal — MT & MS",
    "mapa_pearson_administracao_publica_MT_MS"
)

# Indústria
df_ind = df_from_csv(CSV_IND, "pearson_valor_industria")
gdf_ind = merge_ibge(gdf_ibge, df_ind, "pearson_valor_industria")
plot_export(
    gdf_ibge, gdf_ind, "pearson_valor_industria",
    "Correlação de Pearson\n(desmatado × valor indústria)",
    "Correlação de Pearson por município (desmatado × valor indústria)\nPantanal — MT & MS",
    "mapa_pearson_valor_industria_MT_MS"
)
