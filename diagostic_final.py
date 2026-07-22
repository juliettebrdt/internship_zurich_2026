### Diagnostic — Label distribution (Gold Standard only)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path     # Used for robust and cross-platform file path management
import pandas as pd     
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)



# Core columns to extract from Progenetix biosample summaries
BIOSAMPLE_COLS = [
    "biosample_id", "individual_id", "analysis_id",
    "platform_id", "histological_diagnosis_id",
    "icdo_topography_id", "icdo_morphology_id",
    "pathological_stage_id", "sample_origin_type_id",
    "cohorts"
]


# Raw Data Inputs (Downloads folder)
DOWNLOADS_DIR     = Path("/Users/bgadmin/Downloads/")
BIOSAMPLE_SUMMARY = DOWNLOADS_DIR / "biosample_summary.csv"
GENE_PANEL        = DOWNLOADS_DIR / "gene_cnv_cancer_panel.tsv"
ILLUMINA_MANIFEST = DOWNLOADS_DIR / "infinium-methylationepic-v-1-0-b5-manifest-file.csv"
# wget https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz
NCBI_GENE_INFO = Path("/Users/bgadmin/Downloads/Homo_sapiens.gene_info")

# Intermediate Storage (Raw Omics)
GDC_RNA_INPUT    = Path("cohort_matrices")
GDC_METH_INPUT   = Path("cohort_methylation_matrices")
CBIO_BASE_INPUT  = Path("cbioportal_data")
PROGENETIX_RAW   = Path("progenetix_cnv")

# Cleaned Data Folders
GDC_CLEAN_OUTPUT       = Path("cohort_matrices_cleaned")
CBIO_CLEAN_OUTPUT = Path("cbioportal_cleaned")

#CNV bridge ID
OUTPUT_BRIDGE     = Path("cnv_id_bridge.csv")

# Mapping & Final Results
METADATA_FILE    = "all_progenetix_metadata.csv"
CNV_FILE         = PROGENETIX_RAW / "GLOBAL_cnv_gene_panel.parquet"
CNV_BRIDGE       = Path("cnv_id_bridge.csv")

# Cleaned directories for Joining
GDC_RNA_DIR      = GDC_CLEAN_OUTPUT / "rnaseq/"
CBIO_RNA_DIR     = CBIO_CLEAN_OUTPUT / "rnaseq/"
GDC_METH_DIR     = GDC_CLEAN_OUTPUT / "methylation/"
CBIO_METH_DIR    = CBIO_CLEAN_OUTPUT / "methylation/"

# ── Charger le gold standard ──────────────────────────────────
gold_path = sorted(Path(".").glob("gold_standard_20260512_1315.parquet"))
if not gold_path:
    raise FileNotFoundError("Aucun fichier gold_standard_*.parquet trouvé.")
df = pd.read_parquet(gold_path[-1])
log.info(f"Gold standard chargé : {df.shape[0]:,} samples × {df.shape[1]:,} colonnes")

# ── Identifier les colonnes omiques ───────────────────────────
cnv_cols       = [c for c in df.columns if c.startswith("cnv_")]
rna_gdc_cols   = [c for c in df.columns if c.startswith("rna_gdc_")]
rna_cbio_cols  = [c for c in df.columns if c.startswith("rna_cbio_")]
meth_gdc_cols  = [c for c in df.columns if c.startswith("meth_gdc_")]
meth_cbio_cols = [c for c in df.columns if c.startswith("meth_cbio_")]
rna_cols       = rna_gdc_cols + rna_cbio_cols
meth_cols      = meth_gdc_cols + meth_cbio_cols

# Nettoyage project_id
if "project_id" in df.columns:
    df["project_id"] = df["project_id"].str.replace("pgx:", "", regex=False)

# ═══════════════════════════════════════════════════════════════
# 1. RÉSUMÉ GLOBAL
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("GOLD STANDARD — LABEL DISTRIBUTION")
print("=" * 60)
print(f"  Total samples     : {len(df):,}")
print(f"  Cancer types      : {df['project_id'].nunique() if 'project_id' in df.columns else 'N/A'}")
print(f"  CNV features      : {len(cnv_cols):,}")
print(f"  RNA GDC genes     : {len(rna_gdc_cols):,}")
print(f"  RNA cBio genes    : {len(rna_cbio_cols):,}")
print(f"  Meth GDC probes   : {len(meth_gdc_cols):,}")
print(f"  Meth cBio probes  : {len(meth_cbio_cols):,}")

if "source_origin" in df.columns:
    print(f"\n  Par source :")
    print(df["source_origin"].value_counts().to_string())

# ═══════════════════════════════════════════════════════════════
# 2. COUVERTURE PAR CANCER TYPE × MODALITÉ
# ═══════════════════════════════════════════════════════════════

if "project_id" in df.columns:
    coverage = df.groupby("project_id").apply(lambda g: pd.Series({
        "n_samples" : len(g),
        "cnv"       : g[cnv_cols].notna().any(axis=1).sum()   if cnv_cols   else 0,
        "rna_gdc"   : g[rna_gdc_cols].notna().any(axis=1).sum()  if rna_gdc_cols  else 0,
        "rna_cbio"  : g[rna_cbio_cols].notna().any(axis=1).sum() if rna_cbio_cols else 0,
        "meth_gdc"  : g[meth_gdc_cols].notna().any(axis=1).sum() if meth_gdc_cols else 0,
        "meth_cbio" : g[meth_cbio_cols].notna().any(axis=1).sum() if meth_cbio_cols else 0,
    })).astype(int).sort_values("n_samples", ascending=False)

    print(f"\n{'─'*65}")
    print("  Couverture par modalité et par cancer type (gold standard)")
    print(f"{'─'*65}")
    print(coverage.to_string())


# ═══════════════════════════════════════════════════════════════
# 3. HELPER PLOT
# ═══════════════════════════════════════════════════════════════

def plot_bar(series: pd.Series, title: str, color="#4C72B0",
             top_n=35, figsize=(14, 7)):
    vc = series.value_counts(dropna=True).head(top_n)
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(vc.index[::-1], vc.values[::-1], color=color, edgecolor="white")
    for bar, val in zip(bars, vc.values[::-1]):
        ax.text(bar.get_width() + max(vc.values) * 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=8)
    ax.set_xlabel("Nombre de samples", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fname = f"gold_dist_{series.name}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    log.info(f"Sauvegardé : {fname}")


# ═══════════════════════════════════════════════════════════════
# 4. GRAPHIQUES DE DISTRIBUTION
# ═══════════════════════════════════════════════════════════════

# ── G1 : samples par cancer type ─────────────────────────────
if "project_id" in df.columns:
    plot_bar(df["project_id"],
             title=f"Gold Standard — Samples par cancer type "
                   f"({df['project_id'].nunique()} types, {len(df):,} samples)",
             color="#4C72B0")

# ── G2 : source origin ───────────────────────────────────────
if "source_origin" in df.columns:
    plot_bar(df["source_origin"],
             title="Gold Standard — Distribution par source",
             color="#55A868", figsize=(8, 4))

# ── G3 : stade pathologique ──────────────────────────────────
if "pathological_stage_id" in df.columns:
    plot_bar(df["pathological_stage_id"],
             title="Gold Standard — Distribution des stades pathologiques",
             color="#C44E52", figsize=(10, 6))

# ── G4 : heatmap couverture % par cancer type × modalité ─────
if "project_id" in df.columns and not coverage.empty:

    # % de samples couverts par modalité
    cov_pct = coverage[["cnv", "rna_gdc", "rna_cbio", "meth_gdc", "meth_cbio"]]\
              .div(coverage["n_samples"], axis=0) * 100

    n_types = len(cov_pct)
    fig, ax = plt.subplots(figsize=(7, max(6, n_types * 0.35)))
    im = ax.imshow(cov_pct.values, aspect="auto", cmap="YlGn", vmin=0, vmax=100)

    ax.set_xticks(range(5))
    ax.set_xticklabels(["CNV", "RNA\nGDC", "RNA\ncBio", "Meth\nGDC", "Meth\ncBio"],
                        fontsize=9)
    ax.set_yticks(range(n_types))
    ax.set_yticklabels(
        [f"{t}  (n={coverage.loc[t,'n_samples']:,})" for t in cov_pct.index],
        fontsize=8
    )

    # Annotations dans les cellules
    for i in range(n_types):
        for j in range(5):
            val = cov_pct.values[i, j]
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=7, color="black" if val < 70 else "white")

    plt.colorbar(im, ax=ax, label="% samples couverts", shrink=0.6)
    ax.set_title("Gold Standard — Couverture par modalité et cancer type",
                 fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig("gold_coverage_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()
    log.info("Sauvegardé : gold_coverage_heatmap.png")

# ── G5 : stacked bar — composition des sources RNA + Meth ────
if "project_id" in df.columns:

    df["has_rna_gdc"]   = df[rna_gdc_cols].notna().any(axis=1)   if rna_gdc_cols   else False
    df["has_rna_cbio"]  = df[rna_cbio_cols].notna().any(axis=1)  if rna_cbio_cols  else False
    df["has_meth_gdc"]  = df[meth_gdc_cols].notna().any(axis=1)  if meth_gdc_cols  else False
    df["has_meth_cbio"] = df[meth_cbio_cols].notna().any(axis=1) if meth_cbio_cols else False

    # Catégorie RNA source
    def rna_source(row):
        if row["has_rna_gdc"] and row["has_rna_cbio"]: return "RNA: GDC+cBio"
        if row["has_rna_gdc"]:                          return "RNA: GDC only"
        if row["has_rna_cbio"]:                         return "RNA: cBio only"
        return "RNA: none"

    def meth_source(row):
        if row["has_meth_gdc"] and row["has_meth_cbio"]: return "Meth: GDC+cBio"
        if row["has_meth_gdc"]:                           return "Meth: GDC only"
        if row["has_meth_cbio"]:                          return "Meth: cBio only"
        return "Meth: none"

    df["rna_source_cat"]  = df.apply(rna_source,  axis=1)
    df["meth_source_cat"] = df.apply(meth_source, axis=1)

    for col, title, colors in [
        ("rna_source_cat",  "Gold Standard — Source RNA par cancer type",
         {"RNA: GDC only": "#4C72B0", "RNA: cBio only": "#55A868",
          "RNA: GDC+cBio": "#C44E52", "RNA: none": "#d3d3d3"}),
        ("meth_source_cat", "Gold Standard — Source Méthylation par cancer type",
         {"Meth: GDC only": "#4C72B0", "Meth: cBio only": "#55A868",
          "Meth: GDC+cBio": "#C44E52", "Meth: none": "#d3d3d3"}),
    ]:
        pivot = df.groupby(["project_id", col]).size().unstack(fill_value=0)
        pivot = pivot.loc[coverage.index]  # même ordre que coverage

        fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.35)))
        bottom = pd.Series(0, index=pivot.index)
        for cat, color in colors.items():
            if cat in pivot.columns:
                ax.barh(pivot.index, pivot[cat], left=bottom,
                        label=cat, color=color, edgecolor="white")
                bottom += pivot[cat]

        ax.set_xlabel("Nombre de samples")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        plt.tight_layout()
        fname = f"gold_source_{col}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.show()
        log.info(f"Sauvegardé : {fname}")

# Nettoyage colonnes temporaires
df.drop(columns=["has_rna_gdc", "has_rna_cbio", "has_meth_gdc",
                  "has_meth_cbio", "rna_source_cat", "meth_source_cat"],
        errors="ignore", inplace=True)

from joblib import Parallel, delayed
import gc

# ── FONCTION GÉNÉRIQUE DE CHARGEMENT PARALLÈLE ───────────────────────────────

def load_csv(f: Path, sep=None) -> tuple[str, pd.DataFrame]:
    """Charge un CSV en détectant automatiquement le séparateur."""
    if sep is None:
        # Détection automatique : lit les 2 premières lignes
        with open(f) as fh:
            first_line = fh.readline()
        sep = "\t" if first_line.count("\t") > first_line.count(",") else ","
    
    df = pd.read_csv(f, index_col=0, sep=sep, engine="pyarrow")
    return f.stem, df


def load_folder_parallel(folder: Path, pattern: str, stem_replace: str = "", n_jobs: int = -1):
    """
    Charge tous les CSV d'un dossier en parallèle.
    n_jobs=-1 utilise tous les cœurs disponibles.
    """
    files = sorted(folder.glob(pattern)) if folder.exists() else []
    if not files:
        print(f"  ✗ Aucun fichier trouvé : {folder}/{pattern}")
        return {}

    print(f"  Chargement de {len(files)} fichiers en parallèle...")
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(load_csv)(f) for f in files
    )

    matrices = {}
    for stem, df in results:
        key = stem.replace(stem_replace, "") if stem_replace else stem
        matrices[key] = df
        print(f"  ✓ {key} : {df.shape[0]} features × {df.shape[1]} samples")

    return matrices


# ── CHARGEMENT PARALLÈLE DE TOUTES LES MODALITÉS ─────────────────────────────

print("=" * 55)
print("CHARGEMENT DES DONNÉES LOCALES (parallèle)")
print("=" * 55)

print("\n── GDC RNA-seq ──")
gdc_rna_matrices = load_folder_parallel(
    GDC_RNA_INPUT, "*_STAR_counts_matrix.csv", "_STAR_counts_matrix"
)

print("\n── GDC Méthylation ──")
gdc_meth_matrices = load_folder_parallel(
    GDC_METH_INPUT, "*_methylation_beta_matrix.csv", "_methylation_beta_matrix"
)

print("\n── cBioPortal RNA-seq ──")
cbio_rna_matrices = load_folder_parallel(
    CBIO_BASE_INPUT / "rnaseq", "*.csv"
)

print("\n── cBioPortal Méthylation ──")
cbio_meth_matrices = load_folder_parallel(
    CBIO_BASE_INPUT / "methylation", "*.csv"
)

# ── CNV (fichier unique — pas besoin de parallélisme) ─────────────────────────
print("\n── CNV Progenetix ──")
cnv_global = None
cnv_global_path = PROGENETIX_RAW / "GLOBAL_cnv_gene_panel.csv"
if cnv_global_path.exists():
    cnv_global = pd.read_csv(cnv_global_path, low_memory=False)
    print(f"  ✓ GLOBAL_cnv : {cnv_global.shape[0]} lignes × {cnv_global.shape[1]} colonnes")
elif CNV_FILE.exists():
    cnv_global = pd.read_parquet(CNV_FILE)
    print(f"  ✓ GLOBAL_cnv (parquet) : {cnv_global.shape[0]} lignes × {cnv_global.shape[1]} colonnes")
else:
    print(f"  ✗ CNV introuvable → lance run_CNV()")

# ── Nettoyage mémoire ─────────────────────────────────────────────────────────
gc.collect()

# ── Résumé ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("RÉSUMÉ")
print("=" * 55)
print(f"  GDC RNA        : {len(gdc_rna_matrices)} cohortes")
print(f"  GDC Méthyl     : {len(gdc_meth_matrices)} cohortes")
print(f"  cBio RNA       : {len(cbio_rna_matrices)} études")
print(f"  cBio Méthyl    : {len(cbio_meth_matrices)} études")
print(f"  CNV global     : {'✓' if cnv_global is not None else '✗ manquant'}")
print("=" * 55)