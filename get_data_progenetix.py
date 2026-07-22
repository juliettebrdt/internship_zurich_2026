"""
CNV Pipeline — Adapté de l'approche biosample_summary + gene_panel
===================================================================
Reproduit et étend la logique de la tutrice :
  1. Charge biosample_summary.csv (dump Progenetix complet)
  2. Charge gene_cnv_cancer_panel.tsv (CNV déjà résumés par gène)
  3. Filtre : analysis_id présents dans le panel + exclusion plateformes non-CNV
  4. Filtre additionnel : sources TCGA + cBioPortal uniquement
  5. Joint les métadonnées aux données CNV
  6. Sauvegarde par cohorte + global
"""

import pandas as pd
import logging
from pathlib import Path

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("cnv_pipeline.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────
BIOSAMPLE_SUMMARY = Path("/Users/bgadmin/Downloads/biosample_summary.csv")
GENE_PANEL        = Path("/Users/bgadmin/Downloads/gene_cnv_cancer_panel.tsv")
OUTPUT_DIR        = Path("progenetix_cnv")

# Plateformes à exclure (non-CNV : expression arrays, etc.)
# EFO:0001456 = Affymetrix expression array
# EFO:0002701 = RNA expression array
EXCLUDED_PLATFORMS = {"EFO:0001456", "EFO:0002701"}

# Colonnes à garder depuis la biosample_summary
BIOSAMPLE_COLS = [
    "biosample_id", "individual_id", "analysis_id",
    "platform_id", "histological_diagnosis_id",
    "icdo_topography_id", "icdo_morphology_id",
    "pathological_stage_id", "sample_origin_type_id",
    "cohorts"
]


# ══════════════════════════════════════════════════════════════════
# STEP 1 — Charger et filtrer les biosamples (logique tutrice)
# ══════════════════════════════════════════════════════════════════

def load_and_filter_biosamples() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reproduit exactement la logique de la tutrice + filtre TCGA/cBioPortal.

    Filtres appliqués :
      1. analysis_id présent dans le gene panel (a des données CNV)
      2. platform_id hors des plateformes non-CNV
      3. Source = TCGA ou cBioPortal (filtre additionnel)
    """
    log.info("Loading biosample_summary.csv...")

    # Garder uniquement les colonnes disponibles dans le fichier
    # (certaines peuvent être absentes selon la version du dump)
    available_cols = pd.read_csv(BIOSAMPLE_SUMMARY, nrows=0).columns.tolist()
    cols_to_load   = [c for c in BIOSAMPLE_COLS if c in available_cols]
    missing        = [c for c in BIOSAMPLE_COLS if c not in available_cols]

    if missing:
        log.warning(f"Columns not found in biosample_summary: {missing}")

    bios = pd.read_csv(
        BIOSAMPLE_SUMMARY,
        sep=",",
        usecols=cols_to_load,
        dtype=str,
        low_memory=False
    )
    log.info(f"Total biosamples in summary: {len(bios)}")

    # ── Charger le gene panel ─────────────────────────────────────
    log.info("Loading gene_cnv_cancer_panel.tsv...")
    gene_panel = pd.read_csv(
        GENE_PANEL,
        sep="\t",
        dtype={
            "analysis_id": str,
            "gene_symbol":  str,
            "gene_id":      str,
            "chrom":        str,
            "start":        int,
            "end":          int,
            "dup_frac":     float,
            "del_frac":     float,
            "hldup_frac":   float,
            "hldel_frac":   float,
        }
    )
    log.info(f"Gene panel: {len(gene_panel)} rows, "
             f"{gene_panel['analysis_id'].nunique()} unique analyses, "
             f"{gene_panel['gene_symbol'].nunique()} unique genes")

    panel_ana_ids = set(gene_panel["analysis_id"].unique())

    # ── Filtre 1 : analysis_id présent dans le panel ──────────────
    bios_panel = bios[bios["analysis_id"].isin(panel_ana_ids)].copy()
    log.info(f"After analysis_id filter: {len(bios_panel)} biosamples")

    # ── Filtre 2 : exclure les plateformes non-CNV ────────────────
    if "platform_id" in bios_panel.columns:
        bios_panel = bios_panel[
            ~bios_panel["platform_id"].isin(EXCLUDED_PLATFORMS)
        ].copy()
        log.info(f"After platform filter: {len(bios_panel)} biosamples")

    import re

# ── Filtre 3 : TCGA + cBioPortal via colonne cohorts ─────────────
    if "cohorts" in bios_panel.columns:

        is_tcga = bios_panel["cohorts"].str.contains("TCGA",       case=False, na=False)
        is_cbio = bios_panel["cohorts"].str.contains("cbioportal", case=False, na=False)

        bios_panel = bios_panel[is_tcga | is_cbio].copy()
        log.info(f"After cohorts filter: {len(bios_panel)} biosamples")

        # Extraire les vrais IDs depuis la colonne cohorts
        def extract_tcga_project(cohorts_str: str) -> str | None:
            """Extrait 'TCGA-BRCA' depuis une string comme 'TCGA-BRCA::Breast...' """
            if pd.isna(cohorts_str):
                return None
            m = re.search(r"(TCGA-[A-Z]+)", str(cohorts_str))
            return m.group(1) if m else None

        def extract_cbio_id(cohorts_str: str) -> str | None:
            """Extrait 'cbioportal:acyc_mskcc_2013' depuis la colonne cohorts"""
            if pd.isna(cohorts_str):
                return None
            m = re.search(r"(cbioportal:[a-z0-9_]+)", str(cohorts_str))
            return m.group(1) if m else None

        bios_panel["tcgaproject_id"] = bios_panel["cohorts"].apply(extract_tcga_project)
        bios_panel["cbioportal_id"]  = bios_panel["cohorts"].apply(extract_cbio_id)

        # Stats
        n_tcga = bios_panel["tcgaproject_id"].notna().sum()
        n_cbio = (bios_panel["cbioportal_id"].notna() & bios_panel["tcgaproject_id"].isna()).sum()

    log.info("\n" + "="*40 + "\n--- BIOSAMPLE FILTRATION STATS ---")
    log.info(f"Total entries in raw summary : {len(bios)}")
    log.info(f"Total matching the CNV Panel : {len(panel_ana_ids)}")
    log.info(f"Final Cohorts Kept           : {len(bios_panel)}")
    log.info(f"  -> TCGA Samples            : {n_tcga}")
    log.info(f"  -> cBioPortal-Only Samples : {n_cbio}\n" + "="*40)
    return bios_panel, gene_panel


# ══════════════════════════════════════════════════════════════════
# STEP 2 — Joindre biosamples + CNV genes
# ══════════════════════════════════════════════════════════════════

def merge_cnv_with_metadata(
    bios_panel: pd.DataFrame,
    gene_panel: pd.DataFrame
) -> pd.DataFrame:
    """
    Joint les données CNV du gene panel avec les métadonnées des biosamples.

    Clé de jointure : analysis_id
    Résultat : une ligne par (gène × biosample) avec toutes les métadonnées
    """
    log.info("Merging CNV data with biosample metadata...")

    merged = gene_panel.merge(
        bios_panel,
        on="analysis_id",
        how="inner"   # inner = on garde uniquement les analyses présentes des deux côtés
    )

    log.info(f"Merged shape: {merged.shape}")
    log.info(f"  Unique biosamples : {merged['biosample_id'].nunique()}")
    log.info(f"  Unique analyses   : {merged['analysis_id'].nunique()}")
    log.info(f"  Unique genes      : {merged['gene_symbol'].nunique()}")

    return merged


# ══════════════════════════════════════════════════════════════════
# STEP 3 — Sauvegarder par cohorte TCGA + global cBioPortal
# ══════════════════════════════════════════════════════════════════

def save_by_cohort(merged: pd.DataFrame) -> None:
    """
    Sauvegarde les données CNV par cohorte TCGA (TCGA-BRCA, TCGA-LUAD...)
    et en un fichier global pour cBioPortal.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── TCGA : un fichier par projet ──────────────────────────────
    if "tcgaproject_id" in merged.columns:
        tcga_dir = OUTPUT_DIR / "TCGA"
        tcga_dir.mkdir(exist_ok=True)

        tcga_data = merged[merged["tcgaproject_id"].notna()]
        for project_id, group in tcga_data.groupby("tcgaproject_id"):
            out = tcga_dir / f"{project_id}_cnv_gene_panel.csv"
            group.to_csv(out, index=False)
            log.info(f"  ✅ TCGA {project_id}: {group['biosample_id'].nunique()} samples "
                     f"→ {out.name}")

    # ── cBioPortal : un fichier global ───────────────────────────
    if "cbioportal_id" in merged.columns:
        cbio_dir  = OUTPUT_DIR / "CBIOPORTAL"
        cbio_dir.mkdir(exist_ok=True)

        # Samples purement cBioPortal (pas dans TCGA)
        if "tcgaproject_id" in merged.columns:
            cbio_data = merged[merged["tcgaproject_id"].isna() & merged["cbioportal_id"].notna()]
        else:
            cbio_data = merged[merged["cbioportal_id"].notna()]

        if not cbio_data.empty:
            out = cbio_dir / "cBioPortal_cnv_gene_panel.csv"
            cbio_data.to_csv(out, index=False)
            log.info(f"  ✅ cBioPortal: {cbio_data['biosample_id'].nunique()} samples "
                     f"→ {out.name}")

    # ── Global : tous les samples ─────────────────────────────────
    global_out = OUTPUT_DIR / "GLOBAL_cnv_gene_panel.csv"
    merged.to_csv(global_out, index=False)
    log.info(f"\n  💾 Global file: {global_out} ({len(merged)} rows)")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def run():
    OUTPUT_DIR.mkdir(exist_ok=True)

    log.info("=" * 55)
    log.info("CNV Pipeline — biosample_summary + gene_panel")
    log.info("=" * 55)

    # Step 1 — charger et filtrer
    bios_panel, gene_panel = load_and_filter_biosamples()

    if bios_panel.empty:
        log.error("No biosamples after filtering. Check your input files.")
        return

    # Step 2 — merger CNV + métadonnées
    merged = merge_cnv_with_metadata(bios_panel, gene_panel)

    if merged.empty:
        log.error("Merge produced no results. Check analysis_id compatibility.")
        return

    # Step 3 — sauvegarder par cohorte
    save_by_cohort(merged)

    log.info("\n🎉 Pipeline complete.")


if __name__ == "__main__":
    run()