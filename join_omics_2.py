"""
join_omics_data.py
─────────────────────────────────────────────────────────────
Joins all omics data sources on a common sample identifier.

Key fixes vs previous version:
  1. GDC RNA-seq matrices: genes are index, samples are columns
     → NO transposition needed, just merge on column names
  2. cBioPortal ID matching: only keep files whose column IDs
     match Progenetix biosample_name (P-XXXXXXX format)
     → Files with MSKPCa8_ORG, MDA-PCa-117, TP_2077 are skipped
  3. TCGA barcodes kept at full length (16 chars: TCGA-XX-XXXX-01A)
     → No truncation to 12 chars
  4. Methylation handled for both GDC and cBioPortal

Sources:
  - Progenetix metadata          → all_progenetix_metadata.csv
  - GDC RNA-seq (filtered)       → cohort_matrices_cleaned/rnaseq/
  - GDC Methylation (filtered)   → cohort_matrices_cleaned/methylation/
  - cBioPortal RNA-seq           → cbioportal_cleaned/rnaseq/
  - cBioPortal Methylation       → cbioportal_cleaned/methylation/
  - CNV (Progenetix)             → progenetix_cnv/

Output:
  id_mapping.csv              → UUID → TCGA barcode
  combined_rnaseq.parquet     → all RNA-seq (genes × samples)
  combined_methylation.parquet→ all methylation (CpG × samples)
  final_joined_table.parquet  → full joined table
  gold_standard_only.parquet  → samples with RNA + Meth + CNV

Usage:
  python join_omics_data.py
  nohup python join_omics_data.py > join_omics.log 2>&1 &
"""

import requests
import pandas as pd
import time
import logging
import mygene
from pathlib import Path

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("join_omics.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────
METADATA_FILE = "all_progenetix_metadata.csv"
CNV_DIR       = Path("progenetix_cnv/")
GDC_RNA_DIR   = Path("cohort_matrices_cleaned/rnaseq/")
GDC_METH_DIR  = Path("cohort_matrices_cleaned/methylation/")
CBIO_RNA_DIR  = Path("cbioportal_cleaned/rnaseq/")
CBIO_METH_DIR = Path("cbioportal_cleaned/methylation/")

GDC_CASES_URL = "https://api.gdc.cancer.gov/cases"
CHUNK_SIZE    = 200


# =========================================================
# STEP 1 — UUID → TCGA BARCODE MAPPING
# =========================================================
def build_uuid_to_barcode_mapping(uuids: list[str]) -> pd.DataFrame:
    """
    Convert GDC UUIDs to TCGA barcodes via the GDC API.
    Keeps full barcode: TCGA-XX-XXXX-01A (16 chars).
    """
    mapping_file = Path("id_mapping.csv")
    if mapping_file.exists():
        log.info("ID mapping already built — loading from disk.")
        return pd.read_csv(mapping_file)

    log.info(f"Building UUID → barcode mapping for {len(uuids)} UUIDs...")
    all_rows  = []
    n_chunks  = (len(uuids) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for i in range(0, len(uuids), CHUNK_SIZE):
        chunk     = uuids[i:i + CHUNK_SIZE]
        chunk_num = i // CHUNK_SIZE + 1
        log.info(f"  Chunk {chunk_num}/{n_chunks}...")

        filters = {
            "op": "in",
            "content": {"field": "samples.sample_id", "value": chunk}
        }
        params = {
            "filters": filters,
            "fields": "samples.sample_id,samples.submitter_id,submitter_id",
            "format": "JSON",
            "size": CHUNK_SIZE
        }

        try:
            r = requests.post(GDC_CASES_URL, json=params, timeout=60)
            r.raise_for_status()
            hits = r.json().get("data", {}).get("hits", [])
            for hit in hits:
                case_barcode = hit.get("submitter_id", "")
                for sample in hit.get("samples", []):
                    all_rows.append({
                        "gdc_uuid":     sample.get("sample_id", ""),
                        "tcga_barcode": sample.get("submitter_id", ""),
                        "tcga_patient": case_barcode
                    })
            log.info(f"    {len(hits)} cases returned.")
        except Exception as e:
            log.error(f"  Error in chunk {chunk_num}: {e}")

        time.sleep(0.3)

    df_map = pd.DataFrame(all_rows).drop_duplicates(subset=["gdc_uuid"])
    df_map.to_csv(mapping_file, index=False)
    log.info(f"Mapping: {len(df_map)} UUIDs resolved.")
    return df_map


# =========================================================
# STEP 2 — ENRICH METADATA
# =========================================================
def enrich_metadata(df_meta: pd.DataFrame, df_map: pd.DataFrame) -> pd.DataFrame:
    """
    Add tcga_barcode column to metadata.
    - TCGA: UUID → barcode via mapping (full 16-char barcode)
    - cBioPortal: biosample_name IS the join key
    """
    log.info("Enriching metadata with barcodes...")

    # TCGA: merge on UUID
    df_tcga = df_meta[df_meta["source_origin"] == "TCGA"].copy()
    df_tcga = df_tcga.merge(
        df_map[["gdc_uuid", "tcga_barcode"]],
        left_on="biosample_name",
        right_on="gdc_uuid",
        how="left"
    )

    # cBioPortal: biosample_name = sample ID
    df_cbio = df_meta[df_meta["source_origin"] == "cBioPortal"].copy()
    df_cbio["tcga_barcode"] = df_cbio["biosample_name"]

    df_enriched = pd.concat([df_tcga, df_cbio], ignore_index=True)

    n_tcga_matched = df_tcga["tcga_barcode"].notna().sum()
    log.info(f"TCGA barcodes resolved : {n_tcga_matched} / {len(df_tcga)}")
    log.info(f"cBioPortal IDs set     : {len(df_cbio)}")

    return df_enriched


# =========================================================
# STEP 3 — LOAD GDC MATRICES
# =========================================================
def load_gdc_matrices(
    directory: Path,
    source_label: str
) -> pd.DataFrame | None:
    """
    Load and concatenate GDC matrices.

    GDC format (genes × samples):
      index   = ENSEMBL gene IDs  (ENSG00000002016.18)
      columns = TCGA barcodes     (TCGA-3C-AAAU-01A)

    → genes stay as index, samples stay as columns
    → concatenate along axis=1 (add more sample columns)
    """
    files = sorted(directory.glob("*.csv"))
    if not files:
        log.warning(f"No files in {directory}/")
        return None

    log.info(f"[{source_label}] Loading {len(files)} GDC matrices...")
    all_matrices = []

    for f in files:
        try:
            df = pd.read_csv(f, index_col=0, low_memory=False)

            # Drop non-numeric columns
            meta_cols = ["analysis_id", "group_id", "source",
                         "Entrez_Gene_Id", "Hugo_Symbol"]
            df = df.drop(columns=[c for c in meta_cols if c in df.columns],
                         errors="ignore")
            df = df.select_dtypes(include="number")

            if df.empty:
                log.warning(f"  {f.name}: no numeric data — skipping.")
                continue

            # Handle duplicate gene rows
            if df.index.duplicated().any():
                df = df.groupby(df.index).mean()

            all_matrices.append(df)
            log.info(f"  {f.name}: {df.shape[0]} genes × {df.shape[1]} samples")

        except Exception as e:
            log.error(f"  Could not load {f.name}: {e}")

    if not all_matrices:
        return None

    # Concatenate along axis=1 (add sample columns)
    combined = pd.concat(all_matrices, axis=1)

    # Remove duplicate sample columns
    before = combined.shape[1]
    combined = combined.loc[:, ~combined.columns.duplicated(keep="first")]
    if combined.shape[1] < before:
        log.info(f"  Removed {before - combined.shape[1]} duplicate sample columns.")

    log.info(f"[{source_label}] Combined: {combined.shape[0]} genes × {combined.shape[1]} samples")
    return combined


# =========================================================
# STEP 4 — LOAD CBIOPORTAL MATRICES (WITH ID FILTER)
# =========================================================
def load_cbioportal_matrices(
    directory: Path,
    valid_ids: set,
    source_label: str
) -> pd.DataFrame | None:
    """
    Load cBioPortal matrices, keeping only files whose column IDs
    match Progenetix biosample_name values.

    cBioPortal format (genes × samples):
      index   = gene symbols  (TP53, BRCA1)
      columns = sample IDs    (P-0039208-T01-IM6 or MSKPCa8_ORG ...)

    Only files where >50% of column IDs match valid_ids are kept.
    → This skips files with MSKPCa8_ORG, MDA-PCa-117, TP_2077 etc.
    """
    files = sorted(directory.glob("*.csv"))
    if not files:
        log.warning(f"No files in {directory}/")
        return None

    log.info(f"[{source_label}] Checking {len(files)} cBioPortal matrices...")
    all_matrices = []

    for f in files:
        try:
            df = pd.read_csv(f, index_col=0, low_memory=False)

            # Drop non-numeric columns
            if "Entrez_Gene_Id" in df.columns:
                df = df.drop(columns=["Entrez_Gene_Id"])
            df = df.select_dtypes(include="number")

            if df.empty:
                log.warning(f"  {f.name}: no numeric data — skipping.")
                continue

            # Check how many sample IDs match Progenetix
            cols = set(df.columns.astype(str))
            n_match = len(cols & valid_ids)
            match_rate = n_match / len(cols) if cols else 0

            if match_rate < 0.5:
                log.info(
                    f"  ⚠️  {f.name}: only {n_match}/{len(cols)} IDs match "
                    f"Progenetix ({match_rate:.0%}) — skipping."
                )
                continue

            # Keep only columns that match valid_ids
            matched_cols = [c for c in df.columns if c in valid_ids]
            df = df[matched_cols]

            # Handle duplicate gene rows
            if df.index.duplicated().any():
                df = df.groupby(df.index).mean()

            all_matrices.append(df)
            log.info(
                f"  ✅ {f.name}: {df.shape[0]} genes × {df.shape[1]} samples "
                f"({match_rate:.0%} match)"
            )

        except Exception as e:
            log.error(f"  Could not load {f.name}: {e}")

    if not all_matrices:
        log.warning(f"[{source_label}] No matching files found.")
        return None

    # Concatenate along axis=1
    combined = pd.concat(all_matrices, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated(keep="first")]

    log.info(f"[{source_label}] Combined: {combined.shape[0]} genes × {combined.shape[1]} samples")
    return combined


# =========================================================
# STEP 5 — CONVERT ENSEMBL → GENE SYMBOL (CORRIGÉ)
# =========================================================
def convert_ensembl_to_symbol(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert ENSEMBL IDs (with version) to gene symbols via MyGene.info.
    ENSG00000141510.15 → TP53
    Keeps ENSEMBL ID as fallback if no symbol found.
    """
    if df is None or df.empty:
        return df

    log.info("Converting ENSEMBL IDs → gene symbols via MyGene.info...")

    # Strip version: ENSG00000141510.15 → ENSG00000141510
    ensembl_ids = [str(g).split(".")[0] for g in df.index]

    mg = mygene.MyGeneInfo()
    results = mg.querymany(
        ensembl_ids,
        scopes="ensembl.gene",
        fields="symbol",
        species="human",
        verbose=False
    )

    mapping = {
        r["query"]: r["symbol"]
        for r in results
        if "symbol" in r and "notfound" not in r
    }

    log.info(f"  Converted: {len(mapping)} / {len(ensembl_ids)} genes")

    # Apply mapping
    df.index = [mapping.get(str(g).split(".")[0], g) for g in df.index]

    # FIX: Gestion correcte des doublons d'index (gènes)
    if df.index.duplicated().any():
        n_dups = df.index.duplicated().sum()
        log.info(f"  Removing {n_dups} duplicate gene symbols (keeping highest mean expression)...")
        
        # Tri des lignes par moyenne d'expression décroissante
        row_means = df.mean(axis=1)
        df = df.iloc[row_means.argsort()[::-1]]
        
        # On ne garde que la première occurrence (la plus exprimée) de chaque gène
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()

    log.info(f"  Final gene count: {len(df)}")
    return df


# =========================================================
# STEP 6 — LOAD CNV MATRICES
# =========================================================
def load_cnv_matrices(cnv_dir: Path) -> pd.DataFrame | None:
    files = sorted(cnv_dir.glob("**/*.csv"))
    if not files:
        log.warning(f"No CNV files in {cnv_dir}/")
        return None

    log.info(f"[CNV] Loading {len(files)} CNV files...")
    all_matrices = []

    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)

            # Format long : pivot sur biosample_id × gene_symbol
            # Valeur : dup_frac (fraction de duplication)
            if "biosample_id" in df.columns and "gene_symbol" in df.columns:
                df_pivot = df.pivot_table(
                    index="biosample_id",      # ← pgxbs-xxx
                    columns="gene_symbol",
                    values="dup_frac",
                    aggfunc="mean"
                )
                df_pivot.columns.name = None
                log.info(f"  {f.name}: pivoted → {df_pivot.shape}")
                all_matrices.append(df_pivot)
            else:
                log.warning(f"  {f.name}: missing biosample_id or gene_symbol — skipping.")

        except Exception as e:
            log.error(f"  Could not load {f.name}: {e}")

    if not all_matrices:
        return None

    combined = pd.concat(all_matrices, axis=0, sort=False)

    if combined.index.duplicated().any():
        combined = combined.groupby(combined.index).mean()

    log.info(f"[CNV] Final: {combined.shape[0]} samples × {combined.shape[1]} genes")
    return combined


# =========================================================
# STEP 7 — JOIN ALL SOURCES
# =========================================================
def join_metadata_with_omics(df_meta, df_rnaseq, df_methylation, df_cnv):
    log.info("Joining metadata with omics data...")
    df = df_meta.copy()
    df["biosample_id"] = df["biosample_id"].astype(str).str.strip()
    df["tcga_barcode"] = df["tcga_barcode"].astype(str).str.strip()

    # Barcode tronqué à 15 chars pour RNA-seq/méthylation GDC
    # TCGA-BH-A0DP-01A (16) → TCGA-BH-A0DP-01 (15)
    df["tcga_barcode_15"] = df["tcga_barcode"].apply(
        lambda x: x[:15] if str(x).startswith("TCGA") else x
    )

    # ── RNA-seq ───────────────────────────────────────────────────
    if df_rnaseq is not None:
        log.info(f"  Joining RNA-seq ({df_rnaseq.shape[1]} samples)...")
        rna_T = df_rnaseq.T.copy()
        rna_T.index = rna_T.index.astype(str).str.strip()
        rna_T.columns = [f"rna_{c}" for c in rna_T.columns]
        rna_T.index.name = "tcga_barcode_15"

        df = df.merge(rna_T.reset_index(), on="tcga_barcode_15", how="left")
        rna_cols = [c for c in df.columns if c.startswith("rna_")]
        n = df[rna_cols[0]].notna().sum() if rna_cols else 0
        log.info(f"  RNA-seq: {n} samples matched.")

    # ── Methylation ───────────────────────────────────────────────
    if df_methylation is not None:
        log.info(f"  Joining methylation ({df_methylation.shape[1]} samples)...")
        meth_T = df_methylation.T.copy()
        meth_T.index = meth_T.index.astype(str).str.strip()
        meth_T.columns = [f"meth_{c}" for c in meth_T.columns]
        meth_T.index.name = "tcga_barcode_15"

        df = df.merge(meth_T.reset_index(), on="tcga_barcode_15", how="left")
        meth_cols = [c for c in df.columns if c.startswith("meth_")]
        n = df[meth_cols[0]].notna().sum() if meth_cols else 0
        log.info(f"  Methylation: {n} samples matched.")

    # ── CNV ───────────────────────────────────────────────────────
    if df_cnv is not None:
        log.info(f"  Joining CNV ({len(df_cnv)} samples)...")
        cnv_ready = df_cnv.copy()
        cnv_ready.index = cnv_ready.index.astype(str).str.strip()
        cnv_ready.columns = [f"cnv_{c}" for c in cnv_ready.columns]
        cnv_ready.index.name = "biosample_id"  # ← pgxbs-xxx

        df = df.merge(cnv_ready.reset_index(), on="biosample_id", how="left")
        cnv_cols = [c for c in df.columns if c.startswith("cnv_")]
        n = df[cnv_cols[0]].notna().sum() if cnv_cols else 0
        log.info(f"  CNV: {n} samples matched.")

    log.info(f"Final table shape: {df.shape}")
    return df


# =========================================================
# MAIN PIPELINE
# =========================================================
def run():
    log.info("=" * 55)
    log.info("STARTING OMICS JOIN PIPELINE")
    log.info("=" * 55)

    # ── Load metadata ─────────────────────────────────────────────
    log.info("\nLoading metadata...")
    df_meta = pd.read_csv(METADATA_FILE, low_memory=False)
    log.info(f"Metadata: {df_meta.shape}")

    # ── Step 1: UUID → barcode mapping ───────────────────────────
    log.info("\n" + "="*55)
    log.info("STEP 1 — UUID → barcode mapping")
    log.info("="*55)

    tcga_uuids = (
        df_meta[df_meta["source_origin"] == "TCGA"]["biosample_name"]
        .dropna().unique().tolist()
    )
    df_map = build_uuid_to_barcode_mapping(tcga_uuids)

    # ── Step 2: Enrich metadata ───────────────────────────────────
    log.info("\n" + "="*55)
    log.info("STEP 2 — Enrich metadata")
    log.info("="*55)

    df_meta_enriched = enrich_metadata(df_meta, df_map)

    # Valid cBioPortal IDs for filtering
    valid_cbio_ids = set(
        df_meta_enriched[df_meta_enriched["source_origin"] == "cBioPortal"]
        ["biosample_name"].dropna().unique()
    )
    log.info(f"Valid cBioPortal IDs: {len(valid_cbio_ids)}")

    # ── Step 3: Load RNA-seq ──────────────────────────────────────
    log.info("\n" + "="*55)
    log.info("STEP 3 — RNA-seq matrices")
    log.info("="*55)

    rnaseq_file = Path("combined_rnaseq.parquet")
    if rnaseq_file.exists():
        log.info("RNA-seq already combined — loading from disk.")
        df_rnaseq = pd.read_parquet(rnaseq_file)
    else:
        # GDC: genes × samples (ENSEMBL IDs → convert to symbols)
        df_gdc_rna = load_gdc_matrices(GDC_RNA_DIR, "GDC RNA-seq")
        if df_gdc_rna is not None:
            df_gdc_rna = convert_ensembl_to_symbol(df_gdc_rna)

        # cBioPortal: genes × samples (gene symbols, filter by valid IDs)
        df_cbio_rna = load_cbioportal_matrices(
            CBIO_RNA_DIR, valid_cbio_ids, "cBioPortal RNA-seq"
        )

        # Combine on gene index (outer join — keeps all genes)
        sources = [df for df in [df_gdc_rna, df_cbio_rna] if df is not None]
        if sources:
            # FIX: Sécurité pour forcer l'unicité de l'index avant concat
            sources = [
                df.groupby(df.index).mean() if df.index.duplicated().any() else df 
                for df in sources
            ]
            df_rnaseq = pd.concat(sources, axis=1, sort=False)
            df_rnaseq = df_rnaseq.loc[:, ~df_rnaseq.columns.duplicated()]
            df_rnaseq.to_parquet(rnaseq_file)
            log.info(f"Saved: {rnaseq_file} — {df_rnaseq.shape}")
        else:
            df_rnaseq = None

    # ── Step 4: Load Methylation ──────────────────────────────────
    log.info("\n" + "="*55)
    log.info("STEP 4 — Methylation matrices")
    log.info("="*55)

    meth_file = Path("combined_methylation.parquet")
    if meth_file.exists():
        log.info("Methylation already combined — loading from disk.")
        df_methylation = pd.read_parquet(meth_file)
    else:
        # GDC: CpG probes × samples
        df_gdc_meth = load_gdc_matrices(GDC_METH_DIR, "GDC Methylation")

        # cBioPortal: CpG or gene × samples (filter by valid IDs)
        df_cbio_meth = load_cbioportal_matrices(
            CBIO_METH_DIR, valid_cbio_ids, "cBioPortal Methylation"
        )

        sources = [df for df in [df_gdc_meth, df_cbio_meth] if df is not None]
        if sources:
            # FIX: Sécurité pour forcer l'unicité de l'index avant concat
            sources = [
                df.groupby(df.index).mean() if df.index.duplicated().any() else df 
                for df in sources
            ]
            df_methylation = pd.concat(sources, axis=1, sort=False)
            df_methylation = df_methylation.loc[
                :, ~df_methylation.columns.duplicated()
            ]
            df_methylation.to_parquet(meth_file)
            log.info(f"Saved: {meth_file} — {df_methylation.shape}")
        else:
            df_methylation = None

    # ── Step 5: Load CNV ──────────────────────────────────────────
    log.info("\n" + "="*55)
    log.info("STEP 5 — CNV matrices")
    log.info("="*55)

    df_cnv = load_cnv_matrices(CNV_DIR)

    # ── Step 6: Join everything ───────────────────────────────────
    log.info("\n" + "="*55)
    log.info("STEP 6 — Join all sources")
    log.info("="*55)

    df_final = join_metadata_with_omics(
        df_meta_enriched, df_rnaseq, df_methylation, df_cnv
    )

    # ── Save full table ───────────────────────────────────────────
    output_file = Path("final_joined_table_2.parquet")
    df_final.to_parquet(output_file, engine="pyarrow", index=False)
    log.info(f"Full table saved: {output_file}")

    # ── Gold standard (all 3 omics) ───────────────────────────────
    rna_cols  = [c for c in df_final.columns if c.startswith("rna_")]
    meth_cols = [c for c in df_final.columns if c.startswith("meth_")]
    cnv_cols  = [c for c in df_final.columns if c.startswith("cnv_")]

    if rna_cols and meth_cols and cnv_cols:
        has_rna  = df_final[rna_cols].notna().any(axis=1)
        has_meth = df_final[meth_cols].notna().any(axis=1)
        has_cnv  = df_final[cnv_cols].notna().any(axis=1)
        gold_df  = df_final[has_rna & has_meth & has_cnv]
        gold_df.to_parquet("gold_standard_only.parquet", engine="pyarrow", index=False)
        log.info(f"Gold standard saved: {len(gold_df)} samples.")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    print(f"  Total samples      : {len(df_final)}")
    print(f"  Total columns      : {df_final.shape[1]}")
    print(f"  RNA-seq genes      : {len(rna_cols)}")
    print(f"  Methylation probes : {len(meth_cols)}")
    print(f"  CNV features       : {len(cnv_cols)}")
    if rna_cols and meth_cols and cnv_cols:
        print(f"  ★ Gold standard    : {len(gold_df)} samples")
    print(f"\n  By source :")
    print(df_final["source_origin"].value_counts().to_string())
    print("="*60)


if __name__ == "__main__":
    run()