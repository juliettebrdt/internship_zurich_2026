"""
join_omics_data.py  (v5 — source-tagged columns)
─────────────────────────────────────────────────────────────────────
Changements vs v4 :
  • RNA et Méthylation ne sont PLUS concaténés puis moyennés avant le merge.
    Chaque source est mergée séparément avec son propre préfixe :
        rna_gdc_*    rna_cbio_*    meth_gdc_*    meth_cbio_*
    → 1 ligne par biosample, NaN là où la source n'a pas de données.
    → les deux sources RNA (ou Meth) peuvent coexister sur la même ligne
      sans être mélangées.

  • Gold standard : CNV + au moins 1 RNA (gdc ou cbio)
                               + au moins 1 Meth (gdc ou cbio)
    La source est indifférente — c'est la modalité qui compte.

  • concatenate_matrices() reçoit maintenant valid_ids pour GDC aussi
    (cohérence), mais n'est plus appelée avec un merge immédiat de sources.

Schéma final des colonnes :
    cnv_*          ← Progenetix
    rna_gdc_*      ← GDC / TCGA
    rna_cbio_*     ← cBioPortal
    meth_gdc_*     ← GDC / TCGA
    meth_cbio_*    ← cBioPortal

Prérequis : lancer d'abord build_cnv_id_bridge.py pour générer
            cnv_id_bridge.csv (analysis_id → biosample_id).
"""

import re
import requests
import pandas as pd
import time
import logging
from pathlib import Path
from datetime import datetime
import mygene

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
METADATA_FILE  = "all_progenetix_metadata.csv"
CNV_FILE       = Path("progenetix_cnv/GLOBAL_cnv_gene_panel.csv")
CNV_BRIDGE     = Path("cnv_id_bridge.csv")


GDC_RNA_DIR    = Path("cohort_matrices_cleaned/rnaseq/")
CBIO_RNA_DIR   = Path("cbioportal_cleaned/rnaseq/")
GDC_METH_DIR   = Path("cohort_matrices_cleaned/methylation/")
CBIO_METH_DIR  = Path("cbioportal_cleaned/methylation/")

GDC_CASES_URL  = "https://api.gdc.cancer.gov/cases"
CHUNK_SIZE     = 200

MATCH_RATE_WARNING_THRESHOLD = 0.10
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")

_TCGA_BARCODE_RE = re.compile(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-", re.IGNORECASE)


# =========================================================
# UTILS
# =========================================================

def clean_tcga_id(id_val: str) -> str:
    """
    Vrais barcodes TCGA → upper + tronqué à 15 car.
    Tous les autres IDs → strip uniquement (pas de troncature).
    """
    s = str(id_val).strip()
    if _TCGA_BARCODE_RE.match(s):
        return s.upper()[:15]
    return s


def check_match_rate(label: str, n_matched: int, n_total: int) -> None:
    if n_total == 0:
        log.warning(f"  [{label}] No samples to match.")
        return
    rate = n_matched / n_total
    msg  = f"  [{label}] Match rate: {n_matched:,}/{n_total:,} ({rate:.1%})"
    if rate < MATCH_RATE_WARNING_THRESHOLD:
        log.warning(f"WARNING LOW MATCH RATE — {msg}")
    else:
        log.info(msg)


# =========================================================
# STEP 1 — UUID → TCGA BARCODE MAPPING
# =========================================================

def build_uuid_to_barcode_mapping(uuids: list[str]) -> pd.DataFrame:
    mapping_file = Path("id_mapping.csv")
    if mapping_file.exists():
        log.info("ID mapping already built — loading from disk.")
        return pd.read_csv(mapping_file)

    log.info(f"Querying GDC API for {len(uuids):,} UUIDs...")
    all_rows = []
    n_chunks = (len(uuids) + CHUNK_SIZE - 1) // CHUNK_SIZE

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
            "fields":  "samples.sample_id,samples.submitter_id,submitter_id",
            "format":  "JSON",
            "size":    CHUNK_SIZE
        }
        try:
            r = requests.post(GDC_CASES_URL, json=params, timeout=60)
            r.raise_for_status()
            hits = r.json().get("data", {}).get("hits", [])
            for hit in hits:
                for sample in hit.get("samples", []):
                    all_rows.append({
                        "gdc_uuid":     sample.get("sample_id", ""),
                        "tcga_barcode": sample.get("submitter_id", ""),
                        "tcga_patient": hit.get("submitter_id", "")
                    })
        except Exception as e:
            log.error(f"  Error in chunk {chunk_num}: {e}")
        time.sleep(0.3)

    df_map = pd.DataFrame(all_rows).drop_duplicates(subset=["gdc_uuid"])
    df_map.to_csv(mapping_file, index=False)
    log.info(f"Mapping built: {len(df_map):,} UUIDs resolved.")
    return df_map


# =========================================================
# STEP 2 — LOAD CNV BRIDGE  (analysis_id → biosample_id)
# =========================================================

def load_cnv_bridge() -> dict[str, str]:
    if not CNV_BRIDGE.exists():
        raise FileNotFoundError(
            f"\nBridge file not found: {CNV_BRIDGE}\n"
            "  Run first:  python build_cnv_id_bridge.py\n"
        )
    df = pd.read_csv(CNV_BRIDGE, dtype=str)
    required = {"analysis_id", "biosample_id"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Bridge file must contain columns {required}. "
            f"Found: {df.columns.tolist()}"
        )
    df = df.dropna(subset=["analysis_id", "biosample_id"])
    mapping = dict(zip(df["analysis_id"].str.strip(),
                       df["biosample_id"].str.strip()))
    n_pgxcs = sum(1 for k in mapping if k.startswith("pgxcs-"))
    n_pgxbs = sum(1 for v in mapping.values() if v.startswith("pgxbs-"))
    log.info(f"CNV bridge loaded: {len(mapping):,} pairs "
             f"({n_pgxcs:,} pgxcs- → {n_pgxbs:,} pgxbs-)")
    return mapping


# =========================================================
# STEP 3 — CONCATENATE MATRICES
# =========================================================

def concatenate_matrices(
    directory: Path | str,
    glob_pattern: str = "*.csv",
    source_label: str = "",
    valid_ids: set | None = None
) -> pd.DataFrame | None:

    directory = Path(directory)
    files = [directory] if directory.is_file() else sorted(directory.glob(glob_pattern))

    if not files:
        log.warning(f"No files found in {directory} with pattern '{glob_pattern}'")
        return None

    log.info(f"[{source_label}] Processing {len(files)} file(s)...")

    all_wide: list[pd.DataFrame] = []
    all_long: list[pd.DataFrame] = []

    for f in files:
        try:
            header = pd.read_csv(f, nrows=0).columns.tolist()

            # Long CNV format
            if "gene_symbol" in header and "analysis_id" in header:
                val_col = next(
                    (c for c in ["dup_frac", "value", "cn", "log2"] if c in header),
                    None
                )
                cols = ["analysis_id", "gene_symbol"] + ([val_col] if val_col else [])
                df   = pd.read_csv(f, usecols=cols, low_memory=False)
                if val_col is None:
                    df["dup_frac"] = 1.0
                else:
                    df = df.rename(columns={val_col: "dup_frac"})
                all_long.append(df)

            # Wide format (RNA / Meth)
            else:
                df = pd.read_csv(f, low_memory=False)
                df = df.set_index(df.columns[0])
                df = df.drop(
                    columns=[c for c in ["Entrez_Gene_Id", "analysis_id",
                                         "group_id", "source", "group_label"]
                             if c in df.columns],
                    errors="ignore"
                )
                df = df.select_dtypes(include="number")
                if df.empty:
                    continue

                if valid_ids is not None:
                    keep = [c for c in df.columns
                            if clean_tcga_id(str(c)) in valid_ids]
                    if not keep:
                        log.info(f"  {f.name}: 0 matches — skipping.")
                        continue
                    df = df[keep]

                df = df.T
                df.index = [clean_tcga_id(str(i)) for i in df.index]
                if df.index.duplicated().any():
                    df = df.groupby(df.index).mean()
                all_wide.append(df)

        except Exception as e:
            log.error(f"  Could not load {f.name}: {e}")

    # Pivot long CNV
    if all_long:
        combined_long = pd.concat(all_long, ignore_index=True)
        log.info(
            f"[{source_label}] Pivoting {combined_long.shape[0]:,} rows "
            f"({combined_long['analysis_id'].nunique():,} analyses)..."
        )
        df_pivot = combined_long.pivot_table(
            index="analysis_id",
            columns="gene_symbol",
            values="dup_frac",
            aggfunc="mean"
        )
        df_pivot.columns.name = None
        all_wide.append(df_pivot)

    if not all_wide:
        return None

    combined = pd.concat(all_wide, axis=0, sort=False)
    if combined.index.duplicated().any():
        combined = combined.groupby(combined.index).mean()

    log.info(
        f"[{source_label}] Combined: "
        f"{combined.shape[0]:,} samples × {combined.shape[1]:,} features"
    )
    return combined


# =========================================================
# STEP 4 — ENSEMBL → GENE SYMBOL
# =========================================================

def convert_ensembl_to_symbol(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Converting ENSEMBL IDs → gene symbols...")
    ensembl_ids = [str(g).split(".")[0] for g in df.columns]
    mg = mygene.MyGeneInfo()
    results = mg.querymany(ensembl_ids, scopes="ensembl.gene",
                           fields="symbol", species="human", verbose=False)
    mapping = {r["query"]: r["symbol"]
               for r in results if "symbol" in r and "notfound" not in r}
    log.info(f"  Converted: {len(mapping):,}/{len(ensembl_ids):,}")
    df.columns = [mapping.get(str(g).split(".")[0], g) for g in df.columns]
    if df.columns.duplicated().any():
        df = df.T.groupby(level=0).mean().T
    return df


# =========================================================
# STEP 5 — MERGE ONE OMICS MODALITY (source-specific prefix)
# =========================================================

def merge_omics_modality(
    df: pd.DataFrame,
    omics_df: pd.DataFrame,
    prefix: str,          # ex: "rna_gdc_", "rna_cbio_", "meth_gdc_", "meth_cbio_"
    cbio_lookup: dict[str, int],
) -> pd.DataFrame:
    """
    Merge une matrice omique dans df avec un préfixe de source explicite.
    Chaque appel correspond à UNE source → pas de mélange inter-sources.
    Les doublons résiduels INTRA-source (réplicats techniques sur le même
    biosample) sont moyennés.
    """
    log.info(f"  Merging [{prefix.rstrip('_')}] ({len(omics_df):,} samples)...")
    omics = omics_df.copy()

    # Préfixer toutes les colonnes features avec le label source
    omics.columns = [
        f"{prefix}{c}" if not str(c).startswith(prefix) else c
        for c in omics.columns
    ]

    # Résoudre chaque sample_id vers l'index de df_meta
    idx_map: list[int | None] = []
    for sample_id in omics.index:
        s_clean = clean_tcga_id(str(sample_id))
        if s_clean in cbio_lookup:
            idx_map.append(cbio_lookup[s_clean])
        else:
            matches = df[df["tcga_barcode_clean"] == s_clean].index
            idx_map.append(matches[0] if not matches.empty else None)

    omics["_meta_index"] = idx_map
    omics = omics.dropna(subset=["_meta_index"])
    omics["_meta_index"] = omics["_meta_index"].astype(int)
    omics = omics.set_index("_meta_index")

    # Moyenne des réplicats techniques intra-source sur le même biosample
    if omics.index.duplicated().any():
        n_dup = omics.index.duplicated().sum()
        log.info(f"    Averaging {n_dup:,} intra-source duplicate(s) for [{prefix.rstrip('_')}]")
        omics = omics.groupby(omics.index).mean()

    df = df.merge(omics, left_index=True, right_index=True, how="left")

    feat_cols = [c for c in df.columns if c.startswith(prefix)]
    n_matched = df[feat_cols[0]].notna().sum() if feat_cols else 0
    check_match_rate(prefix.rstrip("_"), n_matched, len(omics_df))
    return df


# =========================================================
# STEP 6 — MERGE CNV  (via bridge file)
# =========================================================

def merge_cnv(
    df: pd.DataFrame,
    df_cnv: pd.DataFrame,
    ana_to_bio: dict[str, str],
) -> pd.DataFrame:

    log.info(f"  Merging CNV ({len(df_cnv):,} samples)...")
    cnv = df_cnv.copy()
    cnv.columns = [
        f"cnv_{c}" if not str(c).startswith("cnv_") else c
        for c in cnv.columns
    ]

    original = cnv.index.tolist()
    cnv.index = [ana_to_bio.get(str(i), "") for i in original]

    n_translated = sum(1 for i in original if str(i) in ana_to_bio)
    log.info(
        f"  CNV translation: {n_translated:,}/{len(original):,} "
        f"analysis_ids resolved to biosample_ids"
    )

    cnv = cnv[cnv.index != ""]
    cnv.index.name = "biosample_id"

    if cnv.index.duplicated().any():
        cnv = cnv.groupby(cnv.index).mean()

    df = df.merge(cnv, left_on="biosample_id", right_index=True, how="left")

    cnv_cols  = [c for c in df.columns if c.startswith("cnv_")]
    n_matched = df[cnv_cols[0]].notna().sum() if cnv_cols else 0
    check_match_rate("CNV", n_matched, len(cnv))
    return df


# =========================================================
# MAIN PIPELINE
# =========================================================

def run():
    log.info("=" * 55)
    log.info("STARTING OMICS JOIN PIPELINE  (v5)")
    log.info("=" * 55)

    df_meta = pd.read_csv(METADATA_FILE, low_memory=False)
    log.info(f"Metadata loaded: {df_meta.shape}")
    log.info(f"Columns: {df_meta.columns.tolist()}")

    ana_to_bio = load_cnv_bridge()

    # ── Step 1: UUID → barcode ────────────────────────────────────
    log.info("\n" + "=" * 55)
    log.info("STEP 1 — UUID → barcode mapping")
    log.info("=" * 55)

    tcga_uuids = (
        df_meta[df_meta["source_origin"] == "TCGA"]["biosample_name"]
        .dropna().unique().tolist()
    )
    df_map = build_uuid_to_barcode_mapping(tcga_uuids)

    df_meta = df_meta.merge(
        df_map[["gdc_uuid", "tcga_barcode"]],
        left_on="biosample_name", right_on="gdc_uuid",
        how="left", suffixes=("", "_gdc")
    )
    if "tcga_barcode_gdc" in df_meta.columns:
        df_meta["tcga_barcode"] = df_meta["tcga_barcode"].fillna(
            df_meta["tcga_barcode_gdc"]
        )
        df_meta = df_meta.drop(columns=["tcga_barcode_gdc"])

    df_meta.loc[df_meta["source_origin"] == "cBioPortal", "tcga_barcode"] = (
        df_meta["biosample_name"]
    )

    valid_cbio_ids = {
        clean_tcga_id(str(x))
        for x in df_meta[df_meta["source_origin"] == "cBioPortal"][
            "biosample_name"
        ].dropna().unique()
    }
    log.info(f"Valid cBioPortal IDs: {len(valid_cbio_ids):,}")

    # ── Step 2: Charger les matrices RNA (sans les fusionner) ─────
    log.info("\n" + "=" * 55)
    log.info("STEP 2 — RNA-seq (GDC et cBioPortal séparément)")
    log.info("=" * 55)

    df_gdc_rna = concatenate_matrices(GDC_RNA_DIR, "*.csv", "GDC RNA")
    if df_gdc_rna is not None:
        df_gdc_rna = convert_ensembl_to_symbol(df_gdc_rna)
        log.info(f"  GDC RNA ready: {df_gdc_rna.shape[0]:,} samples × {df_gdc_rna.shape[1]:,} genes")

    df_cbio_rna = concatenate_matrices(CBIO_RNA_DIR, "*.csv", "cBio RNA",
                                       valid_ids=None)
    if df_cbio_rna is not None:
        log.info(f"  cBio RNA ready: {df_cbio_rna.shape[0]:,} samples × {df_cbio_rna.shape[1]:,} genes")

    # ── Step 3: Charger les matrices Méthylation ──────────────────
    log.info("\n" + "=" * 55)
    log.info("STEP 3 — Methylation (GDC et cBioPortal séparément)")
    log.info("=" * 55)

    df_gdc_meth = concatenate_matrices(GDC_METH_DIR, "*.csv", "GDC Meth")
    if df_gdc_meth is not None:
        if any("ENSG" in str(c) for c in df_gdc_meth.columns[:10]):
            df_gdc_meth = convert_ensembl_to_symbol(df_gdc_meth)
        log.info(f"  GDC Meth ready: {df_gdc_meth.shape[0]:,} samples × {df_gdc_meth.shape[1]:,} probes")

    df_cbio_meth = concatenate_matrices(CBIO_METH_DIR, "*.csv", "cBio Meth",
                                        valid_ids=None)
    if df_cbio_meth is not None:
        log.info(f"  cBio Meth ready: {df_cbio_meth.shape[0]:,} samples × {df_cbio_meth.shape[1]:,} probes")

    # ── Step 4: CNV ───────────────────────────────────────────────
    log.info("\n" + "=" * 55)
    log.info("STEP 4 — CNV")
    log.info("=" * 55)
    def load_cnv_four_fractions(cnv_file: Path) -> pd.DataFrame:
        log.info(f"Loading CNV with 4 fractions from {cnv_file}...")

        df = pd.read_csv(
            cnv_file,
            sep="\t" if str(cnv_file).endswith(".tsv") else ",",
            dtype={"analysis_id": str, "gene_symbol": str,
                "dup_frac": float, "del_frac": float,
                "hldup_frac": float, "hldel_frac": float},
            low_memory=False
        )
        log.info(f"  Raw panel: {len(df):,} rows | "
                f"{df['analysis_id'].nunique():,} analyses | "
                f"{df['gene_symbol'].nunique():,} genes")

        frames = []
        for col in ["dup_frac", "del_frac", "hldup_frac", "hldel_frac"]:
            suffix = col.replace("_frac", "")
            piv = df.pivot_table(
                index="analysis_id",
                columns="gene_symbol",
                values=col,
                aggfunc="mean"
            )
            # Pas de préfixe cnv_ ici — merge_cnv() l'ajoute lui-même
            piv.columns = [f"{g}__{suffix}" for g in piv.columns]
            piv.columns.name = None
            frames.append(piv)

        wide = pd.concat(frames, axis=1).fillna(0.0)
        log.info(f"  CNV wide: {wide.shape[0]:,} samples × {wide.shape[1]:,} features "
                f"({wide.shape[1]//4} genes × 4 fractions)")
        return wide
    df_cnv = load_cnv_four_fractions(CNV_FILE)


    # ── Step 5: Final join ────────────────────────────────────────
    # Chaque source est mergée séparément avec son propre préfixe.
    # Résultat : rna_gdc_*, rna_cbio_*, meth_gdc_*, meth_cbio_*, cnv_*
    # sur la même ligne — NaN là où la source n'a pas de données.
    # ─────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 55)
    log.info("STEP 5 — Final join (source-tagged columns)")
    log.info("=" * 55)

    df = df_meta.copy()
    df["tcga_barcode_clean"] = df["tcga_barcode"].apply(clean_tcga_id)

    # Lookup rapide : tout identifiant connu → index de df_meta
    cbio_lookup: dict[str, int] = {}
    for col in ["biosample_name", "tcga_barcode", "biosample_id"]:
        if col not in df.columns:
            continue
        for idx, row in df[[col]].dropna().iterrows():
            v = str(row.iloc[0]).strip()
            cbio_lookup.setdefault(v, idx)
            cbio_lookup.setdefault(clean_tcga_id(v), idx)

    # Merge source par source — ordre fixe pour la lisibilité des colonnes
    if df_gdc_rna is not None:
        df = merge_omics_modality(df, df_gdc_rna,  "rna_gdc_",  cbio_lookup)
    if df_cbio_rna is not None:
        df = merge_omics_modality(df, df_cbio_rna, "rna_cbio_", cbio_lookup)
    if df_gdc_meth is not None:
        df = merge_omics_modality(df, df_gdc_meth,  "meth_gdc_",  cbio_lookup)
    if df_cbio_meth is not None:
        df = merge_omics_modality(df, df_cbio_meth, "meth_cbio_", cbio_lookup)
    if df_cnv is not None:
        df = merge_cnv(df, df_cnv, ana_to_bio)

    df = df.drop(columns=["tcga_barcode_clean"], errors="ignore")
    log.info(f"Final combined shape: {df.shape}")

    # ── Save full table ───────────────────────────────────────────
    output = Path(f"final_joined_table_{TIMESTAMP}.parquet")
    df.to_parquet(output, engine="pyarrow", index=False)
    log.info(f"Full table saved: {output}")

    # ── Gold standard ─────────────────────────────────────────────
    # Critère : CNV + au moins 1 RNA (gdc ou cbio) + au moins 1 Meth (gdc ou cbio)
    # La source est indifférente — c'est la modalité qui compte.
    # Les réplicats techniques intra-source ont déjà été moyennés dans
    # merge_omics_modality → 1 valeur par (biosample, source, feature).
    # ─────────────────────────────────────────────────────────────
    cnv_cols       = [c for c in df.columns if c.startswith("cnv_")]
    rna_gdc_cols   = [c for c in df.columns if c.startswith("rna_gdc_")]
    rna_cbio_cols  = [c for c in df.columns if c.startswith("rna_cbio_")]
    meth_gdc_cols  = [c for c in df.columns if c.startswith("meth_gdc_")]
    meth_cbio_cols = [c for c in df.columns if c.startswith("meth_cbio_")]

    rna_cols  = rna_gdc_cols  + rna_cbio_cols
    meth_cols = meth_gdc_cols + meth_cbio_cols

    if cnv_cols and rna_cols and meth_cols:
        has_cnv  = df[cnv_cols].notna().any(axis=1)
        has_rna  = df[rna_cols].notna().any(axis=1)
        has_meth = df[meth_cols].notna().any(axis=1)

        gold_mask = has_cnv & has_rna & has_meth
        gold_df   = df[gold_mask].copy()

        # Breakdown par combinaison de sources RNA × Meth
        has_rna_gdc   = df[rna_gdc_cols].notna().any(axis=1)   if rna_gdc_cols   else pd.Series(False, index=df.index)
        has_rna_cbio  = df[rna_cbio_cols].notna().any(axis=1)  if rna_cbio_cols  else pd.Series(False, index=df.index)
        has_meth_gdc  = df[meth_gdc_cols].notna().any(axis=1)  if meth_gdc_cols  else pd.Series(False, index=df.index)
        has_meth_cbio = df[meth_cbio_cols].notna().any(axis=1) if meth_cbio_cols else pd.Series(False, index=df.index)

        def _count(mask): return int((gold_mask & mask).sum())

        breakdown = {
            "RNA_gdc  + Meth_gdc"  : _count( has_rna_gdc  &  has_meth_gdc  & ~has_rna_cbio & ~has_meth_cbio),
            "RNA_gdc  + Meth_cbio" : _count( has_rna_gdc  &  has_meth_cbio & ~has_rna_cbio & ~has_meth_gdc),
            "RNA_cbio + Meth_gdc"  : _count( has_rna_cbio &  has_meth_gdc  & ~has_rna_gdc  & ~has_meth_cbio),
            "RNA_cbio + Meth_cbio" : _count( has_rna_cbio &  has_meth_cbio & ~has_rna_gdc  & ~has_meth_gdc),
            "RNA_both + Meth_gdc"  : _count( has_rna_gdc  &  has_rna_cbio  &  has_meth_gdc  & ~has_meth_cbio),
            "RNA_both + Meth_cbio" : _count( has_rna_gdc  &  has_rna_cbio  &  has_meth_cbio & ~has_meth_gdc),
            "RNA_gdc  + Meth_both" : _count( has_rna_gdc  & ~has_rna_cbio  &  has_meth_gdc  &  has_meth_cbio),
            "RNA_cbio + Meth_both" : _count(~has_rna_gdc  &  has_rna_cbio  &  has_meth_gdc  &  has_meth_cbio),
            "RNA_both + Meth_both" : _count( has_rna_gdc  &  has_rna_cbio  &  has_meth_gdc  &  has_meth_cbio),
        }
        for label, n in breakdown.items():
            if n > 0:
                log.info(f"  Gold [{label}] : {n:,}")

        gold_out = Path(f"gold_standard_{TIMESTAMP}.parquet")
        gold_df.to_parquet(gold_out, engine="pyarrow", index=False)
        log.info(f"Gold standard saved: {len(gold_df):,} samples → {gold_out}")
    else:
        gold_df   = pd.DataFrame()
        breakdown = {}
        missing = []
        if not cnv_cols:  missing.append("CNV")
        if not rna_cols:  missing.append("RNA")
        if not meth_cols: missing.append("Meth")
        log.warning(f"Gold standard skipped — missing modalities: {', '.join(missing)}")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY  (v5)")
    print("=" * 60)
    print(f"  Total samples        : {len(df):,}")
    print(f"  Total columns        : {df.shape[1]:,}")
    print(f"  RNA GDC genes        : {len(rna_gdc_cols):,}")
    print(f"  RNA cBio genes       : {len(rna_cbio_cols):,}")
    print(f"  Meth GDC probes      : {len(meth_gdc_cols):,}")
    print(f"  Meth cBio probes     : {len(meth_cbio_cols):,}")
    print(f"  CNV features         : {len(cnv_cols):,}")
    if not gold_df.empty:
        print(f"  Gold standard        : {len(gold_df):,} samples")
        print("    Source breakdown (CNV always present):")
        for label, n in breakdown.items():
            if n > 0:
                print(f"    ├─ {label} : {n:,}")
    if "source_origin" in df.columns:
        print("\n  By source_origin:")
        print(df["source_origin"].value_counts().to_string())
    print("=" * 60)


if __name__ == "__main__":
    run()