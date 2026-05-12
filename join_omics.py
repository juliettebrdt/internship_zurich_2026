import requests
import pandas as pd
import time
import logging
from pathlib import Path
import mygene
import pyarrow.parquet as pq

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("join_omics.log"), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────
METADATA_FILE   = "all_progenetix_metadata.csv"
CNV_DIR         = Path("progenetix_cnv/")

GDC_RNA_DIR     = Path("cohort_matrices_cleaned/rnaseq/")
CBIO_RNA_DIR    = Path("cbioportal_cleaned/rnaseq/")
GDC_METH_DIR    = Path("cohort_matrices_cleaned/methylation/")
CBIO_METH_DIR   = Path("cbioportal_cleaned/methylation/")

GDC_CASES_URL   = "https://api.gdc.cancer.gov/cases"
CHUNK_SIZE      = 200

# =========================================================
# UTILS: NORMALISATION DES BARCODES (PATIENT LEVEL)
# =========================================================
def clean_tcga_id(id_val):
    if pd.isna(id_val):
        return id_val
    s = str(id_val).strip()
    if s.upper().startswith("TCGA"):
        return s.upper()[:12]
    return s # Retourne l'ID pgxbs intact

# =========================================================
# STEP 1 — BUILD UUID → TCGA BARCODE MAPPING
# =========================================================
def build_uuid_to_barcode_mapping(uuids):
    mapping_file = Path("id_mapping.csv")
    if mapping_file.exists():
        log.info("ID mapping déjà existant, chargement...")
        return pd.read_csv(mapping_file)

    log.info(f"Requête GDC API pour {len(uuids)} UUIDs...")
    all_rows = []
    for i in range(0, len(uuids), CHUNK_SIZE):
        chunk = uuids[i:i + CHUNK_SIZE]
        filters = {"op": "in", "content": {"field": "samples.sample_id", "value": chunk}}
        params = {"filters": filters, "fields": "samples.sample_id,samples.submitter_id,submitter_id", "format": "JSON", "size": CHUNK_SIZE}
        try:
            r = requests.post(GDC_CASES_URL, json=params, timeout=60)
            hits = r.json().get("data", {}).get("hits", [])
            for hit in hits:
                case_barcode = hit.get("submitter_id", "")
                for sample in hit.get("samples", []):
                    all_rows.append({
                        "gdc_uuid": sample.get("sample_id", ""),
                        "tcga_barcode": sample.get("submitter_id", ""),
                        "tcga_patient": case_barcode
                    })
            time.sleep(0.3)
        except Exception as e:
            log.error(f"Erreur chunk: {e}")

    df_map = pd.DataFrame(all_rows).drop_duplicates(subset=["gdc_uuid"])
    df_map.to_csv(mapping_file, index=False)
    return df_map

# =========================================================
# STEP 3 — CONCATENATION AVEC DÉTECTION DE FORMAT (LONG/LARGE)
# =========================================================
def concatenate_matrices(directory, glob_pattern="*.csv", source_label=""):
    files = sorted(directory.glob(glob_pattern))
    if not files: return None
    log.info(f"[{source_label}] Traitement de {len(files)} fichiers...")

    all_matrices = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            
            if "biosample_id" in df.columns and "gene_symbol" in df.columns:
                # Format Long (CNV)
                df = df.pivot_table(index="biosample_id", columns="gene_symbol", values="dup_frac", aggfunc="mean")
            else:
                # Format Large (RNA/Meth/CNV Large)
                df = df.set_index(df.columns[0])
                # On transpose si les colonnes contiennent des échantillons (TCGA ou pgx/pgxbs)
                col_str = df.columns.astype(str)
                if col_str.str.contains("TCGA|pgx", case=False).any():
                    df = df.T
            
            # --- CRUCIAL : Nettoyage de l'index des samples ---
            df.index = [clean_tcga_id(str(i)) for i in df.index]
            
            # On ne garde que le numérique et on gère les doublons d'index
            df = df.select_dtypes(include=['number'])
            if df.index.duplicated().any():
                df = df.groupby(df.index).mean()
            
            all_matrices.append(df)
        except Exception as e:
            log.error(f"Erreur sur {f.name}: {e}")

    if not all_matrices: return None
    
    combined = pd.concat(all_matrices, axis=0, sort=False)
    
    if combined.index.duplicated().any():
        combined = combined.groupby(combined.index).mean()
        
    return combined

def convert_ensembl_to_symbol(df):
    if df is None or df.empty: return df
    
    log.info("Conversion Ensembl -> Symbol via MyGene...")
    ensembl_ids = [str(g).split(".")[0] for g in df.columns]
    
    mg = mygene.MyGeneInfo()
    results = mg.querymany(ensembl_ids, scopes="ensembl.gene", fields="symbol", species="human", verbose=False)
    mapping = {r["query"]: r["symbol"] for r in results if "symbol" in r}
    
    df.columns = [mapping.get(str(g).split(".")[0], g) for g in df.columns]
    
    if df.columns.duplicated().any():
        log.info("  Fusion des colonnes gènes dupliquées...")
        df = df.T.groupby(level=0).mean().T
        
    return df

# =========================================================
# STEP 6 — JOIN FINAL AVEC NORMALISATION
# =========================================================
def join_metadata_with_omics(df_meta, df_rnaseq, df_methylation, df_cnv):
    log.info("Jointure finale des données...")
    df = df_meta.copy()
    
    # Normalisation des IDs dans les métadonnées
    df["tcga_barcode"] = df["tcga_barcode"].apply(clean_tcga_id)
    df["biosample_id"] = df["biosample_id"].astype(str).str.strip()

    # Join RNA
    if df_rnaseq is not None:
        log.info("  Merging RNA...")
        df_rnaseq.columns = [f"rna_{c}" if not str(c).startswith("rna_") else c for c in df_rnaseq.columns]
        df = df.merge(df_rnaseq, left_on="tcga_barcode", right_index=True, how="left")

    # Join Methylation
    if df_methylation is not None:
        log.info("  Merging Methylation...")
        df_methylation.columns = [f"meth_{c}" if not str(c).startswith("meth_") else c for c in df_methylation.columns]
        df = df.merge(df_methylation, left_on="tcga_barcode", right_index=True, how="left")

    # Join CNV
    if df_cnv is not None:
        log.info("  Merging CNV...")
        df_cnv_ready = df_cnv.copy()
        df_cnv_ready.index = df_cnv_ready.index.astype(str).str.strip()
        df_cnv_ready.columns = [f"cnv_{c}" if not str(c).startswith("cnv_") else c for c in df_cnv_ready.columns]
        df = df.merge(df_cnv_ready, left_on="biosample_id", right_index=True, how="left")
        
        # Diagnostic immédiat dans le log
        count = df[df.columns[df.columns.str.startswith('cnv_')][0]].notna().sum()
        log.info(f"  CNV joined: {count} samples matched.")

    return df

# =========================================================
# MAIN RUN
# =========================================================
def run():
    log.info("Démarrage du pipeline...")
    df_meta = pd.read_csv(METADATA_FILE, low_memory=False)
    
    # Éviter les conflits de colonnes si "tcga_barcode" existe déjà dans le fichier de métadonnées
    if "tcga_barcode" in df_meta.columns:
        df_meta = df_meta.drop(columns=["tcga_barcode"])

    # Mapping UUIDs
    tcga_uuids = df_meta[df_meta["source_origin"] == "TCGA"]["biosample_name"].dropna().unique().tolist()
    df_map = build_uuid_to_barcode_mapping(tcga_uuids)
    
    # Enrichir Metadata
    df_meta = df_meta.merge(df_map[["gdc_uuid", "tcga_barcode"]], left_on="biosample_name", right_on="gdc_uuid", how="left")
    df_meta.loc[df_meta["source_origin"] == "cBioPortal", "tcga_barcode"] = df_meta["biosample_name"]

    # ── RNA-seq ──────────────────────────────────────────────────
    df_gdc_rna = concatenate_matrices(GDC_RNA_DIR, "*.csv", "GDC RNA")
    if df_gdc_rna is not None: 
        df_gdc_rna = convert_ensembl_to_symbol(df_gdc_rna)
        
    df_cbio_rna = concatenate_matrices(CBIO_RNA_DIR, "*.csv", "cBio RNA")
    df_rnaseq = pd.concat([df for df in [df_gdc_rna, df_cbio_rna] if df is not None], axis=0, sort=False)

    # ── Methylation ──────────────────────────────────────────────
    df_gdc_meth = concatenate_matrices(GDC_METH_DIR, "*.csv", "GDC Meth")
    if df_gdc_meth is not None:
        if any("ENSG" in str(c) for c in df_gdc_meth.columns[:10]):
            df_gdc_meth = convert_ensembl_to_symbol(df_gdc_meth)
            
    df_cbio_meth = concatenate_matrices(CBIO_METH_DIR, "*.csv", "cBio Meth")
    df_meth = pd.concat([df for df in [df_gdc_meth, df_cbio_meth] if df is not None], axis=0, sort=False)
    
    # ── CNV ──────────────────────────────────────────────────────
    df_cnv = concatenate_matrices(CNV_DIR, "**/*.csv", "CNV")
    
    # Jointure finale
    df_final = join_metadata_with_omics(df_meta, df_rnaseq, df_meth, df_cnv)
    
    # 1. Sauvegarde de la table complète
    output = "final_joined_table.parquet"
    log.info(f"Sauvegarde vers {output}...")
    df_final.to_parquet(output, engine='pyarrow', index=False)

    # 2. Filtrage et sauvegarde Gold Standard (Le vrai !)
    log.info("Filtrage du Gold Standard (3 couches)...")
    rna_cols = [c for c in df_final.columns if c.startswith("rna_")]
    meth_cols = [c for c in df_final.columns if c.startswith("meth_")]
    cnv_cols = [c for c in df_final.columns if c.startswith("cnv_")]

    if rna_cols and meth_cols and cnv_cols:
        gold_df = df_final[
            df_final[rna_cols].notna().any(axis=1) & 
            df_final[meth_cols].notna().any(axis=1) & 
            df_final[cnv_cols].notna().any(axis=1)
        ]
        gold_df.to_parquet("gold_standard_only.parquet", engine='pyarrow', index=False)
        log.info(f"Gold Standard sauvegardé : {len(gold_df)} échantillons réels.")
    
    log.info("Pipeline terminé avec succès !")

if __name__ == "__main__":
    run()