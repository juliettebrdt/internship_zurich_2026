import requests
import pandas as pd
import re
import time
import logging
from io import StringIO
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("metadata_extraction.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────
OUTPUT_CSV     = "all_progenetix_metadata.csv"
OUTPUT_IDS_TXT = "gdc_sample_ids.txt"

UUID_PATTERN = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I
)

# ── HTTP Session with retry ────────────────────────────────────────
def make_session() -> requests.Session:  
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session

SESSION = make_session()


# =========================================================
# 1. FETCH TCGA SAMPLES (TSV — simple and reliable)
# =========================================================
def fetch_tcga_metadata() -> pd.DataFrame:
    """
    Fetch all TCGA samples from Progenetix via the sampletable service.
    Adds a source_origin column.
    """
    url = "https://progenetix.org/services/sampletable/?filters=pgx:cohort-TCGAcancers&limit=0"
    log.info("Fetching TCGA metadata from Progenetix...")

    resp = SESSION.get(url, timeout=300)
    resp.raise_for_status()

    df = pd.read_csv(StringIO(resp.text), sep="\t")
    df["source_origin"] = "TCGA"

    # Clean project_id prefix
    if "project_id" in df.columns:
        df["project_id"] = df["project_id"].str.replace("pgx:", "", regex=False)

    log.info(f"TCGA : {len(df)} samples retrieved.")
    return df


# =========================================================
# 2. FETCH CBIOPORTAL SAMPLES (study by study)
# =========================================================
def get_cbioportal_study_ids() -> list[str]:
    """
    Get all cBioPortal study IDs referenced in Progenetix.
    """
    r = SESSION.get(
        "https://progenetix.org/services/collations",
        params={"collationTypes": "cbioportal"},
        timeout=30
    )
    r.raise_for_status()
    results = r.json().get("response", {}).get("results", [])

    studies = [
        entry["id"].replace("cbioportal:", "")
        for entry in results
        if "cbioportal:" in entry.get("id", "")
    ]
    log.info(f"Found {len(studies)} cBioPortal studies in Progenetix.")
    return sorted(studies)


def fetch_cbioportal_metadata() -> pd.DataFrame:
    """
    Fetch metadata for all cBioPortal studies from Progenetix,
    one study at a time using the sampletable service.
    """
    studies = get_cbioportal_study_ids()
    all_dfs = []

    for i, study_id in enumerate(studies, 1):
        log.info(f"  [{i}/{len(studies)}] Fetching : {study_id}")

        url = (
            f"https://progenetix.org/services/sampletable/"
            f"?filters=cbioportal:{study_id}&limit=0"
        )
        try:
            resp = SESSION.get(url, timeout=120)
            resp.raise_for_status()

            df = pd.read_csv(StringIO(resp.text), sep="\t")

            if df.empty:
                log.info(f"    No samples for {study_id}.")
                continue

            df["source_origin"] = "cBioPortal"
            df["cbioportal_study"] = study_id
            all_dfs.append(df)
            log.info(f"    {len(df)} samples.")

        except Exception as e:
            log.error(f"    Failed for {study_id}: {e}")

        time.sleep(0.2)

    if not all_dfs:
        log.warning("No cBioPortal data retrieved.")
        return pd.DataFrame()

    df_cbio = pd.concat(all_dfs, ignore_index=True)
    log.info(f"cBioPortal total : {len(df_cbio)} samples.")
    return df_cbio


# =========================================================
# 3. REMOVE CELL LINES
# =========================================================
def remove_cell_lines(df: pd.DataFrame) -> pd.DataFrame:
    col = "biosample_status_label"
    if col not in df.columns:
        log.warning("Column 'biosample_status_label' not found — skipping cell line filter.")
        return df

    before = len(df)
    df = df[~df[col].str.contains("cell line", case=False, na=False)].copy()
    log.info(f"Cell lines removed : {before - len(df)} samples.")
    return df


# =========================================================
# 4. DETECT ID FORMAT FOR GDC
# =========================================================
def detect_id_format(df: pd.DataFrame) -> str:
    """
    Detect whether biosample_name contains UUIDs or TCGA barcodes.
    Returns the correct GDC API field name.
    """
    sample_values = df["biosample_name"].dropna().head(10).astype(str).tolist()
    is_uuid = all(UUID_PATTERN.match(v) for v in sample_values)
    gdc_field = "cases.samples.sample_id" if is_uuid else "cases.samples.submitter_id"

    log.info(f"ID format detected : {'UUID' if is_uuid else 'Barcode'}")
    log.info(f"GDC field : {gdc_field}")
    return gdc_field


# =========================================================
# 5. MAIN PIPELINE
# =========================================================
def run_extraction():
    log.info("=== PROGENETIX METADATA EXTRACTION (TCGA + cBioPortal) ===")

    # ── Step 1 : Fetch both sources ───────────────────────────────
    df_tcga  = fetch_tcga_metadata()
    df_cbio  = fetch_cbioportal_metadata()

    # ── Step 2 : Combine ──────────────────────────────────────────
    all_dfs = [df for df in [df_tcga, df_cbio] if not df.empty]
    if not all_dfs:
        log.error("No data retrieved from any source.")
        return

    df = pd.concat(all_dfs, ignore_index=True)
    log.info(f"Total before deduplication : {len(df)} samples.")

    # Deduplicate on biosample_id
    if "biosample_id" in df.columns:
        df = df.drop_duplicates(subset=["biosample_id"], keep="first")
        log.info(f"Total after deduplication : {len(df)} samples.")

    # ── Step 3 : Remove cell lines ────────────────────────────────
    df = remove_cell_lines(df)

    # ── Step 4 : Save combined metadata ───────────────────────────
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    log.info(f"Metadata saved : {OUTPUT_CSV}")

    # ── Step 5 : Save TCGA IDs for GDC queries ───────────────────
    df_tcga_clean = df[df["source_origin"] == "TCGA"].copy()
    if not df_tcga_clean.empty and "biosample_name" in df_tcga_clean.columns:
        gdc_field = detect_id_format(df_tcga_clean)
        unique_ids = df_tcga_clean["biosample_name"].dropna().unique().tolist()

        with open(OUTPUT_IDS_TXT, "w") as f:
            f.write(f"# Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# GDC field : {gdc_field}\n")
            f.write(f"# Total IDs : {len(unique_ids)}\n")
            for id_ in unique_ids:
                f.write(f"{id_}\n")
        log.info(f"GDC IDs saved : {OUTPUT_IDS_TXT} ({len(unique_ids)} IDs)")

    # ── Step 6 : Summary ──────────────────────────────────────────
    print("\n" + "="*55)
    print("SUMMARY")
    print("="*55)
    print(f"  Total samples          : {len(df)}")
    print(f"\n  By source :")
    print(df["source_origin"].value_counts().to_string())
    if "biosample_status_label" in df.columns:
        print(f"\n  By biosample status :")
        print(df["biosample_status_label"].value_counts().to_string())
    if "project_id" in df.columns:
        print(f"\n  Top 10 projects :")
        print(df["project_id"].value_counts().head(10).to_string())
    print("="*55)


if __name__ == "__main__":
    run_extraction()