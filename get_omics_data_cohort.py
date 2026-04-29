"""
download_by_cohort.py
─────────────────────────────────────────────────────────────
Downloads STAR - Counts RNA-seq data from GDC for all TCGA cohorts.
Builds one gene × samples matrix per cohort.

Features:
  - Automatic pagination → retrieves ALL samples per cohort, no limit
  - Skips already-downloaded cohorts (resume support)
  - Filters STAR metadata rows (N_unmapped, N_multimapping, etc.)
  - Efficient matrix build with pd.concat (not iterative merge)
  - Detailed progress log (pipeline.log)

Usage:
  python download_by_cohort.py

Background (survives screen lock on Mac):
  nohup python download_by_cohort.py > pipeline.log 2>&1 &
  tail -f pipeline.log
"""

import requests
import pandas as pd
import io
import time
import logging
from pathlib import Path

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────
GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT  = "https://api.gdc.cancer.gov/data"
OUTPUT_DIR         = Path("cohort_matrices")
PAGE_SIZE          = 500   # files per pagination request
SLEEP_BETWEEN      = 0.2   # seconds between file downloads


# =========================================================
# 1. GET TCGA COHORTS FROM PROGENETIX
# =========================================================
def get_progenetix_cohorts() -> list[str]:
    """
    Retrieve all TCGA cohort IDs from Progenetix.
    Returns a sorted list of project IDs like ['TCGA-ACC', 'TCGA-BLCA', ...]
    """
    url = "https://progenetix.org/services/collations?collationTypes=TCGAproject"
    log.info("Fetching TCGA cohort list from Progenetix...")

    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()

    results = data.get("response", {}).get("results", [])

    # Strip the "pgx:" prefix → "pgx:TCGA-ACC" becomes "TCGA-ACC"
    cohorts = sorted([
        entry["id"].replace("pgx:", "")
        for entry in results
        if "TCGA-" in entry.get("id", "")
    ])

    log.info(f"Found {len(cohorts)} TCGA cohorts: {cohorts}")
    return cohorts


# =========================================================
# 2. GET ALL FILE IDs FOR A COHORT (WITH PAGINATION)
# =========================================================
def get_all_files_for_cohort(project_id: str) -> list[dict]:
    """
    Retrieve ALL STAR - Counts file IDs for a given TCGA project.
    Uses pagination to go beyond GDC's single-request limit.
    Returns a list of dicts with file_id and sample_id.
    """
    log.info(f"  Fetching file list for {project_id}...")

    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": [project_id]
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "data_type",          # ← no "files." prefix
                    "value": "Gene Expression Quantification"
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "analysis.workflow_type",
                    "value": "STAR - Counts"
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "access",             # ← no "files." prefix
                    "value": "open"
                }
            }
        ]
    }

    all_files = []
    from_index = 0          # pagination cursor

    while True:
        params = {
            "filters": filters,
            "fields": "file_id,cases.samples.sample_id,cases.samples.submitter_id",
            "format": "JSON",
            "size": PAGE_SIZE,
            "from": from_index
        }

        try:
            r = requests.post(
                GDC_FILES_ENDPOINT,
                headers={"Content-Type": "application/json"},
                json=params,
                timeout=60
            )
            r.raise_for_status()

            hits      = r.json()["data"]["hits"]
            paginated = r.json()["data"]["pagination"]
            total     = paginated["total"]

            for h in hits:
                if "cases" not in h or not h["cases"]:
                    continue
                samples = h["cases"][0].get("samples", [])
                if not samples:
                    continue
                all_files.append({
                    "file_id":   h["file_id"],
                    "sample_id": samples[0].get("submitter_id")
                              or samples[0].get("sample_id", h["file_id"])
                })

            from_index += len(hits)
            log.info(f"    {from_index} / {total} files retrieved...")

            # Stop when all pages have been fetched
            if from_index >= total:
                break

        except Exception as e:
            log.error(f"  Pagination error at offset {from_index}: {e}")
            break

    log.info(f"  Total files found for {project_id}: {len(all_files)}")
    return all_files


# =========================================================
# 3. DOWNLOAD ONE FILE AND PARSE IT
# =========================================================
def download_and_parse(file_id: str, sample_id: str) -> pd.DataFrame | None:
    """
    Download one STAR - Counts file from GDC and return it as a
    two-column DataFrame: gene_id | sample_id.
    Filters out STAR summary rows (N_unmapped, N_multimapping, etc.).
    """
    try:
        r = requests.post(
            GDC_DATA_ENDPOINT,
            json={"ids": [file_id]},
            timeout=120
        )
        r.raise_for_status()

        content = r.content.decode("utf-8")
        df = pd.read_csv(io.StringIO(content), sep="\t", comment="#")

        # Keep only the first two columns: gene_id + unstranded counts
        df = df.iloc[:, :2].copy()
        df.columns = ["gene_id", sample_id]

        # Remove STAR summary rows (not actual genes)
        df = df[~df["gene_id"].str.startswith("N_")]

        return df

    except Exception as e:
        log.warning(f"    Failed to download {file_id}: {e}")
        return None


# =========================================================
# 4. BUILD MATRIX FOR ONE COHORT
# =========================================================
def build_cohort_matrix(project_id: str, file_list: list[dict]) -> None:
    """
    Download all files for a cohort and concatenate them into
    a single gene × samples matrix saved as CSV.
    """
    output_file = OUTPUT_DIR / f"{project_id}_STAR_counts_matrix.csv"

    # Resume support: skip if already done
    if output_file.exists():
        log.info(f"  ⏭️  {project_id} already downloaded — skipping.")
        return

    log.info(f"  Downloading {len(file_list)} files for {project_id}...")

    sample_series = []

    for i, item in enumerate(file_list, 1):
        log.info(f"    [{i}/{len(file_list)}] {item['sample_id']}")

        df = download_and_parse(item["file_id"], item["sample_id"])

        if df is not None:
            # Set gene_id as index for efficient concat later
            df = df.set_index("gene_id")
            sample_series.append(df)

        time.sleep(SLEEP_BETWEEN)

    if not sample_series:
        log.warning(f"  No data retrieved for {project_id}.")
        return

    # ── Build matrix efficiently with a single pd.concat ──────────
    log.info(f"  Building matrix ({len(sample_series)} samples)...")
    matrix = pd.concat(sample_series, axis=1)   # genes × samples
    matrix = matrix.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

    matrix.to_csv(output_file)
    log.info(f"  ✅ Saved: {output_file}  "
             f"({matrix.shape[0]} genes × {matrix.shape[1]} samples)")


# =========================================================
# 5. FULL PIPELINE
# =========================================================
def run():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Step 1 — get cohort list from Progenetix
    cohorts = get_progenetix_cohorts()

    if not cohorts:
        log.error("No TCGA cohorts found. Check the Progenetix endpoint.")
        return

    log.info(f"\nPipeline starting — {len(cohorts)} cohorts to process.")

    for idx, cohort in enumerate(cohorts, 1):
        log.info(f"\n{'='*55}")
        log.info(f"COHORT {idx}/{len(cohorts)}: {cohort}")
        log.info(f"{'='*55}")

        # Step 2 — get ALL file IDs with pagination
        file_list = get_all_files_for_cohort(cohort)

        if not file_list:
            log.warning(f"  No files found for {cohort} — skipping.")
            continue

        # Step 3 — download + build matrix
        build_cohort_matrix(cohort, file_list)

    log.info("\nPipeline complete.")
    log.info(f"Matrices saved in: {OUTPUT_DIR}/")


# =========================================================
# 6. ENTRY POINT
# =========================================================
if __name__ == "__main__":
    run()