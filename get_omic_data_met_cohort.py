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
        logging.FileHandler("methylation_pipeline.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────
GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT  = "https://api.gdc.cancer.gov/data"
OUTPUT_DIR         = Path("cohort_methylation_matrices")
PAGE_SIZE          = 500   # files per pagination request
SLEEP_BETWEEN      = 0.3   # seconds between file downloads (slightly more than RNA-seq)
                            # methylation files are larger (~500KB vs ~1MB uncompressed)

# Methylation-specific GDC parameters
EXPERIMENTAL_STRATEGY = "Methylation Array"
DATA_TYPE             = "Methylation Beta Value"
WORKFLOW_TYPE         = "SeSAMe Methylation Beta Estimation"


#------ GET TCGA COHORTS FROM PROGENETIX ---------

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


# ----- GET ALL FILE IDs FOR A COHORT (WITH PAGINATION) ------

def get_all_files_for_cohort(project_id: str) -> list[dict]:
    """
    Retrieve ALL methylation file IDs for a given TCGA project.
    Uses pagination to go beyond GDC's single-request limit.
    Returns a list of dicts with file_id and sample_id.
    """
    log.info(f"  Fetching methylation file list for {project_id}...")

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
                    "field": "experimental_strategy",
                    "value": EXPERIMENTAL_STRATEGY      # "Methylation Array"
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "data_type",
                    "value": DATA_TYPE                  # "Methylation Beta Value"
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "analysis.workflow_type",
                    "value": WORKFLOW_TYPE              # "SeSAMe Methylation Beta Estimation"
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "access",
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

    log.info(f"  Total methylation files found for {project_id}: {len(all_files)}")
    return all_files



# ----- DOWNLOAD ONE FILE AND PARSE IT ------

def download_and_parse(file_id: str, sample_id: str) -> pd.DataFrame | None:
    """
    Download one methylation beta value file from GDC and return it as a
    two-column DataFrame: composite_element_ref | sample_id.

    GDC methylation file format (TSV):
      Column 1 : composite_element_ref  → CpG probe ID (e.g. cg00000029)
      Column 2 : beta_value             → float between 0 and 1
                                          (0 = fully unmethylated,
                                           1 = fully methylated)
      Column 3+: additional annotations (chromosome, position, etc.) — ignored

    Unlike RNA-seq, there are no N_ summary rows to filter.
    Beta values are floats, not integers.
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

        # Keep only CpG probe ID + beta value columns
        df = df.iloc[:, :2].copy()
        df.columns = ["cpg_id", sample_id]

        # Remove rows with missing probe IDs
        df = df.dropna(subset=["cpg_id"])

        # Beta values must be numeric floats in [0, 1]
        df[sample_id] = pd.to_numeric(df[sample_id], errors="coerce")

        return df

    except Exception as e:
        log.warning(f"    Failed to download {file_id}: {e}")
        return None



# ----- BUILD MATRIX FOR ONE COHORT --------

def build_cohort_matrix(project_id: str, file_list: list[dict]) -> None:
    """
    Download all methylation files for a cohort and concatenate them into
    a single CpG sites × samples matrix saved as CSV.
    """
    output_file = OUTPUT_DIR / f"{project_id}_methylation_beta_matrix.csv"

    # Resume support: skip if already done
    if output_file.exists():
        log.info(f"  ⏭️  {project_id} already downloaded — skipping.")
        return

    log.info(f"  Downloading {len(file_list)} methylation files for {project_id}...")

    sample_series = []

    for i, item in enumerate(file_list, 1):
        log.info(f"    [{i}/{len(file_list)}] {item['sample_id']}")

        df = download_and_parse(item["file_id"], item["sample_id"])

        if df is not None:
            # Set cpg_id as index for efficient concat later
            df = df.set_index("cpg_id")
            sample_series.append(df)

        time.sleep(SLEEP_BETWEEN)

    if not sample_series:
        log.warning(f"  No data retrieved for {project_id}.")
        return

    # ── Build matrix efficiently with a single pd.concat ──────────
    log.info(f"  Building matrix ({len(sample_series)} samples)...")
    matrix = pd.concat(sample_series, axis=1)   # CpG sites × samples

    # Beta values stay as floats — do NOT cast to int
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    # NaN = probe not measured in that sample — keep as NaN (don't fill with 0
    # as 0 would incorrectly mean "fully unmethylated")
    matrix.index.name = "cpg_id"

    matrix.to_csv(output_file)
    log.info(f"  ✅ Saved: {output_file}  "
             f"({matrix.shape[0]} CpG probes × {matrix.shape[1]} samples)")



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
            log.warning(f"  No methylation files found for {cohort} — skipping.")
            continue

        # Step 3 — download + build matrix
        build_cohort_matrix(cohort, file_list)

    log.info("\nPipeline complete.")
    log.info(f"Matrices saved in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    run()