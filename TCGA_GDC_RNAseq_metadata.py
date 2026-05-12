import requests
import pandas as pd
import io
import time
from pathlib import Path

# --- CONFIGURATION ---
GDC_API_URL = "https://api.gdc.cancer.gov/files"
IDS_FILE = "gdc_sample_ids.txt"
FIELD_FILE = "gdc_id_field.txt"
#OUTPUT_FILE = "gdc_rnaseq_metadata.csv"


CHUNK_SIZE = 500   # Number of IDs sent per request
PAGE_SIZE = 5000   # Max results returned per request

# Expanded fields to get better biological context
FIELDS = [
    "file_id",
    "file_name",
    "file_size",
    "data_type",
    "analysis.workflow_type",
    "cases.case_id",
    "cases.submitter_id",
    "cases.samples.sample_id",
    "cases.samples.submitter_id", 
    "cases.samples.sample_type",    # Crucial to distinguish Tumor vs Normal
    "cases.project.project_id"
]

def run_gdc_extraction(strategy, data_type, workflow, output_name):
    print(f"\n=== GDC EXTRACTION: {strategy} ===")

    # 1. LOAD INPUT FILES
    if not Path(IDS_FILE).exists():
        print(f"ERROR: {IDS_FILE} not found.")
        return

    # Detect field
    id_field = Path(FIELD_FILE).read_text().strip() if Path(FIELD_FILE).exists() else "cases.samples.sample_id"
    
    # Load IDs
    all_ids = [line.strip() for line in Path(IDS_FILE).read_text().splitlines() 
               if line.strip() and not line.startswith("#")]
    print(f"-> {len(all_ids)} IDs loaded.")

    # 2. EXTRACTION IN CHUNKS
    all_dfs = []
    total_chunks = (len(all_ids) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for i in range(0, len(all_ids), CHUNK_SIZE):
        current_chunk = all_ids[i:i + CHUNK_SIZE]
        chunk_idx = i // CHUNK_SIZE + 1
        print(f"Processing Chunk {chunk_idx}/{total_chunks}...")

        filters = {
            "op": "and",
            "content": [
                {"op": "in", "content": {"field": id_field, "value": current_chunk}},
                {"op": "=", "content": {"field": "experimental_strategy", "value": strategy}},
                {"op": "=", "content": {"field": "data_type", "value": data_type}},
                {"op": "=", "content": {"field": "analysis.workflow_type", "value": workflow}},
                {"op": "=", "content": {"field": "access", "value": "open"}}
            ]
        }

        params = {
            "filters": filters,
            "fields": ",".join(FIELDS),
            "format": "TSV",
            "size": str(PAGE_SIZE)
        }

        try:
            response = requests.post(GDC_API_URL, headers={"Content-Type": "application/json"}, json=params, timeout=60)
            response.raise_for_status()
            data_text = response.content.decode("utf-8").strip()
            
            if len(data_text) > 50: 
                df_chunk = pd.read_csv(io.StringIO(data_text), sep="\t")
                if len(df_chunk) >= PAGE_SIZE:
                    print(f"   ⚠️ WARNING: page size limit ({PAGE_SIZE}) reached — some files may be missing!")
                all_dfs.append(df_chunk)
                print(f"   Successfully retrieved {len(df_chunk)} files.")
            else:
                print(f"   No files found for this chunk.")
        except Exception as e:
            print(f"   ERROR in chunk {chunk_idx}: {e}")

        time.sleep(0.3)

    # 3. CONSOLIDATION AND EXPORT
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True).drop_duplicates(subset=["file_id"])
        #n_found = final_df["cases.submitter_id"].nunique()
        #print(f"Samples couverts : {n_found} / {len(all_ids)}")
        final_df.to_csv(output_name, index=False)
        print(f"DONE! {len(final_df)} unique files saved to '{output_name}'.")
    else:
        print("FINISHED: No data retrieved.")

# --- EXECUTION ---
if __name__ == "__main__":
    # To get RNA-Seq
    run_gdc_extraction(
        strategy="RNA-Seq", 
        data_type="Gene Expression Quantification", 
        workflow="STAR - Counts", 
        output_name="rnaseq_metadata.csv"
    )

    # To get Methylation
    run_gdc_extraction(
        strategy="Methylation Array", 
        data_type="Methylation Beta Value", 
        workflow="SeSAMe Methylation Beta Estimation", 
        output_name="methylation_metadata.csv"
    )