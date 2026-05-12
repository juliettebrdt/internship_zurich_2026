import requests
import pandas as pd
import io
import time
from pathlib import Path

# --- CONFIGURATION ---
GDC_API_URL = "https://api.gdc.cancer.gov/files"
IDS_FILE = "gdc_sample_ids.txt"
FIELD_FILE = "gdc_id_field.txt"
OUTPUT_FILE = "gdc_methylation_metadata.csv"

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

def run_gdc_extraction():
    print("=== GDC METHYLATION METADATA EXTRACTION ===")

    # 1. LOAD INPUT FILES
    if not Path(IDS_FILE).exists():
        print(f"ERROR: {IDS_FILE} not found. Please run the Progenetix script first.")
        return

    # Automatically detect which GDC field to use (from the mapping file)
    if Path(FIELD_FILE).exists():
        id_field = Path(FIELD_FILE).read_text().strip()
        print(f"-> Field mapping loaded: {id_field}")
    else:
        id_field = "cases.samples.sample_id" # Default fallback
        print(f"-> WARNING: {FIELD_FILE} not found, using default: {id_field}")

    # Load all IDs from the text file (ignoring comments)
    all_ids = [
        line.strip() for line in Path(IDS_FILE).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    print(f"-> {len(all_ids)} IDs loaded for processing.")

    # 2. EXTRACTION IN CHUNKS
    all_dfs = []
    total_chunks = (len(all_ids) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for i in range(0, len(all_ids), CHUNK_SIZE):
        current_chunk = all_ids[i:i + CHUNK_SIZE]
        chunk_idx = i // CHUNK_SIZE + 1
        print(f"Processing Chunk {chunk_idx}/{total_chunks}...")

        # Constructing the filters
        filters = {
            "op": "and",
            "content": [
                {"op": "in", "content": {"field": id_field, "value": current_chunk}},
                {"op": "=", "content": {"field": "files.experimental_strategy", "value": "Methylation"}},
                {"op": "=", "content": {"field": "files.data_type", "value": "Methylation Beta Value"}},
                {"op": "=", "content": {"field": "files.access", "value": "open"}},
                {"op": "=", "content": {"field": "files.analysis.workflow_type", "value": "STAR - Counts"}} #to have the standard HTseq-compatibleraw coumts. 
    
            ]
        }

        params = {
            "filters": filters,
            "fields": ",".join(FIELDS),
            "format": "TSV",
            "size": str(PAGE_SIZE)
        }

        try:
            response = requests.post(
                GDC_API_URL,
                headers={"Content-Type": "application/json"},
                json=params,
                timeout=60
            )
            response.raise_for_status()

            # Parse TSV response
            data_text = response.content.decode("utf-8").strip()
            if len(data_text) > 50: # Check if there is actual data beyond the header
                df_chunk = pd.read_csv(io.StringIO(data_text), sep="\t")
                
                # Check if we hit the API limit
                if len(df_chunk) >= PAGE_SIZE:
                    print(f"   ! WARNING: Chunk {chunk_idx} hit the {PAGE_SIZE} limit. Some files might be missing.")

                all_dfs.append(df_chunk)
                print(f"   Successfully retrieved {len(df_chunk)} files.")
            else:
                print(f"   No RNA-Seq files found for this chunk.")

        except Exception as e:
            print(f"   ERROR in chunk {chunk_idx}: {e}")

        # Sleep briefly to be polite to the API
        time.sleep(0.3)

    # 3. CONSOLIDATION AND EXPORT
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        # Remove duplicates (sometimes one sample has multiple analysis versions)
        final_df = final_df.drop_duplicates(subset=["file_id"])
        
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nDONE! {len(final_df)} unique RNA-Seq files saved to '{OUTPUT_FILE}'.")
        
        # Display small summary
        if "cases.project.project_id" in final_df.columns:
            print("\nTop 5 Projects in results:")
            print(final_df["cases.project.project_id"].value_counts().head(5))
    else:
        print("\nFINISHED: No data was retrieved. Check your input IDs and GDC field.")

if __name__ == "__main__":
    run_gdc_extraction()