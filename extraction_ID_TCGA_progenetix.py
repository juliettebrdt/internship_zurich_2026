import requests
import pandas as pd
import re
from io import StringIO
from datetime import datetime

# --- CONFIGURATION ---
PROGENETIX_URL = "https://progenetix.org/services/sampletable/?filters=pgx:cohort-TCGAcancers&limit=0"
OUTPUT_TXT = "gdc_sample_ids.txt"
OUTPUT_CSV = "progenetix_metadata.csv"

# Regex to detect UUID format (e.g., d3222a2c-6715-4eea-88b0-4006f61c736b)
UUID_PATTERN = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I
)

def run_extraction():
    print("=== PROGENETIX TCGA ID EXTRACTION ===")
    
    # 1. DOWNLOAD DATA
    try:
        print(f"Step 1: Fetching data from Progenetix...")
        response = requests.get(PROGENETIX_URL, timeout=120)
        response.raise_for_status()
    except Exception as e:
        print(f"ERROR: Could not reach Progenetix. {e}")
        return

    # 2. LOAD INTO DATAFRAME
    df_tcga = pd.read_csv(StringIO(response.text), sep="\t")
    print(f"-> {len(df_tcga)} raw samples retrieved.")

    # 3. DATA CLEANING
    # Remove rows with empty IDs
    df_tcga_clean = df_tcga.dropna(subset=['biosample_name'])
    df = df_tcga_clean[df_tcga_clean['biosample_name'].str.strip() != '']
    
    # 4. ID FORMAT DETECTION (The "Smart" part from Script 1)
    # Check the first 10 samples to determine the format
    sample_values = df['biosample_name'].head(10).astype(str).tolist()
    is_uuid = all(UUID_PATTERN.match(v) for v in sample_values)
    
    # Map to the correct GDC API field
    gdc_field = "cases.samples.sample_id" if is_uuid else "cases.samples.submitter_id"
    
    print(f"-> Detection: {'UUID' if is_uuid else 'Barcode/Submitter ID'} detected.")
    print(f"-> Target GDC Field: {gdc_field}")

    # 5. SAVE UNIQUE IDS TO TEXT FILE
    unique_ids = df['biosample_name'].unique().tolist()
    with open(OUTPUT_TXT, "w") as f:
        # Headers to remember the context later
        f.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Detected Format: {'UUID' if is_uuid else 'Barcode'}\n")
        f.write(f"# Target Field for GDC API: {gdc_field}\n")
        for sample_id in unique_ids:
            f.write(f"{sample_id}\n")
    
    print(f"-> Successfully saved {len(unique_ids)} unique IDs to '{OUTPUT_TXT}'.")

    # 6. SAVE ENRICHED METADATA TO CSV
    # Clean project names (remove 'pgx:' prefix)
    if 'project_id' in df.columns:
        df['project_id'] = df['project_id'].str.replace("pgx:", "", regex=False)
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"-> Metadata saved to '{OUTPUT_CSV}'.")

    # 7. SUMMARY
    print("\n--- Summary ---")
    if 'project_id' in df.columns:
        print("Top 5 Projects found:")
        print(df['project_id'].value_counts().head(5))
    print("\nDone. You can now use the .txt file for your GDC queries.")

if __name__ == "__main__":
    run_extraction()