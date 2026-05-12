import requests
import pandas as pd
import io
import time
import logging
from pathlib import Path

# ── Logging Configuration ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("cbioportal_download_report.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────
PROGENETIX_URL = "https://progenetix.org/services/collations?collationTypes=cbioportal"
CBIO_API_BASE  = "https://www.cbioportal.org/api"
OUTPUT_DIR     = Path("cbioportal_data")

# 1. FETCH STUDY LIST
def get_cbioportal_study_ids():
    """Retrieves the list of cBioPortal study IDs from the Progenetix service."""
    log.info("Fetching study IDs from Progenetix...")
    try:
        r = requests.get(PROGENETIX_URL, timeout=30)
        r.raise_for_status()
        data = r.json()
        results = data.get("response", {}).get("results", [])
        
        study_ids = []
        for entry in results:
            raw_id = entry.get("id", "")
            if "cbioportal:" in raw_id:
                # Clean prefix 'cbioportal:acc_2019' -> 'acc_2019'
                study_ids.append(raw_id.split(":")[-1].strip())
                
        log.info(f"✅ Successfully retrieved {len(study_ids)} studies.")
        return study_ids
    except Exception as e:
        log.error(f"Failed to fetch studies from Progenetix: {e}")
        return []

# 2. IDENTIFY MOLECULAR PROFILES
def get_molecular_profiles(study_id):
    """
    Scans the cBioPortal API for all available RNA-seq and Methylation profiles 
    for a specific study. Returns two lists of Profile IDs.
    """
    url = f"{CBIO_API_BASE}/studies/{study_id}/molecular-profiles"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return [], []
        
        profiles = r.json()
        rna_profiles = []
        meth_profiles = []
        
        for p in profiles:
            p_id = p['molecularProfileId'].lower()
            
            # Identify RNA-seq profiles (excluding Z-scores to get raw/normalized counts)
            if ("rna_seq" in p_id or "mrna" in p_id) and "zscores" not in p_id:
                rna_profiles.append(p['molecularProfileId'])
                
            # Identify Methylation profiles
            if "methylation" in p_id:
                meth_profiles.append(p['molecularProfileId'])
                
        return rna_profiles, meth_profiles
    except Exception as e:
        log.debug(f"Error fetching profiles for {study_id}: {e}")
        return [], []

# 3. DOWNLOAD AND CONVERT DATA
def download_matrix(study_id, profile_id, dtype):
    """
    Downloads the matrix from cBioPortal's GitHub DataHub.
    Handles Git LFS redirection and provides Resume Support.
    """
    # Create subfolders (rnaseq/ or methylation/)
    save_path = OUTPUT_DIR / dtype / f"{profile_id}.csv"
    
    # --- RESUME SUPPORT ---
    if save_path.exists():
        log.info(f"  ⏭️  Skipping: {profile_id} (Already downloaded)")
        return True

    # Map profile ID to DataHub filename
    # e.g., 'acc_2019_rna_seq_v2_mrna' -> 'data_rna_seq_v2_mrna.txt'
    suffix = profile_id.replace(study_id + "_", "")
    filename = f"data_{suffix}.txt"
    
    # Use the /raw/ URL to force GitHub to resolve Git LFS pointers
    url = f"https://github.com/cBioPortal/datahub/raw/master/public/{study_id}/{filename}"
    
    try:
        r = requests.get(url, timeout=180, allow_redirects=True)
        
        if r.status_code == 200:
            # SAFETY CHECK: Ensure we didn't download a 3-line LFS pointer file
            if r.text.startswith("version https://git-lfs.github.com"):
                log.error(f"  ❌ LFS Error: Could not resolve actual data for {profile_id}")
                return False

            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Read tab-separated data and save as CSV
            df = pd.read_csv(io.StringIO(r.text), sep="\t", low_memory=False)
            df.to_csv(save_path, index=False)
            log.info(f"  ✅ Saved: {profile_id} ({len(df)} genes found)")
            return True
        else:
            log.debug(f"  ⚠️  File {filename} not found on DataHub.")
            return False
            
    except Exception as e:
        log.error(f"  ❌ Error processing {profile_id}: {e}")
        return False

# 4. MAIN PIPELINE
def run():
    """Executes the full pipeline: Fetch -> Detect -> Download."""
    log.info("Starting cBioPortal Multimodal Pipeline...")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    study_ids = get_cbioportal_study_ids()
    if not study_ids:
        log.error("No studies to process. Exiting.")
        return

    for idx, study in enumerate(study_ids, 1):
        log.info(f"[{idx}/{len(study_ids)}] Processing Study: {study}")
        
        rna_profs, meth_profs = get_molecular_profiles(study)
        
        # Download all RNA profiles found
        for r_prof in rna_profs:
            download_matrix(study, r_prof, "rnaseq")
            
        # Download all Methylation profiles found
        for m_prof in meth_profs:
            download_matrix(study, m_prof, "methylation")
        
        # Polite delay for the API
        time.sleep(0.2) 

    log.info("Pipeline Complete. Data stored in: " + str(OUTPUT_DIR))

if __name__ == "__main__":
    run()