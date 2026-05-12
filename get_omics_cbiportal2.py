import requests
import pandas as pd
import io
import time
import logging
from pathlib import Path

# ── Logging ───────────────────────────────────────────────────────
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
API_DELAY      = 0.2
API_CHUNK_SIZE = 500

# =========================================================
# STEP 2 — IDENTIFICATION
# =========================================================

def get_molecular_profiles(study_id: str) -> dict:
    url = f"{CBIO_API_BASE}/studies/{study_id}/molecular-profiles"
    res = {"rnaseq": [], "methylation": []}
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200: return res
        profiles = r.json()
    except: return res

    for p in profiles:
        p_id = p.get("molecularProfileId", "").lower()
        alt_type = str(p.get("molecularAlterationType", "")).upper()
        datatype = str(p.get("datatype", "")).upper()
        is_z = (datatype == "Z-SCORE")

        if "MRNA_EXPRESSION" in alt_type or any(kw in p_id for kw in ["rna_seq", "mrna", "transcriptome"]):
            res["rnaseq"].append(p)
        elif "METHYLATION" in alt_type or "methylation" in p_id:
            res["methylation"].append(p)
    return res

# =========================================================
# STEP 3a — DATAHUB
# =========================================================

def download_via_datahub(study_id: str, profile_id: str, save_path: Path) -> bool:
    suffix = profile_id.replace(study_id + "_", "", 1)
    base = f"https://github.com/cBioPortal/datahub/raw/master/public/{study_id}"
    candidates = [f"data_{suffix}.txt", f"data_{suffix}_mrna.txt", f"{suffix}.txt"]
    
    try:
        r_meta = requests.get(f"{base}/meta_{suffix}.txt", timeout=10)
        if r_meta.status_code == 200:
            for line in r_meta.text.split('\n'):
                if 'data_filename:' in line:
                    candidates.insert(0, line.split(':')[-1].strip())
    except: pass

    for filename in list(dict.fromkeys(candidates)):
        try:
            r = requests.get(f"{base}/{filename}", timeout=60)
            if r.status_code == 200 and not r.text.startswith("version https://git-lfs"):
                df = pd.read_csv(io.StringIO(r.text), sep="\t", low_memory=False)
                if not df.empty:
                    df.rename(columns={df.columns[0]: "gene_symbol"}, inplace=True)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(save_path, index=False)
                    log.info(f"    [DataHub] Success: {filename}")
                    return True
        except: continue
    return False

# =========================================================
# STEP 3b — API
# =========================================================

def download_via_api(study_id: str, profile_id: str, save_path: Path) -> bool:
    try:
        r_s = requests.get(f"{CBIO_API_BASE}/studies/{study_id}/samples", params={"pageSize": 10000}, timeout=30)
        s_ids = [s["sampleId"] for s in r_s.json()]
    except: return False

    chunks = []
    # On tente le POST sans gènes (Mode "Fetch All")
    for i in range(0, len(s_ids), API_CHUNK_SIZE):
        body = {"sampleIds": s_ids[i:i + API_CHUNK_SIZE]}
        try:
            r = requests.post(f"{CBIO_API_BASE}/molecular-profiles/{profile_id}/molecular-data/fetch", json=body, timeout=120)
            if r.status_code == 200 and r.json():
                c_df = pd.DataFrame(r.json())
                if 'gene' in c_df.columns:
                    c_df['gene_symbol'] = c_df['gene'].apply(lambda x: x.get('hugoGeneSymbol', '') if isinstance(x, dict) else '')
                else:
                    c_df['gene_symbol'] = c_df.get('entrezGeneId', 'unknown')
                chunks.append(c_df[['sampleId', 'gene_symbol', 'value']])
            time.sleep(API_DELAY)
        except: continue

    if not chunks: return False
    try:
        final_df = pd.concat(chunks).pivot_table(index="gene_symbol", columns="sampleId", values="value", aggfunc="mean").reset_index()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(save_path, index=False)
        log.info(f"    [API ✓] Success")
        return True
    except: return False

# =========================================================
# MAIN
# =========================================================

def run():
    log.info("=== STARTING PIPELINE v3.5 (FINAL) ===")
    
    # Nettoyer le dossier avant de commencer pour éviter les "Skipping"
    import shutil
    if OUTPUT_DIR.exists():
        log.info("Cleaning old data directory...")
        shutil.rmtree(OUTPUT_DIR)
    
    try:
        res = requests.get(PROGENETIX_URL).json()
        study_ids = [s["id"].split(":")[-1].strip() for s in res["response"]["results"] if "cbioportal:" in s["id"]]
    except Exception as e:
        log.error(f"Failed to fetch study list: {e}")
        return

    stats = {"rnaseq": 0, "methylation": 0, "from_hub": 0, "from_api": 0}

    for idx, study in enumerate(study_ids, 1):
        profs = get_molecular_profiles(study)
        
        total = len(profs["rnaseq"]) + len(profs["methylation"])
        if total == 0: continue

        log.info(f"[{idx}/{len(study_ids)}] {study} (Found {len(profs['rnaseq'])} RNA, {len(profs['methylation'])} Meth)")

        for dtype in ["rnaseq", "methylation"]:
            for p in profs[dtype]:
                p_id = p["molecularProfileId"]
                is_z = p.get("datatype", "").upper() == "Z-SCORE"
                save_path = OUTPUT_DIR / dtype / f"{p_id}{'_zscore' if is_z else ''}.csv"

                if download_via_datahub(study, p_id, save_path):
                    stats[dtype] += 1; stats["from_hub"] += 1
                elif download_via_api(study, p_id, save_path):
                    stats[dtype] += 1; stats["from_api"] += 1
        
    log.info(f"RESULT -> RNA: {stats['rnaseq']} | Meth: {stats['methylation']} | Hub: {stats['from_hub']} | API: {stats['from_api']}")

if __name__ == "__main__":
    run()