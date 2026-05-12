"""
extract_bridge_from_biosample_summary.py
─────────────────────────────────────────
Extrait les colonnes analysis_id + biosample_id depuis biosample_summary.csv
et génère cnv_id_bridge.csv utilisé par join_omics_data.py.

Usage :
    python extract_bridge_from_biosample_summary.py

Entrée  : biosample_summary.csv  (chemin configurable ci-dessous)
Sortie  : cnv_id_bridge.csv
"""

import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Config — adapte le chemin si besoin ───────────────────────────
BIOSAMPLE_SUMMARY = Path("/Users/bgadmin/Downloads/biosample_summary.csv")
OUTPUT_BRIDGE     = Path("cnv_id_bridge.csv")


def run():
    if not BIOSAMPLE_SUMMARY.exists():
        log.error(f"File not found: {BIOSAMPLE_SUMMARY}")
        return

    log.info(f"Reading {BIOSAMPLE_SUMMARY}...")

    # On lit uniquement les deux colonnes utiles — rapide même sur un gros fichier
    available = pd.read_csv(BIOSAMPLE_SUMMARY, nrows=0).columns.tolist()
    log.info(f"Available columns: {available}")

    for col in ["analysis_id", "biosample_id"]:
        if col not in available:
            log.error(
                f"Column '{col}' not found in biosample_summary.csv. "
                f"Available: {available}"
            )
            return

    df = pd.read_csv(
        BIOSAMPLE_SUMMARY,
        usecols=["analysis_id", "biosample_id"],
        dtype=str,
        low_memory=False
    )
    log.info(f"Loaded {len(df):,} rows.")

    # Nettoyage
    df = df.dropna(subset=["analysis_id", "biosample_id"])
    df["analysis_id"]  = df["analysis_id"].str.strip()
    df["biosample_id"] = df["biosample_id"].str.strip()
    df = df.drop_duplicates(subset=["analysis_id"])

    log.info(f"After cleaning: {len(df):,} unique analysis_id entries.")

    # Sanity check
    n_pgxcs = df["analysis_id"].str.startswith("pgxcs-").sum()
    n_pgxbs = df["biosample_id"].str.startswith("pgxbs-").sum()
    log.info(f"  pgxcs- analysis_ids : {n_pgxcs:,}")
    log.info(f"  pgxbs- biosample_ids: {n_pgxbs:,}")

    if n_pgxcs == 0:
        log.warning("No pgxcs- IDs found in analysis_id — check column content.")
    if n_pgxbs == 0:
        log.warning("No pgxbs- IDs found in biosample_id — check column content.")

    df.to_csv(OUTPUT_BRIDGE, index=False)
    log.info(f"\n✅ Bridge saved: {OUTPUT_BRIDGE}  ({len(df):,} pairs)")
    log.info("You can now run: python join_omics_data.py")


if __name__ == "__main__":
    run()