from pathlib import Path
import pandas as pd
import numpy as np

# ── Chemins ──────────────────────────────────────────────
golden_path        = Path("gold_standard_20260602_1455.parquet")
biosample_summary_path = Path("/Users/bgadmin/Downloads/biosample_summary.csv")

# ── 1. Charger le golden set ──────────────────────────────
print("Loading golden set...")
gold = pd.read_parquet(golden_path)
print(f"Shape : {gold.shape}")

# Colonnes CNV par fraction
dup_cols   = [c for c in gold.columns if c.endswith("__dup")]
del_cols   = [c for c in gold.columns if c.endswith("__del")]
hldup_cols = [c for c in gold.columns if c.endswith("__hldup")]
hldel_cols = [c for c in gold.columns if c.endswith("__hldel")]
print(f"Gènes × fractions : {len(dup_cols)} dup | {len(del_cols)} del | "
      f"{len(hldup_cols)} hldup | {len(hldel_cols)} hldel")

# ── 2. Extraire uniquement biosample_id + colonnes CNV ────
cnv_cols = dup_cols + del_cols + hldup_cols + hldel_cols
df_golden_cnv = gold[["biosample_id"] + cnv_cols].copy()
print(f"\ndf_golden_cnv shape : {df_golden_cnv.shape}")

# ── 3. Vérification rapide ────────────────────────────────
vals_dup = df_golden_cnv[dup_cols].values.flatten()
vals_del = df_golden_cnv[del_cols].values.flatten()
print(f"\n% dup > 0.1  : {(vals_dup > 0.1).mean()*100:.1f}%  (attendu ~18%)")
print(f"% del > 0.1  : {(vals_del > 0.1).mean()*100:.1f}%  (attendu ~8%)")
print(f"% dup == 0   : {(vals_dup == 0).mean()*100:.1f}%")
print(f"Max dup      : {vals_dup.max():.2f}")

# ── 4. Charger les métadonnées ────────────────────────────
biosample_cols = [
    "biosample_id", "individual_id", "analysis_id",
    "platform_id", "histological_diagnosis_id",
    "icdo_topography_id", "icdo_morphology_id",
    "pathological_stage_id", "sample_origin_type_id",
    "cohorts",
]
bios = pd.read_csv(biosample_summary_path, sep=",", usecols=biosample_cols, dtype=str)

# Aligner sur les biosample_id du golden set
bios_panel = bios[bios["biosample_id"].isin(df_golden_cnv["biosample_id"])].copy()
print(f"\nMétadonnées alignées : {bios_panel.shape[0]} lignes "
      f"/ {df_golden_cnv['biosample_id'].nunique()} samples golden")

# ── 5. Sauvegarder le CNV-only corrigé ───────────────────
out_path = Path("/Users/bgadmin/Downloads/golden_set_cnv_v2.parquet")
df_golden_cnv.to_parquet(out_path, index=False)
print(f"\nSauvegardé → {out_path}")