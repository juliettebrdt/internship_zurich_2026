import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

print("=== Patch CNV du golden set ===\n")

# ── 1. Charger le golden set ──────────────────────────────────
print("Chargement golden set...")
gold = pd.read_parquet("gold_standard_20260602_1455.parquet")
golden_ids = set(gold["biosample_id"].astype(str))
print(f"Golden set : {len(gold):,} samples × {gold.shape[1]:,} colonnes")

# ── 2. Séparer colonnes CNV et non-CNV ───────────────────────
cnv_cols_old = [c for c in gold.columns if c.startswith("cnv_")]
non_cnv_cols = [c for c in gold.columns if not c.startswith("cnv_")]
print(f"Colonnes CNV à remplacer : {len(cnv_cols_old)}")
print(f"Colonnes non-CNV à garder : {len(non_cnv_cols)}")

# ── 3. Bridge analysis_id → biosample_id ─────────────────────
bridge = pd.read_csv("cnv_id_bridge.csv", dtype=str)
ana_to_bio = dict(zip(
    bridge["analysis_id"].str.strip(),
    bridge["biosample_id"].str.strip()
))
bio_to_ana = {}
for ana, bio in ana_to_bio.items():
    bio_to_ana.setdefault(bio, []).append(ana)

golden_analysis_ids = set()
for bio in golden_ids:
    golden_analysis_ids.update(bio_to_ana.get(bio, []))
print(f"\nanalysis_id du golden set : {len(golden_analysis_ids):,}")

# ── 4. Charger gene panel filtré sur golden set ───────────────
print("Chargement gene panel (peut prendre 1-2 min)...")
gene_panel = pd.read_csv(
    "/Users/bgadmin/Downloads/gene_cnv_cancer_panel.tsv",
    sep="\t",
    usecols=["analysis_id", "gene_symbol",
             "dup_frac", "del_frac", "hldup_frac", "hldel_frac"],
    dtype={
        "analysis_id": str, "gene_symbol": str,
        "dup_frac": float, "del_frac": float,
        "hldup_frac": float, "hldel_frac": float,
    }
)
panel_golden = gene_panel[
    gene_panel["analysis_id"].isin(golden_analysis_ids)
].copy()
print(f"Lignes pour golden set : {len(panel_golden):,}")
print(f"% dup_frac > 0         : {(panel_golden['dup_frac'] > 0).mean()*100:.2f}%")

# ── 5. Pivot → format wide ────────────────────────────────────
print("\nConstruction matrice CNV wide...")
frames = []
for col in ["dup_frac", "del_frac", "hldup_frac", "hldel_frac"]:
    suffix = col.replace("_frac", "")
    piv = panel_golden.pivot_table(
        index="analysis_id",
        columns="gene_symbol",
        values=col,
        aggfunc="mean"
    ).fillna(0.0)
    piv.columns = [f"cnv_{g}__{suffix}" for g in piv.columns]
    piv.columns.name = None
    frames.append(piv)

cnv_wide = pd.concat(frames, axis=1)

# ── 6. Traduire analysis_id → biosample_id ───────────────────
cnv_wide.index = [ana_to_bio.get(str(i), "") for i in cnv_wide.index]
cnv_wide = cnv_wide[cnv_wide.index != ""]
cnv_wide.index.name = "biosample_id"

if cnv_wide.index.duplicated().any():
    print(f"Doublons : {cnv_wide.index.duplicated().sum()} → moyenne")
    cnv_wide = cnv_wide.groupby(cnv_wide.index).mean()

print(f"CNV wide final : {cnv_wide.shape}")
print(f"% > 0          : {(cnv_wide.values > 0).mean()*100:.2f}%")
print(f"Golden couverts: {len(set(cnv_wide.index) & golden_ids):,} / {len(golden_ids):,}")

# ── 7. Merger avec le golden set ─────────────────────────────
print("\nMerge avec le golden set...")
gold_fixed = gold[non_cnv_cols].merge(
    cnv_wide.reset_index(),
    on="biosample_id",
    how="left"
)

# Vérification
cnv_cols_new = [c for c in gold_fixed.columns if c.startswith("cnv_")]
pct = (gold_fixed[cnv_cols_new].fillna(0).values > 0).mean() * 100
samples_ok = (gold_fixed[cnv_cols_new].fillna(0) > 0).any(axis=1).sum()

print(f"\n=== Golden set corrigé ===")
print(f"Shape            : {gold_fixed.shape}")
print(f"Colonnes CNV     : {len(cnv_cols_new)}")
print(f"% CNV > 0        : {pct:.2f}%")
print(f"Samples avec CNV : {samples_ok:,} / {len(gold_fixed):,}")

# Vérifier quelques gènes connus
for gene in ["TP53", "MYC", "CDKN2A", "EGFR"]:
    col = f"cnv_{gene}__dup"
    if col in gold_fixed.columns:
        n = (gold_fixed[col] > 0).sum()
        print(f"  {gene}__dup > 0 : {n:,} samples")

# ── 8. Sauvegarder ───────────────────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
out_path = f"gold_standard_{timestamp}.parquet"
gold_fixed.to_parquet(out_path, engine="pyarrow", index=False)
print(f"\nSauvegardé → {out_path}")
print("✓ Golden set corrigé avec CNV continus depuis gene_cnv_cancer_panel.tsv")