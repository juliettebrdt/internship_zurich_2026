"""
qc_check.py
===========
Validation des données nettoyées (cBioPortal / GDC).
Lance avec : python3 qc_check.py
Produit    : qc_report.txt  +  affichage console
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# PATHS  — adapter si besoin
# ─────────────────────────────────────────────────────────────

GENE_PANEL_FILE  = Path("/Users/bgadmin/Downloads/gene_cnv_cancer_panel.tsv")

CBIO_CLEANED     = Path("cbioportal_cleaned")
GDC_CLEANED      = Path("cohort_matrices_cleaned")

REPORT_FILE      = Path("qc_report.txt")

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

SEP  = "=" * 65
SEP2 = "-" * 55

lines = []   # accumulateur pour le rapport


def log(msg=""):
    print(msg)
    lines.append(msg)


def section(title):
    log()
    log(SEP)
    log(f"  {title}")
    log(SEP)


def subsection(title):
    log()
    log(SEP2)
    log(f"  {title}")
    log(SEP2)


def check(label, ok, detail=""):
    icon = "✅" if ok else "❌"
    msg  = f"  {icon}  {label}"
    if detail:
        msg += f"  ({detail})"
    log(msg)
    return ok


# ─────────────────────────────────────────────────────────────
# 1. CHARGE LE PANEL DE RÉFÉRENCE
# ─────────────────────────────────────────────────────────────

section("1. PANEL DE RÉFÉRENCE")

panel = pd.read_csv(GENE_PANEL_FILE, sep="\t")
panel_symbols = set(panel["gene_symbol"].dropna().astype(str).str.strip())
log(f"  Gènes dans le panel : {len(panel_symbols)}")
log(f"  Exemples            : {sorted(panel_symbols)[:8]}")


# ─────────────────────────────────────────────────────────────
# 2. FONCTION DE QC PAR FICHIER
# ─────────────────────────────────────────────────────────────

def qc_file(path: Path, panel_symbols: set) -> dict:
    """Retourne un dict de métriques QC pour un fichier nettoyé."""

    result = {"path": path, "ok": True, "issues": []}

    try:
        df = pd.read_csv(path, index_col=0)
    except Exception as e:
        result["ok"] = False
        result["issues"].append(f"Lecture impossible : {e}")
        return result

    n_genes, n_samples = df.shape
    result["n_genes"]   = n_genes
    result["n_samples"] = n_samples

    # ── Index = gene symbols ? ────────────────────────────────
    idx = df.index.astype(str).str.strip()

    # Pas de valeurs numériques dans l'index
    n_numeric_idx = sum(v.replace(".", "").lstrip("-").isdigit() for v in idx)
    if n_numeric_idx > 0:
        result["ok"] = False
        result["issues"].append(
            f"Index contient encore {n_numeric_idx} valeurs numériques (Entrez non mappés ?)"
        )

    # Pas de miRNA
    n_mirna = sum(v.lower().startswith(("hsa-", "mir-")) for v in idx)
    if n_mirna > 0:
        result["ok"] = False
        result["issues"].append(f"{n_mirna} miRNA dans l'index")

    # Tous les symboles sont dans le panel
    idx_set      = set(idx)
    in_panel     = idx_set & panel_symbols
    out_of_panel = idx_set - panel_symbols
    result["n_in_panel"]     = len(in_panel)
    result["n_out_of_panel"] = len(out_of_panel)
    result["pct_panel"]      = round(100 * len(in_panel) / len(idx_set), 1) if idx_set else 0

    if out_of_panel:
        result["issues"].append(
            f"{len(out_of_panel)} symboles hors panel : {sorted(out_of_panel)[:5]}"
        )

    # ── Doublons ──────────────────────────────────────────────
    n_dup = df.index.duplicated().sum()
    result["n_duplicates"] = n_dup
    if n_dup > 0:
        result["ok"] = False
        result["issues"].append(f"{n_dup} index dupliqués")

    # ── Valeurs numériques ────────────────────────────────────
    numeric_df = df.apply(pd.to_numeric, errors="coerce")

    pct_nan = round(100 * numeric_df.isna().values.mean(), 2)
    result["pct_nan"] = pct_nan
    if pct_nan > 50:
        result["ok"] = False
        result["issues"].append(f"{pct_nan}% de NaN (données trop creuses)")

    # Colonnes non numériques
    non_num_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    result["n_non_numeric_cols"] = len(non_num_cols)
    if non_num_cols:
        result["ok"] = False
        result["issues"].append(
            f"{len(non_num_cols)} colonnes non numériques : {non_num_cols[:3]}"
        )

    # Variance nulle (gènes constants)
    var = numeric_df.var(axis=1)
    n_zero_var = (var == 0).sum()
    result["n_zero_var"] = int(n_zero_var)

    # Stats de base
    result["val_min"]  = round(float(numeric_df.min().min()), 4)
    result["val_max"]  = round(float(numeric_df.max().max()), 4)
    result["val_mean"] = round(float(numeric_df.mean().mean()), 4)

    return result


# ─────────────────────────────────────────────────────────────
# 3. QC DE TOUS LES FICHIERS NETTOYÉS
# ─────────────────────────────────────────────────────────────

all_results = []

for base, label in [(CBIO_CLEANED, "cBioPortal"), (GDC_CLEANED, "GDC")]:
    if not base.exists():
        continue
    for omic_dir in sorted(base.iterdir()):
        if not omic_dir.is_dir():
            continue
        files = sorted(omic_dir.glob("*.csv"))
        if not files:
            continue

        section(f"2. {label.upper()} — {omic_dir.name.upper()}")
        log(f"  Fichiers trouvés : {len(files)}")

        for f in files:
            subsection(f.stem)
            r = qc_file(f, panel_symbols)
            all_results.append({**r, "source": label, "omic": omic_dir.name})

            check("Lecture OK",           r.get("ok", False) or not any("Lecture" in i for i in r["issues"]))
            check("Index = gene symbols", r.get("n_numeric_idx", 0) == 0 and r.get("n_mirna_idx", 0) == 0,
                  f"ex: {list(pd.read_csv(f, index_col=0).index[:3])}")
            check("Pas de doublons",      r.get("n_duplicates", 0) == 0,
                  f"{r.get('n_duplicates',0)} dupliqués")
            check("Colonnes numériques",  r.get("n_non_numeric_cols", 0) == 0,
                  f"{r.get('n_non_numeric_cols',0)} col. non-num.")
            check("NaN < 50%",            r.get("pct_nan", 0) < 50,
                  f"{r.get('pct_nan',0)}% NaN")
            check("Gènes dans le panel",  r.get("n_out_of_panel", 0) == 0,
                  f"{r.get('pct_panel',0)}% dans le panel")

            log(f"  → Dimensions     : {r.get('n_genes','?')} gènes × {r.get('n_samples','?')} samples")
            log(f"  → Valeurs        : min={r.get('val_min','?')}  max={r.get('val_max','?')}  mean={r.get('val_mean','?')}")
            log(f"  → Variance nulle : {r.get('n_zero_var','?')} gènes constants")

            if r["issues"]:
                log(f"  ⚠️  Problèmes détectés :")
                for issue in r["issues"]:
                    log(f"      • {issue}")
            else:
                log("  → Aucun problème détecté.")


# ─────────────────────────────────────────────────────────────
# 4. RÉSUMÉ GLOBAL
# ─────────────────────────────────────────────────────────────

section("3. RÉSUMÉ GLOBAL")

if not all_results:
    log("  Aucun fichier nettoyé trouvé. Vérifie les dossiers de sortie.")
else:
    total      = len(all_results)
    n_ok       = sum(1 for r in all_results if r["ok"])
    n_fail     = total - n_ok
    total_genes   = sum(r.get("n_genes", 0)   for r in all_results)
    total_samples = sum(r.get("n_samples", 0) for r in all_results)

    log(f"  Fichiers traités     : {total}")
    log(f"  ✅ Sans problème      : {n_ok}")
    log(f"  ❌ Avec problèmes     : {n_fail}")
    log(f"  Total gènes (cumul)  : {total_genes}")
    log(f"  Total samples (cumul): {total_samples}")

    log()
    if n_fail == 0:
        log("  🎉 TOUTES LES DONNÉES SONT PROPRES — tu peux continuer l'analyse.")
    else:
        log("  ⚠️  Des problèmes ont été détectés — voir le détail ci-dessus.")
        log()
        log("  Fichiers en échec :")
        for r in all_results:
            if not r["ok"]:
                log(f"    • {r['path'].name}")
                for issue in r["issues"]:
                    log(f"        → {issue}")

# ─────────────────────────────────────────────────────────────
# 5. SAUVEGARDE DU RAPPORT
# ─────────────────────────────────────────────────────────────

REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
print()
print(f"Rapport sauvegardé : {REPORT_FILE.resolve()}")