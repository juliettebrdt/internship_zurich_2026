"""
explore_tcga_data.py
--------------------
Script d'exploration des données TCGA disponibles sur GDC et cBioPortal.
Pour chaque projet TCGA, liste les types de données RNA-seq et méthylation disponibles.
 
Usage:
    pip install requests pandas
    python explore_tcga_data.py
 
Outputs:
    - gdc_summary.csv       : résumé des données disponibles sur GDC par projet
    - cbioportal_summary.csv: résumé des données disponibles sur cBioPortal par étude
    - data_availability.csv : tableau croisé final (un projet = une ligne)
"""
 
import requests
import pandas as pd
import json
import time
from collections import defaultdict
 
# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
 
GDC_BASE      = "https://api.gdc.cancer.gov"
CBIO_BASE     = "https://www.cbioportal.org/api"
HEADERS       = {"Content-Type": "application/json", "Accept": "application/json"}
 
# Types de données qui nous intéressent sur GDC
DATA_TYPES_OF_INTEREST = {
    "RNA-seq"     : {"data_type": "Gene Expression Quantification", "experimental_strategy": "RNA-Seq"},
    "Methylation" : {"data_type": "Methylation Beta Value",         "experimental_strategy": "Methylation Array"},
}
 
 
# ─────────────────────────────────────────────
# PARTIE 1 — GDC
# ─────────────────────────────────────────────
 
def get_tcga_projects_gdc() -> list[dict]:
    """Récupère la liste de tous les projets TCGA sur GDC."""
    url = f"{GDC_BASE}/projects"
    params = {
        "filters": json.dumps({
            "op": "in",
            "content": {"field": "program.name", "value": ["TCGA"]}
        }),
        "fields": "project_id,name,summary.case_count,summary.file_count",
        "size": 100,
        "format": "json"
    }
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    hits = r.json()["data"]["hits"]
    print(f"[GDC] {len(hits)} projets TCGA trouvés")
    return hits
 
 
def count_files_for_project(project_id: str, data_label: str, data_type: str, experimental_strategy: str) -> dict:
    """
    Pour un projet TCGA donné, compte le nombre de fichiers et de cases
    disponibles pour un type de données (ex: RNA-seq ou Méthylation).
    """
    url = f"{GDC_BASE}/files"
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id",    "value": [project_id]}},
            {"op": "in", "content": {"field": "data_type",                   "value": [data_type]}},
            {"op": "in", "content": {"field": "experimental_strategy",       "value": [experimental_strategy]}},
            {"op": "in", "content": {"field": "access",                      "value": ["open"]}},  # données open-access
        ]
    }
    params = {
        "filters": json.dumps(filters),
        "fields":  "file_id,cases.case_id,data_format,platform",
        "size":    1,   # on veut juste le total (pagination)
        "format":  "json"
    }
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()["data"]
    total_files = data["pagination"]["total"]
 
    # Récupère aussi le format et la plateforme du premier fichier si dispo
    platform = ""
    data_format = ""
    if data["hits"]:
        first = data["hits"][0]
        data_format = first.get("data_format", "")
        platform    = first.get("platform", "")
 
    return {
        "project_id"   : project_id,
        "data_label"   : data_label,
        "n_files"      : total_files,
        "data_format"  : data_format,
        "platform"     : platform,
        "available"    : total_files > 0
    }
 
 
def explore_gdc() -> pd.DataFrame:
    """
    Point d'entrée GDC :
    - Liste tous les projets TCGA
    - Pour chaque projet, vérifie la disponibilité RNA-seq et Méthylation
    Retourne un DataFrame avec une ligne par (projet, type_de_données).
    """
    print("\n=== EXPLORATION GDC ===")
    projects = get_tcga_projects_gdc()
 
    rows = []
    for proj in projects:
        pid   = proj["project_id"]
        name  = proj["name"]
        cases = proj.get("summary", {}).get("case_count", 0)
        print(f"  → {pid} ({cases} cases)...")
 
        for label, params in DATA_TYPES_OF_INTEREST.items():
            result = count_files_for_project(pid, label, params["data_type"], params["experimental_strategy"])
            rows.append({
                "project_id"   : pid,
                "project_name" : name,
                "n_cases_total": cases,
                "data_type"    : label,
                "n_files"      : result["n_files"],
                "data_format"  : result["data_format"],
                "platform"     : result["platform"],
                "available_gdc": result["available"]
            })
            time.sleep(0.2)   # respecte le rate limit GDC
 
    df = pd.DataFrame(rows)
    df.to_csv("gdc_summary.csv", index=False)
    print(f"\n[GDC] Résultats sauvegardés → gdc_summary.csv")
    return df

def make_summary(df_gdc: pd.DataFrame) -> pd.DataFrame:
    """
    Crée un tableau croisé final corrigé.
    """
    # 1. Pivot GDC : transforme le format long en format large
    gdc_pivot = df_gdc.pivot_table(
        index=["project_id", "project_name", "n_cases_total"],
        columns="data_type",
        values=["n_files", "available_gdc"],
        aggfunc="first"
    ).reset_index()
    
    # Nettoyage des noms de colonnes après le pivot
    gdc_pivot.columns = ["_".join(c).strip("_") for c in gdc_pivot.columns]
 
    # --- LES LIGNES À AJOUTER SONT ICI ---
    # On crée les colonnes manquantes que le print essaie d'utiliser
    summary = gdc_pivot.copy()
    summary["has_rnaseq"] = summary["available_gdc_RNA-seq"] == True
    summary["has_methylation"] = summary["available_gdc_Methylation"] == True
    # -------------------------------------

    # Extrait le cancer type (ex: TCGA-BRCA -> BRCA)
    summary["cancer_type"] = summary["project_id"].str.replace("TCGA-", "")
    
    summary.to_csv("data_availability.csv", index=False)
 
    print("\n=== RÉSUMÉ FINAL ===")
    # Maintenant, ce print ne plantera plus !
    cols_to_show = [
        "project_id", "n_cases_total",
        "n_files_RNA-seq", "available_gdc_RNA-seq",
        "n_files_Methylation", "available_gdc_Methylation",
        "has_rnaseq", "has_methylation"
    ]
    print(summary[cols_to_show].to_string(index=False))
    
    print(f"\nFichier final sauvegardé → data_availability.csv")
    return summary

if __name__ == "__main__":
    print("=" * 60)
    print("  Exploration données TCGA — GDC + cBioPortal")
    print("=" * 60)
 
    df_gdc  = explore_gdc()
    summary = make_summary(df_gdc)
 
    print("\nDone. Fichiers générés :")
    print("  - gdc_summary.csv        (détail par projet et type de données)")

    print("  - data_availability.csv  (tableau croisé final)")