import pandas as pd
from pathlib import Path

def full_audit():
    dirs = {
        "GDC (TCGA)": Path("cohort_matrices_cleaned"),
        "cBio RNA-seq": Path("cbioportal_cleaned/rnaseq"),
        "cBio Meth": Path("cbioportal_cleaned/methylation")
    }

    report = []

    for label, dir_path in dirs.items():
        if not dir_path.exists():
            continue
            
        files = list(dir_path.glob("*.csv"))
        print(f"--- Auditing {label} ({len(files)} files) ---")

        for f in files:
            try:
                # On ne charge que les colonnes et l'index pour économiser la RAM
                df = pd.read_csv(f, index_col=0, nrows=2) 
                
                # Vérif 1: Doublons de gènes
                # (Ici on doit charger tout l'index pour être sûr)
                full_index = pd.read_csv(f, usecols=[0], index_col=0)
                has_dup = full_index.index.duplicated().any()
                
                # Vérif 2: Longueur des IDs Patients
                sample_ids = df.columns.tolist()
                max_id_len = max([len(str(sid)) for sid in sample_ids]) if sample_ids else 0
                
                status = "✅ OK" if (not has_dup and (label != "GDC (TCGA)" or max_id_len <= 15)) else "⚠️ ISSUE"
                
                report.append({
                    "Source": label,
                    "File": f.name,
                    "Genes": len(full_index),
                    "Samples": len(sample_ids),
                    "Max_ID_Len": max_id_len,
                    "Duplicates": has_dup,
                    "Status": status
                })
                
                if status == "⚠️ ISSUE":
                    print(f"  ❌ Issue found in {f.name}: DUP={has_dup}, ID_LEN={max_id_len}")
                else:
                    print(f"  ✅ {f.name} verified.")

            except Exception as e:
                print(f"  🔥 Error reading {f.name}: {e}")

    # Résumé final
    df_report = pd.DataFrame(report)
    print("\n" + "="*60)
    print("FINAL AUDIT SUMMARY")
    print("="*60)
    if not df_report.empty:
        print(df_report.groupby(['Source', 'Status']).size().unstack(fill_value=0))
        df_report.to_csv("full_cleaning_audit_report.csv", index=False)
    else:
        print("No files were audited.")

if __name__ == "__main__":
    full_audit()