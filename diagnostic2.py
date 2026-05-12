import pandas as pd

# 1. Charge ton fichier fusionné final
# (Si c'est un fichier très lourd, utilise low_memory=False)
path = "final_joined_table_20260512_1315.parquet" 
path_to_gold = "gold_standard_20260512_1315.parquet"
try:
    # Lecture du fichier Parquet
    df = pd.read_parquet(path)
    df_gold=pd.read_parquet(path_to_gold)
    print(f"✅ Fichier chargé avec succès ! Dimensions : {df.shape}")
    print(f"✅ Fichier chargé avec succès ! Dimensions : {df_gold.shape}")
    
    # --- Début du diagnostic ---
    
    # Identification des colonnes
    cnv_cols = [c for c in df.columns if c.startswith("cnv_")]
    rna_cols = [c for c in df.columns if c.startswith("rna_")]
    meth_cols = [c for c in df.columns if c.startswith("meth_")]

    # Masques de présence de données
    has_cnv  = df[cnv_cols].notna().any(axis=1)
    has_rna  = df[rna_cols].notna().any(axis=1)
    has_meth = df[meth_cols].notna().any(axis=1)

    # Samples avec CNV mais sans RNA ni Meth (Orphelins)
    orphans = df[has_cnv & ~has_rna & ~has_meth]
    print(f"\nNombre total d'orphelins (CNV seul) : {len(orphans)}")

    # Analyse de la provenance
    if "source_origin" in df.columns:
        print("\nOrigine des orphelins :")
        print(orphans["source_origin"].value_counts())
    else:
        # Diagnostic par le nom de l'ID (Index)
        tcga_count = orphans.index.str.contains("TCGA").sum()
        print(f"Orphelins de type TCGA : {tcga_count}")
        print(f"Autres orphelins : {len(orphans) - tcga_count}")

except Exception as e:
    print(f"❌ Erreur lors de la lecture du Parquet : {e}")

#cbio_orphans = df[has_cnv & ~has_rna & ~has_meth & (df["source_origin"] == "cBioPortal")]

#print(cbio_orphans["cbioportal_study"].value_counts().head(30))
target_cols = [
    'histological_diagnosis_label', 
    'icdo_topography_label', 
    'icdo_morphology_label',
    'biosample_status_label'
]

print("=== STATISTIQUES DE LA COHORTE FINALISÉE ===")

for col in target_cols:
    if col in df.columns:
        print(f"\n--- Top 15 : {col} ---")
        # On affiche le décompte en excluant les NaN
        print(df[col].value_counts().head(15))

# Petit bonus pour voir la pureté de ta multi-omique sur ces types de cancer
print("\n=== COUVERTURE MULTI-OMIQUE PAR TYPE DE CANCER ===")
# On crée un indicateur multi-omique (au moins 2 types de données)
has_rna = df[[c for c in df.columns if c.startswith("rna_")]].notna().any(axis=1)
has_meth = df[[c for c in df.columns if c.startswith("meth_")]].notna().any(axis=1)

multi_omique = df[has_rna | has_meth]
print(f"Total samples avec RNA ou Meth : {len(multi_omique)}")
print(multi_omique['icdo_topography_label'].value_counts().head(10))