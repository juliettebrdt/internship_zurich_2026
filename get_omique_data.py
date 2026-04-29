import pandas as pd

# Charger ton manifest extrait précédemment
df = pd.read_csv("rnaseq_metadata.csv")

# Le GDC Data Transfer Tool attend exactement ces 5 colonnes
manifest = pd.DataFrame({
    "id":       df["file_id"],
    "filename": df["file_name"],
    "md5":      "",        # vide, GDC le vérifie automatiquement
    "size":     df["file_size"] if "file_size" in df.columns else "",
    "state":    "validated"
})

manifest.to_csv("gdc_manifest_rnaseq.txt", sep="\t", index=False)
print(f"Manifest généré : {len(manifest)} fichiers")