"""
SCRIPT 0 — Diagnostic : Vérifier le format des biosample_name de Progenetix
           et trouver le bon champ GDC correspondant.
"""
import requests
import pandas as pd
import re
from io import StringIO

print("=== DIAGNOSTIC : FORMAT DES IDs PROGENETIX ===\n")

url = "https://progenetix.org/services/sampletable/?filters=pgx:cohort-TCGAcancers&limit=0"
response = requests.get(url, timeout=120)
response.raise_for_status()
df = pd.read_csv(StringIO(response.text), sep="\t")

print(f"Colonnes disponibles : {df.columns.tolist()}\n")

# Afficher un échantillon des valeurs de biosample_name
sample_values = df['biosample_name'].dropna().head(20).tolist()
print("=== 20 premières valeurs de 'biosample_name' ===")
for v in sample_values:
    print(f"  {v}")

# Patterns possibles
UUID_PATTERN    = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I)
SUBMITTER_PATTERN = re.compile(r'^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2}[A-Z]-[0-9]{2}[A-Z]-[A-Z0-9]+-[0-9]{2}$')
BARCODE_PATTERN = re.compile(r'^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}$')

counts = {"GDC UUID (sample)": 0, "TCGA Barcode complet": 0, 
          "TCGA Barcode court": 0, "Autre format": 0}

for v in df['biosample_name'].dropna():
    if UUID_PATTERN.match(str(v)):
        counts["GDC UUID (sample)"] += 1
    elif SUBMITTER_PATTERN.match(str(v)):
        counts["TCGA Barcode complet"] += 1
    elif BARCODE_PATTERN.match(str(v)):
        counts["TCGA Barcode court"] += 1
    else:
        counts["Autre format"] += 1

print("\n=== Distribution des formats ===")
for fmt, count in counts.items():
    print(f"  {fmt:30s}: {count}")

# Recommandation automatique
print("\n=== RECOMMANDATION ===")
dominant = max(counts, key=counts.get)
mapping = {
    "GDC UUID (sample)":     ("cases.samples.sample_id",    "UUID GDC natif — champ correct ✅"),
    "TCGA Barcode complet":  ("cases.samples.submitter_id", "Barcode complet TCGA ✅"),
    "TCGA Barcode court":    ("cases.submitter_id",         "Barcode patient TCGA (sans aliquot)"),
    "Autre format":          (None,                          "Format inconnu — inspection manuelle nécessaire ⚠️"),
}
field, advice = mapping[dominant]
print(f"  Format dominant : {dominant}")
print(f"  Champ GDC à utiliser : {field}")
print(f"  Note : {advice}")

# Vérification croisée GDC sur 3 IDs
if field:
    print("\n=== VÉRIFICATION CROISÉE GDC (3 IDs) ===")
    test_ids = df['biosample_name'].dropna().head(3).tolist()
    for test_id in test_ids:
        r = requests.post(
            "https://api.gdc.cancer.gov/cases",
            headers={"Content-Type": "application/json"},
            json={
                "filters": {"op": "=", "content": {"field": field, "value": test_id}},
                "fields": "submitter_id,case_id",
                "size": 1
            }
        )
        hits = r.json().get("data", {}).get("hits", [])
        status = f"✅ Trouvé → {hits[0]['submitter_id']}" if hits else "❌ Non trouvé dans GDC"
        print(f"  {test_id[:40]:<42} {status}")