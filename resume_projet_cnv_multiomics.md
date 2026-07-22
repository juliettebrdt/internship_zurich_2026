# Résumé du projet — Stage UZH Baudis Group — Analyse CNV multi-omique

## Contexte
Stage ingénieur fin d'études, Polytech Nice Sophia, spécialité Bioinformatique.
UZH, groupe Baudis. Projet : analyser les CNV pour relier lignées cellulaires
cancéreuses aux tumeurs réelles, via deux axes :
1. CNV seul → pipeline preprocessing → clustering → validation → inférence
2. Multi-omique (CNV+RNA+Méthylation, MOFA) → comparaison avec axe 1

---

## Évolution de la classification des groupes (icdot_root_grouped / cancer_group)

### V1 (point de départ)
Groupement simple par topographie ICD-O (organe), ex: C00_C14, C15_C17_C21_C26,
C30_C39, etc. ~21 groupes.

### V2 — Séparation C15 par morphologie
Demande du groupe Baudis : séparer C15 (œsophage) de C17/C21/C26 (intestin) par
TISSU. Problème : C17/C21/C26 n'a que 12 samples → trop petit pour clustériser.
Solution adoptée : séparer C15 par MORPHOLOGIE plutôt :
- C15_SCC (carcinome épidermoïde, code morpho 8070-8079) — 93 samples
- C15_ADENO (adénocarcinome, code morpho 8140-8149) — 85 samples
- C15_other / C15_NOS — résiduel
C17_C21_C26 reste exclu (n=12 < seuil 50).

### V3 — Séparation C00_C14 par tissu
Le tuteur a demandé : Head & Neck (C00-C14) n'est pas un tissu homogène
(cavité orale ≠ pharynx ≠ nasopharynx EBV-driven ≠ glandes salivaires).
Vérification des effectifs sur golden set multi-omique (n=9592) :
- Oral_cavity (C00-C06) : 256 samples
- Pharynx (C07-C14) : 120 samples
Les deux passent le seuil de 50 → séparation adoptée par TISSU (topographie),
contrairement à C15 qui est séparé par MORPHOLOGIE.

### V4 — Classification transversale Organe × Histologie (étape actuelle, FINALE)
Le tuteur a demandé une dimension supplémentaire : ne pas regrouper QUE par
organe, mais aussi par grande catégorie histologique transversale
(carcinome vs glioma vs sarcome etc.), comme pour C15 SCC/ADENO mais
généralisé à TOUS les organes quand statistiquement viable.

#### Fonction `histology_class_from_label()`
Classification basée sur mots-clés dans `icdo_morphology_label` (PAS sur les
codes numériques ICD-O — pas de mapping officiel fiable trouvé, voir note
ci-dessous) :
```python
def histology_class_from_label(label: str) -> str:
    if not isinstance(label, str):
        return "Other_rare"
    label_lower = label.lower()
    if "squamous" in label_lower:
        return "SCC"
    elif "adenocarcinoma" in label_lower:
        return "Adenocarcinoma"
    elif any(k in label_lower for k in ["glioblastoma", "astrocytoma", "oligodendroglioma", "glioma"]):
        return "Glioma"
    elif "melanoma" in label_lower:
        return "Melanoma"
    elif "sarcoma" in label_lower:
        return "Sarcoma"
    elif "lymphoma" in label_lower:
        return "Lymphoma"
    elif "leukemia" in label_lower or "leukaemia" in label_lower:
        return "Leukemia"
    elif "mesothelioma" in label_lower:
        return "Mesothelioma"
    elif any(k in label_lower for k in ["germ cell", "teratoma", "seminoma"]):
        return "Germ_cell_tumor"
    elif "carcinoma" in label_lower:
        return "Carcinoma_other"
    else:
        return "Other_rare"
```

NOTE IMPORTANTE : `icdo_morphology_label` n'existe QUE dans le golden set
multi-omique (`meta_final.tsv` / `golden_clean`), PAS dans `biosample_summary.csv`
utilisé par le pipeline CNV seul. Pour le pipeline CNV, il faut récupérer
le mapping `icdo_morphology_id -> icdo_morphology_label` depuis
`multiomics_preproc/meta_final.tsv` et faire un `.map()` :
```python
morph_mapping_df = pd.read_csv(
    "/Users/bgadmin/Downloads/multiomics_preproc/meta_final.tsv",
    sep="\t", dtype=str,
    usecols=["icdo_morphology_id", "icdo_morphology_label"]
).drop_duplicates()
morph_id_to_label = dict(zip(
    morph_mapping_df["icdo_morphology_id"],
    morph_mapping_df["icdo_morphology_label"]
))
bios_tumour["icdo_morphology_label"] = bios_tumour["icdo_morphology_id"].map(morph_id_to_label)
# ~4-10 labels manquants sur ~9600, négligeable
```

#### Combinaisons organe × histologie viables (seuil strict n≥50, calculé sur
golden set multi-omique n=9592, COHÉRENT avec le seuil utilisé partout
ailleurs dans le pipeline — ne JAMAIS assouplir ce seuil pour rester
méthodologiquement cohérent) :

**Version MULTI-OMICS (noms longs avec parenthèses, ex "Lung (C30-C39)") :**
```python
VIABLE_ORGAN_HISTOLOGY_COMBOS = {
    ("Lung (C30-C39)", "SCC"),
    ("Lung (C30-C39)", "Adenocarcinoma"),
    ("Lung (C30-C39)", "Other_rare"),
    ("Lung (C30-C39)", "Mesothelioma"),
    ("Lung (C30-C39)", "Carcinoma_other"),
    ("Gynecologic (C51-C58)", "Adenocarcinoma"),
    ("Gynecologic (C51-C58)", "SCC"),
    ("Male genital (C60-C63)", "Adenocarcinoma"),
    ("Male genital (C60-C63)", "Carcinoma_other"),
    ("Male genital (C60-C63)", "Germ_cell_tumor"),
    ("Urinary (C64-C68)", "Adenocarcinoma"),
    ("Urinary (C64-C68)", "Carcinoma_other"),
    ("Thyroid/Endocrine (C73-C75)", "Adenocarcinoma"),
    ("Thyroid/Endocrine (C73-C75)", "Carcinoma_other"),
    ("Thyroid/Endocrine (C73-C75)", "Other_rare"),
    ("SoftTissue (C49)", "Sarcoma"),
    ("SoftTissue (C49)", "Melanoma"),
    ("C16", "Adenocarcinoma"),
    ("C16", "Carcinoma_other"),
    ("Bone/SoftTissue (C40-C48)", "Sarcoma"),
}

def cancer_group_combined(organ: str, histology: str) -> str:
    if organ is None:
        return None
    if organ.startswith("Esophagus") or organ.startswith("C15"):
        return organ  # C15 déjà géré en amont, ne pas recombiner
    if (organ, histology) in VIABLE_ORGAN_HISTOLOGY_COMBOS:
        organ_clean = organ.split(" (")[0]
        return f"{organ_clean}_{histology}"
    return organ
```

**Version CNV SEUL (codes courts, ex "C30_C39") :**
```python
VIABLE_ORGAN_HISTOLOGY_COMBOS_CNV = {
    ("C30_C39", "SCC"), ("C30_C39", "Adenocarcinoma"), ("C30_C39", "Other_rare"),
    ("C30_C39", "Mesothelioma"), ("C30_C39", "Carcinoma_other"),
    ("C51_C58", "Adenocarcinoma"), ("C51_C58", "SCC"),
    ("C60_C63", "Adenocarcinoma"), ("C60_C63", "Carcinoma_other"), ("C60_C63", "Germ_cell_tumor"),
    ("C64_C68", "Adenocarcinoma"), ("C64_C68", "Carcinoma_other"),
    ("C73_C75", "Adenocarcinoma"), ("C73_C75", "Carcinoma_other"), ("C73_C75", "Other_rare"),
    ("C49", "Sarcoma"), ("C49", "Melanoma"),
    ("C16", "Adenocarcinoma"), ("C16", "Carcinoma_other"),
    ("C40_C41_C48", "Sarcoma"),
}

def cancer_group_combined_cnv(organ: str, histology: str) -> str:
    if organ is None:
        return None
    if organ.startswith("C15"):
        return organ
    if (organ, histology) in VIABLE_ORGAN_HISTOLOGY_COMBOS_CNV:
        return f"{organ}_{histology}"
    return organ
```

#### Résultat final : 45-46 groupes au total (selon pipeline), incluant :
- Groupes combinés organe_histologie (20 nouvelles combinaisons)
- Groupes organe-seul déjà homogènes (Brain→Glioma, Breast→Carcinoma_other,
  Colorectal→Adenocarcinoma, Oral_cavity→SCC, Pharynx→SCC, etc. — pas de
  séparation nécessaire car une seule histologie dominante)
- Groupes résiduels sous le seuil 50 (ex: C30_C39 résiduel=11, C16 résiduel=5,
  Thyroid/Endocrine résiduel=5) → automatiquement exclus plus loin par
  MIN_SAMPLES_REQUIRED=50

C15_SCC (93) et C15_ADENO (85) et C00_C06 (257) et C07_C14 (122) restent
INCHANGÉS par rapport à V2/V3 — la fonction cancer_group_combined() les laisse
passer tels quels (court-circuit explicite pour C15 ; pas de combo viable pour
C00_C06/C07_C14 car déjà mono-histologiques).

---

## Pipeline CNV seul — état d'avancement

### Notebooks dans l'ordre :
1. **Preprocessing** (`cnv_preproc_release_v2/`) — TERMINÉ avec V4
   - Charge golden_cnv (gold_standard_*.parquet) + biosample_summary.csv
   - Calcule bios_tumour["icdot_root_grouped"] avec cancer_group_combined_cnv()
   - 45 roots obtenus, build_feature_matrix_for_root_golden() par root
   - PCA/UMAP first-pass + HDBSCAN technical cluster removal
   - Stage2A (L2-norm + PCA) / Stage2B (platform effect correction, pas
     nécessaire ici car golden set quasi-exclusivement TCGA)
   - Sauvegarde : `final_space/{root}_PCs_final.parquet` (48 fichiers générés,
     dont 3 obsolètes à nettoyer : C00_C14, C15_C17_C21_C26, C17_C21_C26)
   - BUG CORRIGÉ : `fit_pca_umap()` doit avoir un fallback `init="random"`
     pour UMAP quand n_samples < 4*n_neighbors (sinon erreur
     `scipy.linalg.eigh` sur petits groupes)
   - BUG CORRIGÉ : garde anti-matrice-vide dans la boucle Stage2A (sinon
     `ValueError: zero-size array` sur groupes résiduels dégénérés)

2. **KNN / retrieval_similarity** (notebook séparé) — TERMINÉ avec V4
   - Charge `final_space/manifest_final.tsv` (généré par preprocessing,
     PAS le même fichier que celui généré PAR CE notebook — confusion de noms)
   - Construit kNN basique (K=50, euclidean, 20 PCs) → `knn_index/`
   - Stability sweep (metric × n_pcs) → `stability_df`
   - Sélectionne meilleur setting par root → `best_by_root_v1.tsv`
     (fichier CRITIQUE utilisé par le notebook de clustering ensuite)
   - ATTENTION : le sweep skip les roots avec n≤K_SWEEP=50 (normal, ce sont
     les résidus sous le seuil)
   - Construit Graph A (`knn_index_subtype/perroot_best/`) et Graph B
     (`knn_index_subtype/baseline_euc10/`)
   - Sauvegarde `cnv_purity_per_group.tsv` (utilisé dans la comparaison finale)

3. **Clustering Leiden** (`subtyping_leiden_A/`) — TERMINÉ avec V4 après nettoyage
   - Charge `best_by_root_v1.tsv` pour construire les graphes pondérés (RBF
     sur distances kNN)
   - Scan résolution Leiden CPM (broad puis fine grid), 5 seeds, choix par
     ARI de stabilité ≥0.80
   - Sauvegarde : `subtyping_leiden_A/clusters_by_root/{root}_clusters_leiden_cpm.parquet`
   - PROBLÈME RÉCURRENT RÉSOLU : fichiers obsolètes (C00_C14, C15_C17_C21_C26)
     qui persistent sur disque entre les itérations et polluent best_by_root_v1.tsv
     et les dossiers de clusters → TOUJOURS nettoyer manuellement avant de
     relancer une chaîne de notebooks après changement de icdot_root_grouped()
   - Section "EXPÉRIENCES" en bas du notebook = optionnelle (sensibilité aux
     paramètres kNN), PAS requise pour le pipeline principal — source de
     confusion ("fichier kNN manquant" peut venir de là, pas du clustering
     principal)
   - ÉTAT ACTUEL : 44 fichiers clusters générés, 2 obsolètes restants à
     supprimer (C00_C14_clusters_leiden_cpm.parquet,
     C15_C17_C21_C26_clusters_leiden_cpm.parquet)

4. **Subtyping / signatures** — pas encore relancé avec V4
5. **Validation** (prediction_validation, trust metrics) — pas encore relancé avec V4

---

## Pipeline multi-omics — état d'avancement

1. **Preprocessing** (`multiomics_preproc/`) — TERMINÉ avec V3 (Oral_cavity/Pharynx)
   mais PAS ENCORE avec V4 (combos organe×histologie) — À FAIRE
   - Charge gold_standard parquet, sépare X_cnv/X_rna/X_meth, nettoie/impute/normalise
   - icdot_root_grouped() V3 déjà appliqué (Esophagus_SCC/ADENO, Oral_cavity,
     Pharynx) — bloc manuel SCC/ADENO à la fin du notebook supprimé (inutile
     car déjà géré par icdot_root_grouped + relance complète du pipeline)
   - MOFA tourne GLOBALEMENT sur tout meta_final (20 facteurs, scale_views=True)
   - PCA séparée par omique (CNV/RNA/Meth, 20 PCs chacune après comparaison
     PCA séparée vs PCA globale concaténée)
   - Sauvegarde meta_final.tsv, pcs_*.parquet, mofa_factors.parquet

2. **KNN par groupe** (`knn_index/knn_by_cancer_group/pergroup_best/`) —
   À RELANCER avec V4
3. **Clustering Leiden** (`clustering_leiden_A/clusters_by_group/`) — TERMINÉ
   avec V3 (45 groupes après ajout Oral_cavity/Pharynx), À RELANCER avec V4
   - Résultat V3 notable : Oral_cavity et Pharynx → 1 seul cluster chacun
     ("no structure" / "no clear winner" dans l'espace MOFA), biologiquement
     cohérent car déjà homogènes en SCC
4. **Subtyping** — pas encore relancé avec V4
5. **Validation** — pas encore relancé avec V4

---

## Notebook de comparaison finale (CNV seul vs MOFA)

Dictionnaire `cnv_to_mofa` (mapping nom court CNV -> nom long MOFA) À
REFAIRE ENTIÈREMENT avec les 45 nouveaux groupes V4 (actuellement seulement
21 entrées de la V3).

Dictionnaires de fichiers `mofa_group_to_file` et `cnv_root_to_file` à
régénérer aussi.

### Résultats obtenus avec V3 (avant V4, à refaire) :
Classification finale en 6 catégories : CNV sufficient / Multi-omic adds
value / Multi-omic adds value (global structure only) / Neutral (no clear
winner) / Neutral (conflicting signals) / Too small — unreliable (n<150)

4 métriques de consensus : kNN purity morphologique, ARI vs morphologie,
Homogeneity/Completeness/V-measure, test de Wilcoxon apparié sur purity
per-sample.

Résultats V3 notables :
- Oral_cavity (C00-C06) → CNV sufficient (4/4 métriques d'accord, consensus -1.00)
- Pharynx (C07-C14) → Too small — unreliable (n=122 < 150)
- Découverte importante : la séparation C00_C14/C15_C17_C21_C26 ne change
  PAS la conclusion finale par rapport aux groupes fusionnés (même verdict
  global CNV-suffisant ou trop-petit), MAIS améliore la rigueur méthodologique
  et l'interprétabilité biologique (groupes anatomiquement cohérents au lieu
  de mélanges hétérogènes type "Head & Neck" qui combinait cavité orale et
  nasopharynx EBV-driven).

---

## TODO immédiat (prochaines étapes dans l'ordre)

1. [FAIT] Nettoyer les fichiers obsolètes du pipeline CNV (C00_C14,
   C15_C17_C21_C26, C17_C21_C26) dans final_space/, knn_index_subtype/
   perroot_best/, baseline_euc10/, subtyping_leiden_A/clusters_by_root/
2. [FAIT] best_by_root_v1.tsv régénéré proprement avec les groupes V4 valides
3. [FAIT] Clustering CNV relancé avec V4, 44 fichiers clusters (à nettoyer
   les 2 derniers obsolètes : C00_C14_clusters_leiden_cpm.parquet,
   C15_C17_C21_C26_clusters_leiden_cpm.parquet)
4. [À FAIRE] Vérifier/nettoyer leiden_manifest_stats.tsv (subtyping_leiden_A)
5. [À FAIRE] Relancer subtyping CNV (signatures Track1/Track2) avec V4
6. [À FAIRE] Relancer validation CNV (prediction_validation, trust_manifest) avec V4
7. [À FAIRE] Appliquer cancer_group_combined() (version longue) au
   preprocessing multi-omics, relancer MOFA + sauvegarde meta_final
8. [À FAIRE] Relancer KNN par groupe multi-omics avec V4
9. [À FAIRE] Relancer clustering Leiden multi-omics avec V4
10. [À FAIRE] Relancer subtyping + validation multi-omics avec V4
11. [À FAIRE] Refaire cnv_to_mofa, mofa_group_to_file, cnv_root_to_file
    avec les 45 groupes V4
12. [À FAIRE] Relancer tout le notebook de comparaison finale (ARI, HCV,
    Wilcoxon, consensus score, narrative summary)
13. [À FAIRE] Rédaction rapport : Discussion, Conclusion, résumés FR/EN
    (250 mots + 5 mots-clés), bibliographie format Cell, bilan compétences

## Leçon méthodologique apprise
Après CHAQUE modification de icdot_root_grouped() / cancer_group_combined(),
il faut :
(a) relancer TOUT le pipeline depuis le preprocessing (Kernel → Restart & Run All)
(b) NETTOYER manuellement les fichiers obsolètes sur disque dans TOUS les
    sous-dossiers concernés AVANT de relancer, car les notebooks n'écrasent
    que les fichiers des groupes actuellement traités — les anciens fichiers
    de groupes qui n'existent plus persistent silencieusement et polluent
    les manifests générés dynamiquement par glob() sur le système de fichiers.