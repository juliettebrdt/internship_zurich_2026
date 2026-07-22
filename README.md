# CNV & Multi-Omics — UZH Baudis Group Internship

Final-year engineering internship (Polytech Nice Sophia, Bioinformatics track) carried out within the Baudis group (UZH). The goal of the project is to link cancer cell lines to real tumors based on copy number variation (CNV) profiles, along two complementary axes:

1. **CNV-only pipeline**: preprocessing → retrieval/similarity (KNN) → clustering (Leiden) → subtyping → validation → inference
2. **Multi-omics pipeline** (CNV + RNA-seq + methylation, MOFA integration): same steps, compared against the CNV-only axis

Samples are grouped using a transversal **organ × histology** classification (e.g. `Lung_SCC`, `Lung_Adenocarcinoma`), built iteratively throughout the internship to balance biological homogeneity with sample size (minimum threshold of 50 samples per group).

## Notebooks

| Notebook | Role |
|---|---|
| `preprocessing.ipynb` | Multi-omics data preparation and cleaning |
| `retrieval_similarity_omics.ipynb` | Similarity search (KNN) |
| `clustering_omics.ipynb` | Clustering (Leiden) |
| `subtyping_omics.ipynb` | Subtyping of organ × histology groups |
| `validation_omics.ipynb` | Cluster/subtype validation |
| `CNV_only_vs_multi_omics.ipynb`, `CNV_only_vs_multi_omics_morpholy_histolgy.ipynb` | Comparison of the two pipelines |
| `investigation_cluster.ipynb` | In-depth cluster exploration |
| `inference.ipynb`, `inference_morpholy_histology.ipynb`, `inference_multi_omics.ipynb` | Inference on new samples |

## Main scripts

- `get_omics_cbioportal.py`, `get_omics_data_RNAseq_cohort.py`, `get_data_progenetix.py` — data retrieval from cBioPortal / GDC / Progenetix
- `bridge_CCLE.ipynb` — bridge to CCLE cell line data
- `cleaned_omics_data.py`, `join_omics3.py` — cleaning and joining of multi-omics tables
- `correction_golden_set.py`, `extraction_CNV_goldentset.py` — golden set construction

## Data

Raw and intermediate datasets (CSV, Parquet, PKL, expression matrices, etc.) are **not versioned** in this repo (see `.gitignore`) due to their size. They are available locally in the Baudis group working environment.

## Environment

Python environment managed via `venv_omics/` (not versioned). Main dependencies: `pandas`, `numpy`, `scikit-learn`, `scanpy`/`leidenalg`, `mofapy2`.
