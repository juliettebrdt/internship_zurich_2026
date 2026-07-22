import pandas as pd
import logging
import shutil
import requests

from pathlib import Path

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("omics_cleaner.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────

GENE_PANEL_FILE   = Path("/Users/bgadmin/Downloads/gene_cnv_cancer_panel.tsv")

ILLUMINA_MANIFEST = Path(
    "/Users/bgadmin/Downloads/"
    "infinium-methylationepic-v-1-0-b5-manifest-file.csv"
)

# Télécharger une fois :
# wget https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz
NCBI_GENE_INFO = Path("/Users/bgadmin/Downloads/Homo_sapiens.gene_info")

GDC_RNA_INPUT    = Path("cohort_matrices_2")
GDC_METH_INPUT   = Path("cohort_methylation_matrices")
#CBIO_BASE_INPUT  = Path("cbioportal_data")

GDC_OUTPUT       = Path("cohort_matrices_cleaned")
#CBIO_BASE_OUTPUT = Path("cbioportal_cleaned")

# ─────────────────────────────────────────────────────────────
# LOAD GENE PANEL
# ─────────────────────────────────────────────────────────────

def load_gene_panel(filepath: Path):
    """
    Retourne :
      ensembl            : set Ensembl IDs avec version  (ENSG00000008130.15)
      ensembl_no_version : set Ensembl IDs sans version  (ENSG00000008130)
      symbols            : set Hugo symbols               (NADK, GNB1 ...)
      entrez             : dict {entrez_str -> hugo_symbol}
                           construit depuis NCBI gene_info, filtré sur le panel.
                           Couvre tous les Entrez IDs correspondant aux gènes du panel.
    """

    log.info(f"Chargement panel : {filepath}")
    df = pd.read_csv(filepath, sep="\t")
    log.info(f"Colonnes panel   : {df.columns.tolist()}")

    ensembl = set(df["gene_id"].dropna().astype(str).str.strip())
    ensembl_no_version = {e.split(".")[0] for e in ensembl}
    symbols = set(df["gene_symbol"].dropna().astype(str).str.strip())

    log.info(f"Ensembl IDs  : {len(ensembl)}")
    log.info(f"Gene symbols : {len(symbols)}")
    log.info(f"Exemples     : {list(symbols)[:8]}")

    entrez: dict[str, str] = {}

    if NCBI_GENE_INFO.exists():
        log.info(f"Chargement NCBI gene_info : {NCBI_GENE_INFO}")
        try:
            gene_info = pd.read_csv(
                NCBI_GENE_INFO,
                sep="\t",
                usecols=["GeneID", "Symbol"],
                dtype=str,

            )
            gene_info["GeneID"] = gene_info["GeneID"].str.strip()
            gene_info["Symbol"] = gene_info["Symbol"].str.strip()

            # Ne garder que les gènes du panel
            panel_rows = gene_info[gene_info["Symbol"].isin(symbols)]
            entrez = dict(zip(panel_rows["GeneID"], panel_rows["Symbol"]))

            log.info(f"Entrez mappings (panel) : {len(entrez)}")
            log.info(f"Exemple mappings        : {list(entrez.items())[:10]}")

            missing = symbols - set(entrez.values())
            log.info(f"Symboles couverts : {len(symbols)-len(missing)}/{len(symbols)}")
            if missing:
                log.warning(f"Sans Entrez ({len(missing)}) : {sorted(missing)[:20]}")

        except Exception as e:
            log.error(f"Erreur NCBI gene_info : {e}")

    else:
        log.warning(
            f"NCBI gene_info introuvable : {NCBI_GENE_INFO}\n"
            "  Télécharge-le : wget https://ftp.ncbi.nlm.nih.gov/gene/DATA/"
            "GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz"
        )
        log.info(f"Fallback MyGeneInfo pour {len(symbols)} symboles...")
        try:
            sym_list = sorted(symbols)
            for i in range(0, len(sym_list), 1000):
                chunk = sym_list[i : i + 1000]
                r = requests.post(
                    "https://mygene.info/v3/querymany",
                    data={
                        "q": ",".join(chunk),
                        "scopes": "symbol",
                        "fields": "symbol,entrezgene",
                        "species": "human",
                    },
                    timeout=60,
                )
                if r.status_code != 200:
                    log.warning(f"MyGeneInfo status={r.status_code}")
                    continue
                for g in r.json():
                    if isinstance(g, dict) and "entrezgene" in g and "symbol" in g:
                        entrez[str(g["entrezgene"]).strip()] = str(g["symbol"]).strip()
            log.info(f"MyGeneInfo : {len(entrez)} mappings")
        except Exception as e:
            log.error(f"MyGeneInfo inaccessible : {e}")

    return ensembl, ensembl_no_version, symbols, entrez


# ─────────────────────────────────────────────────────────────
# METHYLATION MAPPING
# ─────────────────────────────────────────────────────────────

def get_or_create_probe_mapping(symbols: set) -> dict | None:

    import hashlib
    h     = hashlib.md5(",".join(sorted(symbols)).encode()).hexdigest()[:8]
    cache = Path(f"probe_to_gene_mapping_{h}.csv")

    for old in Path(".").glob("probe_to_gene_mapping_*.csv"):
        if old != cache:
            old.unlink()

    if cache.exists():
        log.info(f"Cache méthylation : {cache}")
        df = pd.read_csv(cache)
        return dict(zip(df["Name"], df["Gene_Symbol"]))

    if not ILLUMINA_MANIFEST.exists():
        log.error(f"Manifest Illumina introuvable : {ILLUMINA_MANIFEST}")
        return None

    log.info("Lecture manifest Illumina EPIC...")
    try:
        df = pd.read_csv(
            ILLUMINA_MANIFEST, skiprows=7,
            usecols=["Name", "UCSC_RefGene_Name"], low_memory=False,
        )
    except ValueError:
        df = pd.read_csv(
            ILLUMINA_MANIFEST, skiprows=7,
            usecols=["IlmnID", "UCSC_RefGene_Name"], low_memory=False,
        )
        df.rename(columns={"IlmnID": "Name"}, inplace=True)

    df = df.dropna(subset=["UCSC_RefGene_Name"])
    df["Gene_Symbol"] = df["UCSC_RefGene_Name"].apply(
        lambda x: next(
            (g.strip() for g in str(x).split(";") if g.strip() in symbols), None
        )
    )
    df = df.dropna(subset=["Gene_Symbol"])
    df[["Name", "Gene_Symbol"]].to_csv(cache, index=False)
    log.info(f"Mapping méthylation : {len(df)} sondes → {cache}")
    return dict(zip(df["Name"], df["Gene_Symbol"]))


# ─────────────────────────────────────────────────────────────
# FILTER BY PANEL
# ─────────────────────────────────────────────────────────────

def filter_by_panel(
    matrix: pd.DataFrame,
    ensembl: set,
    ensembl_no_version: set,
    symbols: set,
    entrez: dict,
    name: str,
) -> pd.DataFrame | None:

    matrix.index = (
        matrix.index.astype(str).str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    idx    = set(matrix.index)
    sample = list(idx)[:5]
    log.info(f"  Exemple IDs : {sample}")

    # miRNA
    n_mirna = sum(1 for s in sample if s.lower().startswith(("hsa-", "mir-")))
    if n_mirna >= 3:
        log.warning(f"  [SKIP miRNA] {name}")
        return None

    # Stats
    log.info(
        f"  Match → Entrez:{len(idx & set(entrez.keys()))} "
        f"| Symbol:{len(idx & symbols)} "
        f"| SymbolUpper:{len({s.upper() for s in idx} & {s.upper() for s in symbols})} "
        f"| Ensembl:{len(idx & ensembl)} "
        f"| EnsemblNoVer:{len({e.split('.')[0] for e in idx} & ensembl_no_version)}"
    )

    # 1. Entrez
    matched = idx & set(entrez.keys())
    if matched:
        log.info(f"  [Entrez] {len(matched)} gènes retenus")
        m = matrix.loc[list(matched)].copy()
        m.index = m.index.map(entrez)
        return m

    # 2. Symbol exact
    matched = idx & symbols
    if matched:
        log.info(f"  [Symbol] {len(matched)} gènes retenus")
        return matrix.loc[list(matched)].copy()

    # 3. Symbol normalisé (casse)
    upper_panel = {s.upper(): s for s in symbols}
    upper_idx   = {s.upper(): s for s in idx}
    matched_up  = set(upper_idx) & set(upper_panel)
    if matched_up:
        log.info(f"  [Symbol normalisé] {len(matched_up)} gènes retenus")
        m = matrix.loc[[upper_idx[u] for u in matched_up]].copy()
        m.index = [upper_panel[u] for u in matched_up]
        return m

    # 4. Ensembl avec version
    matched = idx & ensembl
    if matched:
        log.info(f"  [Ensembl] {len(matched)} gènes retenus")
        return matrix.loc[list(matched)].copy()

    # 5. Ensembl sans version
    idx_nv = {e.split(".")[0]: e for e in idx}
    matched_nv = set(idx_nv) & ensembl_no_version
    if matched_nv:
        log.info(f"  [Ensembl sans version] {len(matched_nv)} gènes retenus")
        return matrix.loc[[idx_nv[e] for e in matched_nv]].copy()

    log.warning(f"  [REJET] {name} — aucun match panel | exemples : {sample}")
    return None


# ─────────────────────────────────────────────────────────────
# PROCESS MATRIX
# ─────────────────────────────────────────────────────────────

def process_matrix(
    path: Path,
    ensembl: set,
    ensembl_no_version: set,
    symbols: set,
    entrez: dict,
    output: Path,
    source: str,
    probe_dict: dict | None = None,
) -> None:

    log.info(f"Traitement : {path.stem}")

    try:
        if source == "cBioPortal":
            matrix = pd.read_csv(path, low_memory=False)
        else:
            matrix = pd.read_csv(path, index_col=0, low_memory=False)
    except Exception as e:
        log.error(f"  Lecture ERROR : {e}")
        return

    # ── cBioPortal ───────────────────────────────────────────────────────────
    if source == "cBioPortal":

        matrix.columns = [str(c).strip() for c in matrix.columns]

        symbol_candidates = ["Hugo_Symbol", "HUGO_SYMBOL", "gene_symbol", "Gene_Symbol", "symbol"]
        entrez_candidates = ["Entrez_Gene_Id", "entrez_gene_id", "entrez_id"]

        sym_col = next((c for c in symbol_candidates if c in matrix.columns), None)
        ent_col = next((c for c in entrez_candidates if c in matrix.columns), None)

        chosen_col = None

        if sym_col:
            # Détecter si la colonne contient des Entrez IDs (numériques) ou des symbols
            sample_vals = matrix[sym_col].dropna().astype(str).str.strip().head(20).tolist()
            n_numeric   = sum(1 for v in sample_vals if v.replace(".", "").lstrip("-").isdigit())
            kind        = "Entrez (colonne mal nommée)" if n_numeric >= len(sample_vals) // 2 else "Symbol"
            log.info(f"  Index cBioPortal → '{sym_col}' ({kind})")
            chosen_col = sym_col

        elif ent_col:
            log.info(f"  Index cBioPortal → '{ent_col}' (Entrez)")
            chosen_col = ent_col

        else:
            log.warning(f"  Aucune colonne détectée | colonnes : {list(matrix.columns[:15])}")

        if chosen_col:
            matrix = matrix.set_index(chosen_col)

        # Drop metadata
        metadata_cols = [
            "Entrez_Gene_Id", "entrez_gene_id", "entrez_id",
            "Hugo_Symbol", "HUGO_SYMBOL", "gene_symbol", "Gene_Symbol", "symbol",
            "DESCRIPTION", "Description", "NAME", "Name",
        ]
        matrix.drop(columns=[c for c in matrix.columns if c in metadata_cols],
                    errors="ignore", inplace=True)

        matrix = matrix[matrix.index.notna()]
        matrix.index = (
            matrix.index.astype(str).str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )

    # ── GDC ──────────────────────────────────────────────────────────────────
    if source == "GDC":
        matrix.columns = [str(c)[:15] for c in matrix.columns]
        if matrix.columns.duplicated().any():
            matrix = matrix.T.groupby(matrix.columns).mean().T

    # ── Méthylation : sondes → gènes ─────────────────────────────────────────
    try:
        if str(matrix.index[0]).startswith(("cg", "ch")):
            if not probe_dict:
                log.warning(f"  [SKIP méthylation] probe_dict absent")
                return
            mapped = matrix.index.map(probe_dict)
            matrix = matrix.loc[mapped.notna()].copy()
            matrix.index = mapped[mapped.notna()]
            matrix = matrix.groupby(matrix.index).mean()
            log.info(f"  Méthylation : {len(matrix)} gènes après mapping")
    except Exception:
        pass

    # ── Filtrage panel ────────────────────────────────────────────────────────
    matrix = filter_by_panel(
        matrix, ensembl, ensembl_no_version, symbols, entrez, path.stem
    )
    if matrix is None:
        return

    # ── Dédoublonnage ─────────────────────────────────────────────────────────
    if matrix.index.duplicated().any():
        n = len(matrix)
        matrix = matrix.groupby(matrix.index).mean()
        log.info(f"  Dédoublonnage : {n} → {len(matrix)} gènes")

    matrix = matrix.apply(pd.to_numeric, errors="coerce")

    output.parent.mkdir(parents=True, exist_ok=True)
    matrix.sort_index().to_csv(output)
    log.info(f"  ✓ {len(matrix)} gènes → {output}")


# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────

def run():

    ensembl, ensembl_no_version, symbols, entrez = load_gene_panel(GENE_PANEL_FILE)

    print("\n── TEST ENTREZ ──")
    for tid in ["523", "5198", "1666", "9659", "3371", "6628"]:
        print(f"  {tid} → {entrez.get(tid, 'ABSENT')}")
    print(f"  Total : {len(entrez)}\n")

    probe_dict = get_or_create_probe_mapping(symbols)

    for folder in [GDC_OUTPUT]:
        if folder.exists():
            shutil.rmtree(folder)

    # STEP 1 : GDC
    log.info("\n" + "=" * 55 + "\nSTEP 1 : GDC / TCGA\n" + "=" * 55)
    for folder, out_sub, p_dict in [
        (GDC_RNA_INPUT,  "rnaseq",      None),
        (GDC_METH_INPUT, "methylation", probe_dict),
    ]:
        if not folder.exists():
            log.warning(f"Dossier introuvable : {folder}")
            continue
        for f in sorted(folder.glob("*.csv")):
            process_matrix(
                f, ensembl, ensembl_no_version, symbols, entrez,
                GDC_OUTPUT / out_sub / f"{f.stem}_clean.csv",
                "GDC", p_dict,
            )

    # STEP 2 : cBioPortal
    '''log.info("\n" + "=" * 55 + "\nSTEP 2 : cBioPortal\n" + "=" * 55)
    for omic in ["rnaseq", "methylation"]:
        src = CBIO_BASE_INPUT / omic
        if not src.exists():
            log.warning(f"Dossier introuvable : {src}")
            continue
        for f in sorted(src.glob("*.csv")):
            process_matrix(
                f, ensembl, ensembl_no_version, symbols, entrez,
                CBIO_BASE_OUTPUT / omic / f"{f.stem}_clean.csv",
                "cBioPortal",
                probe_dict if omic == "methylation" else None,
            )'''

    log.info("\n✅ TERMINÉ")


if __name__ == "__main__":
    run()