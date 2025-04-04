# 📘 Thesis Citation Analyzer and Metadata Enrichment Tool

This script enriches and validates bibliographic entries in a LaTeX thesis project by automating metadata retrieval and comparing cited text with publication abstracts. It uses APIs like Crossref and PubMed, matches statements to abstracts using both TF-IDF and biomedical embeddings (BioBERT), and flags weakly supported claims.

## 📄 Overview

This Python script analyzes a LaTeX thesis manuscript and its corresponding BibTeX library. Its primary goals are:

1. **Metadata Enrichment:** To fetch and add missing metadata (DOI, PMID, Abstract, Reference Count, Citation Count) to the BibTeX entries that are cited in the LaTeX document. It uses APIs like Crossref, NCBI Eutils, and OpenCitations, and falls back to web scraping in some cases.  
2. **Statement-Abstract Similarity Analysis:** To extract sentences (statements) preceding `\cite{...}` commands in the LaTeX file and compare their semantic similarity to the abstracts of the cited references. It uses various methods, including simple word overlap, TF-IDF, and advanced sentence embeddings (BERT, BioBERT).  
3. **Reporting & Cleanup:** To identify potential issues like missing references, duplicate entries (by DOI), and references lacking key metadata. It outputs an enriched BibTeX file and a detailed CSV report of the similarity analysis.

## ✨ Features

* **BibTeX Parsing:** Reads and parses `.bib` files using regular expressions.
* **LaTeX Citation Extraction:** Identifies `\cite{...}` commands and attempts to extract the preceding statement.
* **Metadata Fetching:**
  * Retrieves DOIs via Crossref API (using title/author).
  * Retrieves PMIDs via NCBI Eutils API (using DOI or title).
  * Retrieves Abstracts via PubMed (PMID) or DOI landing pages (scraping).
  * Retrieves Reference Counts via Crossref API.
  * Retrieves Citation Counts via OpenCitations API.
* **Web Caching:** Caches downloaded web content (`HTML`, `JSON`) locally to speed up subsequent runs and reduce API load.
* **Text Cleaning:** Cleans text extracted from BibTeX and LaTeX, removing common artifacts.
* **Similarity Scoring:** Calculates statement-abstract similarity using:
  * Simple Word Overlap
  * TF-IDF Cosine Similarity
  * BERT Sentence Embeddings (`paraphrase-MiniLM-L6-v2`) Cosine Similarity
  * BioBERT Sentence Embeddings (`pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb`) Cosine Similarity
* **Duplicate Detection:** Identifies BibTeX entries with identical DOIs.
* **Interactive Updates:** Prompts the user to resolve discrepancies if fetched metadata (DOI, PMID, Abstract) differs from existing values in the BibTeX file. Original values can be backed up.
* **Output Generation:**
  * Creates a new, enriched BibTeX file containing only cited references (`*_Fred.bib`).
  * Generates a comprehensive CSV report (`statement_vs_abstract_match_scores.csv`) with similarity scores, statements, abstracts, and metadata.
  * Produces a histogram plot (`hist_bib_years.png`) showing the distribution of publication years for cited references.

## 🧭 Workflow

The script generally follows these steps (when the main execution block is enabled):

1. **Load Data:** Reads the specified `.bib` and `.tex` files.
2. **Extract Citations:** Parses the `.tex` file to find citations and their preceding statements.
3. **Identify Cited Subset:** Filters the full bibliography to include only those entries cited in the manuscript.
4. **Fetch & Enrich:** Iteratively fetches DOIs, PMIDs, reference/citation counts, and abstracts for the cited subset, filling in missing data.
5. **Handle Discrepancies:** If fetched data conflicts with existing BibTeX data, prompts the user for confirmation before updating.
6. **Clean Data:** Applies text cleaning routines to BibTeX fields.
7. **Analyze Similarity:** Calculates various similarity scores between LaTeX statements and fetched abstracts.
8. **Generate Reports:**
   * Saves the similarity scores and associated data to `statement_vs_abstract_match_scores.csv`.
   * Identifies and prints lists of entries missing abstracts, citation counts, or DOIs.
   * Identifies and prints lists of duplicate entries based on DOI.
   * Generates `hist_bib_years.png`.
9. **Save Enriched BibTeX:** Writes the processed subset of BibTeX entries (with added metadata) to a new file (e.g., `original_filename_Fred.bib`).

## 📦 Requirements

* Python 3.x
* Required Python packages:

    ```bash
    pip install crossref_commons tabulate tqdm pandas scikit-learn sentence-transformers matplotlib dynamic_multiprocessing
    ```

    *(Note: `dynamic_multiprocessing` might be a custom library or require specific installation steps if not on PyPI. If it's custom, it should be included alongside the script.)*

* A suitable backend for Matplotlib if the default doesn't work (e.g., `pip install PyQt5` for the "Qt5Agg" backend used in the script).

## 🛠 Setup

1. **Place Files:** Put the script (`your_script_name.py`), your LaTeX manuscript (`.tex`), and your BibTeX library (`.bib`) in the same directory, or update the file paths in the script.
2. **Install Dependencies:** Run the `pip install` command listed above.
3. **Email for APIs:** The script uses a hardcoded email address (`frederik.bay2@gmail.com`) for API politeness policies (Crossref, NCBI). **It is highly recommended to replace this with your own email address** within the script code (search for `mailto=` and `email=`).
4. **Enable Main Block:** The main execution logic is currently inside an `if False and __name__ == '__main__':` block. Change `False` to `True` to enable it.
5. **Update Filenames:** Modify the `bibtex_filename` and `latex_filename` variables within the main block to match your input files.

## ▶️ Usage

1. Ensure setup is complete.
2. Run the script from your terminal:

    ```bash
    python your_script_name.py
    ```

3. The script will print progress bars and status messages.
4. If discrepancies are found between existing and fetched metadata, you will be prompted interactively in the console to choose which entries to update.
5. Upon completion, the following outputs will be generated in the script's directory:
   * `your_bib_filename_Fred.bib`: The enriched BibTeX file.
   * `statement_vs_abstract_match_scores.csv`: The similarity analysis report.
   * `hist_bib_years.png`: The publication year histogram.
   * `cached_urls/`: A directory containing cached web requests (will be created if it doesn't exist).

## ⚙️ Configuration

* **Input/Output Files:** Change `bibtex_filename`, `latex_filename` in the main block. The output BibTeX name is derived from the input name. CSV and PNG filenames are hardcoded.
* **API Email:** Change the email address used for APIs (essential).
* **Embedding Models:** The `sentence_transformers` models used for BERT/BioBERT scoring can be changed in `get_BERT_scores` and `get_BioBERT_scores`.
* **Multiprocessing:** The `run_go` function has a `multiprocessing` flag (currently unused in the main block example). The `get_...` functions run sequentially.
* **Caching:** Caching is enabled by default in `get_html_from_url`.


