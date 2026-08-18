# PDF Word Extraction

Extract clean, POS-filtered word lists from batches of PDF files. The Streamlit app, marimo app, and command-line wrapper all use the shared `pdf_word_extraction.py` module in the repository root.

## Install

From the repository root:

```bash
python3 -m pip install -r streamlit/requirements.txt
```

The first run downloads the required NLTK tagger, stopword, and WordNet data if it is not already available.

## Run the Streamlit app

```bash
streamlit run streamlit/gui.py
```

## Run the marimo app

```bash
marimo run extraction.py
```

## What the apps do

1. Upload one or more PDFs by drag-and-drop or file browsing.
2. Choose the parts of speech to keep, such as nouns, adjectives, verbs, adverbs, or proper nouns.
3. Optionally add advanced NLTK POS tags.
4. Exclude English stopwords, exact words, or words that start with selected prefixes.
5. Optionally protect exact words or prefixes so they remain included even when they match an exclusion.
6. Optionally use an allow-list with "Only include these words".
7. Download a ZIP containing either:
   - one `.txt` word list per PDF, or
   - one `.csv` frequency table per PDF.

For word-list downloads, duplicate removal is controlled by the "Remove duplicates" checkbox. For frequency-table downloads, duplicates are always counted so repeated terms are represented as `word,count` rows instead of repeated text.

If one uploaded file cannot be parsed as a PDF, the rest of the batch still runs and the ZIP includes an `_errors.txt` report.

## Command-line usage

```bash
python3 streamlit/pdf_text_extraction.py path/to/file-or-folder -o output-folder
```

Useful options:

```text
--pos-groups "Nouns,Adjectives,Verbs"
--valid-tags "NN,NNS,JJ,VBG"
--filter-tokens "et,al,doi"
--prefix-filter-tokens "pre,anti"
--protected-tokens "press"
--protected-prefixes "prefrontal"
--valid-tokens "year,study,result"
--no-lemmatize
--no-deduplicate
```

`--valid-tags` overrides the friendly `--pos-groups` selection when provided.
