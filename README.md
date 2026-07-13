# PDF Word Extraction

This repository contains two matching interfaces for extracting words from PDFs:

- `extraction.py`: a marimo app.
- `streamlit/gui.py`: a Streamlit app.

Both apps use the same shared extraction module, `pdf_word_extraction.py`, so uploads processed through either interface produce the same output for the same options.

## Install

```bash
python3 -m pip install -r streamlit/requirements.txt
```

## Run

Marimo:

```bash
marimo run extraction.py
```

Streamlit:

```bash
streamlit run streamlit/gui.py
```

Command line:

```bash
python3 streamlit/pdf_text_extraction.py path/to/file-or-folder -o output-folder
```

## Extraction Options

The apps let users:

- upload multiple PDFs;
- choose POS groups such as nouns, adjectives, verbs, adverbs, or proper nouns;
- add advanced NLTK POS tags when needed;
- exclude English stopwords, exact words, or word prefixes;
- protect exact words or prefixes from those exclusions;
- optionally allow only a supplied list of words;
- choose between word-list `.txt` files and frequency-count `.csv` files;
- download all generated files as a ZIP archive.

If an uploaded file cannot be parsed as a PDF, the apps keep processing the rest of the batch and add an `_errors.txt` report to the ZIP.

The extraction logic cleans common PDF text issues before tagging, including line-break hyphenation, URLs, emails, DOI fragments, `(cid:...)` artifacts, publisher-note fragments, accent-split words, punctuation, and duplicate output when deduplication is enabled.
