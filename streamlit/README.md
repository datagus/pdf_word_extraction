# Python PDF Text Extraction


## Prerequisites
```bash
pip install -r requirements.txt
```

## Usage
```bash
$ python3 pdf_text_extraction.py -h
usage: pdf_text_extraction.py [-h] [-o OUTPUT_PATH] [-l] [-ft FILTER_TOKENS] [-vto VALID_TOKENS] [-vta VALID_TAGS] input_path

Extract text from a PDF file.

positional arguments:
  input_path            Path to the input PDF file.

options:
  -h, --help            show this help message and exit
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        Path to save the extracted text. If not provided, it will be stored alongside the input file with an added .txt suffix.
  -l, --lemmatize       Whether to lemmatize the extracted tokens. Default is False.
  -ft FILTER_TOKENS, --filter-tokens FILTER_TOKENS
                        Comma-separated list of tokens to filter out. Default is 'et,al,etc,ie,ISSN,http,https'.
  -vto VALID_TOKENS, --valid-tokens VALID_TOKENS
                        Comma-separated list of valid tokens. Default is an empty set.
  -vta VALID_TAGS, --valid-tags VALID_TAGS
                        Comma-separated list of valid POS tags. Default is 'NN,NNS,JJ'.
```


## GUI
```bash
streamlit run gui.py
```
