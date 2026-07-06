import os
import sys
import glob
import argparse
from typing import List
from pathlib import Path

from unstructured.partition.pdf import partition_pdf

import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, words, wordnet


def nltk_download(resource, path):
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource)


DATA = Path.home() / "nltk_data"

for pkg in [
    "punkt",
    "punkt_tab",
    "averaged_perceptron_tagger_eng",
    "maxent_ne_chunker",
    "maxent_ne_chunker_tab",
    "words",
    "stopwords",
    "wordnet",
]:
    nltk.download(pkg, download_dir=str(DATA), quiet=True)


def text_from_pdf(filepath: str, categories: tuple = ("NarrativeText")) -> List[str]:
    """
    Reads the pdf document at the given path, partitions it and returns the text for each paragraph.

    Inputs:
        filepath:   str:        Path to the PDF file
        categories: tuple:      Categories to parse from the file (Options: Title, NarrativeText, ListItem). Default: NarrativeText

    Output:
        paragraphs: List[str]:  The paragraphs found in the file, each as text.
    """
    elements = partition_pdf(
        filename=filepath,  # mandatory
        strategy="auto",    # mandatory to use ``hi_res`` strategy
        # extract_images_in_pdf=True,   # mandatory to set as ``True``
        # extract_image_block_types=["Image", "Table"],   # optional
        # infer_table_structure = True
        # model_name ="yolox"
    )

    paragraphs = [e.text for e in elements if e.category in categories]

    return paragraphs


def tokenize(text: str) -> List[str]:
    return word_tokenize(text)


def tag_tokens(tokens: str | list) -> List[tuple[str, str]]:
    input = tokens
    if isinstance(tokens, str):
        input = tokenize(tokens)

    return pos_tag(input)


def lemmatize_token(token: str, pos=None) -> str:
    lemmatizer = WordNetLemmatizer()

    if isinstance(pos, str) and pos.lower() in "nvars":
        return lemmatizer.lemmatize(token.lower(), pos=pos.lower())
    else:
        return lemmatizer.lemmatize(token.lower())


def filter_tokens(tokens: list, additionalFilterTokens: tuple = ('et', 'al', 'etc', 'ie', 'ISSN', 'http', 'https'), validTags: tuple = ('NN', 'NNS', 'JJ'), validTokens: tuple = (), lemmatize: bool = False) -> list[str]:
    """
    Filters the given tokens.
    Inputs:
        additionalFilterTokens: Tuple containing additional stopwords to remove.
        validTags:              Set of tags to include. Any token with a tag not in validTags will be ignored. If empty set is passed, all is included.
        validTokens:            Set of tokens to include. Any token not in validTokens will be ignored. If empty set is passed, all is included.
        lemmatize:              Whether or not to lemmatize the input before filtering.
    Outputs:
        List[str] contining filtered tokens
    """
    if len(tokens) == 0:
        return []

    if type(tokens[0]) == tuple:
        taggedInput = tokens
    elif type(tokens[0]) == str:
        # if only tokens are passed -> tag
        taggedInput = tag_tokens(tokens)
    else:
        raise TypeError(
            f"Invalid type. Pass list[str] or list[tuple[str, str]]"
        )

    if lemmatize:
        taggedInput = [
            (lemmatize_token(t[0], t[1]), t[1])
            for t in taggedInput
        ]

    filterWords = set()

    # filter stopwords
    filterWords.update(stopwords.words('english'))

    # filter named entities
    for chunk in nltk.ne_chunk(taggedInput):
        if hasattr(chunk, "label"):
            filterWords.update(c[0] for c in chunk)

    # user defined filter words
    filterWords.update(additionalFilterTokens)

    # all lower case
    filterWords = {w.lower() for w in filterWords}

    def filter_token(taggedToken: tuple[str, str]) -> bool:
        token, tag = taggedToken
        return (
            token.isalpha() and
            len(token) > 2 and
            (tag in validTags or not validTags) and
            (token in validTokens or not validTokens) and
            token.lower() not in filterWords
        )

    return [token for token, _ in list(filter(filter_token, taggedInput))]


def process_file(filepath: str, additionalFilterTokens: tuple = ('et', 'al', 'etc', 'ie', 'ISSN', 'http', 'https'), validTags: tuple = ('NN', 'NNS', 'JJ'), validTokens: tuple = (), lemmatize: bool = False) -> List[str]:
    """
    Process the PDF at filepath. Includes tokenization, tagging and filtering.
    """
    allTokens: list[str] = []
    paragraphs = text_from_pdf(filepath)
    for p in paragraphs:
        tokens = tokenize(p)
        tokensTagged = tag_tokens(tokens)
        tokensFiltered = filter_tokens(
            tokensTagged,
            additionalFilterTokens=additionalFilterTokens,
            validTags=validTags,
            validTokens=validTokens,
            lemmatize=lemmatize
        )
        # print(f"{len(tokensFiltered)} filtered tokens")
        allTokens.extend(tokensFiltered)

    print(f"{len(allTokens)} tokens total")

    return allTokens


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Extract text from a PDF file."
    )
    parser.add_argument(
        "input_path",
        help="Path to the input PDF file."
    )
    parser.add_argument(
        '-o', '--output-path',
        required=False,
        help="Path to save the extracted text. If not provided, it will be stored alongside the input file with an added .txt suffix."
    )
    parser.add_argument(
        '-l', '--lemmatize',
        action='store_true',
        default=False,
        help="Whether to lemmatize the extracted tokens. Default is False."
    )
    parser.add_argument(
        '-ft', '--filter-tokens',
        default="et,al,etc,ie,ISSN,http,https",
        type=lambda x: set(x.split(",") if x else set()),
        help="Comma-separated list of tokens to filter out. Default is 'et,al,etc,ie,ISSN,http,https'."
    )
    parser.add_argument(
        '-vto', '--valid-tokens',
        default="",
        type=lambda x: set(x.split(",") if x else set()),
        help="Comma-separated list of valid tokens. Default is an empty set."
    )
    parser.add_argument(
        '-vta', '--valid-tags',
        default="NN,NNS,JJ",
        type=lambda x: set(x.split(",") if x else set()),
        help="Comma-separated list of valid POS tags. Default is 'NN,NNS,JJ'."
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    inputpath: str = args.input_path
    outputpath = args.output_path

    if os.path.isfile(inputpath):
        if not outputpath:
            outputpath = inputpath + '.txt'
        if os.path.isdir(outputpath):
            outputpath = outputpath + os.path.basename(inputpath) + '.txt'
        result_tokens = process_file(
            inputpath,
            outputpath,
            additionalFilterTokens=args.filter_tokens,
            validTags=args.valid_tags,
            validTokens=args.valid_tokens,
            lemmatize=args.lemmatize
        )
        with open(outputpath, 'wt') as f:
            f.write(' '.join(result_tokens))
    elif os.path.isdir(inputpath):
        files = glob.glob(f"{inputpath.rstrip('/')}/*.pdf")
        if not outputpath:
            outputpath = inputpath
        for f in files:
            result_tokens = process_file(
                f,
                additionalFilterTokens=args.filter_tokens,
                validTags=args.valid_tags,
                validTokens=args.valid_tokens,
                lemmatize=args.lemmatize
            )
            with open(f"{outputpath}/{os.path.basename(f)}.txt", 'wt') as f:
                f.write(' '.join(result_tokens))
    else:
        print(f"File / Folder {inputpath} not found.")
