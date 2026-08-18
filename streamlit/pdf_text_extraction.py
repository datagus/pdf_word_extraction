from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pdf_word_extraction import (  # noqa: E402
    DEFAULT_EXCLUDED_WORDS,
    DEFAULT_POS_TAGS,
    DEFAULT_POS_GROUPS,
    ExtractionOptions,
    csv_to_tuple,
    extract_text_from_pdf,
    extract_words_from_text,
    pos_tags_from_groups,
    process_file as _process_file,
)


def text_from_pdf(filepath: str, categories: tuple = ()) -> list[str]:
    del categories
    return [extract_text_from_pdf(filepath)]


def tokenize(text: str) -> list[str]:
    from pdf_word_extraction import tokenize_words

    return tokenize_words(text)


def tag_tokens(tokens: str | list[str]) -> list[tuple[str, str]]:
    from nltk import pos_tag

    if isinstance(tokens, str):
        tokens = tokenize(tokens)
    return pos_tag(tokens)


def filter_tokens(
    tokens: list[str] | list[tuple[str, str]],
    additionalFilterTokens: tuple[str, ...] = DEFAULT_EXCLUDED_WORDS,
    prefixFilterTokens: tuple[str, ...] = (),
    validTags: tuple[str, ...] = DEFAULT_POS_TAGS,
    validTokens: tuple[str, ...] = (),
    lemmatize: bool = True,
    protectedWords: tuple[str, ...] = (),
    protectedPrefixes: tuple[str, ...] = (),
) -> list[str]:
    if not tokens:
        return []
    if isinstance(tokens[0], tuple):
        text = " ".join(token for token, _tag in tokens)  # type: ignore[misc]
    else:
        text = " ".join(tokens)  # type: ignore[arg-type]
    options = ExtractionOptions(
        pos_tags=tuple(validTags),
        excluded_words=tuple(additionalFilterTokens),
        excluded_prefixes=tuple(prefixFilterTokens),
        protected_words=tuple(protectedWords),
        protected_prefixes=tuple(protectedPrefixes),
        include_only_words=tuple(validTokens),
        lemmatize=lemmatize,
    )
    return extract_words_from_text(text, options)


def process_file(
    filepath: str,
    additionalFilterTokens: tuple[str, ...] = DEFAULT_EXCLUDED_WORDS,
    prefixFilterTokens: tuple[str, ...] = (),
    validTags: tuple[str, ...] = DEFAULT_POS_TAGS,
    validTokens: tuple[str, ...] = (),
    lemmatize: bool = True,
    deduplicate: bool = True,
    protectedWords: tuple[str, ...] = (),
    protectedPrefixes: tuple[str, ...] = (),
) -> list[str]:
    options = ExtractionOptions(
        pos_tags=tuple(validTags),
        excluded_words=tuple(additionalFilterTokens),
        excluded_prefixes=tuple(prefixFilterTokens),
        protected_words=tuple(protectedWords),
        protected_prefixes=tuple(protectedPrefixes),
        include_only_words=tuple(validTokens),
        lemmatize=lemmatize,
        deduplicate=deduplicate,
    )
    return _process_file(filepath, options)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract POS-filtered words from PDF files.")
    parser.add_argument("input_path", help="Path to a PDF file or folder of PDF files.")
    parser.add_argument("-o", "--output-path", help="Output text file or folder.")
    parser.add_argument("-l", "--lemmatize", action="store_true", default=True)
    parser.add_argument("--no-lemmatize", action="store_false", dest="lemmatize")
    parser.add_argument("--no-deduplicate", action="store_false", dest="deduplicate")
    parser.add_argument("-ft", "--filter-tokens", default=",".join(DEFAULT_EXCLUDED_WORDS))
    parser.add_argument("--prefix-filter-tokens", default="")
    parser.add_argument("-pt", "--protected-tokens", default="")
    parser.add_argument("--protected-prefixes", default="")
    parser.add_argument("-vto", "--valid-tokens", default="")
    parser.add_argument("-pg", "--pos-groups", default=",".join(DEFAULT_POS_GROUPS))
    parser.add_argument("-vta", "--valid-tags", default="")
    return parser.parse_args()


def _write_words(pdf_path: str, output_path: str | None, args: argparse.Namespace) -> bool:
    try:
        words = process_file(
            pdf_path,
            additionalFilterTokens=csv_to_tuple(args.filter_tokens),
            prefixFilterTokens=csv_to_tuple(args.prefix_filter_tokens),
            validTags=csv_to_tuple(args.valid_tags) or pos_tags_from_groups(csv_to_tuple(args.pos_groups)),
            validTokens=csv_to_tuple(args.valid_tokens),
            protectedWords=csv_to_tuple(args.protected_tokens),
            protectedPrefixes=csv_to_tuple(args.protected_prefixes),
            lemmatize=args.lemmatize,
            deduplicate=args.deduplicate,
        )
    except Exception as exc:
        print(f"{Path(pdf_path).name}: ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return False

    path = Path(pdf_path)
    if output_path is None:
        destination = Path(f"{pdf_path}.txt")
    elif Path(output_path).is_dir():
        destination = Path(output_path) / f"{path.stem}.txt"
    else:
        destination = Path(output_path)
    destination.write_text(" ".join(words), encoding="utf-8")
    print(f"{path.name}: {len(words)} words -> {destination}")
    return True


if __name__ == "__main__":
    args = _parse_args()
    input_path = Path(args.input_path)
    if input_path.is_file():
        if not _write_words(str(input_path), args.output_path, args):
            raise SystemExit(1)
    elif input_path.is_dir():
        output_dir = Path(args.output_path or input_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        failures = 0
        for pdf_file in glob.glob(str(input_path / "*.pdf")):
            if not _write_words(pdf_file, str(output_dir), args):
                failures += 1
        if failures:
            raise SystemExit(1)
    else:
        print(f"File or folder not found: {input_path}")
        raise SystemExit(1)
