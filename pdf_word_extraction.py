from __future__ import annotations

import io
import csv
import re
import unicodedata
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import nltk
import pdfplumber
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pypdf import PdfReader


DEFAULT_EXCLUDED_WORDS = (
    "et",
    "al",
    "etc",
    "ie",
    "issn",
    "http",
    "https",
    "www",
    "doi",
    "org",
    "com",
    "edu",
    "gmail",
    "mail",
    "cid",
)
POS_PRESETS = {
    "Nouns": ("NN", "NNS"),
    "Proper nouns": ("NNP", "NNPS"),
    "Adjectives": ("JJ", "JJR", "JJS"),
    "Verbs": ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ"),
    "Adverbs": ("RB", "RBR", "RBS"),
}
DEFAULT_POS_GROUPS = ("Nouns", "Adjectives")
DEFAULT_POS_TAGS = ("NN", "NNS", "JJ", "JJR", "JJS")
WORD_RE = re.compile(r"[A-Za-z]+")


@dataclass(frozen=True)
class ExtractionOptions:
    pos_tags: tuple[str, ...] = DEFAULT_POS_TAGS
    excluded_words: tuple[str, ...] = DEFAULT_EXCLUDED_WORDS
    excluded_prefixes: tuple[str, ...] = ()
    protected_words: tuple[str, ...] = ()
    protected_prefixes: tuple[str, ...] = ()
    include_only_words: tuple[str, ...] = ()
    use_stopwords: bool = True
    lemmatize: bool = True
    deduplicate: bool = True
    lowercase: bool = True
    min_length: int = 3


@dataclass(frozen=True)
class ExtractionResult:
    filename: str
    output_filename: str
    words: list[str]
    text: str
    error: str | None = None

    @property
    def counts(self) -> list[tuple[str, int]]:
        counts = Counter(self.words)
        return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def csv_to_tuple(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = value.replace("\n", ",").split(",")
    else:
        parts = value
    return tuple(part.strip() for part in parts if str(part).strip())


def pos_tags_from_groups(
    groups: Sequence[str] | None = None,
    extra_tags: str | Sequence[str] | None = None,
) -> tuple[str, ...]:
    tags: list[str] = []
    seen: set[str] = set()
    selected_groups = DEFAULT_POS_GROUPS if groups is None else groups

    for group in selected_groups:
        for tag in POS_PRESETS.get(group, ()):
            normalized = tag.upper()
            if normalized not in seen:
                tags.append(normalized)
                seen.add(normalized)

    for tag in csv_to_tuple(extra_tags):
        normalized = tag.upper()
        if normalized not in seen:
            tags.append(normalized)
            seen.add(normalized)

    return tuple(tags)


def ensure_nltk_data() -> None:
    resources = {
        "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
    }
    for package, resource_path in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.data.find(f"{resource_path}.zip")
            except LookupError:
                nltk.download(package, quiet=True)


def clean_pdf_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+([\u0300-\u036f])", r"\1", text)
    text = re.sub(r"([\u0300-\u036f])\s+(?=[A-Za-z])", r"\1", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.\w+\b", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdoi\s*:?\s*\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\(cid:\d+\)", " ", text)
    text = re.sub(r"published maps and institutional affil", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\biations\b\.?", " ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"Publisher'?s Note:.*?institutional\s+affil(?:-|iations).*?iations\.",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(
        r"neutral with regard to jurisdictional claims in published maps and institutional\s+affil-.*?iations\.",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    # Join words broken by PDF line wrapping, e.g. "environ-\nmental".
    text = re.sub(r"(?<=[A-Za-z])-\s*\n\s*(?=[A-Za-z])", "", text)
    text = re.sub(r"(?<=[A-Za-z])-\s+(?=[a-z])", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def _cluster_words_into_lines(words: list[dict], y_tolerance: float = 3) -> list[list[dict]]:
    """Group pdfplumber word dicts into visual lines by vertical position."""
    if not words:
        return []
    words_sorted = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines: list[list[dict]] = [[words_sorted[0]]]
    current_top = words_sorted[0]["top"]

    for word in words_sorted[1:]:
        if abs(word["top"] - current_top) <= y_tolerance:
            lines[-1].append(word)
        else:
            lines.append([word])
            current_top = word["top"]

    return [sorted(line, key=lambda w: w["x0"]) for line in lines]


def _extract_page_text_two_column(
    page,
    x_tolerance: float = 1.5,
    y_tolerance: float = 3,
    min_side_fraction: float = 0.15,
) -> str:
    """Extract a page's text, handling two-column academic layouts.

    Plain ``page.extract_text(layout=False, ...)`` reads a page purely by
    vertical position. In a two-column layout set on a shared baseline
    grid, the left column's line N and the right column's line N often
    sit at (almost) the same height -- so a naive y-based reader treats
    them as a single row and concatenates left-column text directly
    against unrelated right-column text. When either side ends mid-word
    (a hyphenated line wrap), that produces fused nonsense tokens like
    "patcleveland" or "stanness".

    Classifying by the bounding box of a whole *line* doesn't fix this,
    because the merged row's bbox spans the full page width and looks
    like a legitimate full-width heading. The fix has to happen at the
    *word* level, before lines are ever assembled: every word is
    assigned to the left or right column purely by its own x-center,
    each column's words are then grouped into lines and sorted
    top-to-bottom independently, and the two column blocks are
    concatenated left-then-right. A word's own bounding box is never
    split, so this step cannot itself truncate or fuse a word -- at
    worst a wide centered heading gets its words distributed across
    both blocks, which is a readability quirk, not a data-quality bug.

    If the page doesn't actually look two-column (too few words
    confined to one side), this falls back to plain top-to-bottom
    reading order instead of forcing a spurious split.
    """
    words = page.extract_words(x_tolerance=x_tolerance, y_tolerance=y_tolerance)
    if not words:
        return ""

    def lines_text(word_list: list[dict]) -> str:
        lines = _cluster_words_into_lines(word_list, y_tolerance=y_tolerance)
        return "\n".join(" ".join(w["text"] for w in line) for line in lines)

    midpoint = page.width / 2
    left_words = [w for w in words if (w["x0"] + w["x1"]) / 2 < midpoint]
    right_words = [w for w in words if (w["x0"] + w["x1"]) / 2 >= midpoint]

    min_side = max(5, len(words) * min_side_fraction)
    if len(left_words) < min_side or len(right_words) < min_side:
        # Not a genuine two-column page -- splitting would scramble
        # normal single-column reading order, so don't.
        return lines_text(words)

    return lines_text(left_words) + "\n" + lines_text(right_words)

def extract_text_from_pdf(pdf: str | Path | bytes) -> str:
    if isinstance(pdf, (str, Path)):
        path_or_stream = str(pdf)
        fallback_source = str(pdf)
    else:
        path_or_stream = io.BytesIO(pdf)
        fallback_source = io.BytesIO(pdf)

    page_texts: list[str] = []
    with pdfplumber.open(path_or_stream) as document:
        for page in document.pages:
             page_texts.append(_extract_page_text_two_column(page, x_tolerance=1.5, y_tolerance=3))

    text = "\n".join(page_texts)
    if text.strip():
        return clean_pdf_text(text)

    reader = PdfReader(fallback_source)
    fallback_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return clean_pdf_text(fallback_text)


def tokenize_words(text: str) -> list[str]:
    text = clean_pdf_text(text)
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(char if char.isascii() else " " for char in normalized)
    return WORD_RE.findall(ascii_text)


def _tag_to_wordnet_pos(tag: str) -> str:
    if tag.startswith("J"):
        return "a"
    if tag.startswith("V"):
        return "v"
    if tag.startswith("R"):
        return "r"
    return "n"


def _normalized_set(values: Sequence[str]) -> set[str]:
    return {value.strip().lower() for value in values if value.strip()}


def _matches_prefix(forms: set[str], prefixes: set[str]) -> bool:
    return any(form.startswith(prefix) for form in forms for prefix in prefixes)


def extract_words_from_text(text: str, options: ExtractionOptions | None = None) -> list[str]:
    ensure_nltk_data()
    options = options or ExtractionOptions()
    tokens = tokenize_words(text)
    if not tokens:
        return []

    tagged_tokens = pos_tag(tokens)
    allowed_tags = {tag.upper() for tag in options.pos_tags}
    explicit_exclusions = _normalized_set(options.excluded_words)
    excluded_prefixes = _normalized_set(options.excluded_prefixes)
    protected = _normalized_set(options.protected_words)
    protected_prefixes = _normalized_set(options.protected_prefixes)
    include_only = _normalized_set(options.include_only_words)
    excluded = set(explicit_exclusions)

    if options.use_stopwords:
        excluded.update(word.lower() for word in stopwords.words("english"))

    lemmatizer = WordNetLemmatizer()
    output: list[str] = []
    seen: set[str] = set()

    for token, tag in tagged_tokens:
        raw = token.lower()
        word = raw if options.lowercase else token
        if options.lemmatize:
            word = lemmatizer.lemmatize(raw, pos=_tag_to_wordnet_pos(tag))
        elif options.lowercase:
            word = raw

        comparison = word.lower()
        token_forms = {raw, comparison}
        if allowed_tags and tag.upper() not in allowed_tags:
            continue
        if len(comparison) < options.min_length:
            continue
        if include_only and token_forms.isdisjoint(include_only):
            continue
        is_protected = not token_forms.isdisjoint(protected) or _matches_prefix(token_forms, protected_prefixes)
        is_excluded = not token_forms.isdisjoint(excluded) or _matches_prefix(token_forms, excluded_prefixes)
        if is_excluded and not is_protected:
            continue
        if options.deduplicate and comparison in seen:
            continue

        output.append(word)
        seen.add(comparison)

    return output


def output_name_for_pdf(filename: str, extension: str = ".txt") -> str:
    stem = Path(filename).stem or "extracted_words"
    safe = re.sub(r"[^\w .-]+", "_", stem).strip(" .")
    extension = extension if extension.startswith(".") else f".{extension}"
    return f"{safe or 'extracted_words'}{extension}"


def process_pdf_bytes(contents: bytes, filename: str, options: ExtractionOptions | None = None) -> ExtractionResult:
    text = extract_text_from_pdf(contents)
    words = extract_words_from_text(text, options)
    return ExtractionResult(
        filename=filename,
        output_filename=output_name_for_pdf(filename),
        words=words,
        text=" ".join(words),
    )


def process_file(filepath: str | Path, options: ExtractionOptions | None = None) -> list[str]:
    text = extract_text_from_pdf(filepath)
    return extract_words_from_text(text, options)


def process_files(file_items: Iterable[tuple[str, bytes]], options: ExtractionOptions | None = None) -> list[ExtractionResult]:
    results: list[ExtractionResult] = []
    for filename, contents in file_items:
        try:
            results.append(process_pdf_bytes(contents, filename, options))
        except Exception as exc:
            results.append(
                ExtractionResult(
                    filename=filename,
                    output_filename=output_name_for_pdf(filename),
                    words=[],
                    text="",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return results


def result_to_counts_csv(result: ExtractionResult) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["word", "count"])
    writer.writerows(result.counts)
    return buffer.getvalue()


def results_to_zip(results: Sequence[ExtractionResult], output_format: str = "words") -> bytes:
    buffer = io.BytesIO()
    used_names: set[str] = set()
    errors: list[str] = []
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for result in results:
            if result.error:
                errors.append(f"{result.filename}: {result.error}")
                continue
            if output_format == "counts":
                name = output_name_for_pdf(result.filename, ".csv")
                contents = result_to_counts_csv(result)
            else:
                name = result.output_filename
                contents = result.text
            if name in used_names:
                stem = Path(name).stem
                suffix = Path(name).suffix
                index = 2
                while f"{stem}-{index}{suffix}" in used_names:
                    index += 1
                name = f"{stem}-{index}{suffix}"
            used_names.add(name)
            archive.writestr(name, contents)
        if errors:
            archive.writestr("_errors.txt", "\n".join(errors))
    return buffer.getvalue()
