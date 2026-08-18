from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pdf_word_extraction import (  # noqa: E402
    DEFAULT_EXCLUDED_WORDS,
    DEFAULT_POS_GROUPS,
    ExtractionOptions,
    ExtractionResult,
    POS_PRESETS,
    csv_to_tuple,
    output_name_for_pdf,
    pos_tags_from_groups,
    process_files,
    result_to_counts_csv,
    results_to_zip,
)


@st.cache_data(show_spinner="Processing PDFs...")
def process_uploaded_files(
    files: tuple[tuple[str, bytes], ...],
    pos_groups: tuple[str, ...],
    extra_pos_tags_csv: str,
    excluded_words_csv: str,
    excluded_prefixes_csv: str,
    protected_words_csv: str,
    protected_prefixes_csv: str,
    include_only_words_csv: str,
    use_stopwords: bool,
    lemmatize: bool,
    deduplicate: bool,
    output_format: str,
) -> list[ExtractionResult]:
    options = ExtractionOptions(
        pos_tags=pos_tags_from_groups(pos_groups, extra_pos_tags_csv),
        excluded_words=csv_to_tuple(excluded_words_csv),
        excluded_prefixes=csv_to_tuple(excluded_prefixes_csv),
        protected_words=csv_to_tuple(protected_words_csv),
        protected_prefixes=csv_to_tuple(protected_prefixes_csv),
        include_only_words=csv_to_tuple(include_only_words_csv),
        use_stopwords=use_stopwords,
        lemmatize=lemmatize,
        deduplicate=deduplicate and output_format == "words",
    )
    return process_files(files, options)


def main() -> None:
    st.set_page_config(page_title="PDF Word Extraction", layout="wide")
    st.title("PDF Word Extraction")

    with st.sidebar:
        st.header("Word filters")
        pos_groups = st.multiselect(
            "Parts of speech to keep",
            options=list(POS_PRESETS),
            default=list(DEFAULT_POS_GROUPS),
            help="Choose friendly POS groups such as nouns, verbs, adjectives, or adverbs.",
        )
        extra_pos_tags = st.text_input(
            "Additional NLTK POS tags",
            value="",
            help="Optional comma-separated tags, for example CD or FW.",
        )
        use_stopwords = st.checkbox("Exclude English stopwords", value=True)
        lemmatize = st.checkbox("Lemmatize words", value=True)
        deduplicate = st.checkbox("Remove duplicates", value=True)
        output_format = st.radio(
            "Download format",
            options=("words", "counts"),
            format_func=lambda value: "Word list (.txt)" if value == "words" else "Frequency counts (.csv)",
            help="Use counts when you want readable word frequencies instead of repeated words.",
        )
        excluded_words = st.text_area(
            "Additional words to exclude",
            value=",".join(DEFAULT_EXCLUDED_WORDS),
            help="Comma- or line-separated words to exclude.",
            height=90,
        )
        excluded_prefixes = st.text_area(
            "Word prefixes to exclude",
            value="",
            help="Comma- or line-separated beginnings to exclude, for example pre or anti.",
            height=80,
        )
        protected_words = st.text_area(
            "Words not to exclude",
            value="",
            help="Comma- or line-separated exact words to keep even when they are stopwords or exclusions.",
            height=90,
        )
        protected_prefixes = st.text_area(
            "Prefixes not to exclude",
            value="",
            help="Optional exception prefixes that override prefix exclusions.",
            height=80,
        )
        include_only_words = st.text_area(
            "Only include these words (optional)",
            value="",
            help="Optional allow-list. Leave blank to keep all words matching the other filters.",
            height=90,
        )

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Drag and drop or browse for one or more PDF files.",
    )

    if not uploaded_files:
        st.info("Upload one or more PDF files to extract words.")
        return

    files = tuple((uploaded_file.name, uploaded_file.getvalue()) for uploaded_file in uploaded_files)
    results = process_uploaded_files(
        files,
        tuple(pos_groups),
        extra_pos_tags,
        excluded_words,
        excluded_prefixes,
        protected_words,
        protected_prefixes,
        include_only_words,
        use_stopwords,
        lemmatize,
        deduplicate,
        output_format,
    )

    zip_bytes = results_to_zip(results, output_format=output_format)
    failed_count = sum(1 for result in results if result.error)
    if failed_count:
        st.warning(f"{failed_count} file(s) could not be processed. The ZIP includes _errors.txt.")
    st.download_button(
        "Download ZIP",
        data=zip_bytes,
        file_name="extracted_words.zip",
        mime="application/zip",
    )

    st.divider()
    for result in results:
        output_name = result.output_filename if output_format == "words" else output_name_for_pdf(result.filename, ".csv")
        status = "error" if result.error else f"{len(result.words)} words"
        with st.expander(f"{output_name} - {status}", expanded=False):
            if result.error:
                st.error(result.error)
                continue
            if output_format == "counts":
                st.dataframe(
                    [{"word": word, "count": count} for word, count in result.counts],
                    use_container_width=True,
                    hide_index=True,
                )
                file_data = result_to_counts_csv(result)
                mime = "text/csv"
            else:
                st.text_area("Extracted words", value=result.text, height=240, key=result.output_filename)
                file_data = result.text
                mime = "text/plain"
            st.download_button(
                f"Download {output_name}",
                data=file_data,
                file_name=output_name,
                mime=mime,
                key=f"download-{output_name}",
            )


if __name__ == "__main__":
    main()
