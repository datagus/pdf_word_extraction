# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "nltk",
#     "numpy",
#     "pdfplumber",
#     "pypdf",
# ]
# [tool.marimo.runtime]
# auto_instantiate = true
# on_cell_change = "autorun"
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from pdf_word_extraction import (
        DEFAULT_EXCLUDED_WORDS,
        DEFAULT_POS_GROUPS,
        ExtractionOptions,
        POS_PRESETS,
        csv_to_tuple,
        output_name_for_pdf,
        pos_tags_from_groups,
        process_files,
        results_to_zip,
    )

    return (
        DEFAULT_EXCLUDED_WORDS,
        DEFAULT_POS_GROUPS,
        ExtractionOptions,
        POS_PRESETS,
        csv_to_tuple,
        mo,
        output_name_for_pdf,
        pos_tags_from_groups,
        process_files,
        results_to_zip,
    )


@app.cell
def _(DEFAULT_EXCLUDED_WORDS, DEFAULT_POS_GROUPS, POS_PRESETS, mo):
    files = mo.ui.file(
        filetypes=[".pdf"],
        multiple=True,
        kind="area",
        max_size=200_000_000,
        label="PDF files",
    )
    pos_groups = mo.ui.multiselect(
        options=list(POS_PRESETS),
        value=list(DEFAULT_POS_GROUPS),
        label="Parts of speech to keep",
        full_width=True,
    )
    extra_pos_tags = mo.ui.text(
        value="",
        label="Additional NLTK POS tags",
        full_width=True,
    )
    use_stopwords = mo.ui.checkbox(value=True, label="Exclude English stopwords")
    lemmatize = mo.ui.checkbox(value=True, label="Lemmatize words")
    deduplicate = mo.ui.checkbox(value=True, label="Remove duplicates")
    output_format = mo.ui.radio(
        options={"Word list (.txt)": "words", "Frequency counts (.csv)": "counts"},
        value="Word list (.txt)",
        label="Download format",
    )
    excluded_words = mo.ui.text_area(
        value=",".join(DEFAULT_EXCLUDED_WORDS),
        label="Additional words to exclude",
        rows=3,
        full_width=True,
    )
    excluded_prefixes = mo.ui.text_area(
        value="",
        label="Word prefixes to exclude",
        rows=3,
        full_width=True,
    )
    protected_words = mo.ui.text_area(
        value="",
        label="Words not to exclude",
        rows=3,
        full_width=True,
    )
    protected_prefixes = mo.ui.text_area(
        value="",
        label="Prefixes not to exclude",
        rows=3,
        full_width=True,
    )
    include_only_words = mo.ui.text_area(
        value="",
        label="Only include these words (optional)",
        rows=3,
        full_width=True,
    )
    run = mo.ui.run_button(label="Extract words", kind="success")
    return (
        deduplicate,
        excluded_prefixes,
        excluded_words,
        extra_pos_tags,
        files,
        include_only_words,
        lemmatize,
        output_format,
        pos_groups,
        protected_prefixes,
        protected_words,
        run,
        use_stopwords,
    )


@app.cell
def _(
    deduplicate,
    excluded_prefixes,
    excluded_words,
    extra_pos_tags,
    files,
    include_only_words,
    lemmatize,
    mo,
    output_format,
    pos_groups,
    protected_prefixes,
    protected_words,
    run,
    use_stopwords,
):
    mo.vstack(
        [
            mo.md("# PDF Word Extraction"),
            files,
            mo.md("### Filters"),
            pos_groups,
            extra_pos_tags,
            mo.hstack([use_stopwords, lemmatize, deduplicate], justify="start"),
            output_format,
            excluded_words,
            excluded_prefixes,
            protected_words,
            protected_prefixes,
            include_only_words,
            run,
        ],
        gap=1,
    )
    return


@app.cell
def _(
    ExtractionOptions,
    csv_to_tuple,
    deduplicate,
    excluded_prefixes,
    excluded_words,
    extra_pos_tags,
    files,
    include_only_words,
    lemmatize,
    mo,
    output_format,
    output_name_for_pdf,
    pos_groups,
    pos_tags_from_groups,
    process_files,
    protected_prefixes,
    protected_words,
    results_to_zip,
    run,
    use_stopwords,
):

    if not files.value:
        output = mo.md("Upload one or more PDFs to begin.")
    elif not run.value:
        output = mo.md(f"{len(files.value)} PDF file(s) ready.")
    else:
        options = ExtractionOptions(
            pos_tags=pos_tags_from_groups(pos_groups.value, extra_pos_tags.value),
            excluded_words=csv_to_tuple(excluded_words.value),
            excluded_prefixes=csv_to_tuple(excluded_prefixes.value),
            protected_words=csv_to_tuple(protected_words.value),
            protected_prefixes=csv_to_tuple(protected_prefixes.value),
            include_only_words=csv_to_tuple(include_only_words.value),
            use_stopwords=use_stopwords.value,
            lemmatize=lemmatize.value,
            deduplicate=deduplicate.value and output_format.value == "words",
        )
        uploaded = [(item.name, item.contents) for item in files.value]
        results = process_files(uploaded, options)
        zip_bytes = results_to_zip(results, output_format=output_format.value)
        failed_count = sum(1 for result in results if result.error)
        rows = [
            {
                "file": result.output_filename
                if output_format.value == "words"
                else output_name_for_pdf(result.filename, ".csv"),
                "status": result.error or f"{len(result.words)} words",
                "preview": result.text[:240]
                if output_format.value == "words"
                else ", ".join(f"{word}: {count}" for word, count in result.counts[:8]),
            }
            for result in results
        ]
        output = mo.vstack(
            [
                mo.download(
                    zip_bytes,
                    filename="extracted_words.zip",
                    mimetype="application/zip",
                    label="Download ZIP",
                ),
                mo.md(f"{failed_count} file(s) could not be processed; see _errors.txt in the ZIP.")
                if failed_count
                else mo.md(""),
                mo.ui.table(rows, label="Extracted files"),
            ],
            gap=1,
        )
    output
    return


if __name__ == "__main__":
    app.run()
