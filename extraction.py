# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "nltk",
#     "numpy",
#     "pdfplumber",
#     "pypdf",
# ]
# ///

import marimo

__generated_with = "0.23.14"
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
<<<<<<< HEAD
def _(os):
    folder = "leuphana_pdfs/"
    pdfs = []
    names = []
    for filename in os.listdir(folder):
        if not filename.startswith('.') and filename.lower().endswith('.pdf'):
            names.append(filename[:-4])
            file_path = os.path.join(folder, filename)
            pdfs.append(file_path)
        else:
            pass

    #only for henrik case
    #names = [name[:3].rstrip(" ") for name in names]

    len(pdfs), len(names)
    return folder, names, pdfs


@app.cell
def _(folder, mo, names, partition_pdf, pdfs):
    documents = []

    for pdf in mo.status.progress_bar(pdfs, title="working", subtitle="Please wait", show_eta=True, show_rate=True):

        if not pdf.startswith(f'{folder}.') and pdf.lower().endswith('.pdf'):
            try:
                elements = partition_pdf(
                    filename=pdf,                  # mandatory
                    strategy="auto",                                     # mandatory to use ``hi_res`` strategy
                    #extract_images_in_pdf=False,                            # mandatory to set as ``True``
                    #extract_image_block_types=["Image", "Table"],          # optional
                    #infer_table_structure = False
                    #model_name ="yolox"
                    )
                documents.append(elements)
            except:
                index = pdfs.index(pdf)
                print(f"{pdf} could not be partitioned. Its index is {index}")
                name_to_remove = pdf[len(folder):-4]
                try:
                    names.remove(name_to_remove)
                except ValueError:
                    print(f"Could not remove {name_to_remove} from names list.")
    return (documents,)


@app.cell
def _(documents):
    import pickle

    # Save to a file
    with open("documents_leuphana.pkl", "wb") as f:
        pickle.dump(documents, f)

    # Load it back
    #with open("data.pkl", "rb") as f:
    #    obj = pickle.load(f)
    return


@app.cell
def _(documents):
    text_elements = []
    for document in documents:
        paragraphs = []
        for i in range(0,len(document)):
            if document[i].category == 'NarrativeText':
                paragraphs.append(document[i].text)
            else:
                pass
        text_elements.append(paragraphs)
    return (text_elements,)


@app.cell
def _(mo, pos_tag, text_elements, word_tokenize):
    token_elements = []
    for text_element in mo.status.progress_bar(text_elements, title="working", subtitle="Please wait", show_eta=True, show_rate=True):
        tagged_tokens = []
        for element in text_element:
            token = word_tokenize(element)
            tagged_tokens.append(pos_tag(token))

        token_elements.append(tagged_tokens)
    return (token_elements,)


@app.cell
def _(WordNetLemmatizer, mo, nltk, stopwords, token_elements, wordnet):
    # Stopwords (normalized to lowercase)
    stop_words = {w.lower() for w in stopwords.words('english')}
    stop_words.update({'et', 'al', 'etc', 'ie', 'issn', 'http', 'https'})

    # WordNet vocabulary (lemma names)
    valid_words = set(lem.name().lower() for syn in wordnet.all_synsets() for lem in syn.lemmas())

    lemmatizer = WordNetLemmatizer()

    # Only singular and plural *common* nouns
    NOUN_TAGS = {'NN', 'NNS'}

    transformed_documents = []


    for token_element in mo.status.progress_bar(token_elements, title="working", subtitle="Please wait", show_eta=True, show_rate=True):
        filtered_paragraphs = []

        for tagged_token in token_element:
            # Named Entity Recognition
            ne_tree = nltk.ne_chunk(tagged_token, binary=False)
            named_entities = set()
            for chunk in ne_tree:
                if hasattr(chunk, 'label'):
                    named_entities.update(w for (w, t) in chunk.leaves())
            named_entities = {w.lower() for w in named_entities}

            filtered_tokens = []
            for word, tag in tagged_token:
                # Lemmatize with POS = noun
                w = lemmatizer.lemmatize(word.lower(), pos='n')

                if (
                    w.isalpha() and len(w) > 2 and
                    tag in NOUN_TAGS and
                    w not in stop_words
                    and w not in named_entities
                    and w in valid_words
                ):
                    filtered_tokens.append(w)

            filtered_paragraphs.append(filtered_tokens)

        transformed_documents.append(filtered_paragraphs)
    return (transformed_documents,)


@app.cell
def _(transformed_documents):
    transformed_documents[1]
    return


@app.cell
def _(names, transformed_documents):
    text_files = []

    for k, transformed_document in enumerate(transformed_documents):
        if not transformed_document:
            print(f"Document at index {k} ('{names[k]}') is empty and couldn't be extracted.")
            text_files.append("")  # or use None if you'd rather keep the spot but mark it as empty
            continue  # skip to the next loop iteration

        outputs = []
        for doc in transformed_document:
            outputs.append(" ".join(doc))

        text = " ".join(outputs)
        text = text.replace("  ", " ")
        text_files.append(text)
    return (text_files,)


@app.cell
def _(names, text_files):
    folder_txt = "leuphana_text_ner/"
    for m in range(len(text_files)):
        #file_name = f'leuphana_text/{i+1:04}.txt'
        with open(f"{folder_txt}{names[m]}.txt", 'w', encoding='utf-8') as file:
        # Write the string to the file
            file.write(text_files[m])
=======
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
>>>>>>> review-pr
    return


if __name__ == "__main__":
    app.run()
