# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "nltk==3.9.4",
#     "pandas==2.3.3",
#     "pdf2image==1.17.0",
#     "pi-heif==1.4.0",
#     "unstructured==0.18.32",
#     "unstructured-inference==1.2.0",
# ]
# ///

import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import os
    from unstructured.partition.pdf import partition_pdf
    from unstructured.staging.base import elements_to_json
    import logging
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    return mo, os, partition_pdf


@app.cell
def _():
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords, words, wordnet
    from nltk.stem import WordNetLemmatizer
    from nltk import pos_tag

    nltk.download("punkt")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("stopwords")
    nltk.download("maxent_ne_chunker")
    nltk.download("words")
    nltk.download("wordnet")
    return WordNetLemmatizer, nltk, pos_tag, stopwords, word_tokenize, wordnet


@app.cell
def _(mo):
    mo.md("""
    ## Various Pdfs
    """)
    return


@app.cell
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
    return


if __name__ == "__main__":
    app.run()
