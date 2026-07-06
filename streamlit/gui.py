import os
import io
import glob
import zipfile

import streamlit as st

from pdf_text_extraction import process_file


def csv_to_list(s: str):
    return [t.strip() for t in s.split(",") if t.strip()] if s else []


@st.cache_data(show_spinner="Processing PDF…")
def process_pdf(file_bytes: bytes,
                filename: str,
                lemmatize: bool,
                filter_tokens_csv: str,
                valid_tokens_csv: str,
                valid_tags_csv: str) -> str:
    """
    Calls the process_file script.
    Wrapping this in a function is necessary, because a click on the download buttons
    triggers a rerun and therefore reexecution of the whole script.
    By caching the output of this function that does not matter.
    """
    temp_file_path = f"./temp_{os.path.basename(filename)}"

    try:
        # Save the uploaded file to a temporary location for processing
        with open(temp_file_path, "wb") as f:
            f.write(file_bytes)

        tokens = process_file(
            temp_file_path,
            # the order of sets are not fixed, hence use list for caching
            additionalFilterTokens=set(csv_to_list(filter_tokens_csv)),
            validTags=set(csv_to_list(valid_tags_csv)),
            validTokens=set(csv_to_list(valid_tokens_csv)),
            lemmatize=lemmatize,
        )
        return tokens
    finally:
        try:
            os.remove(temp_file_path)
        except OSError:
            pass


def main():
    st.title("PDF Text Extraction")

    # Sidebar for configuration
    st.sidebar.header("Configuration")
    lemmatize = st.sidebar.checkbox(
        "Lemmatize",
        value=False,
        help="Reduce words to their base or dictionary form (e.g., 'wolves' → 'wolf').",
    )

    filter_tokens = st.sidebar.text_input(
        "Filter Words",
        value="et,al,etc,ie,ISSN,http,https",
        help="Comma-separated list of tokens to exclude from the output.",
    )
    valid_tokens = st.sidebar.text_input(
        "Valid Words",
        value="",
        help="Comma-separated list of tokens to explicitly include, everything else will be filtered out."
    )
    valid_tags = st.sidebar.text_input(
        "Valid Tags",
        value="NN,NNS,JJ",
        help="Comma-separated list of NLTK POS tags to include (e.g., NN for nouns, JJ for adjectives, VB for verbs)."
    )

    # File upload (accepts multiple files)
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or more PDF files to extract text from."
    )

    # remove any artifacts
    artifacts = glob.glob("./temp*")
    if artifacts:
        for a in artifacts:
            os.remove(a)

    if uploaded_files:
        results = []

        total = len(uploaded_files)
        status = st.empty()
        progress = st.progress(0.0)
        dl_top = st.container()

        st.markdown("---")
        for i, uploaded_file in enumerate(uploaded_files, start=1):
            status.write(f"Processing file {i}/{total}: **{uploaded_file.name}**")

            tokens = process_pdf(
                uploaded_file.getvalue(),
                filename=uploaded_file.name,
                lemmatize=lemmatize,
                filter_tokens_csv=filter_tokens,
                valid_tokens_csv=valid_tokens,
                valid_tags_csv=valid_tags,
            )

            output_text = ' '.join(tokens)
            out_name = f"{uploaded_file.name}.txt"
            # save for the download-all button
            results.append((out_name, output_text))

            with st.expander(f"{uploaded_file.name} — {len(tokens)} words", expanded=False):
                # Display the extracted text
                st.text_area(
                    f"Extracted Text from {uploaded_file.name} ({len(tokens)} words total)",
                    value=output_text,
                    height=300,
                    key=f"ta_{uploaded_file.name}-{i}",
                )

                # download button for the output text
                st.download_button(
                    label=f"Download {out_name}",
                    data=output_text,
                    file_name=out_name,
                    mime="text/plain",
                    key=f"dl_{uploaded_file.name}-{i}"
                )
            progress.progress(i / total)

        progress.progress(1.0)
        status.write("Done.")

        st.markdown("---")
        # collect all results in a compressed buffer for download
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname, text in results:
                zf.writestr(fname, text)
        zip_buffer.seek(0)

        with dl_top:
            # st.markdown("---")
            st.download_button(
                label="Download all",
                data=zip_buffer,
                file_name="extracted_texts.zip",
                mime="application/zip",
                help="Download all extracted texts as a ZIP archive.",
                key="dl_all_top",
            )

        st.download_button(
            label="Download all",
            data=zip_buffer,
            file_name="extracted_texts.zip",
            mime="application/zip",
            help="Download all extracted texts as a ZIP archive.",
            key="dl_all_bottom",
        )


if __name__ == "__main__":
    main()
