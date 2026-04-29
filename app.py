# requirements.txt
# streamlit>=1.35.0
# langchain>=0.2.0
# langchain-google-genai>=1.0.0
# langchain-groq>=0.1.0
# langchain-community>=0.2.0
# faiss-cpu>=1.8.0
# pypdf>=4.0.0
# python-dotenv>=1.0.0

import streamlit as st
import os
import hashlib
import tempfile
import time
from typing import List

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(
    page_title="Patent Summariser (Groq Powered)", page_icon="⚡", layout="wide"
)


def get_file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


def invoke_with_retry(chain, inputs: dict, max_retries: int = 3):
    """
    Standard retry wrapper for network stability.
    """
    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            if attempt < max_retries - 1:
                st.toast(f"⏳ Network blip. Retrying... ({attempt + 1}/{max_retries})")
                time.sleep(2)
            else:
                raise e


@st.cache_resource(show_spinner=False)
def get_vector_store(
    file_hash: str, _file_bytes: bytes, file_name: str, _google_api_key: str
):
    suffix = ".pdf" if file_name.lower().endswith(".pdf") else ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(_file_bytes)
        tmp_path = tmp_file.name

    try:
        if file_name.lower().endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path)

        documents = loader.load()
        first_page_text = documents[0].page_content if documents else ""

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=150
        )
        chunks = text_splitter.split_documents(documents)

        # We keep Google for Embeddings (Free tier limits are very high for embeddings)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", google_api_key=_google_api_key
        )

        vector_store = FAISS.from_documents(chunks, embeddings)

        page_count = len(documents)
        word_count = sum(len(doc.page_content.split()) for doc in documents)

        return vector_store, page_count, word_count, first_page_text
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def extract_metadata(first_page_text: str, groq_api_key: str) -> str:
    # Swapped to Groq's current supported Llama 3.3 model
    llm = ChatGroq(
        temperature=0.1, model_name="llama-3.3-70b-versatile", groq_api_key=groq_api_key
    )

    prompt = ChatPromptTemplate.from_template("""
    You are an expert patent analyst. Extract the following key administrative metadata from the provided text, which represents the first page (cover page) of a patent document.
    
    Required Fields:
    - **Patent Title**
    - **Inventors**
    - **Assignee**
    - **Status** (Infer if this is a Granted Patent, Patent Application, etc., based on standard codes or language)
    - **Timeline** (List any filing dates, publication dates, priority dates found)
    
    Instructions:
    - If a specific piece of information is entirely missing, write "Not explicitly found on the cover page."
    - Output the result as a clean, structured Markdown list.
    
    First Page Text:
    {text}
    
    Output:
    """)

    chain = prompt | llm
    response = invoke_with_retry(chain, {"text": first_page_text})
    return response.content


def generate_section(
    query: str, vector_store, groq_api_key: str, section_name: str
) -> str:
    # Swapped to Groq's current supported Llama 3.3 model
    llm = ChatGroq(
        temperature=0.2, model_name="llama-3.3-70b-versatile", groq_api_key=groq_api_key
    )

    docs = vector_store.similarity_search(query, k=5)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = ChatPromptTemplate.from_template("""
    You are an expert patent analyst. Use the following context from a patent document to generate a structured summary for the section: {section_name}.
    
    Context:
    {context}
    
    Query: {query}
    
    Instructions:
    - Provide a technical and objective response.
    - If the context doesn't contain enough information, state that it's "Not explicitly mentioned in the provided text."
    - Follow specific formatting: bullet points for lists, clear headers if needed.
    
    Output:
    """)

    chain = prompt | llm
    response = invoke_with_retry(
        chain, {"context": context, "query": query, "section_name": section_name}
    )
    return response.content


# --- MAIN UI ---


def main():
    st.title("Patent Summariser")

    st.sidebar.header("API Configuration")

    # Dual API Keys input
    google_api_key_input = st.sidebar.text_input(
        "1. Google API Key (For Embeddings)",
        type="password",
        help="Used to process and index the PDF.",
    )

    groq_api_key_input = st.sidebar.text_input(
        "2. Groq API Key (For Generation)",
        type="password",
        help="Used to generate the summaries at lightning speed.",
    )

    # Bulletproof fallback to check environment variables first
    env_google_key = os.getenv("GOOGLE_API_KEY")
    env_groq_key = os.getenv("GROQ_API_KEY")

    # Final Google key resolution ignoring missing secrets file
    google_secrets_key = None
    try:
        google_secrets_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
    google_api_key = google_api_key_input or env_google_key or google_secrets_key

    # Final Groq key resolution ignoring missing secrets file
    groq_secrets_key = None
    try:
        groq_secrets_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    groq_api_key = groq_api_key_input or env_groq_key or groq_secrets_key

    if not google_api_key or not groq_api_key:
        st.warning("⚠️ Please provide BOTH API keys in the sidebar to proceed.")
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload a patent (PDF or TXT)", type=["pdf", "txt"]
    )

    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name
        file_hash = get_file_hash(file_bytes)

        if (
            "last_file_hash" not in st.session_state
            or st.session_state.last_file_hash != file_hash
        ):
            st.session_state.summary_results = None
            st.session_state.metadata_result = None
            st.session_state.full_summary_text = None
            st.session_state.last_file_hash = file_hash

        with st.spinner("Indexing document (Google Embeddings)..."):
            try:
                vector_store, page_count, word_count, first_page_text = (
                    get_vector_store(file_hash, file_bytes, file_name, google_api_key)
                )
                st.success(f"File indexed: {file_name}")
                st.info(f"Stats Overview: {page_count} pages | ~{word_count} words")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                return

        sections_config = [
            (
                "Simplified Summary",
                "Provide a 3-sentence plain-English summary of the invention.",
            ),
            (
                "Problem & Prior Art",
                "What problem does this patent solve? What existing solutions does it cite?",
            ),
            (
                "Core Invention",
                "What are the key novel claims and technical approach? Explain in plain language.",
            ),
            (
                "Claims Breakdown",
                "List the independent claims as bullet points. Identify and flag which one appears to be the broadest claim.",
            ),
            (
                "Key Technical Terms",
                "Identify up to 10 jargon terms or technical concepts. Present them with clear definitions.",
            ),
            (
                "Novelty & Risk Flags",
                "Analyze what makes this invention novel compared to the prior art. Highlight any obvious overlaps or risks of narrow claims.",
            ),
        ]

        if st.button("Generate Fast Summary", type="primary"):
            total_steps = len(sections_config) + 1
            progress_bar = st.progress(0)

            summary_results = {}
            full_summary_text = f"# Patent Summary Report: {file_name}\n\n"

            with st.status("Extracting Cover Page Metadata...", expanded=False):
                metadata_content = extract_metadata(first_page_text, groq_api_key)
                st.session_state.metadata_result = metadata_content
                full_summary_text += f"## Patent Metadata\n{metadata_content}\n\n"

            progress_bar.progress(1 / total_steps)

            for i, (title, query) in enumerate(sections_config):
                with st.status(f"Synthesising {title}...", expanded=False):
                    section_content = generate_section(
                        query, vector_store, groq_api_key, title
                    )
                    summary_results[title] = section_content
                    full_summary_text += f"## {title}\n{section_content}\n\n"

                progress_bar.progress((i + 2) / total_steps)

            progress_bar.empty()
            st.toast("Summary generation complete!", icon="✅")

            st.session_state.summary_results = summary_results
            st.session_state.full_summary_text = full_summary_text

        if st.session_state.metadata_result and st.session_state.summary_results:
            with st.expander("Patent Metadata", expanded=True):
                st.markdown(st.session_state.metadata_result)

            for title, content in st.session_state.summary_results.items():
                with st.expander(title, expanded=True):
                    st.markdown(content)

            st.download_button(
                label="Export Full Summary (.txt)",
                data=st.session_state.full_summary_text,
                file_name=f"patent_summary_{file_name.replace('.', '_')}.txt",
                mime="text/plain",
            )


if __name__ == "__main__":
    main()
