import streamlit as st
import os
import tempfile
import chromadb
import ollama

from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

def process_document(uploaded_file : UploadedFile) -> list[Document]:
    """
        Process the uploaded PDF document and split it into smaller chunks.
        This function takes an uploaded PDF file, stores it temporarily, and then
        processes it to extract and split the text into smaller chunks using a 
        RecursiveCharacterTextSplitter.
        Args:
            uploaded_file (UploadedFile): The uploaded PDF file to be processed.
        Returns:
            list[Document]: A list of Document objects containing the split text chunks.
    """
    # Store uploaded file as a temporary file 
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.close()

    loader = PyMuPDFLoader(temp_file.name)
    docs = loader.load()
    os.unlink(temp_file.name)


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\r\n", "\n","\r", ".", "\t", "!", "?", " ", ""]
    )
    return text_splitter.split_documents(docs)

def get_vector_collection() -> chromadb.Collection:
    """
        Get the vector database for the RAG model.
        This function loads the vector database for the RAG model from the ChromaDB.
        Returns:
            ChromaDB: The ChromaDB object containing the vector database.
    """
    ollama_ef = OllamaEmbeddingFunction(
        url="https://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )

    chroma_db = chromadb.PersistentClient(path="./demo-rag-chroma")

    return chroma_db.get_or_create_collection(
        name="rag-app",
        embedding_function=ollama_ef,
        metadata={"hnsw": "cosine"},
    )

def add_to_vector_collection(all_splits: list[Document], file_name: str):
    """
        Add the split text chunks to the vector collection.
        This function adds the split text chunks to the vector collection in the ChromaDB.
        Args:
            all_splits (list[Document]): A list of Document objects containing the split text chunks.
            file_name (str): The name of the uploaded file.
    """
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    st.success("📚 Successfully added the document to the vector collection.")

    
if __name__ == "__main__":
    with st.sidebar:
        st.set_page_config(page_title="RAG Document Reader", page_icon=":shark:")
        st.header("RAG Document Reader")

        uploaded_file = st.file_uploader(
            "📑 Upload a PDF file for Q&A", type=["pdf"],
            accept_multiple_files=False)
        
        process = st.button("⚡️ Process")

    if uploaded_file and process:
        normailized_file_name = uploaded_file.name.translate(
            str.maketrans({"-": "_", ".": "_", " ": "_"})
        )
        
        all_splits = process_document(uploaded_file)
        add_to_vector_collection(all_splits, normailized_file_name)
