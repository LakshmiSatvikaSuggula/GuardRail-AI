import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings   # NEW
from langchain_chroma import Chroma                        # NEW

PDF_PATH = "data/college_rules.pdf"
CHROMA_PATH = "chroma_db"

def ingest_pdf():
    print("--- Starting Ingestion ---")
    
    if not os.path.exists(PDF_PATH):
        print(f"Error: Could not find {PDF_PATH}.")
        return

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text  # (fix: use split_text on raw text, or split_documents on docs)
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Creating and saving ChromaDB...")
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=CHROMA_PATH
    )
    print("--- Ingestion Complete! ---")

if __name__ == "__main__":
    ingest_pdf()