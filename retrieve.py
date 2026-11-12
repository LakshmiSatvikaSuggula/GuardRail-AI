from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = "chroma_db"

def get_retriever():
    """
    Loads the ChromaDB database and sets up the search engine.
    """
    # 1. Load the SAME embedding model we used in ingest.py
    # WHY: If we use a different model, the numbers won't match!
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2. Connect to the existing database
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # 3. Create a Retriever
    # WHY k=5? Paper 2 and Paper 5 showed that Top-5 is the sweet spot for balancing context length and accuracy.
    retriever = db.as_retriever(search_kwargs={"k": 5})
    return retriever

if __name__ == "__main__":
    # Test the retriever
    retriever = get_retriever()
    test_query = "What is the minimum percentage required for MBA admission under Category B?"
    
    print(f"Asking: {test_query}\n")
    docs = retriever.invoke(test_query)
    
    for i, doc in enumerate(docs):
        page = doc.metadata.get("page", "Unknown")
        print(f"--- Chunk {i+1} (Page {page}) ---")
        print(doc.page_content[:200] + "...\n") # Print first 200 chars