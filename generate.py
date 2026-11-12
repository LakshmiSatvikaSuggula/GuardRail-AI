from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from retrieve import get_retriever

def format_docs(docs):
    """Combines the retrieved chunks into one big string for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain():
    # 1. Initialize Llama 3
    # WHY temperature=0.1? Paper 6 proved that low temperature makes the model more deterministic and less likely to hallucinate.
    llm = ChatOllama(model="llama3", temperature=0.1)

    # 2. Get the retriever
    retriever = get_retriever()

    # 3. Create the Prompt Template
    # WHY this prompt? It strictly forces the model to ONLY use the provided context.
    template = """You are a helpful college assistant. 
Answer the question using ONLY the following context. 
If you do not know the answer based on the context, say exactly: "I cannot answer this from the provided documents."
Do not use your outside knowledge.

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate.from_template(template)

    # 4. Build the Chain
    # This connects: Question -> Retrieve -> Format Context -> Prompt -> Llama 3 -> Text
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

if __name__ == "__main__":
    chain = build_rag_chain()
    
    print("--- Test: MBA Category B ---")
    query = "What is the minimum percentage required for MBA admission under Category B?"
    print(f"Q: {query}")
    print(f"A: {chain.invoke(query)}\n")

    # Test 2: A question NOT answerable by the PDF (Testing the refusal)
    print("--- Test 2: Unanswerable Question ---")
    query2 = "What is the Wi-Fi password for the cafeteria?"
    print(f"Q: {query2}")
    print(f"A: {chain.invoke(query2)}\n")