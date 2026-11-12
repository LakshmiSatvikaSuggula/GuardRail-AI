import re
import hashlib
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Global embeddings for consistency check (load once)
_embedding_model = None

def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embedding_model


# ============================================================
# GUARDRAIL 1: GROUNDEDNESS
# ============================================================
# Based on Paper 2 (Entity-Aligned Verification) & Paper 3 (EduRAG token grounding)
# Idea: Extract numbers and important keywords from the answer.
#       If they don't appear in the retrieved chunks, the answer is hallucinated.

def check_groundedness(answer: str, retrieved_docs: list) -> dict:
    """
    Returns: {"passed": bool, "score": float, "reason": str, "ungrounded": list}
    """
    context = " ".join(doc.page_content for doc in retrieved_docs).lower()
    answer_lower = answer.lower()

    # If the model refused, that's grounded by definition (it didn't claim anything)
    if "cannot answer" in answer_lower:
        return {"passed": True, "score": 1.0, "reason": "Refusal is always grounded", "ungrounded": []}

    # --- Check 1: Numbers must appear in context ---
    # Numbers are the most common source of hallucination (percentages, fees, dates)
    answer_numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', answer)
    ungrounded_numbers = []
    for num in answer_numbers:
        if num not in context:
            ungrounded_numbers.append(num)

    # --- Check 2: Content words must have at least partial overlap ---
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "of", "to", "in",
                 "for", "on", "with", "and", "or", "by", "as", "at", "be", "this",
                 "that", "these", "those", "it", "its", "from", "according", "provided",
                 "documents", "document", "answer", "question", "based", "minimum",
                 "required", "category", "under"}
    answer_words = [
        w.strip(".,;:()").lower()
        for w in answer_lower.split()
        if len(w) > 3 and w.strip(".,;:()").lower() not in stopwords
    ]
    if not answer_words:
        return {"passed": True, "score": 1.0, "reason": "No content words to check", "ungrounded": []}

    matched = sum(1 for w in answer_words if w in context)
    coverage = matched / len(answer_words)

    # --- Decision ---
    # Fail if ANY number is ungrounded OR coverage is below threshold
    if ungrounded_numbers:
        return {
            "passed": False,
            "score": coverage,
            "reason": f"Ungrounded numbers: {ungrounded_numbers}",
            "ungrounded": ungrounded_numbers,
        }
    if coverage < 0.5:
        return {
            "passed": False,
            "score": coverage,
            "reason": f"Low token coverage: {coverage:.2f}",
            "ungrounded": [],
        }
    return {
        "passed": True,
        "score": coverage,
        "reason": f"Coverage {coverage:.2f}",
        "ungrounded": [],
    }


# ============================================================
# GUARDRAIL 2: SAFETY
# ============================================================
# Based on Paper 7 (Trustworthy RAG survey)
# Idea: Block prompt injection, toxic language, and dangerous requests.

TOXIC_KEYWORDS = [
    "kill", "bomb", "weapon", "terrorist", "hack", "forget", "ignore previous",
    "ignore all previous", "override", "jailbreak", "suicide", "drug", "poison"
]

def check_safety(text: str) -> dict:
    """
    Returns: {"passed": bool, "reason": str, "triggered": list}
    """
    text_lower = text.lower()
    triggered = [kw for kw in TOXIC_KEYWORDS if kw in text_lower]
    if triggered:
        return {"passed": False, "reason": f"Triggered: {triggered}", "triggered": triggered}
    return {"passed": True, "reason": "Safe", "triggered": []}


# ============================================================
# GUARDRAIL 3: CONSISTENCY
# ============================================================
# Based on Paper 4 (Regulatory RAG faithfulness) & Paper 6 (low-temperature stability)
# Idea: Run the same query twice. If answers differ semantically, the model is unstable.

def _cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def check_consistency(chain, query: str, answer_1: str, threshold: float = 0.90) -> dict:
    """
    Runs the chain a second time and compares.
    Returns: {"passed": bool, "score": float, "reason": str}
    """
    # Skip consistency check if the first answer is a refusal
    if "cannot answer" in answer_1.lower():
        return {"passed": True, "score": 1.0, "reason": "Refusal — consistency N/A"}

    answer_2 = chain.invoke(query)
    if "cannot answer" in answer_2.lower():
        return {"passed": False, "score": 0.0, "reason": "Second run refused while first did not"}

    emb = _get_embedding_model()
    vec_1 = emb.embed_query(answer_1)
    vec_2 = emb.embed_query(answer_2)
    sim = _cosine_similarity(vec_1, vec_2)

    if sim < threshold:
        return {"passed": False, "score": sim, "reason": f"Divergence: {sim:.2f} < {threshold}"}
    return {"passed": True, "score": sim, "reason": f"Consistency: {sim:.2f}"}