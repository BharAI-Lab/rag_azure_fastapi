from typing import List, Tuple, Dict, Any
from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_azure_ai.vectorstores.azuresearch import AzureSearch
from langchain_core.documents import Document
from .settings import settings

"""
Three-tier RAG logic with neutral-professional tone:

A) Grounded Answer (GDPR Dataset)
   - Trigger: Query is GDPR-related AND top match strong enough.
   - Behavior: Answer ONLY from retrieved context; include citations.
   - Label prefix: "**Grounded Answer (GDPR Dataset):**"

B) Hybrid GDPR Answer (No/weak match, still GDPR-related)
   - Trigger: Query is GDPR-related BUT top match below threshold OR grounded model returns sentinel.
   - Behavior: Clearly state no dataset answer, then provide concise general guidance (GPT-4o).
   - Label prefix: "No direct answer found in the GDPR dataset." + "**General GDPR Guidance (GPT-4o):**"

C) Off-Topic Neutral Answer (Non-GDPR)
   - Trigger: Query is NOT GDPR-related.
   - Behavior: Respectfully state out of scope, then give a short factual answer (GPT-4o).
   - Label prefix: "Your question is outside the scope of GDPR." + "**General Information (GPT-4o):**"

Return signature remains unchanged: (answer_markdown: str, sources: list[dict])
"""

# -----------------------------
# Clients
# -----------------------------
chat_client = AzureOpenAI(
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_key=settings.AZURE_OPENAI_KEY,
    api_version=settings.AZURE_OPENAI_API_VERSION,
)

embeddings = AzureOpenAIEmbeddings(
    model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    openai_api_key=settings.AZURE_OPENAI_KEY,
)

vs = AzureSearch(
    azure_search_endpoint=settings.AZURE_SEARCH_ENDPOINT,
    azure_search_key=settings.AZURE_SEARCH_API_KEY,
    index_name=settings.AZURE_SEARCH_INDEX_NAME,
    embedding_function=embeddings.embed_query,
)

# -----------------------------
# Config (safe defaults if not present in settings)
# -----------------------------
# Strong retrieval threshold for top-1 score (tune as needed).
#STRONG_SCORE: float = float(getattr(settings, "RAG_STRONG_SCORE", 0.55))
STRONG_SCORE: float = float(0.55)
TOP_K: int = int(getattr(settings, "TOP_K", 4))
TEMPERATURE: float = float(getattr(settings, "TEMPERATURE", 0.2))
MAX_TOKENS: int = int(getattr(settings, "MAX_TOKENS", 800))

# Sentinel emitted by grounded prompt when context lacks explicit answer
SENTINEL_TEXT = "there is no answer based on the retrieval from the vector store"

# -----------------------------
# Prompts
# -----------------------------
CLASSIFIER_SYSTEM = (
    "You label user queries as either GDPR or NON-GDPR.\n"
    "Reply with exactly one token: GDPR or NON-GDPR."
)

GROUNDED_SYSTEM_PROMPT = (
    "You are a GDPR Compliance Expert. Answer ONLY using the provided context.\n"
    "If the answer is not explicitly stated in the context, respond exactly with:\n"
    "'There is no answer based on the retrieval from the vector store.'\n\n"
    "Rules:\n"
    "1) Use only facts from the context.\n"
    "2) Include Article and Recital references when present.\n"
    "3) Be concise and precise."
)

HYBRID_SYSTEM_PROMPT = (
    "You are a GDPR Compliance Expert. The retrieved dataset did not provide a direct answer, "
    "so you may use your own knowledge. Provide a short, clear, professional explanation that "
    "would be useful to someone asking about GDPR."
)

OFFTOPIC_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. The user question is not related to GDPR. "
    "Provide a brief, neutral, factual, and respectful response."
)

# -----------------------------
# Helper functions
# -----------------------------
def classify_query(query: str) -> str:
    """
    Classify query intent: returns 'GDPR' or 'NON-GDPR'.
    Uses a tiny LLM call; exact match (no substring bug).
    """
    resp = chat_client.chat.completions.create(
        model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
        temperature=0,
        max_tokens=3,
        messages=[
            {"role": "system", "content": CLASSIFIER_SYSTEM},
            {"role": "user", "content": f"Query: {query}"},
        ],
    )
    label = (resp.choices[0].message.content or "").strip().upper()
    if label == "GDPR":
        return "GDPR"
    if label == "NON-GDPR":
        return "NON-GDPR"
    # Safe default (bias to GDPR so we try retrieval when unsure)
    return "GDPR"


def _retrieve(query: str, k: int) -> List[Tuple[Document, float]]:
    return vs.similarity_search_with_relevance_scores(query=query, k=k)


def _context(docs: List[Tuple[Document, float]]) -> str:
    return "\n\n---\n\n".join(d[0].page_content for d in docs)


def _sources(docs: List[Tuple[Document, float]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for d, score in docs:
        md = d.metadata or {}
        items.append(
            {
                "article_id": md.get("article_id", ""),
                "article_title": md.get("article_title", ""),
                "chunk_id": md.get("chunk_id", -1),
                "snippet": (d.page_content or "")[:300],
                "score": float(score) if score is not None else None,
            }
        )
    return items


def _top1_score(docs: List[Tuple[Document, float]]) -> float:
    return float(docs[0][1]) if docs else 0.0


def _is_sentinel(text: str) -> bool:
    return SENTINEL_TEXT in (text or "").strip().lower()


# -----------------------------
# Public entrypoint
# -----------------------------
def rag_answer(query: str):
    """
    Main RAG flow:
      1) Classify query as GDPR or NON-GDPR.
      2) Retrieve top-K chunks.
      3) If GDPR & strong match: grounded answer from context (with citations).
      4) If GDPR & weak/no match OR grounded returns sentinel: hybrid GDPR guidance.
      5) If NON-GDPR: respectful factual answer; do not return dataset sources.
    Returns: (answer_markdown, sources_list)
    """
    # 1) Intent classification
    intent = classify_query(query)  # 'GDPR' | 'NON-GDPR'

    # 2) Retrieval
    docs = _retrieve(query, TOP_K)
    sources = _sources(docs)  # prepared for UI
    ctx = _context(docs)
    best_score = _top1_score(docs)

    # 3) Branches
    if intent == "GDPR" and best_score >= STRONG_SCORE:
        # A) Grounded answer (strictly from context)
        grounded = chat_client.chat.completions.create(
            model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {query}"},
            ],
        )
        text = (grounded.choices[0].message.content or "").strip()

        # If model emits sentinel → fallback to hybrid
        if _is_sentinel(text):
            guidance = chat_client.chat.completions.create(
                model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": HYBRID_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
            ).choices[0].message.content.strip()

            answer = (
                "No direct answer found in the GDPR dataset.\n\n"
                "**General GDPR Guidance (GPT-4o):**\n\n" + guidance
            )
            return answer, sources

        # Normal grounded answer
        answer = "**Grounded Answer (GDPR Dataset):**\n\n" + text
        return answer, sources

    if intent == "GDPR":
        # B) Hybrid GDPR answer (dataset weak/no match)
        guidance = chat_client.chat.completions.create(
            model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": HYBRID_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        ).choices[0].message.content.strip()

        answer = (
            "No direct answer found in the GDPR dataset.\n\n"
            "**General GDPR Guidance (GPT-4o):**\n\n" + guidance
        )
        return answer, sources

    # C) Off-topic neutral answer (NON-GDPR)
    general = chat_client.chat.completions.create(
        model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": OFFTOPIC_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    ).choices[0].message.content.strip()

    answer = (
        "Your question is outside the scope of GDPR.\n\n"
        "**General Information (GPT-4o):**\n\n" + general
    )
    # Return no sources to avoid misleading citations on non-GDPR questions
    return answer, []
