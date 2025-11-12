#%% Import libraries
import os
import pandas as pd
import giskard
from openai import AzureOpenAI
from giskard import Model, Dataset
from giskard.rag import KnowledgeBase, generate_testset, evaluate, QATestset
from giskard.rag.metrics.ragas_metrics import (
    ragas_faithfulness, 
    ragas_context_recall, 
    ragas_context_precision, 
    ragas_answer_relevancy
    )
from giskard.llm import set_llm_model, set_embedding_model
from giskard.rag.base import AgentAnswer  
from app.rag import rag_answer             
from app.settings import settings
from openai import AzureOpenAI
from giskard.llm.client.openai import OpenAIClient
from ragas.run_config import RunConfig
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
# from giskard.llm import get_model, get_embedding_model
 
#%% configurations
CSV_PATH = os.getenv("CSV_PATH", "/gdpr_cased_articles_with_recitals.csv")
REPORT_HTML = "rag_eval_report.html"
TESTSET_JSONL = "gdpr_testset.jsonl"
NUM_QUESTIONS = 2
LANG = "en"
REQUIRED_COLS = {"article_id", "article_title", "article_text", "article_recitals"}

#%% Set the OpenAI API Key environment variable.
def setup_azure_judge():
    AZURE_OPENAI_ENDPOINT=os.getenv("NEW_AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_KEY=os.getenv("NEW_AZURE_OPENAI_ENDPOINT_API") 
    AZURE_OPENAI_API_VERSION=os.getenv("AZURE_OPENAI_API_VERSION") 
    AZURE_EMBED_DEPLOYMENT = os.getenv("NEW_AZURE_OPENAI_EMBEDDING_MODEL_DEPLOYMENT_NAME")
    AZURE_OPENAI_CHAT_MODEL_EVALUATION_DEPLOYMENT_NAME=os.getenv("AZURE_OPENAI_CHAT_MODEL_EVALUATION_DEPLOYMENT_NAME")
    
    if AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY and AZURE_OPENAI_API_VERSION and AZURE_EMBED_DEPLOYMENT and AZURE_OPENAI_CHAT_MODEL_EVALUATION_DEPLOYMENT_NAME:
        os.environ["AZURE_API_BASE"] = AZURE_OPENAI_ENDPOINT
        os.environ["AZURE_API_KEY"] = AZURE_OPENAI_KEY
        os.environ["AZURE_API_VERSION"] = AZURE_OPENAI_API_VERSION
        
        set_llm_model(f"azure/{AZURE_OPENAI_CHAT_MODEL_EVALUATION_DEPLOYMENT_NAME}")
        set_embedding_model(f"azure/{AZURE_EMBED_DEPLOYMENT}")
        
        print(f"[info] Using Azure judge model: azure/{AZURE_OPENAI_CHAT_MODEL_EVALUATION_DEPLOYMENT_NAME}")
        print(f"[info] Using Azure embedding model: azure/{AZURE_EMBED_DEPLOYMENT}")
    else:
        print("[warn] Azure judge/embedding env incomplete; using Giskard defaults (may be slower/different).")

# llm = get_model()                 
# emb = get_embedding_model() 
#%%
def _normalize_recitals(value) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    s = str(value).strip().strip('"').strip("'")
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]
#%%
def build_kb_from_csv(csv_path: str) -> KnowledgeBase:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV_PATH not found: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    def row_to_text(r):
        title = (r.get("article_title") or "").strip()
        body = (r.get("article_text") or "").strip()
        recs = _normalize_recitals(r.get("article_recitals"))
        parts = []
        if title: parts.append(title)
        if body: parts.append(body)
        if recs: parts.append("Recitals:\n" + ", ".join(recs))
        return "\n\n".join(parts)

    kb_df = pd.DataFrame({"text": [row_to_text(r) for _, r in df.iterrows()]})
    kb_df["text"] = kb_df["text"].astype(str).str.strip()
    kb_df = kb_df[kb_df["text"].str.len() > 10].drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"[info] KnowledgeBase rows: {len(kb_df)}")
    return KnowledgeBase.from_pandas(kb_df, columns=["text"])

#%%
def answer_fn(question: str) -> AgentAnswer:
    try:
        answer_md, sources = rag_answer(question)
        docs = []
        if sources:
            for src in sources:
                snip = src.get("snippet") if isinstance(src, dict) else None
                docs.append(snip if snip else str(src))
        return AgentAnswer(message=answer_md, documents=docs)
    except Exception as e:
        return AgentAnswer(message=f"[ERROR] {e}", documents=[])

#%%
if __name__ == "__main__":
    setup_azure_judge()
    
    kb = build_kb_from_csv(CSV_PATH)

    if os.path.exists(TESTSET_JSONL):
        testset = QATestset.load(TESTSET_JSONL)
        print(f"[info] Loaded testset: {TESTSET_JSONL} ({len(testset)} Qs)")

    else:
        print(f"[info] Generating testset with {NUM_QUESTIONS} questions...")
        testset = generate_testset(
            kb,
            num_questions=NUM_QUESTIONS,
            language=LANG,
            agent_description="A GDPR compliance bot that answers strictly from GDPR Articles & Recitals.",
        )
        testset.save(TESTSET_JSONL)
        print(f"[info] Saved testset to {TESTSET_JSONL}")

    metrics = [
        ragas_answer_relevancy,
        ragas_faithfulness,
        ragas_context_precision,
        ragas_context_recall,
    ]

    print("[info] Running evaluation...")
  
    report = evaluate(
        answer_fn,
        testset=testset,
        knowledge_base=kb,
        metrics=metrics,
    )
    report.to_html(REPORT_HTML)
    print(f"[info] Saved report to: {REPORT_HTML}")

# %%
