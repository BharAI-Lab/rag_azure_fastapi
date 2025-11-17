import sys
import os

# Update the path based on your repository
PROJECT_ROOT = "/Users/user/Desktop/rag_azure_fastapi"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("Added project root to sys.path:", PROJECT_ROOT)

# Import Required libraries
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
from giskard.rag.metrics import correctness_metric
 
# Variable Configurations
CSV_PATH = os.getenv("CSV_PATH", "/gdpr_cased_articles_with_recitals.csv")
REPORT_HTML = "/Giskard_LLM_Evaluation/reports/Giskard_Metric_Evalution.html"
TESTSET_JSONL = "/Giskard_LLM_Evaluation/Evaluation_Generated_Dataset/gdpr_testset.jsonl"
NUM_QUESTIONS = 50
LANG = "en"
REQUIRED_COLS = {"article_id", "article_title", "article_text", "article_recitals"}

# Set the OpenAI API Key environment variable.
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

def _normalize_recitals(value) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    s = str(value).strip().strip('"').strip("'")
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

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
   
def predict_dataframe(df: pd.DataFrame):
    """
    Adapter for Giskard. Input: DataFrame with a 'question' column.
    Output: list[str] with plain text answers (no sources).
    """
    if "question" not in df.columns:
        raise ValueError("Input DataFrame must have a 'question' column.")
    outputs = []
    for q in df["question"].astype(str).tolist():
        try:
            answer_md, sources = rag_answer(q)
        except Exception as e:
            answer_md = f"[ERROR] {e}"
        outputs.append(answer_md)
    return outputs


if __name__ == "__main__":
    setup_azure_judge()
    
    kb = build_kb_from_csv(CSV_PATH)

    if os.path.exists(TESTSET_JSONL):
        testset = QATestset.load(TESTSET_JSONL)
        print(f"\033[31m[info] Loaded testset: {TESTSET_JSONL} ({len(testset)} Qs)\033[0m")

    else:
        print(f"\033[31m[info] Generating testset with {NUM_QUESTIONS} questions...\033[0m")
        testset = generate_testset(
            kb,
            num_questions=NUM_QUESTIONS,
            language=LANG,
            agent_description="A GDPR compliance bot that answers strictly from GDPR Articles & Recitals.",
        )
        testset.save(TESTSET_JSONL)
        
        print(f"\033[31m[info] Saved testset to {TESTSET_JSONL}\033[0m")

    print("\033[31mRunning Gisckard's scanning for model vulnerabilities.\033[0m")
    
    giskard_model = Model(
    predict_dataframe,
    model_type="text_generation",
    name="GDPR RAG chatbot",
    description=(
        "Rag based chat bot for GDPR dataset with three types of answer the chat answers,\
        1. Strict grounded answers when retrieval is strong; 2. hybrid guidance if weak;\
        3. off-topic guard for non-GDPR queries;"
    ),
    feature_names=["question"])
    
    # Load testset.JSONL into a DataFrame
    df_scan = pd.read_json(TESTSET_JSONL, lines=True)
    # Keep only the column the model expects as input
    df_scan = df_scan[["question"]]
    
    giskard_dataset = Dataset(
    df_scan,
    target=None,                 # no label column for generation
    column_types={"question": "text"}  # tell Giskard what the input column is
    )

    report = giskard.scan(giskard_model,giskard_dataset)
    report.to_html("/Giskard_LLM_Evaluation/reports/Giskard_Scan.html")
    
    print(f"\033[31mSaved Giskard's scan report and procedding for Giskard evaluation metric report.\033[0m")
    
    metrics = [
        correctness_metric,
        ragas_answer_relevancy,
        ragas_faithfulness,
        ragas_context_precision,
        ragas_context_recall,
    ]

    print("\033[31mRunning evaluation...\033[0m")
  
    report = evaluate(
        answer_fn,
        testset=testset,
        knowledge_base=kb,
        metrics=metrics,
    )
    report.to_html(REPORT_HTML)
    
    print(f"\033[31mSaved Giskard metric evaluatio report.\033[0m")

