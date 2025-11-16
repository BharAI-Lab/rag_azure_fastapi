import os
import pandas as pd
import giskard
from giskard import Model, Dataset
from app.rag import rag_answer             
from app.settings import settings


#%%
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

#%%
giskard_model = Model(
    #model=predict_dataframe,
    predict_dataframe,
    model_type="text_generation",
    name="GDPR RAG chatbot",
    description=(
        "Rag based chat bot for GDPR dataset with three types of answer the chat answers,\
        1. Strict grounded answers when retrieval is strong; 2. hybrid guidance if weak;\
        3. off-topic guard for non-GDPR queries;"
    ),
    feature_names=["question"]
)
#%%
# seed_df = pd.DataFrame({
#     "question": [
#         "What are the lawful bases for processing under GDPR?",
#         "When is consent valid under GDPR Article 7?",
#         "What is the capital of Australia?"  # off-topic probe
#     ]
# })

#%%
# giskard_ds = Dataset(
#     df=seed_df,
#     name="seed-questions",
#     column_types={"question": "text"},
#     target=None
# )
#%%
if __name__ == "__main__":
    print("Running Giskard scan… this will call live Azure endpoints used by rag_answer().")
    #report = giskard.scan(giskard_model, giskard_ds,only=["hallucination"])  # dataset optional but recommended
    report = giskard.scan(giskard_model,only=["hallucination"])  # dataset optional but recommended
    #report = giskard.scan(giskard_model,only=["hallucination"])  # dataset optional but recommended
    report_path = "giskard_scan_report2.html"
    report.to_html(report_path)
    print(f"Saved scan report to: {report_path}")
# %%
from IPython.display import display
display(report)
# %%
