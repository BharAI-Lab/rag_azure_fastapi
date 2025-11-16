# LLM Evaluation using Giskard

## Comprehensive Information for Evaluating a RAG Pipeline

This section explains how to evaluate a Retrieval-Augmented Generation (RAG) chatbot using Giskard, including how to run scan (vulnerability & bias detection) and evaluate (metric-based QA evaluation).
It is written so that any developer can follow these instructions and adapt them for their own dataset and their own RAG architecture.

# 🧩 Overview

 ## `1. giskard.scan()` — **Vulnerability & Risk Detection**

- Detects hallucinations

- Detects prompt injection risks

- Detects missing safety alignment

- Detects performance issues and instability

Produces an HTML scan report summarizing vulnerabilities

---
## 2.  giskard.rag.evaluate() — Metric-Based LLM Evaluation

Evaluates the quality of your RAG answers using:

- Ragas Faithfulness

- Ragas Context Recall

- Ragas Context Precision

- Ragas Answer Relevancy

- Correctness Metric

Produces a detailed HTML evaluation report.

Both reports help validate the reliability, correctness, and robustness of your LLM-based system.

# 🚨 1. Giskard Scan Report

## Automated Vulnerability & Risk Analysis

**📄 Report File:** : ![Scan Report]
(https://htmlpreview.github.io/?https://raw.githubusercontent.com/BharAI-Lab/rag_azure_fastapi/main/Giskard_LLM_Evaluation/reports/giskard_scan_report.html)




![Scan Report](Giskard_LLM_Evaluation/reports/screenshots/scan.png "Giskard Scan Report Preview")

# 📐 2. Giskard Metric Evaluation Report

## Quantitative RAG Performance Evaluation (Ragas + Giskard)

| **Metric**                | **Description**                                       |
|---------------------------|-------------------------------------------------------|
| **Ragas Faithfulness**    | Checks if the answer is grounded in the retrieved context |
| **Ragas Context Precision** | Measures how relevant the retrieved documents are      |
| **Ragas Context Recall**  | Measures whether all necessary information was retrieved |
| **Ragas Answer Relevancy** | Verifies if the answer addresses the question          |
| **Correctness Metric**    | Logic + accuracy evaluation                            |


**📄 Report File:** : ![Metric Evaluation](Giskard_LLM_Evaluation/reports/Giskard_Metric_Evalution.html)

![Metric Evaluation](Giskard_LLM_Evaluation/reports/screenshots/eval.png "Metric Evaluation Report Preview")
