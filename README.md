# Fine-Tuning a Small Language Model for Summarization

This project, completed for the CSC 52082 “Introduction to Text Mining and NLP” course at École Polytechnique, explores the fine-tuning of a Small Language Model (SLM) for French text summarization.

**Summary:**  
5,000 French Wikipedia articles were collected and cleaned using the Wikipedia API. Synthetic summaries were generated with Qwen2.5-32B-Instruct through LangChain. The smaller Qwen2.5-0.5B-Instruct model was fine-tuned using LoRA and 4-bit quantization (BitsAndBytes). Evaluation with ROUGE and BERTScore showed consistent improvements over the baseline model.

**Main Files:**  
- `fetch_articles.ipynb` – Wikipedia data collection  
- `generate_summaries.ipynb` – Summary generation using LLMs  
- `slm_finetunig.ipynb` – Fine-tuning pipeline  
- `inference.py` – Inference and evaluation scripts  
- `SLM_summarization_project.pdf` – Full report  

**Authors:**  
Wajdi Maatouk and Aziz Bacha
