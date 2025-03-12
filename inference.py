#!/usr/bin/env python3

import sys
import os
import pandas as pd
import torch
from langchain.schema import Document

# ---------------------
# Model + Pipeline Setup
# ---------------------

import bitsandbytes as bnb
from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from langchain import HuggingFacePipeline
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Some models require an explicit pad token (reuse EOS if needed)
tokenizer.pad_token = tokenizer.eos_token

print("Creating pipeline...")
text_gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    do_sample=False,   # do_sample=False is faster & more deterministic
    temperature=0.1,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    device_map="auto"
)

llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

# ---------------------
# Summarization Chain
# ---------------------
prompt_template = """Vous trouverez ci-dessous un article :
```{text}```

### Objectif :
- Résumez cet article de manière claire, concise et informative.
- Conservez les informations essentielles tout en éliminant les détails superflus.
- Structurez le résumé en plusieurs phrases bien formulées.

### Contraintes :
- Le résumé doit être en français naturel et fluide.
- Ne pas inclure d’opinions personnelles ni d’informations non présentes dans le texte d'origine.
- Maintenir un ton neutre et objectif.

### Résumé :
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
summarize_chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)

def generate_summary(article_text: str) -> str:
    """Generate a summary for the given article text using the pre-initialized pipeline and chain."""
    docs = [Document(page_content=article_text)]
    # Use no_grad/inference_mode for a small speed boost
    with torch.inference_mode():
        try:
            summary = summarize_chain.run(docs)
            return summary
        except Exception as e:
            return f"Error: {e}"

# ---------------------
# Main Inference Logic
# ---------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python inference.py /path/to/split_XX.csv")
        sys.exit(1)

    csv_path = os.path.expanduser(sys.argv[1])
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)

    print(f"Reading chunk CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    num_articles = len(df)
    if num_articles == 0:
        print("No articles found in the provided CSV.")
        sys.exit(0)

    # Prepare output directory and output file name
    output_dir = os.path.expanduser("~/NLP_project/output")
    os.makedirs(output_dir, exist_ok=True)
    chunk_basename = os.path.splitext(os.path.basename(csv_path))[0]
    output_file = os.path.join(output_dir, f"{chunk_basename}_summaries.csv")

    print(f"Will save final results to: {output_file}")

    results = []
    check_interval = max(1, num_articles // 10)  # Save partial results every 10% or so

    for i, row in df.iterrows():
        article_text = row.get("Text", "")
        summary = generate_summary(article_text)
        torch.cuda.empty_cache()

        results.append({
            "article": article_text,
            "summary": summary
        })

        # Save partial results every 10%
        if (i+1) % check_interval == 0:
            partial_df = pd.DataFrame(results)
            partial_df.to_csv(output_file, index=False)
            print(f"Partial save at {i+1}/{num_articles} articles")

    # Final save
    final_df = pd.DataFrame(results)
    final_df.to_csv(output_file, index=False)
    print(f"Done! Final output saved to {output_file}")

if __name__ == "__main__":
    main()
