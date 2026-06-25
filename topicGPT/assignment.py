import pandas as pd
from topicgpt_python.utils import *
from tqdm import tqdm
import regex
import traceback
from sentence_transformers import SentenceTransformer, util
import argparse
import os
import torch
from transformers import pipeline, BitsAndBytesConfig
import gc

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sbert = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

def clear_gpu():
    torch.cuda.empty_cache()
    gc.collect()

class HFClient:
    def __init__(self, model_id):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        print(f"Inicializando pipeline de Atribuição (4-bit): {model_id}")
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={
                "quantization_config": quantization_config,
                "low_cpu_mem_usage": True,
                "device_map": "auto",
            }
        )
        self.tokenizer = self.pipe.tokenizer

    def estimate_token_count(self, text):
        return len(self.tokenizer.encode(text))

    def iterative_prompt(self, prompt, max_tokens, temperature, top_p):
        messages = [{"role": "user", "content": prompt + "\n/no_think"}]
        out = self.pipe(
            messages,
            max_new_tokens=max_tokens,
            temperature=None if temperature == 0.0 else temperature,
            do_sample=False if temperature == 0.0 else True,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return out[0]['generated_text'][-1]['content']

def run_assignment(api_client, topics_root, df, assignment_prompt, context_len, max_tokens, verbose, out_file):
    tree_str = "\n".join(topics_root.to_topic_list(desc=True, count=False))

    start_idx = 0
    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            start_idx = sum(1 for line in f if line.startswith("=== Document "))
        if start_idx > 0:
            print(f"[Checkpoint] Retomando atribuição a partir do documento {start_idx}/{len(df)}")

    responses = []

    for i, (index, row) in enumerate(tqdm(df.iterrows(), total=df.shape[0], desc="Atribuindo", initial=start_idx)):
        if i < start_idx:
            continue

        clear_gpu()
        doc = row["text"]
        doc_id = f"doc_{index}"

        if api_client.estimate_token_count(tree_str) > context_len:
            doc_emb = sbert.encode(doc, convert_to_tensor=True)
            cos_sim = {top: util.cos_sim(sbert.encode(top, convert_to_tensor=True), doc_emb)
                       for top in tree_str.split("\n")}

            top_sorted = sorted(cos_sim, key=cos_sim.get, reverse=True)
            seed_str, current_len = "", 0
            while current_len < context_len and top_sorted:
                topic = top_sorted.pop(0)
                seed_str += topic + "\n"
                current_len += api_client.estimate_token_count(topic + "\n")
        else:
            seed_str = tree_str

        prompt = assignment_prompt.replace("{tree}", seed_str).replace("{Document}", doc)

        try:
            response = api_client.iterative_prompt(prompt, max_tokens, 0.0, 1.0)
            responses.append(response)
            if verbose:
                print(f"\n>>> Doc: {doc_id}\n>>> Atribuição:\n{response}\n" + "-"*40)
        except Exception:
            traceback.print_exc()
            responses.append("Error")

        with open(out_file, "a", encoding="utf-8") as f:
            f.write(f"=== Document {i} ===\n")
            f.write(responses[-1] + "\n\n")

    return responses

def run_assignment_round(api_client, model, data, prompt_file, topic_file, out_file, verbose=True):
    if api_client is None:
        api_client = HFClient(model)

    max_tokens = 1000
    context_limit = 32768
    context_len = context_limit - max_tokens - 5000

    with open(prompt_file, "r", encoding="utf-8") as f:
        assignment_prompt = f.read()

    topics_root = TopicTree().from_topic_list(topic_file, from_file=True)

    df = pd.read_json(data, lines=True)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    responses = run_assignment(api_client, topics_root, df, assignment_prompt, context_len, max_tokens, verbose, out_file=out_file)

    print(f"\nSucesso! Resultados salvos em: {out_file}")
    return api_client

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--data", type=str, default="topicGPT/data/input/sample.jsonl")
    parser.add_argument("--prompt_file", type=str, default="topicGPT/prompt/assignment.txt")
    parser.add_argument("--topic_file", type=str, default="topicGPT/data/output/sample/generation_1.md")
    parser.add_argument("--out_file", type=str, default="topicGPT/data/output/sample/assignment.txt")
    args = parser.parse_args()

    run_assignment_round(None, args.model, args.data, args.prompt_file, args.topic_file, args.out_file)

if __name__ == "__main__":
    main()
