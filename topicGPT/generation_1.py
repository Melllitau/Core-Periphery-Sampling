import pandas as pd
from topicgpt_python.utils import *
from tqdm import tqdm
import regex
import traceback
import argparse
import os
import torch
from transformers import pipeline, BitsAndBytesConfig
import gc

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def clear_gpu():
    torch.cuda.empty_cache()
    gc.collect()

class HFClient:
    def __init__(self, model_id):
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            print("Aviso: HF_TOKEN não encontrado.")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        print(f"Inicializando pipeline (4-bit): {model_id}")
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={
                "quantization_config": quantization_config,
                "low_cpu_mem_usage": True,
                "device_map": "auto",
            },
            token=hf_token
        )
        self.tokenizer = self.pipe.tokenizer

    def estimate_token_count(self, text):
        return len(self.tokenizer.encode(text))

    def truncating(self, text, max_tokens):
        tokens = self.tokenizer.encode(text)
        return self.tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)

    def iterative_prompt(self, prompt, max_tokens, temperature, top_p=1.0, verbose=False):
        messages = [{"role": "user", "content": prompt + "\n/no_think"}]
        do_sample = temperature > 1e-3
        out = self.pipe(
            messages,
            max_new_tokens=max_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return out[0]['generated_text'][-1]['content']

def prompt_formatting(generation_prompt, api_client, doc, topics_list, context_len, verbose):
    topic_str = "\n".join([topic.split(":")[0].strip() for topic in topics_list])

    prompt = generation_prompt.replace("{Document}", doc).replace("{Topics}", topic_str)

    total_len = api_client.estimate_token_count(prompt)
    if verbose:
        print(f"Token count do prompt final: {total_len}")

    return prompt

def generate_topics(topics_root, topics_list, context_len, df, seed_file, api_client, generation_prompt, temperature, max_tokens, top_p, verbose, out_file, topic_file, early_stop=100):
    responses = []
    topic_format = regex.compile(r"^\[(\d+)\] ([\w\s]+):(.+)")

    start_idx = 0
    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            start_idx = sum(1 for _ in f)
        if start_idx > 0:
            print(f"[Checkpoint] Retomando geração a partir do documento {start_idx}/{len(df)}")

    for i, (index, row) in enumerate(tqdm(df.iterrows(), total=df.shape[0], initial=start_idx)):
        if i < start_idx:
            continue

        clear_gpu()

        doc_str = row["text"]
        doc_id = f"doc_{index}"

        prompt = prompt_formatting(generation_prompt, api_client, doc_str, topics_list, context_len, verbose)

        try:
            response = api_client.iterative_prompt(prompt, max_tokens, temperature, top_p=top_p, verbose=verbose)

            topics = [t.strip() for t in response.split("\n")]
            for t in topics:
                match = regex.match(topic_format, t)
                if not match: continue
                lvl, name, desc = int(match[1]), match[2].strip(), match[3].strip()
                if lvl != 1: continue

                dups = topics_root.find_duplicates(name, lvl)
                if dups:
                    dups[0].count += 1
                else:
                    topics_root._add_node(lvl, name, 1, desc, topics_root.root)
                    topics_list = topics_root.to_topic_list(desc=False, count=False)

            if verbose:
                print(f"\n--- Processando: {doc_id} ---")
                print(f"Resposta:\n{response}\n" + "-"*20)
            responses.append(response)

        except Exception:
            traceback.print_exc()
            responses.append("Error")

        row_copy = row.copy()
        row_copy["responses"] = responses[-1]
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(row_copy.to_json() + "\n")
        topics_root.to_file(topic_file)

    return responses, topics_list, topics_root

def generate_topic_lvl1(api_client, model, data, prompt_file, seed_file, out_file, topic_file, verbose):
    if api_client is None:
        api_client = HFClient(model_id=model)

    max_tokens, temperature, top_p = 500, 0.0, 1.0
    context = 32768
    context_len = context - max_tokens

    if not os.path.exists(data):
        raise FileNotFoundError(f"Arquivo {data} não encontrado.")

    with open(data, 'r', encoding='utf-8') as f:
        df = pd.read_json(f, lines=True)

    with open(prompt_file, "r", encoding="utf-8") as f:
        generation_prompt = f.read()

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    if os.path.exists(topic_file) and os.path.exists(out_file):
        n_done = sum(1 for _ in open(out_file, "r", encoding="utf-8"))
        if n_done > 0 and n_done < len(df):
            print(f"[Checkpoint] Carregando árvore de tópicos de {topic_file} ({n_done} docs já processados)")
            topics_root = TopicTree().from_topic_list(topic_file, from_file=True)
        else:
            topics_root = TopicTree().from_seed_file(seed_file)
    else:
        topics_root = TopicTree().from_seed_file(seed_file)

    topics_list = topics_root.to_topic_list(desc=True, count=False)

    responses, topics_list, topics_root = generate_topics(
        topics_root, topics_list, context_len, df, seed_file, api_client,
        generation_prompt, temperature, max_tokens, top_p, verbose,
        out_file=out_file, topic_file=topic_file,
    )

    topics_root.to_file(topic_file)

    return topics_root, api_client

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--data", type=str, default="topicGPT/data/input/sample.jsonl")
    parser.add_argument("--prompt_file", type=str, default="topicGPT/prompt/generation_1.txt")
    parser.add_argument("--seed_file", type=str, default="topicGPT/prompt/seed_1.md")
    parser.add_argument("--out_file", type=str, default="topicGPT/data/output/sample/generation_1.jsonl")
    parser.add_argument("--topic_file", type=str, default="topicGPT/data/output/sample/generation_1.md")
    parser.add_argument("--verbose", type=bool, default=True)
    args = parser.parse_args()

    generate_topic_lvl1(None, args.model, args.data, args.prompt_file, args.seed_file, args.out_file, args.topic_file, args.verbose)
