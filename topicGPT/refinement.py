import pandas as pd
import torch
import os
import regex
import json
import traceback
from topicgpt_python.utils import *
import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sbert = None


def get_sbert():
    global sbert
    if sbert is None:
        from sentence_transformers import SentenceTransformer
        sbert = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return sbert


def clear_gpu():
    torch.cuda.empty_cache()
    gc.collect()


class HFClient:
    def __init__(self, model_id):
        from transformers import pipeline

        print(f"Inicializando pipeline de Refinamento (bfloat16): {model_id}")
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.tokenizer = self.pipe.tokenizer

    def iterative_prompt(self, prompt, max_tokens, temperature, top_p):
        messages = [{"role": "user", "content": prompt + "\n/no_think"}]
        out = self.pipe(
            messages,
            max_new_tokens=max_tokens,
            temperature=None if temperature == 0.0 else temperature,
            do_sample=False if temperature == 0.0 else True,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return out[0]["generated_text"][-1]["content"]


def compute_all_pairs(topic_sent, threshold=0.5):
    """Compute all similar topic pairs once using SentenceTransformer.
    Returns sorted list of (score, i, j) tuples above threshold.
    """
    import numpy as np
    from numpy.linalg import norm

    model = get_sbert()
    embeddings = model.encode(topic_sent, convert_to_numpy=True)

    norms = norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = embeddings / norms
    cosine_scores = normalized @ normalized.T

    n = len(cosine_scores)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            s = float(cosine_scores[i][j])
            if s > threshold:
                pairs.append((s, i, j))

    pairs.sort(reverse=True)
    print(f"  Computed {len(pairs)} pairs above threshold {threshold}")
    return pairs


def select_pairs(sorted_pairs, topic_sent, all_pairs, num_pair=2):
    """Select next pairs to merge from pre-computed sorted pairs list."""
    selected_pairs = []
    for score, i, j in sorted_pairs:
        if len(selected_pairs) >= num_pair:
            break
        if sorted([topic_sent[i], topic_sent[j]]) not in all_pairs:
            selected_pairs.append([topic_sent[i], topic_sent[j]])
            all_pairs.append(sorted([topic_sent[i], topic_sent[j]]))

    return [item for sublist in selected_pairs for item in sublist], all_pairs


def merge_topics(
    topics_root, mapping, refinement_prompt, api_client,
    temperature, max_tokens, top_p, verbose, checkpoint_dir=None,
):
    topic_sent = topics_root.to_topic_list(desc=True, count=False)

    sorted_pairs = compute_all_pairs(topic_sent, threshold=0.5)

    global sbert
    if sbert is not None:
        del sbert
        sbert = None
        gc.collect()

    all_pairs = []
    orig_new = mapping
    ckpt_file = os.path.join(checkpoint_dir, "refine_checkpoint.json") if checkpoint_dir else None
    if ckpt_file and os.path.exists(ckpt_file):
        with open(ckpt_file, "r") as f:
            ckpt = json.load(f)
        all_pairs = ckpt["all_pairs"]
        orig_new = ckpt["mapping"]
        for old_name, new_name in orig_new.items():
            if old_name != new_name:
                topics_root = topics_root.update_tree([(old_name, 1)], new_name, "")
        print(f"  [Checkpoint] Retomando de {len(all_pairs)} pares já processados, {len(orig_new)} merges")

    new_pairs, all_pairs = select_pairs(sorted_pairs, topic_sent, all_pairs, num_pair=2)

    if len(new_pairs) <= 1 and verbose:
        print("  No topic pairs to be merged.")

    responses = []

    pattern_topic = regex.compile(
        r"^\[(\d+)\]\s*([\w\s\-',]+?)\s*:\s*(.+?)\s*\(([^)]+)\)\s*$"
    )

    iteration = 0
    while len(new_pairs) > 1:
        iteration += 1
        clear_gpu()
        refiner_prompt = refinement_prompt.format(Topics="\n".join(new_pairs))
        if verbose:
            print(f"\n  Merging: {new_pairs}")

        try:
            response = api_client.iterative_prompt(
                refiner_prompt, max_tokens, temperature, top_p
            )
            responses.append(response)
            if verbose:
                print(f"  Response: {response}")

            merges = response.split("\n")

            for merge in merges:
                match = pattern_topic.match(merge.strip())
                if match:
                    lvl = int(match.group(1))
                    name = match.group(2).strip()
                    desc = match.group(3).strip()
                    originals_str = match.group(4).strip()

                    orig_topics = []
                    for part in originals_str.split(","):
                        part = part.strip()
                        lvl_match = regex.match(r"\[(\d+)\]\s*", part)
                        if lvl_match:
                            part = part[lvl_match.end():]
                        if ":" in part:
                            part = part.split(":")[0]
                        part = part.strip()
                        if part:
                            orig_topics.append(part)

                    tree_names = {n.name.lower() for n in topics_root.root.descendants}
                    matched = [t for t in orig_topics if t.lower() in tree_names]
                    if len(matched) < 2:
                        pair_pattern = regex.compile(r"^\[(\d+)\]\s*(.+?)(?:\s*:\s*.*)?$")
                        for p in new_pairs:
                            m = pair_pattern.match(p.strip())
                            if m:
                                pname = m.group(2).strip()
                                if pname.lower() in tree_names and pname.lower() != name.lower():
                                    if pname not in orig_topics:
                                        orig_topics.append(pname)

                    original_topics = [(t, lvl) for t in orig_topics]
                    topics_root = topics_root.update_tree(
                        original_topics, name, desc
                    )
                    for orig in original_topics:
                        orig_new[orig[0]] = name
                    print(f"  Merged: {[o[0] for o in original_topics]} -> [{lvl}] {name}")

        except Exception:
            print("  Error when calling API!")
            traceback.print_exc()

        if ckpt_file and iteration % 50 == 0:
            with open(ckpt_file, "w") as f:
                json.dump({"all_pairs": all_pairs, "mapping": orig_new}, f)
            print(f"  [Checkpoint] Salvo após {iteration} iterações")

        new_pairs, all_pairs = select_pairs(
            sorted_pairs, topic_sent, all_pairs, num_pair=2
        )

    if ckpt_file:
        with open(ckpt_file, "w") as f:
            json.dump({"all_pairs": all_pairs, "mapping": orig_new}, f)

    return responses, topics_root, orig_new


def remove_topics(topics_root, verbose, threshold=0.01):
    total_count = sum(node.count for node in topics_root.root.children)
    threshold_count = total_count * threshold
    removed = False

    for node in list(topics_root.root.children):
        if node.count < threshold_count and node.lvl == 1:
            if verbose:
                print(f"  Removing low-freq topic: {node.name} ({node.count} counts)")
            node.parent = None
            removed = True

    if not removed and verbose:
        print("  No topics removed.")

    return topics_root


def replace_topic_key(text, mapping):
    for key, value in mapping.items():
        if key != value and key in text:
            text = text.replace(key, value)
    return text


def update_generation_file(generation_file, updated_file, mapping, mapping_file=None):
    df = pd.read_json(generation_file, lines=True)

    response_column = (
        "refined_responses" if "refined_responses" in df.columns else "responses"
    )
    responses = df[response_column].tolist()
    updated_responses = []
    for response in responses:
        updated_response = "\n".join(
            [replace_topic_key(s, mapping) for s in response.split("\n")]
        )
        updated_responses.append(updated_response)

    df["refined_responses"] = updated_responses
    df.to_json(updated_file, lines=True, orient="records")

    if mapping_file:
        with open(mapping_file, "w") as f:
            json.dump(mapping, f, indent=4)


def refine_topics_round(
    api_client, model, prompt_file, generation_file, topic_file,
    out_file, updated_file, verbose=True, remove=True, mapping_file=None,
):
    if api_client is None:
        api_client = HFClient(model)

    max_tokens, temperature, top_p = 500, 0.0, 1.0

    topics_root = TopicTree().from_topic_list(topic_file, from_file=True)

    print(f"  Tópicos antes do refinamento: {len(list(topics_root.root.children))}")

    with open(prompt_file, "r", encoding="utf-8") as f:
        refinement_prompt = f.read()

    mapping_org = {}
    if mapping_file and os.path.exists(mapping_file):
        with open(mapping_file, "r") as f:
            mapping_org = json.load(f)

    checkpoint_dir = os.path.dirname(out_file)
    responses, updated_topics_root, mapping = merge_topics(
        topics_root, mapping_org, refinement_prompt, api_client,
        temperature, max_tokens, top_p, verbose,
        checkpoint_dir=checkpoint_dir,
    )

    if remove:
        updated_topics_root = remove_topics(updated_topics_root, verbose)

    update_generation_file(
        generation_file, updated_file, mapping, mapping_file
    )

    updated_topics_root.to_file(out_file)

    print(f"  Tópicos após refinamento: {len(list(updated_topics_root.root.children))}")
    print(f"  Salvo em: {out_file}")

    return api_client
