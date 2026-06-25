"""
Runner do pipeline TopicGPT para o dataset Bills.

Uso:
    python topicGPT/run_bills.py --phase generation
    python topicGPT/run_bills.py --phase refinement
    python topicGPT/run_bills.py --phase assignment

Fases executadas em ordem: generation → refinement → assignment.
"""

import sys
import os
import gc
import signal
import traceback
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

def signal_handler(signum, frame):
    print(f"\n[SIGNAL] Recebido sinal {signum} ({signal.Signals(signum).name})", flush=True)
    traceback.print_stack(frame)
    sys.exit(1)

for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT, signal.SIGUSR1, signal.SIGUSR2):
    signal.signal(sig, signal_handler)

sys.path.insert(0, os.path.dirname(__file__))

DATASETS_DIR = "EDA-ESG-Reports/Sampling-Method/data/bills"
OUTPUT_DIR = "topicGPT/data/output/bills"
PROMPT_DIR = "topicGPT/prompt"

GEN_MODEL = "Qwen/Qwen3-14B"
REFINE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
ASSIGN_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

ALL_ROUNDS = [
    {
        "name": "round_0_cps",
        "gen_data": f"{DATASETS_DIR}/generation_bills_CPS.jsonl",
        "assign_data": f"{DATASETS_DIR}/assignment_bills_CPS.jsonl",
    },
    {
        "name": "round_1_random",
        "gen_data": f"{DATASETS_DIR}/generation_bills_random1.jsonl",
        "assign_data": f"{DATASETS_DIR}/assignment_bills_random1.jsonl",
    },
    {
        "name": "round_2_random",
        "gen_data": f"{DATASETS_DIR}/generation_bills_random2.jsonl",
        "assign_data": f"{DATASETS_DIR}/assignment_bills_random2.jsonl",
    },
    {
        "name": "round_3_random",
        "gen_data": f"{DATASETS_DIR}/generation_bills_random3.jsonl",
        "assign_data": f"{DATASETS_DIR}/assignment_bills_random3.jsonl",
    },
]


def run_generation(rounds):
    from topicgpt_python.generation_1 import generate_topic_lvl1

    gen_prompt = f"{PROMPT_DIR}/generation_1_bills.txt"
    seed_file = f"{PROMPT_DIR}/seed_1_bills.md"

    client = None
    for r in rounds:
        out_dir = f"{OUTPUT_DIR}/{r['name']}"
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n>>> Geracao: {r['name']}")

        topics_root, client = generate_topic_lvl1(
            api_client=client,
            model=GEN_MODEL,
            data=r["gen_data"],
            prompt_file=gen_prompt,
            seed_file=seed_file,
            out_file=f"{out_dir}/generation_1.jsonl",
            topic_file=f"{out_dir}/generation_1.md",
            verbose=True,
        )

    del client
    torch.cuda.empty_cache()
    gc.collect()
    print("\nModelo de geracao liberado da VRAM.")


def run_refinement(rounds):
    from topicgpt_python.refinement import refine_topics_round

    refine_prompt = f"{PROMPT_DIR}/refinement_bills.txt"

    client = None
    for r in rounds:
        out_dir = f"{OUTPUT_DIR}/{r['name']}"
        print(f"\n>>> Refinamento: {r['name']}")

        client = refine_topics_round(
            api_client=client,
            model=REFINE_MODEL,
            prompt_file=refine_prompt,
            generation_file=f"{out_dir}/generation_1.jsonl",
            topic_file=f"{out_dir}/generation_1.md",
            out_file=f"{out_dir}/refinement_1.md",
            updated_file=f"{out_dir}/refinement_1.jsonl",
            verbose=True,
            remove=True,
            mapping_file=f"{out_dir}/refiner_mapping.json",
        )

    del client
    torch.cuda.empty_cache()
    gc.collect()
    print("\nModelo de refinamento liberado da VRAM.")


def run_assignment(rounds):
    from topicgpt_python.assignment import run_assignment_round

    assign_prompt = f"{PROMPT_DIR}/assignment_bills.txt"

    client = None
    for r in rounds:
        out_dir = f"{OUTPUT_DIR}/{r['name']}"
        topic_file = f"{out_dir}/refinement_1.md"
        print(f"\n>>> Atribuicao: {r['name']} (topicos: {topic_file})")

        client = run_assignment_round(
            api_client=client,
            model=ASSIGN_MODEL,
            data=r["assign_data"],
            prompt_file=assign_prompt,
            topic_file=topic_file,
            out_file=f"{out_dir}/assignment.txt",
            verbose=True,
        )

    del client
    torch.cuda.empty_cache()
    gc.collect()
    print("\nModelo de atribuicao liberado da VRAM.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline TopicGPT — Bills")
    parser.add_argument("--phase", choices=["generation", "refinement", "assignment", "all"], required=True)
    args = parser.parse_args()

    phases = ["generation", "refinement", "assignment"] if args.phase == "all" else [args.phase]

    for phase in phases:
        print("=" * 60)
        print(f"  BILLS — {phase.upper()} — {len(ALL_ROUNDS)} rounds")
        print("=" * 60)

        if phase == "generation":
            run_generation(ALL_ROUNDS)
        elif phase == "refinement":
            run_refinement(ALL_ROUNDS)
        else:
            run_assignment(ALL_ROUNDS)

        print(f"\n{phase.upper()} CONCLUIDA!")
