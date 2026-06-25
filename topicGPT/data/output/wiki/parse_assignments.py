import re
import json
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ROUNDS = {
    "round_0_cps":    "assignment_wiki_CPS.jsonl",
    "round_1_random": "assignment_wiki_random1.jsonl",
    "round_2_random": "assignment_wiki_random2.jsonl",
    "round_3_random": "assignment_wiki_random3.jsonl",
}

TOPIC_LINE_RE = re.compile(r"^\[1\]\s+(.*?)(?=:|\n|$)", re.MULTILINE)
SECTION_RE = re.compile(
    r"(?i)^(assignment|final\s+assignment|final\s+assignments|final\s+primary\s+assignment)\s*:",
    re.MULTILINE,
)
NA_KEYWORDS = ("not applicable", "not relevant", "not directly relevant", "not directly applicable")


def _clean_topic(raw):
    for sep in (" – ", " — ", " - Not ", " → "):
        idx = raw.find(sep)
        if idx > 0:
            raw = raw[:idx]
    return raw.strip().rstrip(".")


INLINE_TOPIC_RE = re.compile(r"\[1\]\s+(.*?)(?=:|\n|$)")


def _accept_topic(raw_topic, rest_of_line):
    if any(kw in rest_of_line.lower() for kw in NA_KEYWORDS):
        return None
    topic = _clean_topic(raw_topic)
    if not topic or len(topic) > 50 or "?" in topic or "(not in hierarchy)" in topic.lower():
        return None
    return topic


def _extract_topics(text, section_header_line=None):
    topics = []
    if section_header_line:
        for m in INLINE_TOPIC_RE.finditer(section_header_line):
            rest = section_header_line[m.end():]
            topic = _accept_topic(m.group(1), rest)
            if topic:
                topics.append(topic)
    for line in text.split("\n"):
        m = re.match(r"^\[1\]\s+(.*?)(?=:|\n|$)", line)
        if not m:
            continue
        rest_of_line = line[m.end():]
        topic = _accept_topic(m.group(1), rest_of_line)
        if topic:
            topics.append(topic)
    return topics


def parse_assignment_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = re.split(r"=== Document \d+ ===\n?", content)
    blocks = [b for b in blocks if b.strip()]

    predicted = []
    for block in blocks:
        section_matches = list(SECTION_RE.finditer(block))
        if section_matches:
            sec_start = section_matches[-1].start()
            header_line = block[sec_start:].split("\n")[0]
            section_text = block[sec_start:].split("\n", 1)[1] if "\n" in block[sec_start:] else ""
            topics = _extract_topics(section_text, section_header_line=header_line)
        else:
            topics = _extract_topics(block)

        seen = set()
        unique_topics = []
        for t in topics:
            if t.lower() not in seen:
                seen.add(t.lower())
                unique_topics.append(t)

        predicted.append(unique_topics)

    return predicted


def load_jsonl(filepath):
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    for round_name, jsonl_file in ROUNDS.items():
        print(f"\n>>> {round_name}")

        jsonl_path = os.path.join(BASE_DIR, jsonl_file)
        assignment_path = os.path.join(BASE_DIR, round_name, "assignment.txt")
        output_path = os.path.join(BASE_DIR, round_name, "assignment_parsed.csv")

        if not os.path.exists(assignment_path):
            print(f"  SKIP: {assignment_path} not found")
            continue

        docs = load_jsonl(jsonl_path)
        predicted = parse_assignment_file(assignment_path)

        print(f"  Docs no JSONL: {len(docs)}")
        print(f"  Blocos no assignment.txt: {len(predicted)}")

        n = min(len(docs), len(predicted))
        if len(docs) != len(predicted):
            print(f"  AVISO: tamanhos diferentes! Usando {n} documentos")

        rows = []
        for i in range(n):
            gt_topic = docs[i].get("topic", "")
            gt_subtopic = docs[i].get("subtopic", "")
            text = docs[i].get("text", "")
            rows.append({
                "text": text,
                "topic": gt_topic,
                "subtopic": gt_subtopic,
                "topics": [gt_topic] if gt_topic else [],
                "predicted_topics": predicted[i],
            })

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        empty = df["predicted_topics"].apply(lambda x: len(x) == 0).sum()
        avg_topics = df["predicted_topics"].apply(len).mean()
        print(f"  Docs sem tópico predito: {empty}/{n} ({100*empty/n:.1f}%)")
        print(f"  Média de tópicos por doc: {avg_topics:.2f}")
        print(f"  Salvo em: {output_path}")


if __name__ == "__main__":
    main()
