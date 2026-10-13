import re

from .config import LETTERS
from .data import truncate_words

SYSTEM_PROMPT = (
    "You are AfriMed Assistant, a medical question-answering assistant focused on healthcare in African settings. "
    "Give accurate, concise, evidence-based answers. For multiple-choice questions, begin with 'Answer:' followed by "
    "the letter(s) of the correct option(s), then explain briefly."
)
ANSWER_PREFIX = "Answer:"
REFUSAL = (
    "I am a medical assistant trained on African healthcare questions, so I can only answer health-related questions. "
    "Please ask me about symptoms, diseases, treatments, maternal and child health, or other medical topics."
)


def format_user(rec, references=None):
    parts = []
    if references:
        parts.append("Reference notes:\n" + "\n".join(f"- {r}" for r in references))
    if rec.get("context"):
        parts.append(f"Scenario: {rec['context']}")
    parts.append(f"Question: {rec['question']}")
    options = rec.get("options") or []
    if len(options) > 0:
        parts.append("Options:\n" + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options)))
        parts.append("Select the correct option(s).")
    return "\n\n".join(parts)


def format_target(rec):
    options = rec.get("options") or []
    rationale = truncate_words(rec.get("rationale") or "")
    if len(options) > 0:
        idx = list(rec["answer_idx"])
        letters = ", ".join(LETTERS[i] for i in idx)
        text = f"{ANSWER_PREFIX} {letters}"
        if len(idx) == 1:
            text += f". {options[idx[0]]}"
        if rationale:
            text += f"\nExplanation: {rationale}"
        return text
    return rationale


def build_prompt(tokenizer, kind, user, system=SYSTEM_PROMPT):
    if kind == "causal":
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system}\n\n{user}\n\nResponse:"


def parse_letters(text):
    head = text.strip().split("\n")[0]
    match = re.search(r"answer(?:\s+is)?\s*[:\-]?\s*(.*)", head, re.IGNORECASE)
    segment = (match.group(1) if match else head).split(".")[0][:40]
    return sorted(set(re.findall(r"\b([A-E])\b", segment)))
