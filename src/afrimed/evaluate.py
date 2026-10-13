import json
import time

import numpy as np
import pandas as pd
import torch

from .config import LETTERS, PROCESSED_DIR, RESULTS_DIR, SEED
from .data import load_split
from .metrics import expected_calibration_error, generation_metrics, mcq_metrics, set_metrics
from .prompts import ANSWER_PREFIX, build_prompt, format_user, parse_letters


def letter_token_ids(tokenizer):
    prefix = tokenizer(ANSWER_PREFIX, add_special_tokens=False)["input_ids"]
    ids = []
    for letter in LETTERS:
        full = tokenizer(f"{ANSWER_PREFIX} {letter}", add_special_tokens=False)["input_ids"]
        if full[: len(prefix)] != prefix or len(full) != len(prefix) + 1:
            raise ValueError(f"Letter {letter} is not a single token after the answer prefix")
        ids.append(full[-1])
    return prefix, ids


def batches(items, size):
    for i in range(0, len(items), size):
        yield items[i: i + size]


def length_order(prompts):
    return np.argsort([-len(p) for p in prompts])


@torch.no_grad()
def letter_probs(model, tokenizer, kind, prompts, batch_size=16, max_len=1024):
    prefix, ids = letter_token_ids(tokenizer)
    device = next(model.parameters()).device
    order = length_order(prompts)
    out = np.zeros((len(prompts), len(LETTERS)), dtype=np.float32)
    for chunk in batches(list(order), batch_size):
        texts = [prompts[i] for i in chunk]
        if kind == "causal":
            enc = tokenizer([t + ANSWER_PREFIX for t in texts], return_tensors="pt", padding=True,
                            add_special_tokens=False).to(device)
            logits = model(**enc).logits[:, -1, :]
        else:
            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
            start = model.config.decoder_start_token_id
            dec = torch.tensor([[start] + prefix] * len(texts), device=device)
            logits = model(**enc, decoder_input_ids=dec).logits[:, -1, :]
        out[chunk] = torch.softmax(logits[:, ids].float(), dim=-1).cpu().numpy()
    return out


@torch.no_grad()
def generate(model, tokenizer, kind, prompts, max_new_tokens=128, batch_size=16, max_len=1024):
    device = next(model.parameters()).device
    order = length_order(prompts)
    texts_out = [""] * len(prompts)
    n_tokens = [0] * len(prompts)
    for chunk in batches(list(order), batch_size):
        texts = [prompts[i] for i in chunk]
        if kind == "causal":
            enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        else:
            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1,
                             pad_token_id=tokenizer.pad_token_id)
        if kind == "causal":
            gen = gen[:, enc["input_ids"].shape[1]:]
        for i, row in zip(chunk, gen):
            texts_out[i] = tokenizer.decode(row, skip_special_tokens=True).strip()
            n_tokens[i] = int((row != tokenizer.pad_token_id).sum())
    return texts_out, n_tokens


def build_prompts(tokenizer, kind, df, references=None):
    records = df.to_dict("records")
    return [build_prompt(tokenizer, kind, format_user(r, references[i] if references else None))
            for i, r in enumerate(records)]


def load_ood_probe():
    path = RESULTS_DIR / "ood" / "probe_questions.jsonl"
    return pd.read_json(path, lines=True) if path.exists() else None


def evaluate_model(cfg, model, tokenizer, knowledge_index=None, limit=None, use_bertscore=True):
    from .retrieval import retrieve_references

    start = time.time()
    out_dir = RESULTS_DIR / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    test = load_split("test")
    if limit:
        test = pd.concat([g.sample(n=min(limit, len(g)), random_state=SEED) for _, g in test.groupby("question_type")])
    refs = retrieve_references(knowledge_index, test.question.tolist(), k=cfg.rag_k) if cfg.rag else None
    test = test.reset_index(drop=True)
    prompts = build_prompts(tokenizer, cfg.kind, test, refs)
    bs = cfg.eval_batch_size

    mcq_mask = (test.question_type == "mcq").values
    mcq = test[mcq_mask].copy()
    mcq_prompts = [p for p, m in zip(prompts, mcq_mask) if m]
    probs = letter_probs(model, tokenizer, cfg.kind, mcq_prompts, bs)
    for j, letter in enumerate(LETTERS):
        mcq[f"p_{letter}"] = probs[:, j]
    mcq["pred"] = [LETTERS[i] for i in probs.argmax(axis=1)]
    mcq["confidence"] = probs.max(axis=1)
    mcq["gold"] = mcq.answer_letters
    mcq["correct"] = mcq.pred == mcq.gold

    multi_mask = mcq.multi_answer.values
    single = mcq[~multi_mask]
    gold_idx = np.array([LETTERS.index(g) for g in single.gold])
    metrics = {"name": cfg.name, "group": cfg.group, "description": cfg.description, "model_id": cfg.model_id,
               "method": cfg.method, "rag": cfg.rag}
    single_m = mcq_metrics(single.gold.tolist(), single.pred.tolist())
    metrics.update({f"mcq_{k}": v for k, v in single_m.items()})
    metrics["mcq_ece"] = expected_calibration_error(probs[~multi_mask], gold_idx)
    metrics["mcq_n"] = int(len(single))
    metrics["mcq_per_specialty"] = single.groupby("specialty").correct.mean().round(4).to_dict()

    multi_prompts = [p for p, m in zip(mcq_prompts, multi_mask) if m]
    multi_text, _ = generate(model, tokenizer, cfg.kind, multi_prompts, 24, bs) if multi_prompts else ([], [])
    mcq["generated"] = ""
    mcq.loc[mcq.index[multi_mask], "generated"] = multi_text
    multi_pred = [parse_letters(t) for t in multi_text]
    multi_gold = [g.split(",") for g in mcq[multi_mask].gold]
    mcq.loc[mcq.index[multi_mask], "pred"] = [",".join(p) for p in multi_pred]
    mcq.loc[mcq.index[multi_mask], "correct"] = [set(p) == set(g) for p, g in zip(multi_pred, multi_gold)]
    metrics.update({f"mcq_multi_{k}": v for k, v in set_metrics(multi_gold, multi_pred).items()})
    metrics["mcq_multi_n"] = int(multi_mask.sum())

    saq_mask = (test.question_type == "saq").values
    saq = test[saq_mask].copy()
    saq_prompts = [p for p, m in zip(prompts, saq_mask) if m]
    saq_text, saq_tokens = generate(model, tokenizer, cfg.kind, saq_prompts, cfg.gen_max_new_tokens, bs)
    saq["prediction"] = saq_text
    saq["gen_tokens"] = saq_tokens
    saq["hit_max_tokens"] = [n >= cfg.gen_max_new_tokens for n in saq_tokens]
    gen_summary, per_example = generation_metrics(saq_text, saq.rationale.tolist(), use_bertscore)
    for k, v in per_example.items():
        saq[k] = v
    metrics.update({f"saq_{k}": v for k, v in gen_summary.items()})
    metrics["saq_n"] = int(len(saq))

    probe = load_ood_probe()
    if probe is not None:
        probe_prompts = build_prompts(tokenizer, cfg.kind, probe.assign(context="", options=[[]] * len(probe)))
        probe["response"], _ = generate(model, tokenizer, cfg.kind, probe_prompts, min(80, cfg.gen_max_new_tokens), bs)
        probe.to_csv(out_dir / "ood_probe_responses.csv", index=False)

    if refs:
        saq["references"] = [" || ".join(r) for r, m in zip(refs, saq_mask) if m]
    metrics["eval_time_sec"] = time.time() - start
    keep = ["sample_id", "question", "specialty", "question_words", "mentions_africa", "tier", "multi_answer"]
    mcq[keep + ["options", "gold", "pred", "confidence", "correct", "generated"] +
        [f"p_{l}" for l in LETTERS]].to_csv(out_dir / "mcq_predictions.csv", index=False)
    saq.drop(columns=["options", "answer_idx"], errors="ignore").to_csv(out_dir / "saq_predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
