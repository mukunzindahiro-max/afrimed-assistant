import json
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from .config import DATASET_ID, LETTERS, PROCESSED_DIR, RAW_DIR, SEED, ensure_dirs

SCENARIO_RE = re.compile(
    r"complains of (?P<symptoms>.*?) and thinks (?:s?he|they) (?:has|have) (?P<condition>.*?) and is going",
    re.IGNORECASE,
)
SPECIALTY_MERGE = {"Obstetric": "Obstetrics_and_Gynecology"}
MIN_ANSWER_WORDS = {"saq": 2, "consumer_queries": 4}
MAX_RATIONALE_WORDS = 220
NEAR_DUP_THRESHOLD = 0.9


def load_raw(use_cache=True):
    ensure_dirs()
    path = RAW_DIR / "afrimedqa_v2.parquet"
    if use_cache and path.exists():
        return pd.read_parquet(path)
    from datasets import load_dataset

    df = load_dataset(DATASET_ID, split="train").to_pandas()
    df.to_parquet(path, index=False)
    return df


def is_missing(value):
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return str(value).strip() in ("", "None", "nan")


def clean_text(value):
    if is_missing(value):
        return ""
    text = str(value).replace("\r", " ").replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_key(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def word_count(text):
    return len(str(text).split())


def truncate_words(text, limit=MAX_RATIONALE_WORDS):
    words = str(text).split()
    if len(words) <= limit:
        return str(text)
    return " ".join(words[:limit]) + " ..."


def parse_scenario(prompt):
    prompt = clean_text(prompt)
    if not prompt:
        return ""
    match = SCENARIO_RE.search(prompt)
    if match:
        symptoms = match.group("symptoms").strip(" ,.")
        condition = match.group("condition").strip(" ,.")
        subject = "A child" if "child" in prompt.split("complains")[0].lower() else "A patient"
        return f"{subject} has {symptoms} and suspects {condition}."
    return re.sub(r"\s*Question:\s*$", "", prompt).strip()


def parse_options(raw):
    if is_missing(raw):
        return []
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    keys = sorted(data, key=lambda k: int(re.sub(r"\D", "", k) or 0))
    return [clean_text(data[k]) for k in keys if clean_text(data[k])]


def parse_correct(raw, n_options):
    if is_missing(raw):
        return []
    idx = []
    for part in str(raw).split(","):
        digits = re.sub(r"\D", "", part)
        if digits and 1 <= int(digits) <= n_options:
            idx.append(int(digits) - 1)
    return sorted(set(idx))


def build_records(df):
    rows = []
    for r in df.itertuples(index=False):
        qtype = r.question_type
        options = parse_options(r.answer_options) if qtype == "mcq" else []
        answer_idx = parse_correct(r.correct_answer, len(options)) if options else []
        specialty = clean_text(r.specialty) or "Unspecified"
        rows.append(
            {
                "sample_id": r.sample_id,
                "question_type": qtype,
                "official_split": r.split,
                "question": clean_text(r.question_clean) or clean_text(r.question),
                "context": parse_scenario(r.prompt) if qtype == "consumer_queries" else "",
                "options": options,
                "answer_idx": answer_idx,
                "answer_letters": ",".join(LETTERS[i] for i in answer_idx),
                "multi_answer": len(answer_idx) > 1,
                "rationale": clean_text(r.answer_rationale),
                "specialty": SPECIALTY_MERGE.get(specialty, specialty),
                "tier": r.tier,
                "country": r.country,
                "discipline": clean_text(r.discipline) or "Unknown",
                "neg_percent": float(r.neg_percent) if not is_missing(r.neg_percent) else 0.0,
                "quality": None if is_missing(r.quality) else str(r.quality) == "True",
                "mentions_africa": bool(r.mentions_Africa == 1.0),
            }
        )
    out = pd.DataFrame(rows)
    out["question_words"] = out.question.map(word_count)
    out["rationale_words"] = out.rationale.map(lambda s: word_count(s) if s else 0)
    out["is_clean"] = (out.quality != False) & (out.neg_percent <= 0.5)
    return out


def supervised_mask(rec):
    is_mcq = (rec.question_type == "mcq") & (rec.options.map(len) >= 2) & (rec.answer_idx.map(len) >= 1)
    min_words = rec.question_type.map(MIN_ANSWER_WORDS).fillna(10**6)
    is_open = rec.question_type.isin(["saq", "consumer_queries"]) & (rec.rationale_words >= min_words)
    return is_mcq | is_open


def stratify_labels(series, min_count=10):
    counts = series.value_counts()
    return series.where(series.map(counts) >= min_count, "Other_rare")


def assign_splits(sup):
    sup = sup.copy()
    sup["split"] = ""
    mcq = sup[sup.question_type == "mcq"]
    strat = stratify_labels(mcq.specialty + "_" + mcq.multi_answer.astype(str))
    mcq_train, mcq_test = train_test_split(mcq.index, test_size=0.2, random_state=SEED, stratify=strat)
    strat_train = stratify_labels(strat.loc[mcq_train])
    mcq_train, mcq_val = train_test_split(mcq_train, test_size=0.125, random_state=SEED, stratify=strat_train)
    sup.loc[mcq_train, "split"] = "train"
    sup.loc[mcq_val, "split"] = "val"
    sup.loc[mcq_test, "split"] = "test"

    saq = sup[sup.question_type == "saq"]
    saq_train, saq_test = train_test_split(saq.index, test_size=0.2, random_state=SEED, stratify=saq.tier)
    saq_train, saq_val = train_test_split(saq_train, test_size=0.125, random_state=SEED,
                                          stratify=saq.tier.loc[saq_train])
    sup.loc[saq_train, "split"] = "train"
    sup.loc[saq_val, "split"] = "val"
    sup.loc[saq_test, "split"] = "test"

    cq = sup[sup.question_type == "consumer_queries"]
    cq_train, cq_val = train_test_split(cq.index, test_size=0.1, random_state=SEED)
    sup.loc[cq_train, "split"] = "train"
    sup.loc[cq_val, "split"] = "val"
    return sup


def remove_leakage(sup):
    held = sup[sup.split != "train"]
    train = sup[sup.split == "train"]
    held_keys = set(held.question.map(normalize_key))
    exact = train.question.map(normalize_key).isin(held_keys)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit(sup.question)
    sims = (vec.transform(train.question) @ vec.transform(held.question).T).max(axis=1).toarray().ravel()
    near = pd.Series(sims >= NEAR_DUP_THRESHOLD, index=train.index)
    held_answers = set(held[held.rationale_words >= 5].rationale.map(normalize_key))
    same_answer = (train.rationale_words >= 5) & train.rationale.map(normalize_key).isin(held_answers)
    drop = train.index[(exact | near | same_answer).values]
    stats = {"exact_overlap_removed": int(exact.sum()), "near_duplicate_removed": int((near & ~exact).sum()),
             "identical_answer_removed": int((same_answer & ~exact & ~near).sum())}
    return sup.drop(index=drop), sims, stats


def prepare(use_cache=True, save=True):
    raw = load_raw(use_cache)
    rec = build_records(raw)
    funnel = [("Raw rows", len(rec))]
    rec = rec[rec.question.str.len() > 0]
    funnel.append(("Non-empty question", len(rec)))
    rec = rec.assign(_key=rec.question_type + "|" + rec.question.map(normalize_key))
    rec = rec.drop_duplicates("_key").drop(columns="_key")
    funnel.append(("After de-duplication", len(rec)))

    sup_mask = supervised_mask(rec)
    sup = rec[sup_mask]
    funnel.append(("Has usable answer", len(sup)))
    sup = assign_splits(sup)
    sup, train_max_sims, leak_stats = remove_leakage(sup)
    funnel.append(("After leakage removal", len(sup)))

    unlabeled = rec[~sup_mask & (rec.question_type == "consumer_queries")].copy()
    unlabeled["split"] = np.where(unlabeled.official_split == "train", "train", "test")

    report = {
        "funnel": funnel,
        "leakage": leak_stats,
        "split_counts": sup.groupby(["question_type", "split"]).size().unstack(fill_value=0).to_dict(),
        "unlabeled_consumer_queries": unlabeled.split.value_counts().to_dict(),
        "clean_fraction_train": float(sup[sup.split == "train"].is_clean.mean()),
    }
    if save:
        ensure_dirs()
        for name in ("train", "val", "test"):
            sup[sup.split == name].to_json(PROCESSED_DIR / f"{name}.jsonl", orient="records", lines=True)
        unlabeled.to_json(PROCESSED_DIR / "unlabeled_questions.jsonl", orient="records", lines=True)
        np.save(PROCESSED_DIR / "train_max_sim_to_heldout.npy", train_max_sims)
        (PROCESSED_DIR / "data_report.json").write_text(json.dumps(report, indent=2))
    return rec, sup, unlabeled, report


def load_split(name):
    df = pd.read_json(PROCESSED_DIR / f"{name}.jsonl", lines=True, dtype={"answer_letters": str})
    df["answer_letters"] = df.answer_letters.fillna("")
    df["options"] = df.options.map(lambda v: list(v) if isinstance(v, (list, np.ndarray)) else [])
    df["answer_idx"] = df.answer_idx.map(lambda v: list(v) if isinstance(v, (list, np.ndarray)) else [])
    return df


def load_unlabeled():
    return pd.read_json(PROCESSED_DIR / "unlabeled_questions.jsonl", lines=True)


def knowledge_text(rec):
    if rec["options"]:
        answers = "; ".join(rec["options"][i] for i in rec["answer_idx"])
        body = f"{answers}. {rec['rationale']}" if rec["rationale"] else answers
    else:
        body = rec["rationale"]
    return truncate_words(f"Q: {rec['question']} A: {body}", 90)
