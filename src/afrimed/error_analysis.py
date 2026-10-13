import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import RESULTS_DIR
from .plots import save

NEGATION_RE = re.compile(r"\b(not|except|false|incorrect|least likely|untrue)\b", re.IGNORECASE)
NUMERIC_RE = re.compile(r"\d+(\.\d+)?\s*(mg|ml|mmhg|%|kg|g/dl|mmol|weeks?|years?|days?|hours?)", re.IGNORECASE)


def mcq_categories(df):
    return pd.DataFrame({
        "Negation (NOT/EXCEPT)": df.question.str.contains(NEGATION_RE),
        "Long clinical vignette (>60 words)": df.question_words > 60,
        "Numeric / lab values": df.question.str.contains(NUMERIC_RE),
        "Africa-specific context": df.mentions_africa.astype(bool),
        "'All/None of the above' option": df.options.str.contains(r"(all|none) of the above", case=False),
        "Multiple correct answers": df.multi_answer.astype(bool),
    })


def mcq_error_profile(name):
    df = pd.read_csv(RESULTS_DIR / name / "mcq_predictions.csv")
    cats = mcq_categories(df)
    overall_err = 1 - df.correct.mean()
    rows = [{"category": c, "n": int(cats[c].sum()), "error_rate": float(1 - df[cats[c]].correct.mean()),
             "share_of_errors": float((cats[c] & ~df.correct).sum() / max(1, (~df.correct).sum()))}
            for c in cats if cats[c].sum() > 0]
    out = pd.DataFrame(rows).sort_values("error_rate", ascending=False)
    out["overall_error_rate"] = overall_err
    return out


def repeated_ngrams(text, n=3):
    words = str(text).lower().split()
    grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(grams) - len(set(grams)) > 2


def saq_error_category(row, low_threshold):
    pred, ref = str(row.prediction or ""), str(row.rationale)
    if row.semantic_sim >= low_threshold:
        return "Acceptable"
    if len(pred.split()) < 3:
        return "Empty / too short"
    if repeated_ngrams(pred):
        return "Repetition / degenerate"
    if row.hit_max_tokens and row.semantic_sim < low_threshold:
        return "Truncated / rambling"
    if len(ref.split()) <= 6:
        return "Very short reference"
    if len(pred.split()) > 3 * len(ref.split()):
        return "Over-generic / verbose"
    return "Content mismatch (likely factual error)"


def saq_error_profile(name, low_threshold=0.5):
    df = pd.read_csv(RESULTS_DIR / name / "saq_predictions.csv")
    df["error_category"] = df.apply(saq_error_category, axis=1, low_threshold=low_threshold)
    return df


def plot_error_profiles(name):
    mcq = mcq_error_profile(name)
    saq = saq_error_profile(name)
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    axes[0].barh(mcq.category, mcq.error_rate, color="#C44E52")
    axes[0].axvline(mcq.overall_error_rate.iloc[0], c="k", ls="--", label="Overall error rate")
    for i, (r, n) in enumerate(zip(mcq.error_rate, mcq.n)):
        axes[0].text(r, i, f" {r:.2f} (n={n})", va="center", fontsize=8)
    axes[0].set(title=f"MCQ error rate by question category ({name})", xlabel="Error rate", xlim=(0, 1))
    axes[0].legend()
    counts = saq.error_category.value_counts()
    axes[1].pie(counts.values, labels=counts.index, autopct="%1.0f%%", startangle=90,
                colors=plt.cm.Set2(np.arange(len(counts))))
    axes[1].set(title="SAQ answer categories (semantic similarity < 0.5 = error)")
    save(fig, f"res_12_error_profile_{name}")
    return mcq, saq


def example_table(name, n=5):
    mcq = pd.read_csv(RESULTS_DIR / name / "mcq_predictions.csv")
    saq = pd.read_csv(RESULTS_DIR / name / "saq_predictions.csv").sort_values("semantic_sim")
    confident_wrong = mcq[~mcq.correct].sort_values("confidence", ascending=False).head(n)
    confident_right = mcq[mcq.correct].sort_values("confidence", ascending=False).head(n)
    cols = ["question", "prediction", "rationale", "semantic_sim", "rougeL"]
    return {
        "mcq_confident_wrong": confident_wrong[["question", "options", "gold", "pred", "confidence", "specialty"]],
        "mcq_confident_right": confident_right[["question", "options", "gold", "pred", "confidence", "specialty"]],
        "saq_best": saq.tail(n)[cols][::-1],
        "saq_worst": saq.head(n)[cols],
    }


def export_examples(name, n=5):
    tables = example_table(name, n)
    combined = pd.concat([df.assign(table=title) for title, df in tables.items()], ignore_index=True)
    combined.to_csv(RESULTS_DIR / name / "examples.csv", index=False)
    return tables
