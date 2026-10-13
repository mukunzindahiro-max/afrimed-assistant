import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, roc_curve

from .config import FIGURES_DIR, LETTERS, RESULTS_DIR

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {"consumer_queries": "#4C72B0", "mcq": "#DD8452", "saq": "#55A868"}
TYPE_LABEL = {"consumer_queries": "Consumer query", "mcq": "MCQ", "saq": "SAQ"}


def save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=160, bbox_inches="tight")
    return fig


def plot_question_types(rec):
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = rec.groupby(["question_type", "official_split"]).size().unstack(fill_value=0)
    counts.index = counts.index.map(TYPE_LABEL)
    counts.plot(kind="bar", ax=ax, color=["#C44E52", "#4C72B0"], rot=0)
    for c in ax.containers:
        ax.bar_label(c, fontsize=9)
    ax.set(title="AfriMed-QA v2: questions by type and official split", xlabel="", ylabel="Questions")
    return save(fig, "eda_01_question_types")


def plot_answer_availability(rec):
    fig, ax = plt.subplots(figsize=(7, 4))
    has = rec.assign(has_reference=(rec.rationale_words > 0) | (rec.answer_idx.map(len) > 0))
    share = has.groupby("question_type").has_reference.agg(["sum", "count"])
    share["missing"] = share["count"] - share["sum"]
    share.index = share.index.map(TYPE_LABEL)
    share[["sum", "missing"]].rename(columns={"sum": "Has reference answer", "missing": "No reference answer"}).plot(
        kind="barh", stacked=True, ax=ax, color=["#55A868", "#CCCCCC"])
    ax.set(title="Reference answer availability", xlabel="Questions", ylabel="")
    return save(fig, "eda_02_answer_availability")


def plot_specialties(rec, top=25):
    fig, ax = plt.subplots(figsize=(8, 8))
    counts = rec[rec.specialty != "Unspecified"].specialty.value_counts().head(top)[::-1]
    ax.barh(counts.index.str.replace("_", " "), counts.values, color="#4C72B0")
    ax.set(title=f"Top {top} clinical specialties", xlabel="Questions")
    return save(fig, "eda_03_specialties")


def plot_countries(rec):
    fig, ax = plt.subplots(figsize=(8, 4))
    counts = rec.country.value_counts()
    ax.bar(counts.index, counts.values, color="#8172B3")
    ax.set_yscale("log")
    ax.set(title="Contributor country (log scale)", xlabel="Country code", ylabel="Questions")
    return save(fig, "eda_04_countries")


def plot_contributors(rec):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    rec.discipline.value_counts().plot(kind="barh", ax=axes[0], color="#64B5CD")
    axes[0].set(title="Contributor discipline", xlabel="Questions", ylabel="")
    pd.crosstab(rec.question_type.map(TYPE_LABEL), rec.tier).plot(kind="bar", ax=axes[1], rot=0,
                                                                   color=["#937860", "#DA8BC3"])
    axes[1].set(title="Crowdsourced vs expert tier", xlabel="", ylabel="Questions")
    return save(fig, "eda_05_contributors")


def plot_lengths(rec):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for qtype, g in rec.groupby("question_type"):
        sns.kdeplot(np.log10(g.question_words.clip(lower=1)), ax=axes[0], label=TYPE_LABEL[qtype],
                    color=PALETTE[qtype], fill=True, alpha=0.3)
    axes[0].set(title="Question length", xlabel="log10(words)")
    axes[0].legend()
    ans = rec[rec.rationale_words > 0]
    for qtype, g in ans.groupby("question_type"):
        axes[1].hist(g.rationale_words.clip(upper=400), bins=50, alpha=0.6, label=TYPE_LABEL[qtype],
                     color=PALETTE[qtype])
    axes[1].set(title="Reference answer / rationale length (clipped at 400)", xlabel="Words", ylabel="Count")
    axes[1].set_yscale("log")
    axes[1].legend()
    return save(fig, "eda_06_lengths")


def plot_mcq_answers(sup):
    mcq = sup[sup.question_type == "mcq"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    single = mcq[~mcq.multi_answer].answer_letters.value_counts().reindex(LETTERS)
    axes[0].bar(single.index, single.values, color="#DD8452")
    axes[0].axhline(single.sum() / 5, ls="--", c="k", label="Uniform")
    axes[0].set(title="Gold answer position (single-answer MCQs)", xlabel="Option", ylabel="Questions")
    axes[0].legend()
    n_correct = mcq.answer_idx.map(len).value_counts().sort_index()
    axes[1].bar(n_correct.index.astype(str), n_correct.values, color="#C44E52")
    axes[1].set_yscale("log")
    axes[1].set(title="Number of correct options per MCQ", xlabel="Correct options", ylabel="Questions (log)")
    return save(fig, "eda_07_mcq_answers")


def plot_funnel(report):
    stages, counts = zip(*report["funnel"])
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(stages[::-1], counts[::-1], color=sns.color_palette("Blues", len(stages)))
    ax.bar_label(bars, fontsize=9)
    ax.set(title="Data preparation funnel", xlabel="Rows")
    return save(fig, "eda_08_funnel")


def plot_split_composition(sup):
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = pd.crosstab(sup.split, sup.question_type.map(TYPE_LABEL)).reindex(["train", "val", "test"])
    counts.plot(kind="bar", stacked=True, ax=ax, rot=0, color=[PALETTE[k] for k in ("consumer_queries", "mcq", "saq")])
    for c in ax.containers:
        ax.bar_label(c, label_type="center", fontsize=8)
    ax.set(title="Final train / validation / test composition", xlabel="", ylabel="Examples")
    return save(fig, "eda_09_split_composition")


def plot_specialty_by_split(sup, top=15):
    top_spec = sup[sup.specialty != "Unspecified"].specialty.value_counts().head(top).index
    table = pd.crosstab(sup[sup.specialty.isin(top_spec)].specialty, sup.split, normalize="columns")
    table = table[["train", "val", "test"]]
    table.index = table.index.str.replace("_", " ")
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.heatmap(table, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax, cbar_kws={"label": "Share of split"})
    ax.set(title="Specialty distribution per split (stratification check)", xlabel="", ylabel="")
    return save(fig, "eda_10_specialty_by_split")


def plot_quality(rec):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(rec.neg_percent, bins=20, color="#937860")
    axes[0].axvline(0.5, c="r", ls="--", label="Filter threshold")
    axes[0].set(title="Contributor negative-review rate", xlabel="neg_percent", ylabel="Rows")
    axes[0].legend()
    rec.quality.map({True: "Thumbs up", False: "Thumbs down"}).fillna("Not reviewed").value_counts().plot(
        kind="bar", ax=axes[1], rot=0, color=["#CCCCCC", "#55A868", "#C44E52"])
    axes[1].set(title="Reviewer quality flag", ylabel="Rows")
    sup = rec[(rec.rationale_words > 0) | (rec.answer_idx.map(len) > 0)]
    sup.groupby(sup.question_type.map(TYPE_LABEL)).is_clean.mean().plot(kind="bar", ax=axes[2], rot=0,
                                                                         color="#64B5CD")
    axes[2].set(title="Share passing quality filter (answered rows)", ylabel="Fraction", ylim=(0, 1))
    return save(fig, "eda_11_quality")


def plot_token_lengths(sup, tokenizer, max_len=768):
    from .prompts import build_prompt, format_target, format_user

    sample = sup[sup.split == "train"]
    lengths = []
    for rec in sample.to_dict("records"):
        p = len(tokenizer(build_prompt(tokenizer, "causal", format_user(rec)), add_special_tokens=False).input_ids)
        t = len(tokenizer(format_target(rec), add_special_tokens=False).input_ids)
        lengths.append((rec["question_type"], p, t))
    df = pd.DataFrame(lengths, columns=["type", "prompt", "target"])
    df["total"] = df.prompt + df.target
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for qtype, g in df.groupby("type"):
        axes[0].hist(g.total, bins=50, alpha=0.6, label=TYPE_LABEL[qtype], color=PALETTE[qtype])
    axes[0].axvline(max_len, c="r", ls="--", label=f"max_len={max_len}")
    axes[0].set(title="Training example length (Qwen2.5 tokens)", xlabel="Tokens", ylabel="Examples")
    axes[0].legend()
    sns.boxplot(data=df.melt(id_vars="type", value_vars=["prompt", "target"]), x="variable", y="value", hue="type",
                ax=axes[1], showfliers=False)
    axes[1].set(title="Prompt vs target tokens", xlabel="", ylabel="Tokens")
    save(fig, "eda_12_token_lengths")
    return df


def plot_top_terms(sup, n=15):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (qtype, g) in zip(axes, sup.groupby("question_type")):
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3, max_features=5000)
        X = vec.fit_transform(g.question)
        scores = pd.Series(np.asarray(X.mean(axis=0)).ravel(), index=vec.get_feature_names_out()).nlargest(n)[::-1]
        ax.barh(scores.index, scores.values, color=PALETTE[qtype])
        ax.set(title=f"Top TF-IDF terms: {TYPE_LABEL[qtype]}", xlabel="Mean TF-IDF")
    return save(fig, "eda_13_top_terms")


def plot_leakage(train_max_sims, threshold=0.9):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(train_max_sims, bins=60, color="#4C72B0")
    ax.axvline(threshold, c="r", ls="--", label=f"Removal threshold {threshold}")
    ax.set_yscale("log")
    ax.set(title="Max TF-IDF similarity of each training question to any val/test question",
           xlabel="Cosine similarity", ylabel="Training questions (log)")
    ax.legend()
    return save(fig, "eda_14_leakage_check")


def load_all_metrics():
    rows = []
    for path in sorted(RESULTS_DIR.glob("*/metrics.json")):
        if path.parent.name == "ood":
            continue
        m = json.loads(path.read_text())
        log_path = path.parent / "train_log.json"
        if log_path.exists():
            log = json.loads(log_path.read_text())
            m.update({"trainable_params": log["params"]["trainable_params"],
                      "trainable_pct": log["params"]["trainable_pct"],
                      "train_time_min": log["train_time_sec"] / 60, "final_val_loss": log["final_val_loss"]})
        rows.append({k: v for k, v in m.items() if not isinstance(v, dict)})
    return pd.DataFrame(rows)


def results_table(df):
    cols = ["name", "group", "mcq_accuracy", "mcq_macro_f1", "mcq_ece", "mcq_multi_exact_match", "saq_rougeL",
            "saq_bleu", "saq_chrf", "saq_bertscore_f1", "saq_semantic_sim", "trainable_params", "train_time_min",
            "final_val_loss"]
    return df[[c for c in cols if c in df.columns]].sort_values("mcq_accuracy", ascending=False)


def plot_experiment_comparison(df):
    metrics = [m for m in ["mcq_accuracy", "saq_rougeL", "saq_bertscore_f1", "saq_semantic_sim"] if m in df]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6), sharey=True)
    order = df.sort_values("mcq_accuracy").name
    colors = sns.color_palette("tab10", df.group.nunique())
    group_color = dict(zip(sorted(df.group.unique()), colors))
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        d = df.set_index("name").loc[order]
        ax.barh(d.index, d[metric].fillna(0), color=[group_color[g] for g in d.group])
        ax.set(title=metric, xlabel="Score")
        for i, v in enumerate(d[metric]):
            if pd.notna(v):
                ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in group_color.values()]
    fig.legend(handles, group_color.keys(), loc="lower center", ncol=len(group_color), bbox_to_anchor=(0.5, -0.04))
    return save(fig, "res_01_experiment_comparison")


def plot_loss_curves(names):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for name in names:
        path = RESULTS_DIR / name / "train_log.json"
        if not path.exists():
            continue
        hist = pd.DataFrame(json.loads(path.read_text())["log_history"])
        tr = hist.dropna(subset=["loss"]) if "loss" in hist else pd.DataFrame()
        ev = hist.dropna(subset=["eval_loss"]) if "eval_loss" in hist else pd.DataFrame()
        if len(tr):
            axes[0].plot(tr.epoch, tr.loss, label=name, alpha=0.8)
        if len(ev):
            axes[1].plot(ev.epoch, ev.eval_loss, marker="o", label=name)
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].set(title="Validation loss", xlabel="Epoch", ylabel="Cross-entropy")
    for ax in axes:
        ax.legend(fontsize=7)
    return save(fig, "res_02_loss_curves")


def plot_lora_ablation(df):
    rank = df[df.name.isin(["lora_qwen05_r8", "lora_qwen05_r16", "lora_qwen05_r32"])].copy()
    if rank.empty:
        return None
    rank["r"] = rank.name.str.extract(r"_r(\d+)").astype(int)
    rank = rank.sort_values("r")
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(rank.r, rank.mcq_accuracy, "o-", c="#DD8452", label="MCQ accuracy")
    ax1.plot(rank.r, rank.saq_rougeL, "s-", c="#55A868", label="SAQ ROUGE-L")
    ax1.set(xlabel="LoRA rank r", ylabel="Score", xticks=rank.r, title="Effect of LoRA rank")
    ax2 = ax1.twinx()
    ax2.bar(rank.r, rank.trainable_params / 1e6, alpha=0.2, width=3, color="grey", label="Trainable params (M)")
    ax2.set_ylabel("Trainable params (M)")
    ax2.grid(False)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right")
    return save(fig, "res_03_lora_rank")


def plot_specialty_heatmap(names, min_count=10):
    test = pd.concat([pd.read_csv(RESULTS_DIR / n / "mcq_predictions.csv").assign(model=n) for n in names
                      if (RESULTS_DIR / n / "mcq_predictions.csv").exists()])
    test = test[~test.multi_answer]
    counts = test[test.model == names[0]].specialty.value_counts()
    keep = counts[counts >= min_count].index
    table = test[test.specialty.isin(keep)].pivot_table(index="specialty", columns="model", values="correct",
                                                        aggfunc="mean")[names]
    table.index = [f"{s.replace('_', ' ')} (n={counts[s]})" for s in table.index]
    table = table.sort_values(names[-1])
    fig, ax = plt.subplots(figsize=(1.6 * len(names) + 4, 0.38 * len(table) + 2))
    sns.heatmap(table, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax)
    ax.set(title="MCQ accuracy by specialty", xlabel="", ylabel="")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return save(fig, "res_04_specialty_heatmap")


def load_mcq(name):
    df = pd.read_csv(RESULTS_DIR / name / "mcq_predictions.csv")
    return df[~df.multi_answer]


def plot_confusion(name):
    df = load_mcq(name)
    cm = confusion_matrix(df.gold, df.pred, labels=LETTERS, normalize="true")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=LETTERS, yticklabels=LETTERS, ax=axes[0])
    axes[0].set(title=f"Confusion matrix (row-normalised): {name}", xlabel="Predicted", ylabel="Gold")
    dist = pd.DataFrame({"Gold": df.gold.value_counts(normalize=True), "Predicted": df.pred.value_counts(normalize=True)})
    dist.reindex(LETTERS).fillna(0).plot(kind="bar", ax=axes[1], rot=0, color=["#4C72B0", "#DD8452"])
    axes[1].set(title="Answer-position bias", xlabel="Option", ylabel="Share")
    return save(fig, f"res_05_confusion_{name}")


def plot_calibration(names, n_bins=10):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    for name in names:
        df = load_mcq(name)
        bins = pd.cut(df.confidence, np.linspace(0, 1, n_bins + 1))
        g = df.groupby(bins, observed=True).agg(conf=("confidence", "mean"), acc=("correct", "mean"))
        axes[0].plot(g.conf, g.acc, "o-", label=name)
        axes[1].hist(df.confidence, bins=20, alpha=0.5, label=name)
    axes[0].set(title="Reliability diagram (MCQ)", xlabel="Confidence", ylabel="Accuracy")
    axes[1].set(title="Confidence distribution", xlabel="Max option probability", ylabel="Questions")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    return save(fig, "res_06_calibration")


def plot_saq_distributions(names, metric="semantic_sim"):
    frames = [pd.read_csv(RESULTS_DIR / n / "saq_predictions.csv").assign(model=n) for n in names
              if (RESULTS_DIR / n / "saq_predictions.csv").exists()]
    df = pd.concat(frames)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    sns.violinplot(data=df, x="model", y=metric, ax=axes[0], cut=0, inner="quartile")
    axes[0].set(title=f"SAQ per-answer {metric}", xlabel="")
    plt.setp(axes[0].get_xticklabels(), rotation=30, ha="right")
    sns.barplot(data=df, x="model", y=metric, hue="tier", ax=axes[1], errorbar=("ci", 95))
    axes[1].set(title=f"SAQ {metric} by tier (95% CI)", xlabel="")
    plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")
    return save(fig, "res_07_saq_distributions")


def plot_length_scatter(name):
    df = pd.read_csv(RESULTS_DIR / name / "saq_predictions.csv")
    df["pred_words"] = df.prediction.fillna("").str.split().str.len()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(df.rationale_words, df.pred_words, alpha=0.5, c=df.semantic_sim, cmap="viridis")
    lim = max(df.rationale_words.quantile(0.98), df.pred_words.max())
    axes[0].plot([0, lim], [0, lim], "k--")
    axes[0].set(xlim=(0, lim), ylim=(0, lim), title=f"Answer length: generated vs reference ({name})",
                xlabel="Reference words", ylabel="Generated words")
    bins = pd.qcut(df.rationale_words, 4, duplicates="drop")
    df.groupby(bins, observed=True)[["rougeL", "semantic_sim"]].mean().plot(kind="bar", ax=axes[1], rot=15)
    axes[1].set(title="Score by reference-length quartile", xlabel="Reference words", ylabel="Score")
    return save(fig, f"res_08_length_{name}")


def plot_efficiency(df):
    d = df.dropna(subset=["trainable_params"])
    if d.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for _, r in d.iterrows():
        axes[0].scatter(r.trainable_params / 1e6, r.mcq_accuracy, s=80)
        axes[0].annotate(r["name"], (r.trainable_params / 1e6, r.mcq_accuracy), fontsize=7, xytext=(4, 4),
                         textcoords="offset points")
        axes[1].scatter(r.train_time_min, r.mcq_accuracy, s=80)
        axes[1].annotate(r["name"], (r.train_time_min, r.mcq_accuracy), fontsize=7, xytext=(4, 4),
                         textcoords="offset points")
    axes[0].set_xscale("log")
    axes[0].set(title="Accuracy vs trainable parameters", xlabel="Trainable params (M, log)", ylabel="MCQ accuracy")
    axes[1].set(title="Accuracy vs training time", xlabel="Training time (min)", ylabel="MCQ accuracy")
    return save(fig, "res_09_efficiency")


def plot_accuracy_slices(names):
    rows = []
    for name in names:
        df = load_mcq(name)
        df["Length"] = pd.cut(df.question_words, [0, 15, 30, 60, 1000], labels=["<=15", "16-30", "31-60", ">60"])
        df["Mentions Africa"] = df.mentions_africa.map({True: "Yes", False: "No"})
        for col in ("Length", "Mentions Africa"):
            for k, g in df.groupby(col, observed=True):
                rows.append({"model": name, "slice": col, "value": str(k), "accuracy": g.correct.mean(), "n": len(g)})
    d = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, col in zip(axes, ("Length", "Mentions Africa")):
        sns.barplot(data=d[d.slice == col], x="value", y="accuracy", hue="model", ax=ax)
        ax.set(title=f"MCQ accuracy by question {col.lower()}", xlabel=col, ylabel="Accuracy")
        ax.legend(fontsize=7)
    save(fig, "res_10_accuracy_slices")
    return d


def plot_ood(metrics_path=None):
    test = pd.read_csv(RESULTS_DIR / "ood" / "test_scores.csv")
    report = json.loads((metrics_path or RESULTS_DIR / "ood" / "metrics.json").read_text())
    methods = ["tfidf_knn", "embedding_knn", "embedding_logreg"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    for m in methods:
        fpr, tpr, _ = roc_curve(test.label, test[m])
        axes[0].plot(fpr, tpr, label=f"{m} (AUROC={report[m]['auroc']:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set(title="Out-of-domain detection ROC", xlabel="False positive rate (OOD accepted)",
                ylabel="True positive rate (medical accepted)")
    axes[0].legend(fontsize=8)
    for src, g in test.groupby("source"):
        sns.kdeplot(g.embedding_logreg, ax=axes[1], label=src, fill=True, alpha=0.3, clip=(0, 1))
    axes[1].axvline(report["embedding_logreg"]["threshold"], c="r", ls="--", label="Threshold")
    axes[1].set(title="Domain score by source (embedding + logistic regression)", xlabel="P(in-domain)")
    axes[1].legend(fontsize=8)
    acc = pd.DataFrame({m: report[m]["per_source_acceptance"] for m in methods})
    acc.plot(kind="bar", ax=axes[2], rot=15)
    axes[2].set(title="Acceptance rate per source", ylabel="Accepted as medical", ylim=(0, 1.05))
    return save(fig, "res_11_ood_detection")
