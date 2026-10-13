import json

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve

from .config import MODELS_DIR, PROCESSED_DIR, RAW_DIR, RESULTS_DIR, SEED
from .data import load_split, load_unlabeled, normalize_key
from .retrieval import EmbeddingIndex

MEDICAL_TERMS = [
    "medic", "health", "disease", "immun", "pharma", "antibiot", "bacteri", "virus", "infect", "cancer", "cell",
    "hospital", "nurs", "surg", "anatom", "gene", "dna", "brain", "heart", "blood", "drug", "nutri", "vaccin",
    "epidem", "pathogen", "physiolog", "symptom", "tuberculosis", "malaria", "hiv", "aids", "diabet", "biolog",
    "chloroplast", "mitochond", "digest", "doctor", "clinic", "mental", "psych", "plague", "pandemic", "pain",
]
OOD_DIR = MODELS_DIR / "ood"
CACHE = str(RAW_DIR / "hf_cache")


def is_medical(text):
    text = str(text).lower()
    return any(t in text for t in MEDICAL_TERMS)


def load_ood_sources(n_train=6000, n_val=800, n_test=1200):
    from datasets import load_dataset

    rng = np.random.default_rng(SEED)
    squad_train = load_dataset("rajpurkar/squad", split="train", cache_dir=CACHE).to_pandas()
    squad_val = load_dataset("rajpurkar/squad", split="validation", cache_dir=CACHE).to_pandas()
    frames = {}
    train = squad_train[~squad_train.title.map(is_medical) & ~squad_train.question.map(is_medical)]
    train = train.drop_duplicates("question")
    frames["train"] = train.sample(n=min(n_train, len(train)), random_state=SEED).assign(source="squad")

    val_pool = squad_val[~squad_val.title.map(is_medical) & ~squad_val.question.map(is_medical)]
    val_pool = val_pool.drop_duplicates("question")
    titles = val_pool.title.unique()
    rng.shuffle(titles)
    half = set(titles[: len(titles) // 2])
    frames["val"] = val_pool[val_pool.title.isin(half)].sample(n=n_val, random_state=SEED).assign(source="squad")
    frames["test"] = val_pool[~val_pool.title.isin(half)].sample(n=n_test, random_state=SEED).assign(source="squad")
    try:
        web = load_dataset("Stanford/web_questions", split="test", cache_dir=CACHE).to_pandas()
        web = web[~web.question.map(is_medical)].drop_duplicates("question").assign(source="web_questions")
        frames["test"] = pd.concat([frames["test"], web[["question", "source"]]], ignore_index=True)
    except Exception as exc:
        print(f"web_questions unavailable: {exc}")
    return {k: v[["question", "source"]].reset_index(drop=True) for k, v in frames.items()}


def in_domain_sources():
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    unlabeled = load_unlabeled()
    unl_train = unlabeled[unlabeled.split == "train"]
    unl_test = unlabeled[unlabeled.split == "test"].sample(frac=1.0, random_state=SEED)
    mid = len(unl_test) // 2
    pick = lambda df, src: df[["question"]].assign(source=src)
    reference = pd.concat([pick(train, "afrimed_labeled"), pick(unl_train, "afrimed_consumer")], ignore_index=True)
    val_in = pd.concat([pick(val, "afrimed_labeled"), pick(unl_test.iloc[:mid], "afrimed_consumer")],
                       ignore_index=True)
    test_in = pd.concat([pick(test, "afrimed_labeled"), pick(unl_test.iloc[mid:], "afrimed_consumer")],
                        ignore_index=True)
    reference = reference[~reference.question.map(normalize_key).isin(
        set(val_in.question.map(normalize_key)) | set(test_in.question.map(normalize_key)))]
    return reference.reset_index(drop=True), val_in, test_in


def verify_reference_isolation(threshold=0.9):
    reference, val_in, test_in = in_domain_sources()
    held = pd.concat([val_in.assign(split="val"), test_in.assign(split="test")], ignore_index=True)
    ref_keys = set(reference.question.map(normalize_key))
    exact = held.question.map(normalize_key).isin(ref_keys)
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(pd.concat([reference.question, held.question]))
    sims = np.asarray((vec.transform(reference.question) @ vec.transform(held.question).T).max(axis=1).todense()).ravel()
    near = sims >= threshold
    return {
        "reference_size": int(len(reference)),
        "reference_sources": reference.source.value_counts().to_dict(),
        "heldout_size": {"val": int((held.split == "val").sum()), "test": int((held.split == "test").sum())},
        "exact_overlap_val": int(exact[held.split == "val"].sum()),
        "exact_overlap_test": int(exact[held.split == "test"].sum()),
        "reference_with_tfidf_ge_threshold": int(near.sum()),
        "reference_with_tfidf_ge_threshold_by_source": reference[near].source.value_counts().to_dict(),
        "threshold": threshold,
    }, reference.assign(max_sim_to_heldout=sims)


def binary_report(labels, scores, threshold):
    pred = scores >= threshold
    fpr, tpr, _ = roc_curve(labels, scores)
    fpr_at_95 = float(fpr[np.searchsorted(tpr, 0.95)]) if (tpr >= 0.95).any() else 1.0
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr_in": float(average_precision_score(labels, scores)),
        "aupr_out": float(average_precision_score(1 - labels, -scores)),
        "macro_f1": float(f1_score(labels, pred, average="macro")),
        "in_domain_acceptance": float(pred[labels == 1].mean()),
        "ood_rejection": float(1 - pred[labels == 0].mean()),
        "fpr_at_95_tpr": fpr_at_95,
        "threshold": float(threshold),
    }


def best_threshold(labels, scores):
    candidates = np.quantile(scores, np.linspace(0.01, 0.99, 197))
    f1s = [f1_score(labels, scores >= t, average="macro") for t in candidates]
    return float(candidates[int(np.argmax(f1s))])


def run_ood_experiment():
    reference, val_in, test_in = in_domain_sources()
    ood = load_ood_sources()
    val = pd.concat([val_in.assign(label=1), ood["val"].assign(label=0)], ignore_index=True)
    test = pd.concat([test_in.assign(label=1), ood["test"].assign(label=0)], ignore_index=True)

    tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(reference.question)
    ref_tfidf = tfidf.transform(reference.question)
    tfidf_score = lambda q: np.asarray((tfidf.transform(q) @ ref_tfidf.T).max(axis=1).todense()).ravel()

    index = EmbeddingIndex().build(reference.question.tolist(), [{"i": i} for i in range(len(reference))])
    knn_score = lambda q: index.max_similarity(list(q))

    ood_train = ood["train"]
    lr_x = np.vstack([index.embeddings, index.encode(ood_train.question.tolist())])
    lr_y = np.concatenate([np.ones(len(reference)), np.zeros(len(ood_train))])
    logreg = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(lr_x, lr_y)
    lr_score = lambda q: logreg.predict_proba(index.encode(list(q)))[:, 1]

    methods = {"tfidf_knn": tfidf_score, "embedding_knn": knn_score, "embedding_logreg": lr_score}
    report, thresholds = {}, {}
    for name, fn in methods.items():
        val_scores = fn(val.question)
        thresholds[name] = best_threshold(val.label.values, val_scores)
        test[name] = fn(test.question)
        report[name] = binary_report(test.label.values, test[name].values, thresholds[name])
        report[name]["per_source_acceptance"] = (
            test.assign(acc=test[name] >= thresholds[name]).groupby("source").acc.mean().round(4).to_dict())
        val[name] = val_scores

    out_dir = RESULTS_DIR / "ood"
    out_dir.mkdir(parents=True, exist_ok=True)
    test.to_csv(out_dir / "test_scores.csv", index=False)
    val.to_csv(out_dir / "val_scores.csv", index=False)
    report["data"] = {"reference_in_domain": int(len(reference)), "ood_train": int(len(ood_train)),
                      "val_counts": val.groupby("source").size().to_dict(),
                      "test_counts": test.groupby("source").size().to_dict()}
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))

    OOD_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(logreg, OOD_DIR / "logreg.joblib")
    index.save(OOD_DIR / "reference_index")
    (OOD_DIR / "thresholds.json").write_text(json.dumps(thresholds, indent=2))

    probe = pd.concat([ood["test"][ood["test"].source == s].sample(n=10, random_state=SEED)
                       for s in ood["test"].source.unique()], ignore_index=True)
    probe.to_json(out_dir / "probe_questions.jsonl", orient="records", lines=True)
    return report


class DomainGate:
    def __init__(self, directory=OOD_DIR, method="embedding_logreg"):
        self.method = method
        self.index = EmbeddingIndex.load(directory / "reference_index")
        self.logreg = joblib.load(directory / "logreg.joblib")
        self.threshold = json.loads((directory / "thresholds.json").read_text())[method]

    def score(self, question):
        emb = self.index.encode([question])
        if self.method == "embedding_logreg":
            return float(self.logreg.predict_proba(emb)[0, 1])
        return float((emb @ self.index.embeddings.T).max())

    def __call__(self, question):
        s = self.score(question)
        return s >= self.threshold, s
