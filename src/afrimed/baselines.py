import json
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import LETTERS, RESULTS_DIR, SEED
from .data import knowledge_text, load_split
from .metrics import expected_calibration_error, generation_metrics, mcq_metrics, set_metrics


def softmax(x, temperature=1.0):
    z = np.asarray(x, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def save_outputs(name, description, mcq, probs, saq, saq_pred, use_bertscore):
    out_dir = RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    if mcq is None:
        return save_saq_only(name, description, saq, saq_pred, use_bertscore, out_dir)
    mcq = mcq.copy()
    for j, letter in enumerate(LETTERS):
        mcq[f"p_{letter}"] = probs[:, j]
    mcq["confidence"] = probs.max(axis=1)
    mcq["gold"] = mcq.answer_letters
    multi = mcq.multi_answer.values
    mcq["correct"] = [set(p.split(",")) == set(g.split(",")) for p, g in zip(mcq.pred, mcq.gold)]
    single = mcq[~multi]
    gold_idx = np.array([LETTERS.index(g) for g in single.gold])
    metrics = {"name": name, "group": "baseline", "description": description, "model_id": "-", "method": "baseline",
               "rag": False}
    metrics.update({f"mcq_{k}": v for k, v in mcq_metrics(single.gold.tolist(), single.pred.tolist()).items()})
    metrics["mcq_ece"] = expected_calibration_error(probs[~multi], gold_idx)
    metrics["mcq_n"] = int(len(single))
    metrics["mcq_per_specialty"] = single.groupby("specialty").correct.mean().round(4).to_dict()
    multi_df = mcq[multi]
    metrics.update({f"mcq_multi_{k}": v for k, v in set_metrics(
        [g.split(",") for g in multi_df.gold], [p.split(",") for p in multi_df.pred]).items()})
    metrics["mcq_multi_n"] = int(multi.sum())
    mcq["generated"] = ""

    if saq is not None:
        saq = saq.copy()
        saq["prediction"] = saq_pred
        saq["gen_tokens"] = [len(p.split()) for p in saq_pred]
        saq["hit_max_tokens"] = False
        summary, per_example = generation_metrics(saq_pred, saq.rationale.tolist(), use_bertscore)
        for k, v in per_example.items():
            saq[k] = v
        metrics.update({f"saq_{k}": v for k, v in summary.items()})
        metrics["saq_n"] = int(len(saq))
        saq.drop(columns=["options", "answer_idx"], errors="ignore").to_csv(out_dir / "saq_predictions.csv",
                                                                              index=False)
    keep = ["sample_id", "question", "specialty", "question_words", "mentions_africa", "tier", "multi_answer"]
    mcq[keep + ["options", "gold", "pred", "confidence", "correct", "generated"] +
        [f"p_{l}" for l in LETTERS]].to_csv(out_dir / "mcq_predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def save_saq_only(name, description, saq, saq_pred, use_bertscore, out_dir):
    saq = saq.copy()
    saq["prediction"] = saq_pred
    saq["gen_tokens"] = [len(p.split()) for p in saq_pred]
    saq["hit_max_tokens"] = False
    summary, per_example = generation_metrics(saq_pred, saq.rationale.tolist(), use_bertscore)
    for k, v in per_example.items():
        saq[k] = v
    metrics = {"name": name, "group": "baseline", "description": description, "model_id": "-",
               "method": "baseline", "rag": False}
    metrics.update({f"saq_{k}": v for k, v in summary.items()})
    metrics["saq_n"] = int(len(saq))
    saq.drop(columns=["options", "answer_idx"], errors="ignore").to_csv(out_dir / "saq_predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def answer_texts(train):
    knowledge = [knowledge_text(r) for r in train.to_dict("records")]
    return knowledge, [t.split(" A: ", 1)[1] if " A: " in t else t for t in knowledge]


def minilm_retrieval_predict(train, saq):
    from .retrieval import EmbeddingIndex

    _, answers = answer_texts(train)
    index = EmbeddingIndex().build(train.question.tolist(), list(range(len(train))))
    hits = index.search(saq.question.tolist(), k=1)
    return [answers[h[0][1]] for h in hits], [h[0][0] for h in hits]


def tfidf_retrieval_predict(train, mcq, saq, k=5):
    knowledge, answers = answer_texts(train)
    q_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english").fit(train.question)
    train_q = q_vec.transform(train.question)
    content_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english").fit(
        knowledge + [o for opts in mcq.options for o in opts])

    saq_sims = q_vec.transform(saq.question) @ train_q.T
    saq_pred = [answers[int(np.asarray(row.todense()).argmax())] for row in saq_sims]

    mcq_sims = (q_vec.transform(mcq.question) @ train_q.T).toarray()
    probs = np.zeros((len(mcq), len(LETTERS)))
    for i, (question, options) in enumerate(zip(mcq.question, mcq.options)):
        top = np.argsort(-mcq_sims[i])[:k]
        evidence = content_vec.transform([question + " " + " ".join(knowledge[j] for j in top)])
        scores = (content_vec.transform(options) @ evidence.T).toarray().ravel()
        padded = np.full(len(LETTERS), -1.0)
        padded[: len(scores)] = scores
        probs[i] = softmax(padded, temperature=0.1)
    return probs, saq_pred


def run_baselines(use_bertscore=True):
    start = time.time()
    rng = np.random.default_rng(SEED)
    train, test = load_split("train"), load_split("test")
    mcq = test[test.question_type == "mcq"].reset_index(drop=True)
    saq = test[test.question_type == "saq"].reset_index(drop=True)
    train_answers = train[train.question_type != "mcq"].rationale.tolist()
    results = []

    probs = np.full((len(mcq), len(LETTERS)), 1.0 / len(LETTERS))
    rand = mcq.assign(pred=rng.choice(LETTERS, size=len(mcq)))
    saq_rand = list(rng.choice(train_answers, size=len(saq)))
    results.append(save_outputs("baseline_random", "Uniform random option / random training answer",
                                rand, probs, saq, saq_rand, use_bertscore))

    train_single = train[(train.question_type == "mcq") & ~train.multi_answer]
    freq = train_single.answer_letters.value_counts(normalize=True).reindex(LETTERS, fill_value=0).values
    majority = mcq.assign(pred=LETTERS[int(freq.argmax())])
    results.append(save_outputs("baseline_majority", "Always predict the most frequent training answer letter",
                                majority, np.tile(freq, (len(mcq), 1)), None, None, use_bertscore))

    probs, saq_pred = tfidf_retrieval_predict(train, mcq, saq)
    tfidf = mcq.assign(pred=[LETTERS[i] for i in probs.argmax(axis=1)])
    results.append(save_outputs("baseline_tfidf", "TF-IDF nearest-neighbour retrieval over training QA pairs",
                                tfidf, probs, saq, saq_pred, use_bertscore))

    nn_pred, _ = minilm_retrieval_predict(train, saq)
    results.append(save_outputs("baseline_minilm_nn", "MiniLM nearest-neighbour retrieval: answer of the most similar "
                                "training question (SAQ only)", None, None, saq, nn_pred, use_bertscore))
    print(f"baselines finished in {time.time() - start:.1f}s")
    return results
