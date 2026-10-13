import numpy as np
import sacrebleu
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, f1_score

from .config import EMBEDDING_MODEL, LETTERS

_EMBEDDER = {}


def get_embedder(name=EMBEDDING_MODEL):
    if name not in _EMBEDDER:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER[name] = SentenceTransformer(name)
    return _EMBEDDER[name]


def rouge_l(predictions, references):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return np.array([scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(predictions, references)])


def token_f1(predictions, references):
    scores = []
    for p, r in zip(predictions, references):
        p_tok, r_tok = p.lower().split(), r.lower().split()
        common = sum(min(p_tok.count(t), r_tok.count(t)) for t in set(p_tok))
        if not p_tok or not r_tok or common == 0:
            scores.append(0.0)
            continue
        precision, recall = common / len(p_tok), common / len(r_tok)
        scores.append(2 * precision * recall / (precision + recall))
    return np.array(scores)


def semantic_similarity(predictions, references):
    model = get_embedder()
    p = model.encode(list(predictions), normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    r = model.encode(list(references), normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    return (p * r).sum(axis=1)


def bertscore_f1(predictions, references, model_type="roberta-large"):
    from bert_score import score

    _, _, f1 = score(list(predictions), list(references), model_type=model_type, lang="en",
                     rescale_with_baseline=True, verbose=False, batch_size=32)
    return f1.numpy()


def generation_metrics(predictions, references, use_bertscore=True):
    predictions = [p if p.strip() else "." for p in predictions]
    per_example = {
        "rougeL": rouge_l(predictions, references),
        "token_f1": token_f1(predictions, references),
        "semantic_sim": semantic_similarity(predictions, references),
    }
    if use_bertscore:
        per_example["bertscore_f1"] = bertscore_f1(predictions, references)
    summary = {k: float(np.mean(v)) for k, v in per_example.items()}
    summary["bleu"] = float(sacrebleu.corpus_bleu(predictions, [references]).score)
    summary["chrf"] = float(sacrebleu.corpus_chrf(predictions, [references]).score)
    summary["avg_pred_words"] = float(np.mean([len(p.split()) for p in predictions]))
    summary["avg_ref_words"] = float(np.mean([len(r.split()) for r in references]))
    return summary, per_example


def mcq_metrics(gold, pred):
    gold = [g for g in gold]
    pred = [p for p in pred]
    acc = accuracy_score(gold, pred)
    macro_f1 = f1_score(gold, pred, labels=LETTERS, average="macro", zero_division=0)
    return {"accuracy": float(acc), "macro_f1": float(macro_f1)}


def set_metrics(gold_sets, pred_sets):
    exact, jacc = [], []
    for g, p in zip(gold_sets, pred_sets):
        g, p = set(g), set(p)
        exact.append(float(g == p))
        jacc.append(len(g & p) / len(g | p) if g | p else 1.0)
    return {"exact_match": float(np.mean(exact)) if exact else 0.0,
            "jaccard": float(np.mean(jacc)) if jacc else 0.0}


def expected_calibration_error(probs, gold_idx, n_bins=10):
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == gold_idx
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)
