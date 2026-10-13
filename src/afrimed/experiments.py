import gc
import json

import torch

from .config import MODELS_DIR, RESULTS_DIR
from .data import load_split
from .evaluate import evaluate_model
from .modeling import load_model, load_tokenizer
from .retrieval import EmbeddingIndex, build_knowledge_index
from .train import load_experiments, train

INDEX_DIR = MODELS_DIR / "knowledge_index"


def get_knowledge_index():
    if (INDEX_DIR / "embeddings.npy").exists():
        return EmbeddingIndex.load(INDEX_DIR)
    index = build_knowledge_index(load_split("train"))
    index.save(INDEX_DIR)
    return index


def resolve_model(cfg):
    if not cfg.adapter_from:
        return cfg.model_id, None
    path = MODELS_DIR / cfg.adapter_from
    if (path / "adapter_config.json").exists():
        return cfg.model_id, str(path)
    return str(path), None


def run_experiment(cfg, limit=None, max_steps=-1, skip_existing=True, use_bertscore=True):
    metrics_path = RESULTS_DIR / cfg.name / "metrics.json"
    if skip_existing and metrics_path.exists():
        print(f"[skip] {cfg.name} already evaluated")
        return json.loads(metrics_path.read_text())
    print(f"[run] {cfg.name}: {cfg.description}")
    if cfg.method == "none":
        model_id, adapter = resolve_model(cfg)
        tokenizer = load_tokenizer(model_id if adapter is None else adapter, cfg.kind)
        model = load_model(model_id, cfg.kind, adapter_path=adapter)
    else:
        model, tokenizer, _ = train(cfg, limit=limit, max_steps=max_steps)
        tokenizer.padding_side = "left" if cfg.kind == "causal" else "right"
    index = get_knowledge_index() if cfg.rag else None
    metrics = evaluate_model(cfg, model, tokenizer, knowledge_index=index, limit=limit, use_bertscore=use_bertscore)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, dict)}, indent=2))
    return metrics


def run_all(names=None, max_priority=None, **kwargs):
    experiments = load_experiments()
    selected = [experiments[n] for n in names] if names else sorted(experiments.values(), key=lambda c: c.priority)
    if max_priority is not None:
        selected = [c for c in selected if c.priority <= max_priority]
    return [run_experiment(cfg, **kwargs) for cfg in selected]
