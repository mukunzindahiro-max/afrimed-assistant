import time
from pathlib import Path

from .config import LETTERS
from .evaluate import generate, letter_probs
from .modeling import load_model, load_tokenizer
from .prompts import REFUSAL, build_prompt, format_user
from .retrieval import EmbeddingIndex, retrieve_references


class AfriMedAssistant:
    def __init__(self, model_path, kind="causal", adapter_path=None, knowledge_dir=None, gate_dir=None, device=None):
        from .ood import DomainGate

        self.kind = kind
        self.tokenizer = load_tokenizer(adapter_path or model_path, kind)
        self.model = load_model(model_path, kind, adapter_path=adapter_path, device=device)
        self.index = EmbeddingIndex.load(knowledge_dir) if knowledge_dir and Path(knowledge_dir).exists() else None
        self.gate = DomainGate(Path(gate_dir)) if gate_dir and Path(gate_dir).exists() else None

    def check_domain(self, question):
        if self.gate is None:
            return True, None
        return self.gate(question)

    def answer(self, question, options=None, use_rag=True, use_gate=True, k=3, max_new_tokens=200):
        start = time.time()
        options = [o.strip() for o in (options or []) if o and o.strip()]
        in_domain, score = self.check_domain(question) if use_gate else (True, None)
        result = {"in_domain": bool(in_domain), "domain_score": score, "references": [], "letter_probs": None}
        if not in_domain:
            result.update(answer=REFUSAL, latency_sec=time.time() - start)
            return result
        refs = retrieve_references(self.index, [question], k=k)[0] if use_rag and self.index else []
        rec = {"question": question.strip(), "context": "", "options": options}
        prompt = build_prompt(self.tokenizer, self.kind, format_user(rec, refs or None))
        if options:
            probs = letter_probs(self.model, self.tokenizer, self.kind, [prompt], 1)[0][: len(options)]
            probs = probs / probs.sum()
            result["letter_probs"] = {f"{LETTERS[i]}. {o}": float(p) for i, (o, p) in enumerate(zip(options, probs))}
        text, _ = generate(self.model, self.tokenizer, self.kind, [prompt], max_new_tokens, 1)
        result.update(answer=text[0], references=refs, latency_sec=time.time() - start)
        return result
