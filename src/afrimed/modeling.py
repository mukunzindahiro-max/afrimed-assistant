import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tokenizer(model_id, kind):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" if kind == "causal" else "right"
    return tokenizer


def load_model(model_id, kind, adapter_path=None, dtype=None, device=None):
    device = device or get_device()
    if dtype is None:
        dtype = torch.float16 if device == "cuda" and kind == "causal" else torch.float32
    cls = AutoModelForCausalLM if kind == "causal" else AutoModelForSeq2SeqLM
    model = cls.from_pretrained(model_id, torch_dtype=dtype)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.to(device)
    model.eval()
    return model


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": int(total), "trainable_params": int(trainable), "trainable_pct": 100.0 * trainable / total}
