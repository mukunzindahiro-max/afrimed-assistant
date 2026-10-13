import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import torch
import yaml
from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Trainer,
                          TrainingArguments, set_seed)

from .config import CONFIG_DIR, MODELS_DIR, RESULTS_DIR, SEED
from .data import load_split
from .modeling import count_parameters, load_tokenizer
from .prompts import build_prompt, format_target, format_user


@dataclass
class ExperimentConfig:
    name: str
    model_id: str
    group: str = "lora"
    description: str = ""
    kind: str = "causal"
    method: str = "lora"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    learning_rate: float = 2e-4
    epochs: float = 2
    batch_size: int = 8
    grad_accum: int = 2
    max_len: int = 768
    max_target_len: int = 256
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    data_variant: str = "all"
    rag: bool = False
    rag_k: int = 3
    gen_max_new_tokens: int = 128
    eval_batch_size: int = 16
    gradient_checkpointing: bool = False
    adapter_from: Optional[str] = None
    priority: int = 1


def load_experiments(path=None):
    path = path or CONFIG_DIR / "experiments.yaml"
    data = yaml.safe_load(open(path))
    defaults = data.get("defaults", {})
    return {e["name"]: ExperimentConfig(**{**defaults, **e}) for e in data["experiments"]}


def training_frame(cfg, split, limit=None):
    df = load_split(split)
    if cfg.data_variant == "clean":
        df = df[df.is_clean]
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=SEED)
    return df


def encode_causal(tokenizer, cfg, records):
    features, dropped = [], 0
    for rec in records:
        prompt_ids = tokenizer(build_prompt(tokenizer, "causal", format_user(rec)), add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(format_target(rec) + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        target_ids = target_ids[: cfg.max_target_len]
        if len(prompt_ids) + len(target_ids) > cfg.max_len:
            dropped += 1
            continue
        input_ids = prompt_ids + target_ids
        features.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids),
                         "labels": [-100] * len(prompt_ids) + target_ids})
    return features, dropped


def encode_seq2seq(tokenizer, cfg, records):
    features = []
    for rec in records:
        enc = tokenizer(build_prompt(tokenizer, "seq2seq", format_user(rec)), max_length=cfg.max_len, truncation=True)
        labels = tokenizer(text_target=format_target(rec), max_length=cfg.max_target_len, truncation=True)["input_ids"]
        features.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels})
    return features, 0


class CausalCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = lambda seq, value: seq + [value] * (n - len(seq))
        return {
            "input_ids": torch.tensor([pad(b["input_ids"], self.pad_id) for b in batch]),
            "attention_mask": torch.tensor([pad(b["attention_mask"], 0) for b in batch]),
            "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
        }


def attach_lora(model, cfg):
    from peft import LoraConfig, TaskType, get_peft_model

    task = TaskType.CAUSAL_LM if cfg.kind == "causal" else TaskType.SEQ_2_SEQ_LM
    lora = LoraConfig(task_type=task, r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      target_modules=cfg.target_modules, bias="none")
    return get_peft_model(model, lora)


def train(cfg, limit=None, max_steps=-1):
    set_seed(SEED)
    out_dir = MODELS_DIR / cfg.name
    log_dir = RESULTS_DIR / cfg.name
    log_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(cfg.model_id, cfg.kind)
    cls = AutoModelForCausalLM if cfg.kind == "causal" else AutoModelForSeq2SeqLM
    model = cls.from_pretrained(cfg.model_id, torch_dtype=torch.float32)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    if cfg.method == "lora":
        model = attach_lora(model, cfg)
    params = count_parameters(model)

    encode = encode_causal if cfg.kind == "causal" else encode_seq2seq
    train_df = training_frame(cfg, "train", limit)
    val_df = training_frame(cfg, "val", limit and max(8, limit // 5))
    train_features, dropped_train = encode(tokenizer, cfg, train_df.to_dict("records"))
    val_features, dropped_val = encode(tokenizer, cfg, val_df.to_dict("records"))
    collator = CausalCollator(tokenizer.pad_token_id) if cfg.kind == "causal" else \
        DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    cuda = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=str(out_dir / "_checkpoints"),
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs,
        max_steps=max_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=max(10, len(train_features) // (cfg.batch_size * cfg.grad_accum * 4) or 10),
        save_strategy="no",
        report_to="none",
        fp16=cuda and cfg.kind == "causal",
        seed=SEED,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_features, eval_dataset=val_features,
                      data_collator=collator)
    start = time.time()
    trainer.train()
    train_time = time.time() - start
    final_eval = trainer.evaluate()

    if out_dir.exists():
        shutil.rmtree(out_dir)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    log = {
        "config": asdict(cfg),
        "params": params,
        "n_train": len(train_features),
        "n_val": len(val_features),
        "dropped_train_too_long": dropped_train,
        "dropped_val_too_long": dropped_val,
        "train_time_sec": train_time,
        "final_val_loss": final_eval.get("eval_loss"),
        "log_history": trainer.state.log_history,
    }
    (log_dir / "train_log.json").write_text(json.dumps(log, indent=2))

    if cfg.method == "lora":
        model = model.merge_and_unload()
    model.eval()
    if cuda and cfg.kind == "causal":
        model = model.half()
    return model, tokenizer, log
