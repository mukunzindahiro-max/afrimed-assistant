# AfriMed Assistant

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.46-FFD21E)](https://github.com/huggingface/transformers)
[![PEFT LoRA](https://img.shields.io/badge/PEFT-LoRA-FF6F00)](https://github.com/huggingface/peft)
[![Gradio](https://img.shields.io/badge/Gradio-6.17-F97316?logo=gradio&logoColor=white)](https://www.gradio.app/)

**Domain-specific medical question answering for African healthcare, built by fine-tuning Qwen2.5-Instruct with LoRA on AfriMed-QA v2.**

| Resource | Link |
|---|---|
| GitHub repository | [github.com/mukunzindahiro-max/afrimed-assistant](https://github.com/mukunzindahiro-max/afrimed-assistant) |
| Live web app | [afrimed-app-production.up.railway.app](https://afrimed-app-production.up.railway.app) |
| Fine-tuned model | [MukunziJames/afrimed-assistant-qwen2.5-1.5b-lora](https://huggingface.co/MukunziJames/afrimed-assistant-qwen2.5-1.5b-lora) |

> AfriMed Assistant is a student research prototype for the ALU *Machine Learning Techniques I* summative project. It is **not** a medical device and must not replace a qualified health worker.

---

## 1. Problem

Large language models are increasingly used for health questions, but they are trained mostly on Western web text. They have limited exposure to diseases, treatment guidelines, health-system constraints and cultural context that are common in Africa (Olatunji et al., 2024). This project builds a **domain-specific question-answering assistant for African healthcare** that:

* answers open-ended (short-answer) and multiple-choice medical questions,
* is adapted to African clinical data through parameter-efficient fine-tuning,
* refuses questions outside the medical domain with a learned **domain gate**,
* is deployed as a web application anyone can use.

Target users are health students, community health workers and members of the public looking for reliable first-line information.

## 2. Dataset

[**AfriMed-QA v2**](https://huggingface.co/datasets/intronhealth/afrimedqa_v2) (Olatunji et al., 2024; CC BY-SA 4.0) has 15,275 real, human-written questions from 621 contributors in 16 countries across 32+ specialties. No synthetic data is used.

| Type | Rows | Usable (has reference answer) |
|---|---|---|
| Multiple-choice (MCQ, 5 options) | 4,039 | 4,039 |
| Short-answer (SAQ) | 1,236 | 1,222 |
| Consumer queries | 10,000 | 132 (the remaining 9,329 are used as in-domain questions for the domain gate) |

The preparation pipeline (`src/afrimed/data.py`, notebook 01):

* uses `question_clean`, normalises whitespace and parses consumer scenarios,
* removes 540 duplicate rows and unusable answers,
* **re-splits** the data, because the official split puts 3,910 of 4,039 MCQs and all expert SAQs into test. MCQs are split 70/10/20 stratified by specialty and multi-answer flag. SAQs are split 70/10/20 stratified by tier,
* removes training questions that are near-duplicates (TF-IDF cosine ≥ 0.9) of any val/test question, and training rows whose answer is identical to a held-out answer.

Final split: **3,783 train / 541 validation / 1,054 test** (808 MCQ + 246 SAQ in test).

Out-of-domain questions for the domain gate come from SQuAD v1.1 (Rajpurkar et al., 2016) and WebQuestions (Berant et al., 2013), with medical topics filtered out.

## 3. Methodology

```
question ──► Domain gate (MiniLM embeddings + logistic regression) ──► refuse if non-medical
                 │ medical
                 ▼
           Retriever (top-3 similar training QA pairs, MiniLM cosine)
                 ▼
           Qwen2.5-1.5B-Instruct + merged LoRA adapters  ──►  answer / option probabilities
```

| Component | Choice | Why |
|---|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` (final), `Qwen/Qwen2.5-0.5B-Instruct` (ablations) | Modern decoder-only Transformer, instruction-tuned, small enough to fine-tune on one GPU and serve on CPU |
| Adaptation | LoRA (Hu et al., 2021): r=16, α=32, dropout 0.05, all attention and MLP projections | Trains 1.2% of the 1.5B model (18.5M parameters), so it is fast and keeps the pretrained knowledge |
| Objective | Causal LM cross-entropy on response tokens only (prompt masked with -100) | The model learns to answer, not to reproduce questions |
| Training | AdamW, lr 2e-4, cosine schedule, 5% warm-up, 2 epochs, effective batch 16 (4 × 4 accumulation), fp16 | Converges in 2 epochs; validation loss is flat in epoch 2 |
| Contrast model | `google/flan-t5-base`, full fine-tuning | Encoder-decoder (seq2seq) architecture and a different adaptation strategy |
| Retrieval | `all-MiniLM-L6-v2` embeddings over training QA pairs only | Tests whether non-parametric knowledge helps a small model |
| Domain gate | Logistic regression on MiniLM embeddings | Compared with TF-IDF and embedding k-NN detectors |

Baselines: random, majority-letter and TF-IDF retrieval (non-neural), MiniLM nearest-neighbour retrieval (neural, no generation), plus zero-shot Qwen2.5 (pretrained, no fine-tuning).

RNNs, LSTMs and static embeddings (Word2Vec, GloVe) were not used. With 3,783 training examples, a recurrent model trained from scratch cannot learn enough medical knowledge, and static embeddings give one vector per word regardless of context. A pretrained Transformer already contains medical knowledge, and self-attention reads the whole question at once.

## 4. Experiments

All experiments are defined in [`configs/experiments.yaml`](configs/experiments.yaml) and were run in [`notebooks/03_finetuning_colab.ipynb`](notebooks/03_finetuning_colab.ipynb) on a Colab A100 GPU.

| Experiment | What changes | Research question |
|---|---|---|
| `baseline_random` / `baseline_majority` / `baseline_tfidf` | Non-neural | Lower bounds and lexical-overlap baseline |
| `baseline_minilm_nn` | Dense nearest-neighbour retrieval (SAQ) | Is "look up the most similar training question" enough? |
| `zs_qwen05`, `zs_qwen15` | Zero-shot | How much does the pretrained model already know? |
| `lora_qwen05_r16` | Main 0.5B model | Does domain fine-tuning help? |
| `lora_qwen05_r8`, `lora_qwen05_r32` | LoRA rank | Capacity vs efficiency |
| `lora_qwen05_r16_lr5e5` | Learning rate 5e-5 | Sensitivity to learning rate |
| `lora_qwen05_r16_attn` | LoRA on attention only | Which layers matter? |
| `lora_qwen05_r16_clean` | Quality-filtered data only | Does removing low-quality contributors help? |
| `ft_flant5_base` | Encoder-decoder, full fine-tuning | Architecture comparison |
| `zs_qwen05_rag`, `lora_qwen05_r16_rag`, `lora_qwen15_r16_rag` | + retrieval | Does RAG help? |
| `lora_qwen15_r16` | 3× larger base | Does scale help? |

## 5. Results

**Evaluation metrics**

* MCQ: accuracy, macro-F1 and expected calibration error (ECE). Single-answer questions are scored from the option-letter probabilities after `Answer:`. Multi-answer questions are generated and scored by exact match and Jaccard.
* SAQ: ROUGE-L, BLEU, chrF, BERTScore-F1 (rescaled) and embedding semantic similarity.
* Domain gate: AUROC, macro-F1, in-domain acceptance and OOD rejection.

**All experiments (test set: 808 MCQ, 246 SAQ)**

| Model | Trainable params | MCQ acc. | MCQ macro-F1 | ECE ↓ | Multi-answer Jaccard | SAQ ROUGE-L | SAQ BLEU | SAQ BERTScore-F1 | SAQ semantic sim. |
|---|---|---|---|---|---|---|---|---|---|
| Random | – | 0.209 | 0.208 | 0.016 | 0.262 | 0.077 | 1.0 | −0.093 | 0.164 |
| Majority letter | – | 0.241 | 0.078 | 0.004 | 0.215 | – | – | – | – |
| TF-IDF retrieval | – | 0.244 | 0.221 | 0.044 | 0.178 | 0.261 | 11.1 | 0.191 | 0.481 |
| MiniLM nearest neighbour | – | – | – | – | – | 0.243 | 9.9 | 0.206 | 0.556 |
| Flan-T5-base, full fine-tuning | 247.6M | 0.288 | 0.268 | 0.024 | 0.404 | 0.293 | 12.9 | 0.197 | 0.622 |
| Qwen2.5-0.5B zero-shot | – | 0.326 | 0.312 | 0.274 | 0.043 | 0.182 | 5.3 | 0.116 | 0.692 |
| Qwen2.5-0.5B zero-shot + RAG | – | 0.337 | 0.313 | 0.193 | 0.229 | 0.215 | 6.5 | 0.155 | 0.697 |
| 0.5B LoRA r=16 | 8.8M | 0.374 | 0.364 | 0.134 | 0.427 | 0.309 | 14.0 | 0.301 | 0.675 |
| 0.5B LoRA r=8 | 4.4M | 0.382 | 0.372 | 0.086 | 0.450 | 0.308 | 13.3 | 0.297 | 0.672 |
| 0.5B LoRA r=32 | 17.6M | 0.373 | 0.362 | 0.173 | 0.422 | 0.313 | 14.8 | 0.306 | 0.689 |
| 0.5B LoRA r=16, lr 5e-5 | 8.8M | 0.383 | 0.370 | 0.045 | 0.457 | 0.273 | 11.2 | 0.273 | 0.661 |
| 0.5B LoRA r=16, attention only | 2.2M | 0.386 | 0.373 | 0.052 | 0.423 | 0.278 | 12.2 | 0.273 | 0.664 |
| 0.5B LoRA r=16, clean data | 8.8M | 0.374 | 0.360 | 0.128 | 0.378 | 0.276 | 13.1 | 0.255 | 0.666 |
| 0.5B LoRA r=16 + RAG | 8.8M | 0.404 | 0.389 | 0.100 | 0.272 | 0.318 | 16.8 | 0.314 | 0.691 |
| Qwen2.5-1.5B zero-shot | – | 0.473 | 0.468 | 0.339 | 0.202 | 0.185 | 5.0 | 0.136 | 0.705 |
| 1.5B LoRA r=16 | 18.5M | 0.546 | 0.543 | 0.137 | 0.464 | 0.332 | 14.5 | 0.339 | 0.710 |
| **1.5B LoRA r=16 + RAG (deployed)** | 18.5M | **0.556** | **0.554** | 0.128 | 0.384 | **0.337** | **17.3** | 0.338 | **0.716** |

BERTScore is rescaled with a baseline (`rescale_with_baseline=True`), so unrelated text scores about 0 and can go slightly negative. Full numbers are in [`results/summary.csv`](results/summary.csv) and figures in [`reports/figures/`](reports/figures/).

**Main findings**

* **Fine-tuning helps at both sizes.** LoRA adds 4.8 points of MCQ accuracy to the 0.5B model and 7.3 points to the 1.5B model, raises ROUGE-L by about 80% and more than doubles BERTScore, because the fine-tuned model answers in the concise style of the references.
* **Model size matters most.** Zero-shot 1.5B (0.473) already beats every fine-tuned 0.5B variant (best 0.404).
* **More trained parameters is not better.** Flan-T5 trains all 248M parameters and is the weakest neural model; LoRA ranks 8, 16 and 32 differ by about 1 point.
* **RAG helps a little** (+1.1 to +3.0 MCQ points), but part of the SAQ gain comes from copying the retrieved training answer (21 of 246 answers are near-copies).
* **Fine-tuning improves calibration**: ECE drops from 0.34 (zero-shot 1.5B) to 0.13.
* **Weak spots**: 98% of multi-answer MCQs are wrong (the model usually gives one letter), and 42% of negation questions ("Which is NOT…") are wrong.

**Domain gate (out-of-domain detection)**

| Domain gate | AUROC | Medical accepted | Non-medical rejected |
|---|---|---|---|
| TF-IDF k-NN | 0.732 | 71.6% | 61.1% |
| MiniLM k-NN | 0.997 | 95.8% | 99.0% |
| **MiniLM + logistic regression (deployed)** | **0.999** | **98.1%** | **99.6%** |

## 6. Web application

The app (`app/app.py`, Gradio) is deployed on Railway: **[afrimed-app-production.up.railway.app](https://afrimed-app-production.up.railway.app)**.

* **Ask a question:** free-text medical questions. The domain gate checks the question, the retriever shows the 3 most similar training QA pairs, and the fine-tuned model writes the answer.
* **Multiple choice:** enter a question and options A–E. The app shows the model's probability for each option and a generated explanation.
* **How it works:** a short description of the model, data and limitations.

It runs on CPU, so a 100-token answer takes about 25–30 seconds. The model is downloaded from the Hugging Face Hub when the container starts.

## 7. Repository structure

```
afrimed-assistant/
├── app/
│   ├── app.py                  Gradio web application
│   └── Dockerfile              Container used for the Railway deployment
├── configs/experiments.yaml    All experiment configurations
├── notebooks/
│   ├── 01_data_exploration.ipynb        EDA + preparation pipeline (14 figures)
│   ├── 02_baselines_and_ood.ipynb       Non-neural baselines + domain gate
│   ├── 03_finetuning_colab.ipynb        All fine-tuning experiments (Colab GPU)
│   └── 04_evaluation_error_analysis.ipynb  Comparison, plots, error analysis
├── reports/figures/            All generated figures
├── results/                    Metrics, predictions and logs per experiment
├── scripts/                    Command-line versions of every stage
├── src/afrimed/                Python package
│   ├── data.py          loading, cleaning, splitting, leakage removal
│   ├── prompts.py       chat formatting and answer parsing
│   ├── train.py         LoRA / full fine-tuning with HF Trainer
│   ├── evaluate.py      MCQ letter scoring, generation, metrics
│   ├── baselines.py     random / majority / TF-IDF baselines
│   ├── retrieval.py     MiniLM embedding index (RAG)
│   ├── ood.py           domain gate experiments
│   ├── metrics.py       ROUGE, BLEU, chrF, BERTScore, ECE
│   ├── plots.py         all figures
│   ├── error_analysis.py
│   └── inference.py     AfriMedAssistant class used by the app
└── requirements.txt
```

## 8. How to reproduce

```bash
git clone https://github.com/mukunzindahiro-max/afrimed-assistant.git
cd afrimed-assistant
python -m venv .venv && source .venv/bin/activate
pip install torch && pip install -r requirements.txt
hf auth login
```

1. **Data:** `python scripts/prepare_data.py`, or run `notebooks/01_data_exploration.ipynb`.
2. **Baselines + domain gate:** `python scripts/run_baselines.py`, or notebook 02 (CPU, about 20 min).
3. **Fine-tuning:** open `notebooks/03_finetuning_colab.ipynb` in Google Colab with a GPU, or run `python scripts/run_experiments.py --max-priority 3`. Use `--smoke` for a 2-minute pipeline check.
4. **Evaluation:** `notebooks/04_evaluation_error_analysis.ipynb`.
5. **Web app locally:** `python app/app.py`, then open http://localhost:7860. It downloads the fine-tuned model from the Hub.
6. **Publish the model:** `python scripts/push_to_hub.py --experiment lora_qwen15_r16 --repo <user>/<model>` merges the LoRA adapters and uploads the model.
7. **Deploy on Railway:**

   ```bash
   python scripts/deploy_space.py --build-only --model-repo <user>/<model>
   cd build/space
   railway init && railway up
   railway variables --set MODEL_REPO=<user>/<model>
   railway domain --port 7860
   ```

   The service needs about 8 GB of RAM. `OMP_NUM_THREADS=8` is set in the Dockerfile, because PyTorch otherwise starts too many threads on shared CPUs and generation becomes very slow.

On Intel Macs, PyTorch 2.2.2 is the last version with x86_64 wheels: `pip install torch==2.2.2 "numpy<2" "pyarrow<18"`.

## 9. What was used vs what was implemented

| Used (existing resources) | Implemented in this project |
|---|---|
| AfriMed-QA v2 dataset, SQuAD, WebQuestions | Cleaning, scenario parsing, re-splitting, leakage removal, near-duplicate check |
| Qwen2.5-Instruct, Flan-T5, MiniLM, RoBERTa (BERTScore) | Prompt formats, response-only loss masking, LoRA and full fine-tuning setup |
| Hugging Face `transformers`, `peft`, `datasets`, `sentence-transformers`, scikit-learn, Gradio | Baselines, letter-probability MCQ evaluation, all 13 experiments and ablations |
| Google Colab (training), Hugging Face Hub (model hosting), Railway (app hosting) | Domain gate (3 methods compared), RAG pipeline, error-analysis taxonomy, web app, Docker deployment |

## 10. Licence and citation

Code: MIT. Data: AfriMed-QA v2 is CC BY-SA 4.0; derived artifacts (retrieval index) are shared under the same licence.

```
@article{olatunji2024afrimed,
  title={AfriMed-QA: A Pan-African, Multi-Specialty, Medical Question-Answering Benchmark Dataset},
  author={Olatunji, Tobi and Nimo, Charles and Owodunni, Abraham and others},
  journal={arXiv preprint arXiv:2411.15640},
  year={2024}
}
```
