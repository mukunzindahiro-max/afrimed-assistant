# AfriMed Assistant

**Domain-specific medical question answering for African healthcare, built by fine-tuning Qwen2.5-Instruct with LoRA on AfriMed-QA v2.**

| Resource | Link |
|---|---|
| Live web app (Hugging Face Space) | _to be added after deployment_ |
| Fine-tuned model (Hugging Face Hub) | _to be added after training_ |
| Demo video | _to be added_ |
| Report (PDF) | [`reports/AfriMed_Assistant_Report.pdf`](reports/) |

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
           Retriever (optional, top-3 similar training QA pairs, MiniLM cosine)
                 ▼
           Qwen2.5-Instruct + merged LoRA adapters  ──►  answer / option probabilities
```

| Component | Choice | Why |
|---|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` (and 1.5B) | Modern decoder-only Transformer, instruction-tuned, fits a free T4 GPU and a free CPU Space |
| Adaptation | LoRA (Hu et al., 2021): r=16, α=32, dropout 0.05, all attention and MLP projections | Trains about 1.75% of parameters (8.8M of 503M), so it is fast and has low risk of catastrophic forgetting |
| Objective | Causal LM cross-entropy on response tokens only (prompt masked with -100) | The model learns to answer, not to reproduce questions |
| Contrast model | `google/flan-t5-base`, full fine-tuning | Encoder-decoder (seq2seq) architecture |
| Retrieval | `all-MiniLM-L6-v2` embeddings over training QA pairs | Tests whether non-parametric knowledge helps a small model |
| Domain gate | Logistic regression on MiniLM embeddings | Compared with TF-IDF and embedding k-NN detectors |

Baselines: random, majority-letter and TF-IDF retrieval (non-neural), plus zero-shot Qwen2.5 (pretrained, no fine-tuning).

## 4. Experiments

All experiments are defined in [`configs/experiments.yaml`](configs/experiments.yaml).

| Experiment | What changes | Research question |
|---|---|---|
| `baseline_random` / `baseline_majority` / `baseline_tfidf` | Non-neural | Lower bounds and lexical-overlap baseline |
| `zs_qwen05`, `zs_qwen15` | Zero-shot | How much does the pretrained model already know? |
| `lora_qwen05_r16` | Main model | Does domain fine-tuning help? |
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

**Baselines and domain gate (computed locally)**

| Model | MCQ acc. | SAQ ROUGE-L | SAQ semantic sim. |
|---|---|---|---|
| Random | 0.209 | 0.077 | 0.164 |
| Majority letter | 0.241 | – | – |
| TF-IDF retrieval | 0.244 | 0.261 | 0.481 |

| Domain gate | AUROC | Medical accepted | Non-medical rejected |
|---|---|---|---|
| TF-IDF k-NN | 0.732 | 71.6% | 61.1% |
| MiniLM k-NN | 0.997 | 95.8% | 99.0% |
| **MiniLM + logistic regression** | **0.999** | **98.2%** | **99.6%** |

The full table for all fine-tuning experiments is in [`results/summary.csv`](results/summary.csv), and the figures are in [`reports/figures/`](reports/figures/).

## 6. Repository structure

```
afrimed-assistant/
├── app/app.py                  Gradio web application
├── configs/experiments.yaml    All experiment configurations
├── notebooks/
│   ├── 01_data_exploration.ipynb        EDA + preparation pipeline (14 figures)
│   ├── 02_baselines_and_ood.ipynb       Non-neural baselines + domain gate
│   ├── 03_finetuning_colab.ipynb        All fine-tuning experiments (Colab T4 GPU)
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

## 7. How to reproduce

```bash
git clone https://github.com/mukunzindahiro-max/afrimed-assistant.git
cd afrimed-assistant
python -m venv .venv && source .venv/bin/activate
pip install torch && pip install -r requirements.txt
huggingface-cli login
```

1. **Data:** `python scripts/prepare_data.py`, or run `notebooks/01_data_exploration.ipynb`.
2. **Baselines + domain gate:** `python scripts/run_baselines.py`, or notebook 02 (CPU, about 20 min).
3. **Fine-tuning:** open `notebooks/03_finetuning_colab.ipynb` in Google Colab with a T4 GPU, or run `python scripts/run_experiments.py --max-priority 1`. Use `--smoke` for a 2-minute pipeline check.
4. **Evaluation:** `notebooks/04_evaluation_error_analysis.ipynb`.
5. **Web app locally:** `python app/app.py`. Set `MODEL_REPO` to a Hub model, or set `MODEL_REPO=Qwen/Qwen2.5-0.5B-Instruct ADAPTER_PATH=models/lora_qwen05_r16`.
6. **Deploy:** `python scripts/push_to_hub.py --experiment lora_qwen05_r16 --repo <user>/<model>`, then `python scripts/deploy_space.py --space <user>/afrimed-assistant --model-repo <user>/<model>`.

On Intel Macs, PyTorch 2.2.2 is the last version with x86_64 wheels: `pip install torch==2.2.2 "numpy<2" "pyarrow<18"`.

## 8. What was used vs what was implemented

| Used (existing resources) | Implemented in this project |
|---|---|
| AfriMed-QA v2 dataset, SQuAD, WebQuestions | Cleaning, scenario parsing, re-splitting, leakage removal |
| Qwen2.5-Instruct, Flan-T5, MiniLM, RoBERTa (BERTScore) | Prompt formats, response-only loss masking, LoRA and full fine-tuning setup |
| Hugging Face `transformers`, `peft`, `datasets`, `sentence-transformers`, scikit-learn, Gradio | Baselines, letter-probability MCQ evaluation, all experiments and ablations |
| | Domain gate (3 methods compared), RAG pipeline, error-analysis taxonomy, web app, deployment |

## 9. Licence and citation

Code: MIT. Data: AfriMed-QA v2 is CC BY-SA 4.0; derived artifacts (retrieval index) are shared under the same licence.

```
@article{olatunji2024afrimed,
  title={AfriMed-QA: A Pan-African, Multi-Specialty, Medical Question-Answering Benchmark Dataset},
  author={Olatunji, Tobi and Nimo, Charles and Owodunni, Abraham and others},
  journal={arXiv preprint arXiv:2411.15640},
  year={2024}
}
```
