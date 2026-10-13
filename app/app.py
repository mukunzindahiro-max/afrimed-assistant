import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE.parent / "src"):
    if (candidate / "afrimed").exists():
        sys.path.insert(0, str(candidate))

import gradio as gr

from afrimed.inference import AfriMedAssistant

MODEL_REPO = os.environ.get("MODEL_REPO", "mukunzindahiro-max/afrimed-assistant-qwen2.5-0.5b-lora")
ADAPTER = os.environ.get("ADAPTER_PATH") or None
ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", HERE / "artifacts"))
if not ARTIFACTS.exists():
    ARTIFACTS = HERE.parent / "models"

assistant = AfriMedAssistant(MODEL_REPO, adapter_path=ADAPTER, knowledge_dir=ARTIFACTS / "knowledge_index",
                             gate_dir=ARTIFACTS / "ood")

DISCLAIMER = (
    "**Disclaimer:** AfriMed Assistant is a student research prototype. It can be wrong and is **not** a substitute "
    "for a qualified health worker. In an emergency, go to the nearest health facility."
)

EXAMPLES = [
    "What is the first-line treatment for uncomplicated malaria in pregnancy during the second trimester?",
    "What are the danger signs of pre-eclampsia that a pregnant woman should watch for?",
    "How is sickle cell anaemia inherited?",
    "Which vaccines should a newborn receive at birth in most African immunisation schedules?",
    "What causes Lassa fever and how is it transmitted?",
    "Who won the 2010 FIFA World Cup?",
    "How do I bake banana bread?",
]

MCQ_EXAMPLE = [
    "Which of the following infectious diseases, endemic to parts of Africa, is caused by Mycobacterium leprae?",
    "Tuberculosis", "Leprosy", "Yaws", "Buruli ulcer disease", "Lymphatic filariasis",
]


def format_domain(result):
    if result["domain_score"] is None:
        return "Domain gate disabled"
    verdict = "In domain (medical)" if result["in_domain"] else "Out of domain (refused)"
    return f"{verdict}: P(medical) = {result['domain_score']:.3f}"


def format_refs(refs):
    if not refs:
        return "_No references retrieved._"
    return "\n\n".join(f"**{i + 1}.** {r}" for i, r in enumerate(refs))


def ask(question, use_rag, use_gate, max_tokens):
    if not question or not question.strip():
        return "Please type a question.", "", ""
    result = assistant.answer(question, use_rag=use_rag, use_gate=use_gate, max_new_tokens=int(max_tokens))
    footer = f"\n\n<sub>Generated in {result['latency_sec']:.1f}s</sub>"
    return result["answer"] + footer, format_domain(result), format_refs(result["references"])


def ask_mcq(question, a, b, c, d, e, use_rag, use_gate):
    options = [a, b, c, d, e]
    if not question.strip() or sum(bool(o and o.strip()) for o in options) < 2:
        return {}, "Enter a question and at least two options.", ""
    result = assistant.answer(question, options=options, use_rag=use_rag, use_gate=use_gate, max_new_tokens=160)
    return result["letter_probs"] or {}, result["answer"], format_domain(result)


with gr.Blocks(title="AfriMed Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# AfriMed Assistant\n"
        "A medical question-answering assistant for African healthcare contexts. It uses Qwen2.5-Instruct "
        "fine-tuned with LoRA on the AfriMed-QA v2 dataset, with an embedding-based domain gate and optional "
        "retrieval of similar expert-written QA pairs.\n\n" + DISCLAIMER
    )
    with gr.Tab("Ask a question"):
        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(label="Your health question", lines=3,
                                      placeholder="e.g. What are the symptoms of typhoid fever?")
                with gr.Row():
                    use_rag = gr.Checkbox(value=True, label="Use retrieval (RAG)")
                    use_gate = gr.Checkbox(value=True, label="Domain gate (refuse non-medical)")
                    max_tokens = gr.Slider(32, 320, value=200, step=16, label="Max new tokens")
                submit = gr.Button("Ask", variant="primary")
                gr.Examples(EXAMPLES, inputs=question)
            with gr.Column(scale=4):
                answer = gr.Markdown(label="Answer")
                domain = gr.Textbox(label="Domain check", interactive=False)
                with gr.Accordion("Retrieved reference QA pairs", open=False):
                    refs = gr.Markdown()
        submit.click(ask, [question, use_rag, use_gate, max_tokens], [answer, domain, refs])
        question.submit(ask, [question, use_rag, use_gate, max_tokens], [answer, domain, refs])

    with gr.Tab("Multiple choice"):
        with gr.Row():
            with gr.Column():
                mcq_q = gr.Textbox(label="Question", lines=3)
                opts = [gr.Textbox(label=f"Option {l}") for l in "ABCDE"]
                with gr.Row():
                    mcq_rag = gr.Checkbox(value=False, label="Use retrieval (RAG)")
                    mcq_gate = gr.Checkbox(value=True, label="Domain gate")
                mcq_btn = gr.Button("Predict", variant="primary")
                gr.Examples([MCQ_EXAMPLE], inputs=[mcq_q] + opts)
            with gr.Column():
                probs = gr.Label(label="Option probabilities", num_top_classes=5)
                mcq_answer = gr.Markdown(label="Generated answer and explanation")
                mcq_domain = gr.Textbox(label="Domain check", interactive=False)
        mcq_btn.click(ask_mcq, [mcq_q] + opts + [mcq_rag, mcq_gate], [probs, mcq_answer, mcq_domain])

    with gr.Tab("How it works"):
        gr.Markdown(
            "1. **Domain gate:** the question is embedded with `all-MiniLM-L6-v2` and a logistic-regression classifier "
            "decides whether it is medical. Non-medical questions are refused.\n"
            "2. **Retrieval (optional):** the 3 most similar training QA pairs from AfriMed-QA are added to the prompt "
            "as reference notes.\n"
            "3. **Generation:** the fine-tuned Qwen2.5-Instruct model (LoRA adapters merged) generates the answer. For "
            "multiple-choice questions, the probabilities of the option letters after `Answer:` are shown.\n\n"
            "Data: AfriMed-QA v2 (Olatunji et al., 2024), CC BY-SA 4.0. Code: see the GitHub repository."
        )

if __name__ == "__main__":
    demo.queue(max_size=16).launch()
