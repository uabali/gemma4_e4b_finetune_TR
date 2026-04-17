# Turkish Gemma-4 RAG Finetune

This repository contains a Jupyter Notebook for fine-tuning the **Gemma4-e4b** base model specifically for **Turkish Retrieval Augmented Generation (RAG)** tasks. The project utilizes **Unsloth** for efficient, memory-optimized training.

## Overview

The primary goal of this project is to improve the model's ability to answer questions based on a provided Turkish context. By fine-tuning on a specialized dataset, the model learns to stay faithful to the given information and ignore irrelevant background noise.

### Key Technical Specs

* **Base Model:** Gemma-4
* **Optimization:** Unsloth
* **Method:** LoRA / QLoRA (Ranked Adaptation)
* **Language:** Turkish (TR)
* **Task Type:** RAG (Question + Context -> Answer)

## Dataset

The model is trained on **3,000 Turkish examples**. Each example follows a specific chat template:

- **User Instruction:** Contains a "Question" and a "Context".
- **Model Response:** The answer derived strictly from the provided context.

Training focuses exclusively on the model's responses to ensure efficient learning (`train_on_responses_only` is enabled).

## Performance Highlights

* **Training Duration:** ~42 minutes for 2 epochs on an A100 GPU.
* **Trainable Parameters:** ~18.35 Million (0.23% of total parameters).
* **Memory Efficiency:** Smart gradient offloading to minimize VRAM usage.

## Getting Started

### Prerequisites

```bash
pip install unsloth
pip install --no-deps xformers trl peft accelerate bitsandbytes
```

### Inference Example

You can use the following snippet to test the fine-tuned model (adapters):

```python
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "uabali/gemma4-e4b-rag-TR",
    max_seq_length = 2048,
    load_in_4bit = True,
)
# FastModel.for_inference(model)

messages = [
    {"role": "user", "content": "Soru:\n[SORUNUZ]\n\nBağlam:\n[BAĞLAM METNİ]"}
]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

## Export Options

The notebook provides scripts to export the model in several formats:

1. **LoRA Adapters:** Small files containing only the fine-tuned weights.
2. **Merged 16-bit:** The full model with adapters merged (for vLLM/deployment).
3. **GGUF:** Quantized formats for local execution via **Ollama** or **Llama.cpp**.

## License

This project is licensed under the **Apache License 2.0**. See the `LICENSE` file for details.

---

*Created by [uabali](https://huggingface.co/uabali)*
