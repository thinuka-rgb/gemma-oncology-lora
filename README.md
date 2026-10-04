# Gemma 2 LoRA for oncology research emails

A training workflow for drafting outreach emails to oncology researchers. It adapts the instruction-tuned [Gemma 2 2B model](https://huggingface.co/google/gemma-2-2b-it) with a small private prompt/response dataset. The base model is loaded in 4-bit NF4; LoRA adapters target its attention projections.

**Status:** The linked [original Colab notebook](https://colab.research.google.com/drive/1YQM4pBs90pFDkmemgYZV7N_tim6P58Tm) contains 67 loaded examples and a failed training cell. Its saved adapter and generated email do **not** demonstrate successful fine-tuning. This repository provides a corrected training path, but no completed training run or quality improvement is claimed yet. See [the notebook audit](docs/notebook-audit.md).

## Run in Colab

Open [the clean Colab notebook](notebooks/gemma_oncology_email_lora.ipynb) and select a GPU runtime. The notebook clones this repository, installs dependencies, mounts Drive, validates the two private JSON files, trains, evaluates, and saves an adapter **only after** training succeeds. Access to Gemma on Hugging Face requires accepting its model terms and setting an `HF_TOKEN` Colab secret.

The original data files are **not published**. Review private examples for unsupported names, publications, student credentials and other claims before training; the code can validate structure, not truth. Supply your own JSON arrays containing objects like:

```json
[
  {
    "prompt": "Draft a concise email asking to learn about a researcher's verified work.",
    "response": "Subject: Interest in your research\n\nDear Professor..."
  },
  {
    "prompt": "Write a polite follow-up about a verified project.",
    "response": "Subject: Following up\n\nDear Professor..."
  }
]
```

For the original Colab layout, place `cold_email_dataset.json` and `cold_email_dataset_1.json` in `/content/drive/MyDrive/GemmaProject/`. To validate without loading a model:

```bash
python train.py --validate-only /path/to/cold_email_dataset.json /path/to/cold_email_dataset_1.json
```

For a GPU run outside Colab, install PyTorch with CUDA support, then `pip install -r requirements.txt`, authenticate to Hugging Face, and run:

```bash
python train.py /path/to/cold_email_dataset.json /path/to/cold_email_dataset_1.json
python generate.py outputs/gemma-oncology-email-lora/adapter "Draft an email using only these verified facts: ..."
```

The trainer saves `adapter/` and `metrics.json` under `outputs/gemma-oncology-email-lora/`. The adapter does not include the Gemma base weights. Generated emails need human fact checking before use, especially research claims, publication details, and names.

## What the code checks

- Validates the private JSON shape and nonempty prompt/response pairs.
- Uses Gemma's tokenizer chat template through TRL's conversational prompt/completion format.
- Uses completion-only loss so the model learns the response, and switches between BF16 and FP16 based on the GPU.
- Prepares the quantized base model, passes `LoraConfig` to `SFTTrainer` once, and aborts if the trainer does not hold trainable adapter parameters.
- Keeps a seeded 10% evaluation split and records train/evaluation loss. With only 67 original examples, this split is too small to establish reliable email quality or generalization.

This is an email-drafting experiment, **not** an oncology knowledge model or clinical tool.

The repository's MIT license covers its original code and documentation. Gemma model weights have [separate terms](https://huggingface.co/google/gemma-2-2b-it), and the private training datasets are not included.
