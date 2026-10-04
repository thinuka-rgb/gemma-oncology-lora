"""Fine-tune Gemma 2 2B IT with a 4-bit base and trainable LoRA adapters."""

import argparse
import json
from pathlib import Path

from data import load_examples, to_conversations

MODEL_ID = "google/gemma-2-2b-it"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="+", help="Private JSON files with prompt/response objects")
    parser.add_argument("--output", default="outputs/gemma-oncology-email-lora")
    parser.add_argument("--validate-only", action="store_true", help="Check input without loading the model")
    args = parser.parse_args()

    examples = load_examples(args.data)
    print(f"Validated {len(examples)} examples from {len(args.data)} file(s).")
    if args.validate_only:
        return

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("4-bit fine-tuning requires a CUDA GPU; use Colab's GPU runtime")

    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        ),
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    dataset = Dataset.from_list(to_conversations(examples))
    split = dataset.train_test_split(test_size=max(1, round(len(examples) * 0.1)), seed=42)
    output = Path(args.output)
    config = SFTConfig(
        output_dir=str(output / "checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        report_to="none",
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        max_length=768,
        completion_only_loss=True,
        eos_token="<end_of_turn>",
        save_strategy="no",
    )
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
        peft_config=lora,
    )
    if not isinstance(trainer.model, PeftModel):
        raise RuntimeError("LoRA adapter was not attached; refusing to train the quantized base model")
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    if trainable <= 0:
        raise RuntimeError("The LoRA adapter has no trainable parameters")
    print(f"Trainable adapter parameters: {trainable:,}")

    result = trainer.train()
    evaluation = trainer.evaluate()
    adapter_dir = output / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    metrics = {
        "model": MODEL_ID,
        "examples": len(examples),
        "train_examples": len(split["train"]),
        "eval_examples": len(split["test"]),
        "train_loss": result.training_loss,
        "eval_loss": evaluation.get("eval_loss"),
        "trainable_parameters": trainable,
        "seed": 42,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Saved adapter and metrics under {output}")


if __name__ == "__main__":
    main()
