"""Generate a cold-email draft using a trained adapter, never a claimed medical fact."""

import argparse

from train import MODEL_ID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", help="Path to the adapter saved by train.py")
    parser.add_argument("prompt", help="Instructions and verified facts for the email")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("This 4-bit example requires a CUDA GPU")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
        ),
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, args.adapter).eval()
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    print(tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
