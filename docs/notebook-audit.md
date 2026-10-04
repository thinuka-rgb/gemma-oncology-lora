# Audit of the original Colab notebook

Source: [Copy of Working_Gemma_4.0.ipynb](https://colab.research.google.com/drive/1YQM4pBs90pFDkmemgYZV7N_tim6P58Tm), inspected 2026-10-04. The linked notebook is stored in Drive; this repository contains a clean replacement with no copied execution outputs.

| Observation | Consequence | Change here |
| --- | --- | --- |
| `google/gemma-2-2b-it` was loaded with NF4 4-bit quantization; LoRA rank 8 was attached to `q_proj`, `k_proj`, `v_proj`, and `o_proj`. | These are real Gemma attention modules and a valid adapter approach. The notebook printed 3,194,880 trainable parameters. | Retained the model, NF4 and attention targets. |
| The notebook's `SFTTrainer` setup was rerun after `AcceleratorState._reset_state()`. The visible cell fails with “You cannot perform fine-tuning on purely quantized models.” | A private accelerator reset is not a repair, and the failed cell provides no evidence of a training run. | Prepare the quantized base once, then construct one `SFTTrainer` with `peft_config`; check that PEFT is attached before `train()`. |
| The notebook then saved an adapter zip and generated an email despite that failure. A later `model.eval()` displayed a plain `Gemma2ForCausalLM`. | The adapter may be untrained or from earlier runtime state; the sample could be base-model output. Its provenance cannot be established from the notebook. | Save only after `train()` returns successfully; load the saved adapter explicitly for inference. |
| The two datasets were loaded from private Drive and reported 67 total examples. | Their contents, quality, originality and absence of sensitive data cannot be verified from the notebook alone. | Keep them private; validate schema at runtime. |
| The original code set `bf16=True` on a T4 runtime. | NVIDIA T4 does not support native BF16. | Select BF16 only if `torch.cuda.is_bf16_supported()`; otherwise use FP16. |
| Prompts and completions were hand-wrapped with Gemma role markers. The inference prompt was also hand-written. | Easy to duplicate tokens or mismatch the model's chat template. | Use conversational records and `tokenizer.apply_chat_template` for inference. |
| No held-out evaluation, baseline comparison or factuality check was shown. | No quality or medical accuracy claim can be made. | Record a small held-out loss; require human review of generated text. |

Implementation choices follow the [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization), [TRL SFTTrainer docs](https://huggingface.co/docs/trl/sft_trainer), and [Gemma 2 model card](https://huggingface.co/google/gemma-2-2b-it).

## GPU smoke test (2026-10-04)

The published `train.py` ran on a Colab T4 with five synthetic prompt/response examples (four train, one evaluation). It completed with exit code 0 and saved `adapter_config.json` and `adapter_model.safetensors`. Metrics reported 3,194,880 trainable parameters, training loss 6.2043, and evaluation loss 4.6731. These values only establish that the code trains and saves a LoRA adapter; the tiny artificial split says nothing about output quality. Colab's Google Drive mount failed with `ValueError: mount failed`, so the 67 private examples were not used in this test. The separate generation script was not GPU-tested in that session.
