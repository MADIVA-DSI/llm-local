# pip install unsloth
# pip install torch transformers datasets peft trl

from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer


from transformers import TrainingArguments
import json


model = "unsloth/Llama-3.2-3B-Instruct"
# Swap for llama3:8b or mistral:7b if you want stronger baseline quality.

model, tokeniser = FastLanguageModel.from_pretrained(
    model_name  = model,
    max_seq_length = 1024,
    load_in_4bit   = True,    # 4-bit quantisation: fits on one L40S easily
)

[O# ── Add LoRA adapters ────────────────────────────────────────────────────────
# LoRA freezes the base model and trains only small rank-decomposition matrices.
# For 100 training examples, a small rank (r=16) is appropriate.
# We are NOT changing the model's knowledge -- only its output behaviour.

model = FastLanguageModel.get_peft_model(
    model,
    r              = 16,      # LoRA rank -- higher = more capacity, more VRAM
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha     = 16,
    lora_dropout   = 0.05,
    bias           = "none",
    use_gradient_checkpointing = True,
)

# ── Format dataset ───────────────────────────────────────────────────────────
# Alpaca prompt template -- instruction + input + output

PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""+tokeniser.eos_token

def format_prompt(example):
    return {"text": PROMPT_TEMPLATE.format(
        instruction = example["instruction"],
        input       = example["input"],
        output      = example["output"],
    )}

with open("data/finetune_dataset.json") as f:
    raw = json.load(f)

dataset = Dataset.from_list(raw).map(format_prompt)

# ── Train ────────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model        = model,
    tokenizer    = tokeniser,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length     = 1024,
    args = TrainingArguments(
        output_dir          = "./metabolomics_lora",
        num_train_epochs    = 10,
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 2,
        learning_rate       = 2e-4,
        fp16                = False,
        bf16                = True,
        logging_steps       = 10,
        save_strategy       = "epoch",
        warmup_ratio        = 0.1,
        lr_scheduler_type   = "cosine",
        report_to           = "none",
    ),
)

trainer.train()
model.save_pretrained("data/metabolomics_lora")
tokeniser.save_pretrained("data/metabolomics_lora")

model.save_pretrained_merged("data/scott_ft_model", tokeniser, save_method="merged_16bit")
