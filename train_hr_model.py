import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "google/flan-t5-base"
DATA_PATH = "hr_train.jsonl"
OUTPUT_DIR = "hr_finetuned_model"

# 1) Load dataset
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# 2) Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# 3) Apply LoRA
#r is rank how much capacity the LoRA layer has.
#task type is sequence-to-sequence model (question → answer).
#lora_alpha=16 (Scaling factor)This controls how strongly the LoRA updates affect the main model.Higher value → LoRA has more influence.Lower value → very gentle changes
#lora_dropout=0.1.This is like regularization:During training, 10% of LoRA connections are randomly dropped.Prevents the model from memorizing your tiny dataset.

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 4) Preprocess
def preprocess(batch):
    inputs = [f"Question: {q}\nAnswer:" for q in batch["input"]]
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=256)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["output"], truncation=True, padding="max_length", max_length=256)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

# 5) Training setup
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=5,
    fp16=False,              # keep False on CPU/Windows
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    report_to="none"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    data_collator=data_collator,
)

# 6) Train
trainer.train()

# 7) Save LoRA-adapted model
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Fine-tuning complete. Model saved to:", OUTPUT_DIR)
