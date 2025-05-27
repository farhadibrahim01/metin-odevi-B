from pathlib import Path
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling

BASE = Path(__file__).resolve().parent.parent
INPUT = BASE / "data" / "processed" / "corpus_gpt.csv"

df = pd.read_csv(INPUT)
df = df[df["text"].str.split().str.len() > 5]

hf_ds = Dataset.from_pandas(df[["text"]])

model_id = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=100)

tok_ds = hf_ds.map(tokenize, batched=True, remove_columns=["text"])
tok_ds.set_format("torch")

model = AutoModelForCausalLM.from_pretrained(model_id)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

args = TrainingArguments(
    output_dir=str(BASE / "outputs" / "gpt_finetuned"),
    overwrite_output_dir=True,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_strategy="no",
    logging_steps=10,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tok_ds,
    data_collator=data_collator
)

trainer.train()

model.save_pretrained(BASE / "outputs" / "gpt_finetuned")
tokenizer.save_pretrained(BASE / "outputs" / "gpt_finetuned")
print("GPT modeli başarıyla eğitildi ve kaydedildi.")
