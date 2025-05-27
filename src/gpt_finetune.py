from pathlib import Path
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
import torch

BASE = Path(__file__).resolve().parent.parent
INPUT = BASE / "data" / "processed" / "corpus_gpt.csv"

# Veriyi oku ve temizle
raw_df = pd.read_csv(INPUT)
raw_df = raw_df[raw_df["text"].str.split().str.len() > 5]  # 5 kelimeden kısa olanları çıkar

# Huggingface Dataset formatına dönüştür
hf_ds = Dataset.from_pandas(raw_df[["text"]])

# Tokenizer yükle ve uygulama
model_id = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=100
    )

tok_ds = hf_ds.map(tokenize, batched=True, remove_columns=["text"])
tok_ds.set_format("torch")

# Model ve data collator
model = AutoModelForCausalLM.from_pretrained(model_id)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Eğitim ayarları
args = TrainingArguments(
    output_dir=str(BASE / "outputs" / "gpt_finetuned"),
    overwrite_output_dir=True,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_strategy="no",
    logging_steps=10,
    report_to="none"
)

# Trainer başlat ve eğit
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tok_ds,
    data_collator=data_collator
)

trainer.train()

model.save_pretrained(BASE / "outputs" / "gpt_finetuned")
tokenizer.save_pretrained(BASE / "outputs" / "gpt_finetuned")
print("\n✅ GPT modeli başarıyla eğitildi ve kaydedildi.")
