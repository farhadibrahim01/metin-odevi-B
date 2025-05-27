from pathlib import Path
from pdfminer.high_level import extract_text
import pandas as pd, re, unidecode as ud

BASE = Path(__file__).resolve().parent.parent
RAW  = BASE / "data" / "raw"
OUT  = BASE / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

def extract_chunks(text, win=60, stride=40):
    words = text.split()
    for i in range(0, len(words), stride):
        chunk = " ".join(words[i:i + win])
        if len(chunk.split()) >= 10:
            yield chunk.strip()

rows = []
for file in RAW.glob("*.pdf"):
    label = file.stem.lower().split()[0]
    if label == "forensic": label = "paternity"
    text = extract_text(file)
    text = re.sub(r"\s+", " ", ud.unidecode(text))
    for chunk in extract_chunks(text):
        rows.append({"label": label, "text": chunk})

df = pd.DataFrame(rows)
df.to_csv(OUT / "corpus_gpt.csv", index=False)
print("corpus_gpt.csv written:", len(df), "rows")
