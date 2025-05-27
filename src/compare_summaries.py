from pathlib import Path
import json
import pandas as pd
from simhash import Simhash

BASE = Path(__file__).resolve().parent.parent
OUT  = BASE / "outputs"

gpt      = json.loads((OUT / "gpt_summaries.json").read_text(encoding="utf-8"))
chatgpt  = json.loads((OUT / "chatgpt_summaries.json").read_text(encoding="utf-8"))

def similarity(a: str, b: str) -> float:
    return round((1 - Simhash(a).distance(Simhash(b)) / 64) * 100, 2)

df = pd.DataFrame(
    [{"Topic": k, "Similarity (%)": similarity(gpt[k], chatgpt[k])} for k in gpt]
)

print(df.to_string(index=False))
