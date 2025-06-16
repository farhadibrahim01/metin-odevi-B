from pathlib import Path
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE   = Path(__file__).resolve().parent.parent
MODEL  = BASE / "outputs" / "gpt_finetuned"
OUT    = BASE / "outputs" / "gpt_summaries.json"
LABELS = ["alcohol", "dvi", "paternity"]

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model     = AutoModelForCausalLM.from_pretrained(MODEL)

prompts = {
    "alcohol":   "Summarize the traffic-safety risks of alcohol in two sentences:",
    "dvi":       "Summarize Disaster Victim Identification (DVI) in two sentences:",
    "paternity": "Summarize forensic paternity testing in two sentences:"
}

def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_length=100,
            do_sample=False,
            num_beams=5,
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(ids[0], skip_special_tokens=True).replace(prompt, "").strip()

summaries = {lab: generate(prompts[lab]) for lab in LABELS}
OUT.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
print("gpt_summaries.json saved →", OUT)
