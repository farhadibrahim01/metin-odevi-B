from pathlib import Path
import json     # <- ayrı satır

chatgpt = {
    "alcohol": (
        "Alcohol impairs driving by slowing reaction time, reducing vision "
        "clarity and lowering judgement, which raises the risk of accidents."
    ),
    "dvi": (
        "Disaster Victim Identification (DVI) applies forensic techniques such "
        "as DNA, fingerprints and dental records to identify victims of mass "
        "fatality events."
    ),
    "paternity": (
        "DNA-based paternity testing compares genetic markers of a child and "
        "an alleged father to confirm or exclude biological relationship."
    )
}

out_dir = Path(__file__).resolve().parent.parent / "outputs"
out_dir.mkdir(exist_ok=True)
(out_dir / "chatgpt_summaries.json").write_text(
    json.dumps(chatgpt, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
print("chatgpt_summaries.json yazıldı ➜", out_dir)
