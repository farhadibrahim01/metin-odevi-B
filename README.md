# Metin-Ödevi B

Bu projede, üç farklı konuda (`alcohol`, `dvi`, `paternity`) yer alan metinlerden GPT-2 modeliyle özetler üretilmiş ve bu özetler, ChatGPT’nin aynı konular için oluşturduğu özetlerle karşılaştırılmıştır. Amaç, her konuya özel bir modelin genel bir dil modeline kıyasla ne kadar özgün ve başarılı özetler üretebildiğini analiz etmektir.

## Klasör Yapısı

```
metin-odevi-B/
├── data/
│   ├── alcohol.txt
│   ├── dvi.txt
│   └── paternity.txt
├── outputs/
│   ├── corpus_gpt.csv
│   ├── gpt_summaries.json
│   └── gpt_finetuned/
├── src/
│   ├── build_corpus.py
│   ├── gpt_finetune.py
│   ├── generate_summaries.py
│   └── compare_summaries.py
├── chatgpt_summaries.json
├── requirements.txt
└── README.md
```

## Gereksinimlerin Kurulumu

Python ortamı oluşturmak ve gerekli paketleri yüklemek için:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Çalıştırma Sırası (Run Chronology)

Aşağıdaki komutlar sırasıyla çalıştırılır:

```bash
python src/build_corpus.py
python src/gpt_finetune.py
python src/generate_summaries.py
python src/compare_summaries.py
```

## Script Açıklamaları

| Script                    | Açıklama |
|---------------------------|----------|
| `build_corpus.py`         | `data/` klasöründeki metinleri birleştirerek `outputs/corpus_gpt.csv` dosyasını oluşturur. |
| `gpt_finetune.py`         | GPT-2 modelini her konu için `corpus_gpt.csv` üzerinden fine-tune eder. Model çıktıları `outputs/gpt_finetuned/` klasörüne yazılır. |
| `generate_summaries.py`   | Fine-tuned GPT-2 ile her konu için otomatik özetler üretir ve `outputs/gpt_summaries.json` içine kaydeder. |
| `compare_summaries.py`    | Fine-tuned GPT-2'nin ürettiği özetlerle ChatGPT'nin özetlerini karşılaştırarak benzerlik skorlarını hesaplar. |

## Örnek Sonuç

```text
    Topic  Similarity (%)
  alcohol           48.44
      dvi           65.62
paternity           57.81
```

Bu tablo, her konu için GPT-2 ile üretilen özetlerin ChatGPT özetlerine olan benzerlik oranlarını simhash algoritmasıyla vermektedir.

## Kullanılan Teknolojiler

- [Transformers (GPT-2)](https://huggingface.co/gpt2)
- Python 3.10+
- simhash, pandas, torch, tqdm

## Sonuç

Bu projede konuya özel fine-tune edilmiş bir GPT-2 modelinin yetenekleri incelendi. Ürettiği özetlerin ChatGPT ile karşılaştırılması sonucunda bazı konularda özgünlük, bazı konularda ise yüksek benzerlik sağladığı gözlemlendi. Küçük modellerin konuya özgü eğitimle etkili hale getirilebildiği gösterildi.
