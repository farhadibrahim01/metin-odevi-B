# Metin Ödevi – B Kısmı

Bu projede, metin ödevi A kısmında hazırlanan etiketli külliyatlardan yararlanılarak GPT tabanlı özet üretimi gerçekleştirilmiş, ardından ChatGPT ile karşılaştırma yapılmıştır.

## Klasör Yapısı

```
Metin-B/
├── data/
│   ├── raw/                # PDF belgeleri
│   ├── processed/          # corpus_gpt.csv
│   └── corpus/             # metin parçaları (isteğe bağlı)
│
├── outputs/
│   ├── gpt_summaries.json
│   ├── chatgpt_summaries.json
│   └── gpt_finetuned/      # model çıktıları (büyük dosya hariç)
│
├── src/
│   ├── build_corpus.py
│   ├── gpt_finetune.py
│   ├── generate_summaries.py
│   ├── make_chatgpt_json.py
│   └── compare_summaries.py
```

## Adımlar

1. `build_corpus.py`: PDF’lerden metin çıkarır ve corpus_gpt.csv dosyasını üretir.
2. `gpt_finetune.py`: distilgpt2 modeli ile eğitim yapılır.
3. `generate_summaries.py`: Eğitilen modelle her konu için özet üretir.
4. `make_chatgpt_json.py`: Aynı konular için ChatGPT özetlerini JSON olarak kaydeder.
5. `compare_summaries.py`: GPT ve ChatGPT özetlerini simhash yöntemiyle karşılaştırır.

## Kullanılan Kütüphaneler

- transformers
- datasets
- torch
- pandas
- nltk
- simhash
- pdfminer.six

## Not

`outputs/gpt_finetuned/model.safetensors` dosyası GitHub limitini (100MB) aştığı için dahil edilmemiştir. Eğitim kodu ile yeniden üretilebilir.

Bu proje İstanbul Ticaret Üniversitesi "İleri Makine Öğrenmesi" dersi kapsamında hazırlanmıştır.
