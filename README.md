#  NLP Mini Pipeline: OCR → NER → Duygu Analizi → RAG

Türkçe belgeler üzerinde uçtan uca çalışan bir NLP pipeline'ı.  
Görüntüden metin çıkarımı, varlık tanıma, duygu analizi ve soru-cevap sistemini tek bir akışta birleştirir.

---

##  İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Pipeline Mimarisi](#-pipeline-mimarisi)
- [Modüller](#-modüller)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Proje Yapısı](#-proje-yapısı)
- [Kullanılan Teknolojiler](#-kullanılan-teknolojiler)
- [Metrikler](#-metrikler)
- [Katkı](#-katkı)

---

##  Proje Hakkında

Bu proje, Türkçe belgeler üzerinde çalışan dört aşamalı bir NLP pipeline'ı sunar. Her modül bağımsız çalışabildiği gibi `MiniPipeline` sınıfı aracılığıyla uçtan uca da çalıştırılabilir.

**Veri kaynağı:** Türkçe Wikipedia makaleleri (Türkiye, Yapay Zeka, İstanbul, Türk Mutfağı)

**Hedef platform:** Google Colab (CPU)

---

##  Pipeline Mimarisi

```
[Görüntü / Belge]
        │
        ▼
┌───────────────────────────────────────┐
│  MODÜL 1 — OCR                        │
│  Adaptive Thresholding · Noise        │
│  Removal · Tesseract (tur)            │
│  Çıktı: temiz metin + CER/WER         │
└──────────────────┬────────────────────┘
                   │ ham metin
                   ▼
┌───────────────────────────────────────┐
│  MODÜL 2 — NER                        │
│  spaCy tr + Regex Katmanı             │
│  Çıktı: varlık listesi + F1           │
└──────────────────┬────────────────────┘
                   │ varlıklar
                   ▼
┌───────────────────────────────────────┐
│  MODÜL 3 — Duygu Analizi              │
│  BERT Türkçe · Batch Inference        │
│  Çıktı: duygu etiketleri + F1         │
└──────────────────┬────────────────────┘
                   │ zenginleştirilmiş metin
                   ▼
┌───────────────────────────────────────┐
│  MODÜL 4 — RAG                        │
│  SentenceTransformer · FAISS          │
│  RecursiveChunking · Gradio UI        │
│  Çıktı: soru-cevap + JSON rapor       │
└───────────────────────────────────────┘
```

---

##  Modüller

### Bölüm 0 — Kurulum & Ortak Altyapı
Tüm pipeline boyunca kullanılan merkezi yapıları kurar.

| Bileşen | Açıklama |
|---------|----------|
| `make_logger` | Konsola ve dosyaya yazan biçimlendirilmiş logger |
| `ram_info` | Anlık RAM kullanımını MB cinsinden izler |
| `measure` | Her modülün süre ve RAM kullanımını ölçen context manager |
| `PipelineState` | Modüller arası veri taşıyıcısı dataclass'ı |

### Bölüm 1 — OCR Modülü
Görüntüdeki Türkçe metni 7 aşamalı preprocessing ile çıkarır.

- Adaptive thresholding, noise removal, morfoloji
- Tesseract `tur` dil paketi
- **Metrikler:** CER (Character Error Rate), WER (Word Error Rate), güven skoru

### Bölüm 2 — NER Modülü
Metinden isimli varlıkları tanır.

- Hibrit yaklaşım: spaCy tr modeli + kural tabanlı regex katmanı
- Tanınan varlıklar: `PARA`, `TARİH`, `KURUM`, `ŞEHİR`, `ÜLKE`, `KİŞİ`, `SAYI`
- **Metrikler:** Precision, Recall, F1 (varlık türü bazında)

### Bölüm 3 — Duygu Analizi Modülü
Cümle düzeyinde duygu sınıflandırması yapar.

- Model: BERT Türkçe (`savasy/bert-base-turkish-sentiment-cased`)
- Batch inference ile bellek verimli çıkarım
- **Metrikler:** F1 macro, confusion matrix

### Bölüm 4 — RAG Modülü
Metni parçalara böler, vektörize eder ve soru-cevap sistemi sunar.

- `SentenceTransformer` ile embedding (`paraphrase-multilingual-MiniLM-L12-v2`)
- FAISS vektör indeksi ile benzerlik araması
- `RecursiveChunker` ile örtüşen parçalama
- Gradio arayüzü ile interaktif demo

### Bölüm 5 — Bütünleşik Pipeline
`MiniPipeline` sınıfı tüm modülleri sırasıyla çalıştırır ve JSON rapor üretir.

### Bölüm 6 — Akademik Performans Dashboardu
Tüm modüllerin metriklerini tek bir görsel panelde sunar (matplotlib/plotly).

### Bölüm 7 — Özet Rapor
Çalışma sonuçlarını `output/` klasörüne JSON olarak kaydeder.

---

##  Kurulum

Bu proje **Google Colab** ortamı için tasarlanmıştır.

### 1. Repoyu klonla

```bash
git clone https://github.com/sslgulsena7/mini-ai-pipeline-api.git
```

### 2. Google Colab'da aç

[Google Colab](https://colab.research.google.com) → `Dosya` → `Not defteri aç` → `GitHub` sekmesinden repo URL'sini gir.

### 3. Kurulum hücresini çalıştır

Notebook'u açtıktan sonra **Bölüm 0.1** hücresini çalıştır. Gerekli tüm sistem ve Python paketleri otomatik kurulur.

```
Runtime → Run All  (veya sadece Bölüm 0.1 hücresini çalıştır)
```

>  Kurulum tamamlandıktan sonra **Runtime → Restart Runtime** yapılması zorunludur.

### Sistem gereksinimleri

| Gereksinim | Versiyon |
|------------|---------|
| Python | 3.10+ |
| Google Colab | Ücretsiz plan yeterli (CPU) |
| RAM | Min. 4 GB (Colab standart) |

### Bağımlılıklar

```
numpy>=1.24.0,<2.0.0
pytesseract
Pillow
opencv-python-headless
transformers>=4.41.2
torch (CPU)
sentence-transformers>=2.7.0
faiss-cpu
scikit-learn
gradio>=4.40.0
matplotlib
seaborn
plotly
pandas
psutil
tqdm
wikipedia-api
```

Sistem paketleri: `tesseract-ocr`, `tesseract-ocr-tur`, `libtesseract-dev`, `libgl1`

---

##  Kullanım

### Tam pipeline'ı çalıştırma

Bölüm 0.1 → Restart → Bölüm 0.2 → Bölüm 1 → ... → Bölüm 7 sırasıyla çalıştır.  
Ya da `MiniPipeline` sınıfını doğrudan kullan (Bölüm 5.2):

```python
pipeline = MiniPipeline()
state = pipeline.run(image_path="pipeline/images/Türkiye.png")
print(state.raw_text[:200])
print(state.entities)
print(state.sentiment)
```

### Sadece RAG arayüzünü açma

```python
# Bölüm 4.4 hücresini çalıştır
# Gradio arayüzü otomatik açılır
```

### Modülleri bağımsız kullanma

```python
# Sadece OCR
ocr = OcrModule()
result = ocr.run("pipeline/images/belge.png")
print(result.raw_text)

# Sadece NER
ner = NerModule()
entities = ner.run("Atatürk 1923'te Ankara'da Türkiye Cumhuriyeti'ni kurdu.")
print(entities)

# Sadece Duygu Analizi
sentiment = SentimentModule()
label = sentiment.run("Bu ürün gerçekten harika!")
print(label)
```

---

##  Proje Yapısı

```
├── 9-10_Hafta.ipynb          # Ana notebook
├── README.md                  # Bu dosya
├── requirements.txt           # Python bağımlılıkları (opsiyonel)
└── .gitignore                 # Git dışı dosyalar
```

Notebook çalıştırıldığında Colab ortamında aşağıdaki klasörler otomatik oluşur:

```
/content/pipeline/
├── logs/          # Modül logları (.log)
├── data/          # Ara veriler
├── images/        # Wikipedia'dan üretilen PNG belgeler
├── output/        # JSON raporlar ve son çıktılar
└── metrics/       # Modül bazında metrik dosyaları
```

---

##  Kullanılan Teknolojiler

| Alan | Teknoloji |
|------|-----------|
| OCR | Tesseract 5, OpenCV, Pillow |
| NER | spaCy (tr), Regex |
| Duygu Analizi | HuggingFace Transformers, BERT Türkçe |
| Embedding | SentenceTransformers (MiniLM-L12) |
| Vektör Arama | FAISS (CPU) |
| Arayüz | Gradio |
| Görselleştirme | Matplotlib, Seaborn, Plotly |
| Veri | Wikipedia API (Türkçe) |
| İzleme | psutil, logging |

---

##  Metrikler

Her modül aşağıdaki akademik metrikleri hesaplar ve görselleştirir:

| Modül | Metrikler |
|-------|-----------|
| OCR | CER, WER, karakter doğruluğu, kelime doğruluğu, güven skoru |
| NER | Precision, Recall, F1 (macro + varlık türü bazında), TP/FP/FN |
| Duygu Analizi | F1 macro, accuracy, confusion matrix |
| RAG | Top-k retrieval skoru, cevap latansı |
| Pipeline | Toplam süre, modül bazında RAM kullanımı |

---

##  .gitignore

```
# Python
__pycache__/
*.pyc
*.pyo
.env
venv/
.venv/

# Notebook
.ipynb_checkpoints/

# Colab çıktıları (büyük dosyalar)
*.png
*.jpg
*.log
pipeline/

# OS
.DS_Store
Thumbs.db
```

