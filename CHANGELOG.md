# Changelog

Bu dosya projenin versiyon geçmişini takip eder.

---

## [1.0.0] - 2026-05-18

### Eklendi
- OCR modülü: 7 aşamalı preprocessing, Tesseract Türkçe, CER/WER metrikleri
- NER modülü: spaCy tr + regex hibrit katmanı, F1 değerlendirme
- Duygu Analizi modülü: BERT Türkçe, batch inference, confusion matrix
- RAG modülü: SentenceTransformer embedding, FAISS indeks, Gradio arayüzü
- Bütünleşik MiniPipeline sınıfı
- Akademik performans dashboardu (Bölüm 6)
- JSON özet rapor çıktısı (Bölüm 7)
