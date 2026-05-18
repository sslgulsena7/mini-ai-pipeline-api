# Pipeline Mimarisi

## Genel Akış

Görüntü → OCR → NER → Duygu Analizi → RAG → JSON Rapor

## Modüller

### OCR
Girdi olarak PNG veya JPG belge görüntüsü alır. OpenCV ile adaptif
eşikleme, gürültü giderme ve morfolojik işlemler uygulanır. Temizlenen
görüntü Tesseract 5 Türkçe modeline verilir. Çıktı ham metin ve
CER/WER skorlarıdır.

### NER
OCR çıktısı olan ham metin üzerinde çalışır. spaCy Türkçe modeli
temel varlık tanımayı yapar, üstüne regex katmanı eklenerek para,
tarih ve sayı gibi yapısal varlıklar da yakalanır. Çıktı varlık
tipi ve pozisyon bilgisi içeren liste ve F1 skorudur.

### Duygu Analizi
Metin cümlelere bölünür. Her cümle HuggingFace üzerinden yüklenen
BERT Türkçe modeline batch olarak verilir. Çıktı cümle bazında
pozitif/negatif/nötr etiket ve macro F1 skorudur.

### RAG
Metin RecursiveChunker ile örtüşen parçalara bölünür. Her parça
SentenceTransformer ile vektörleştirilir ve FAISS indeksine yazılır.
Sorgu geldiğinde en benzer parçalar getirilir, yanıt üretilir.
Gradio arayüzü ile interaktif kullanım sağlanır.

## Veri Akışı

PipelineState dataclass'ı tüm modüller arasında veri taşır.
Her modül bu nesneyi okur, kendi çıktısını yazar ve bir sonrakine iletir.

## Performans İzleme

Her modül measure() context manager ile sarılıdır.
Süre, RAM kullanımı ve başarı durumu otomatik loglanır.
Tüm metrikler Bölüm 6 dashboardunda görselleştirilir.
