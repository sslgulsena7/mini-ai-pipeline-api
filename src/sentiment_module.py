"""
Duygu Analizi Modülü
--------------------
BERT Türkçe ile cümle bazında duygu sınıflandırması.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class SentimentResult:
    labels: List[str]
    scores: List[float]
    overall: str
    metrics: Dict[str, Any] = field(default_factory=dict)


class SentimentModule:
    """
    Türkçe metinde duygu analizi yapan BERT tabanlı modül.

    Kullanım:
        sentiment = SentimentModule()
        result = sentiment.run("Bu proje gerçekten çok başarılı.")
        print(result.overall)
    """

    MODEL_NAME = "savasy/bert-base-turkish-sentiment-cased"

    def __init__(self, batch_size: int = 16):
        self.batch_size = batch_size
        self._pipeline = None

    def _load(self):
        """Modeli lazy load eder — ilk çağrıda yükler."""
        if self._pipeline is None:
            from transformers import pipeline
            self._pipeline = pipeline(
                "text-classification",
                model=self.MODEL_NAME,
                truncation=True,
                max_length=512
            )

    def _split_sentences(self, text: str) -> List[str]:
        """Metni cümlelere böler."""
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _batch(self, lst: list, size: int):
        """Listeyi batch'lere böler."""
        for i in range(0, len(lst), size):
            yield lst[i:i + size]

    def evaluate(self, predicted: List[str], ground_truth: List[str]) -> Dict[str, float]:
        """Macro F1 hesaplar."""
        from sklearn.metrics import f1_score, accuracy_score
        if not ground_truth or len(predicted) != len(ground_truth):
            return {}
        return {
            "f1_macro": f1_score(ground_truth, predicted, average="macro", zero_division=0),
            "accuracy": accuracy_score(ground_truth, predicted)
        }

    def run(self, text: str, ground_truth: List[str] = None) -> SentimentResult:
        """Metni cümlelere böler ve her cümleye duygu etiketi atar."""
        self._load()
        sentences = self._split_sentences(text)
        if not sentences:
            sentences = [text[:512]]
        labels, scores = [], []
        for batch in self._batch(sentences, self.batch_size):
            results = self._pipeline(batch)
            for r in results:
                labels.append(r["label"])
                scores.append(round(r["score"], 4))
        overall = max(set(labels), key=labels.count) if labels else "neutral"
        metrics = {}
        if ground_truth:
            metrics = self.evaluate(labels, ground_truth)
        return SentimentResult(labels=labels, scores=scores,
                               overall=overall, metrics=metrics)
