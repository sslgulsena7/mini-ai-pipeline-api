"""
NER Modülü
----------
Türkçe metinden isimli varlık tanıma.
spaCy tr modeli + regex hibrit katmanı.
"""

import re
import spacy
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class NerResult:
    entities: List[Dict[str, Any]]
    metrics: Dict[str, Any]


class NerModule:
    """
    Türkçe metinden varlık çıkaran hibrit NER modülü.

    Kullanım:
        ner = NerModule()
        result = ner.run("Atatürk 1923'te Ankara'da cumhuriyeti kurdu.")
        print(result.entities)
    """

    REGEX_PATTERNS = {
        "PARA": r"\b\d+[\.,]?\d*\s*(TL|USD|EUR|lira|dolar|euro)\b",
        "TARIH": r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|\b\d{4}\b",
        "SAYI": r"\b\d+[\.,]?\d*\b",
    }

    CITY_LIST = {
        "ankara", "istanbul", "izmir", "bursa", "antalya",
        "adana", "konya", "gaziantep", "şanlıurfa", "mersin"
    }

    def __init__(self, model: str = "tr_core_news_sm"):
        try:
            self.nlp = spacy.load(model)
        except OSError:
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "spacy", "download", model], check=True)
            self.nlp = spacy.load(model)

    def _regex_entities(self, text: str) -> List[Dict[str, Any]]:
        """Regex katmanı ile yapısal varlıkları bulur."""
        found = []
        for label, pattern in self.REGEX_PATTERNS.items():
            for m in re.finditer(pattern, text, re.IGNORECASE):
                found.append({
                    "text": m.group(),
                    "label": label,
                    "start": m.start(),
                    "end": m.end(),
                    "source": "regex"
                })
        return found

    def _spacy_entities(self, text: str) -> List[Dict[str, Any]]:
        """spaCy modeli ile varlıkları bulur."""
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "source": "spacy"
            })
        for token in doc:
            if token.text.lower() in self.CITY_LIST:
                entities.append({
                    "text": token.text,
                    "label": "SEHIR",
                    "start": token.idx,
                    "end": token.idx + len(token.text),
                    "source": "lookup"
                })
        return entities

    def _deduplicate(self, entities: List[Dict]) -> List[Dict]:
        """Çakışan varlıkları temizler."""
        seen = set()
        result = []
        for ent in sorted(entities, key=lambda x: x["start"]):
            key = (ent["start"], ent["end"])
            if key not in seen:
                seen.add(key)
                result.append(ent)
        return result

    def evaluate(self, predicted: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
        """Precision, Recall, F1 hesaplar."""
        pred_set = {(e["text"], e["label"]) for e in predicted}
        gt_set = {(e["text"], e["label"]) for e in ground_truth}
        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        return {"precision": precision, "recall": recall, "f1": f1,
                "tp": tp, "fp": fp, "fn": fn}

    def run(self, text: str, ground_truth: List[Dict] = None) -> NerResult:
        """Metinden varlıkları çıkarır ve değerlendirir."""
        spacy_ents = self._spacy_entities(text)
        regex_ents = self._regex_entities(text)
        all_ents = self._deduplicate(spacy_ents + regex_ents)
        metrics = {}
        if ground_truth:
            metrics = self.evaluate(all_ents, ground_truth)
        return NerResult(entities=all_ents, metrics=metrics)
