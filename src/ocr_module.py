"""
OCR Modülü
----------
Görüntüden Türkçe metin çıkarımı.
7 aşamalı preprocessing + Tesseract + CER/WER metrikleri.
"""

import cv2
import numpy as np
import pytesseract
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class OcrResult:
    raw_text: str = ""
    confidence: float = 0.0
    cer: float = 0.0
    wer: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)


class OcrModule:
    """
    Türkçe belge görüntülerinden metin çıkaran OCR modülü.

    Kullanım:
        ocr = OcrModule()
        result = ocr.run("images/belge.png")
        print(result.raw_text)
    """

    def __init__(self, lang: str = "tur", psm: int = 6):
        self.lang = lang
        self.config = f"--psm {psm} --oem 3"

    def preprocess(self, image_path: str) -> np.ndarray:
        """7 aşamalı görüntü ön işleme."""
        img = cv2.imread(str(image_path))
        # 1. Griye çevir
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 2. Gürültü giderme
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        # 3. Adaptif eşikleme
        thresh = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        # 4. Morfolojik açma
        kernel = np.ones((1, 1), np.uint8)
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        # 5. Keskinleştirme
        kernel_sharp = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        sharpened = cv2.filter2D(opened, -1, kernel_sharp)
        # 6. Yeniden boyutlandırma (300 DPI hedefi)
        h, w = sharpened.shape
        if w < 1800:
            sharpened = cv2.resize(sharpened, (w * 2, h * 2),
                                   interpolation=cv2.INTER_CUBIC)
        # 7. Deskew (eğim düzeltme)
        coords = np.column_stack(np.where(sharpened > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            if abs(angle) < 10:
                M = cv2.getRotationMatrix2D(
                    (sharpened.shape[1] // 2, sharpened.shape[0] // 2),
                    angle, 1.0
                )
                sharpened = cv2.warpAffine(
                    sharpened, M, (sharpened.shape[1], sharpened.shape[0]),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
        return sharpened

    def _cer(self, ref: str, hyp: str) -> float:
        """Character Error Rate hesaplar."""
        if not ref:
            return 0.0
        ref, hyp = list(ref), list(hyp)
        d = np.zeros((len(ref) + 1, len(hyp) + 1))
        for i in range(len(ref) + 1):
            d[i][0] = i
        for j in range(len(hyp) + 1):
            d[0][j] = j
        for i in range(1, len(ref) + 1):
            for j in range(1, len(hyp) + 1):
                cost = 0 if ref[i-1] == hyp[j-1] else 1
                d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
        return d[len(ref)][len(hyp)] / len(ref)

    def _wer(self, ref: str, hyp: str) -> float:
        """Word Error Rate hesaplar."""
        ref_words = ref.split()
        hyp_words = hyp.split()
        if not ref_words:
            return 0.0
        d = np.zeros((len(ref_words)+1, len(hyp_words)+1))
        for i in range(len(ref_words)+1):
            d[i][0] = i
        for j in range(len(hyp_words)+1):
            d[0][j] = j
        for i in range(1, len(ref_words)+1):
            for j in range(1, len(hyp_words)+1):
                cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
                d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
        return d[len(ref_words)][len(hyp_words)] / len(ref_words)

    def run(self, image_path: str, reference_text: str = "") -> OcrResult:
        """Görüntüyü işler ve metin çıkarır."""
        processed = self.preprocess(image_path)
        data = pytesseract.image_to_data(
            processed, lang=self.lang,
            config=self.config,
            output_type=pytesseract.Output.DICT
        )
        confidences = [int(c) for c in data["conf"] if str(c).isdigit() and int(c) > 0]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        raw_text = pytesseract.image_to_string(
            processed, lang=self.lang, config=self.config
        ).strip()
        cer = self._cer(reference_text, raw_text) if reference_text else 0.0
        wer = self._wer(reference_text, raw_text) if reference_text else 0.0
        return OcrResult(
            raw_text=raw_text,
            confidence=avg_conf,
            cer=cer,
            wer=wer,
            metrics={"char_count": len(raw_text), "word_count": len(raw_text.split())}
        )
