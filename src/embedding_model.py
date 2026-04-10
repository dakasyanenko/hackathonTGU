"""
Модуль эмбеддингов.

Подход к представлению текста из исследовательской работы:
- Символьные n-граммы (1-3) — лучший результат по статье
- Sentence-эмбеддинги через nomic-embed-text или fallback на all-MiniLM-L6-v2
- Конкатенация представлений

В статье показано, что TF-IDF на символьных n-граммах стабильно
превосходит sentence-эмбеддинги BERT для сопоставления вакансий и резюме.
"""

from __future__ import annotations

import re
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


class CharacterNGramVectorizer:
    """
    TF-IDF векторизатор на n-граммах.

    Согласно статье, n-граммы (1-3) дают наивысшую точность
    сопоставления по сравнению со словесными n-граммами и BERT-эмбеддингами.
    """

    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 3),
        max_features: int = 10000,
    ):
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            max_features=max_features,
            strip_accents="unicode",
            lowercase=True,
        )
        self._is_fitted = False

    def fit(self, texts: list[str]) -> None:
        """Обучает векторизатор на корпусе текстов."""
        self.vectorizer.fit(texts)
        self._is_fitted = True

    def transform(self, texts: list[str]) -> np.ndarray:
        """Преобразует тексты в TF-IDF векторы символьных n-грамм."""
        if not self._is_fitted:
            raise RuntimeError("Векторизатор не обучен. Сначала вызовите fit().")
        return self.vectorizer.transform(texts).toarray().astype(np.float32)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        """Обучение и преобразование за один шаг."""
        result = self.vectorizer.fit_transform(texts)
        self._is_fitted = True
        return result.toarray().astype(np.float32)

    @property
    def dimension(self) -> int:
        if not self._is_fitted:
            raise RuntimeError("Векторизатор не обучен.")
        return len(self.vectorizer.get_feature_names_out())


class EmbeddingModel:
    """
    Гибридная модель эмбеддингов, объединяющая:
    1. Sentence-эмбеддинги (nomic-embed-text или fallback-модель)
    2. TF-IDF на символьных n-граммах (лучший результат по статье)

    Итоговое представление может использовать один или оба компонента.
    """

    def __init__(
        self,
        sentence_model_name: str = "all-MiniLM-L6-v2",
        use_char_ngrams: bool = True,
        char_ngram_max_features: int = 10000,
    ):
        print(f"Загрузка sentence-трансформера: {sentence_model_name}...")
        self.sentence_model = SentenceTransformer(sentence_model_name)
        self.sentence_dim = self.sentence_model.get_sentence_embedding_dimension()
        print(f"Размерность sentence-эмбеддинга: {self.sentence_dim}")

        self.use_char_ngrams = use_char_ngrams
        self.char_vectorizer = CharacterNGramVectorizer(
            ngram_range=(1, 3), max_features=char_ngram_max_features
        )
        self.char_dim: int | None = None

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def encode(self, text: str, normalize: bool = True) -> np.ndarray:
        """Создаёт эмбеддинг для одного текста."""
        sent_emb = self.sentence_model.encode(
            [text], normalize_embeddings=normalize
        )[0].astype(np.float32)

        if self.use_char_ngrams and self.char_vectorizer._is_fitted:
            char_emb = self.char_vectorizer.transform([text])[0]
            return np.concatenate([sent_emb, char_emb])

        return sent_emb

    def encode_batch(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """Создаёт эмбеддинги для батча текстов."""
        sent_embs = self.sentence_model.encode(
            texts, normalize_embeddings=normalize
        ).astype(np.float32)

        if self.use_char_ngrams and self.char_vectorizer._is_fitted:
            char_embs = self.char_vectorizer.transform(texts)
            return np.concatenate([sent_embs, char_embs], axis=1)

        return sent_embs

    def build_char_vocab(self, texts: list[str]) -> None:
        """Строит словарь символьных n-грамм из корпуса."""
        self.char_vectorizer.fit(texts)
        self.char_dim = self.char_vectorizer.dimension
        print(f"Размер словаря символьных n-грамм: {self.char_dim}")

    @property
    def dimension(self) -> int:
        """Полная размерность эмбеддинга."""
        dim = self.sentence_dim
        if self.use_char_ngrams and self.char_dim:
            dim += self.char_dim
        return dim

    # ------------------------------------------------------------------
    # Вспомогательные методы подготовки текста
    # ------------------------------------------------------------------

    @staticmethod
    def prepare_vacancy_text(vacancy: dict) -> str:
        """
        Подготавливает текст вакансии для эмбеддинга.
        По статье: суммаризация вакансий (удаление шаблонного текста) улучшает результаты.
        Оставляем только ключевые поля.
        """
        parts: list[str] = []
        data = vacancy.get("data", vacancy)

        if data.get("position"):
            parts.append(f"Position: {data['position']}")
        if data.get("industry"):
            parts.append(f"Industry: {data['industry']}")
        if data.get("mandatoryRequirements"):
            parts.append(f"Requirements: {data['mandatoryRequirements']}")
        if data.get("projectTasks"):
            parts.append(f"Tasks: {data['projectTasks']}")
        if data.get("experienceLevels"):
            parts.append(f"Experience: {data['experienceLevels']}")
        if data.get("description"):
            parts.append(data["description"])
        if vacancy.get("skills"):
            parts.append(f"Skills: {', '.join(vacancy['skills'])}")
        if vacancy.get("dataEng"):
            parts.append(vacancy["dataEng"])

        return " ".join(parts)

    @staticmethod
    def prepare_resume_text(resume: dict) -> str:
        """
        Подготавливает текст резюме для эмбеддинга.
        По статье: НЕ суммаризировать резюме — они уже достаточно краткие.
        """
        parts: list[str] = []

        if resume.get("skill_set"):
            parts.append(f"Skills: {', '.join(resume['skill_set'])}")

        for exp in resume.get("experience", []):
            exp_parts: list[str] = []
            if exp.get("company"):
                exp_parts.append(exp["company"])
            if exp.get("description"):
                exp_parts.append(exp["description"])
            if exp.get("stack"):
                exp_parts.append(exp["stack"])
            if exp_parts:
                parts.append(" ".join(exp_parts))

        if resume.get("education"):
            parts.append(f"Education: {resume['education']}")

        return " ".join(parts)
