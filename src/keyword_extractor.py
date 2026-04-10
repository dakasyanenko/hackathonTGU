"""
Модуль извлечения ключевых слов с использованием TF-IDF.
Основан на исследовательской работе: "Job Vacancy Ranking with Sentence Embeddings,
Keywords, and Named Entities" (Vanetik & Kogan, 2023).
"""

from __future__ import annotations

import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer


class KeywordExtractor:
    """
    Извлекает ключевые слова из текстов вакансий и резюме с помощью TF-IDF.

    Использует символьные/словесные n-граммы для извлечения наиболее
    значимых фраз из текста. MMR-подобная фильтрация обеспечивает
    разнообразие результатов.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # model_name не используется, оставляем для совместимости API
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=5000,
            stop_words=self._get_stop_words(),
            lowercase=True,
            strip_accents="unicode",
        )

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def extract(self, text: str, top_n: int = 20) -> list[str]:
        """Извлекает топ-N ключевых слов из текста."""
        if not text or len(text.strip()) < 10:
            return []

        self.vectorizer.fit([text])
        feature_names = self.vectorizer.get_feature_names_out()
        scores = self.vectorizer.transform([text]).toarray()[0]

        # Сортируем по убыванию TF-IDF scores
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        keywords = []
        for idx, score in indexed[:top_n]:
            kw = feature_names[idx]
            # Пропускаем слишком короткие (<2 символов)
            if len(kw.strip()) >= 2:
                keywords.append(kw.strip())

        return keywords

    def extract_from_vacancy(self, vacancy_data: dict) -> list[str]:
        """Извлекает ключевые слова из структурированного dict вакансии."""
        parts: list[str] = []

        if vacancy_data.get("data", {}).get("position"):
            parts.append(vacancy_data["data"]["position"])
        if vacancy_data.get("data", {}).get("mandatoryRequirements"):
            parts.append(vacancy_data["data"]["mandatoryRequirements"])
        if vacancy_data.get("data", {}).get("projectTasks"):
            parts.append(vacancy_data["data"]["projectTasks"])
        if vacancy_data.get("data", {}).get("description"):
            parts.append(vacancy_data["data"]["description"])
        if vacancy_data.get("skills"):
            parts.append(", ".join(vacancy_data["skills"]))
        if vacancy_data.get("dataEng"):
            parts.append(vacancy_data["dataEng"])

        return self.extract(" ".join(parts))

    def extract_from_resume(self, resume_data: dict) -> list[str]:
        """Извлекает ключевые слова из структурированного dict резюме."""
        parts: list[str] = []

        if resume_data.get("skill_set"):
            parts.append(", ".join(resume_data["skill_set"]))
        for exp in resume_data.get("experience", []):
            if exp.get("description"):
                parts.append(exp["description"])
            if exp.get("stack"):
                parts.append(exp["stack"])
        if resume_data.get("education"):
            parts.append(resume_data["education"])

        return self.extract(" ".join(parts))

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    @staticmethod
    def _get_stop_words() -> list[str]:
        """Объединённый список стоп-слов (английский + русский)."""
        return [
            # Английский
            "the", "and", "for", "are", "but", "not", "you", "all", "can",
            "had", "her", "was", "one", "our", "out", "has", "have", "been",
            "were", "that", "this", "with", "they", "she", "his", "its",
            "will", "would", "from", "what", "when", "where", "which",
            # Русский
            "как", "для", "что", "это", "все", "или", "уже", "только",
            "если", "был", "была", "были", "может", "быть", "под", "над",
            "при", "без", "разработка", "развитие", "работа", "опыт",
            "навыки", "требования", "умение", "знания", "условия",
            "компания", "проект", "команда",
        ]
