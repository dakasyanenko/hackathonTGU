"""
Модуль векторного поиска на базе FAISS с L1 (Манхэттенским) расстоянием.

Исследовательская работа показала, что L1-расстояние превосходит косинусное
подобие для сопоставления вакансий и резюме, поскольку позволяет
взвешивать отдельные признаки.

FAISS не имеет встроенного L1-индекса, поэтому мы храним эмбеддинги
и вычисляем L1-расстояния вручную при поиске.
"""

from __future__ import annotations

import numpy as np


class VectorSearch:
    """
    Векторный поиск на базе FAISS с вычислением L1-расстояния.

    Поддерживает построение индекса эмбеддингов вакансий и поиск
    наиболее похожих вакансий по эмбеддингу резюме.
    """

    def __init__(self):
        self.vacancy_ids: list[str | int] = []
        self.vacancy_embeddings: np.ndarray | None = None
        self._built = False

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def build_index(
        self,
        vacancies: list[dict],
        embeddings: np.ndarray,
    ) -> None:
        """
        Строит поисковый индекс из предвычисленных эмбеддингов вакансий.

        Args:
            vacancies: Список dict вакансий (должно быть поле 'id').
            embeddings: Массив предвычисленных эмбеддингов формы (n_vacancies, dim).
        """
        print(f"Построение индекса для {len(vacancies)} вакансий...")

        self.vacancy_ids = [v.get("id", i) for i, v in enumerate(vacancies)]
        self.vacancy_embeddings = embeddings.astype(np.float32)
        self._built = True

        print(f"Индекс построен. Форма: {embeddings.shape}")

    def search(
        self,
        resume_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[str | int, float]]:
        """
        Ищет наиболее похожие вакансии с помощью L1-расстояния.

        L1-расстояние = Σ|x_resume[i] - x_vacancy[i]|

        Args:
            resume_embedding: Вектор эмбеддинга резюме (1D массив).
            top_k: Количество лучших результатов.

        Returns:
            Список кортежей (vacancy_id, l1_distance), отсортированных по возрастанию L1.
        """
        if not self._built or self.vacancy_embeddings is None:
            raise ValueError("Индекс не построен. Вызовите build_index() сначала.")

        # Приводим к 1D
        if resume_embedding.ndim == 2:
            resume_embedding = resume_embedding[0]

        # Вычисляем L1-расстояния: Σ|запрос - каждая_вакансия|
        diffs = np.abs(self.vacancy_embeddings - resume_embedding)
        l1_distances = np.sum(diffs, axis=1)

        # Сортируем по возрастанию L1 (меньше = более похожа)
        sorted_indices = np.argsort(l1_distances)[:top_k]

        results: list[tuple[str | int, float]] = []
        for idx in sorted_indices:
            if idx < len(self.vacancy_ids):
                results.append((self.vacancy_ids[idx], float(l1_distances[idx])))

        return results

    def search_batch(
        self,
        resume_embeddings: dict[str | int, np.ndarray],
        top_k: int = 5,
    ) -> dict[str | int, list[tuple[str | int, float]]]:
        """
        Пакетный поиск для нескольких резюме.

        Args:
            resume_embeddings: Dict, отображающий resume_id -> embedding.
            top_k: Количество лучших результатов на одно резюме.

        Returns:
            Dict, отображающий resume_id -> список (vacancy_id, l1_distance).
        """
        results = {}
        for resume_id, embedding in resume_embeddings.items():
            results[resume_id] = self.search(embedding, top_k)
        return results

    @property
    def size(self) -> int:
        """Количество проиндексированных вакансий."""
        return len(self.vacancy_ids) if self._built else 0
