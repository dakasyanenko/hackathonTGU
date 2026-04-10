"""
Основной пайплайн: JobCandidateMatcher.

Оркестрирует все модули:
1. Извлечение ключевых слов (KeyBERT)
2. NER-извлечение (spaCy)
3. Улучшение текста (ключевые слова + сущности)
4. Создание эмбеддингов (sentence + символьные n-граммы)
5. Векторный поиск (FAISS с L1-расстоянием)
6. Проверка совместимости (навыки, опыт, обязательные)
7. Генерация отчёта с подсветкой для рекрутера
"""

from __future__ import annotations

import json
from typing import List

from src.compatibility_checker import CompatibilityChecker
from src.embedding_model import EmbeddingModel
from src.keyword_extractor import KeywordExtractor
from src.ner_extractor import NERExtractor
from src.models import CompatibilityResult, MatchResult
from src.vector_search import VectorSearch


class JobCandidateMatcher:
    """
    Главный оркестратор системы сопоставления вакансий и кандидатов.

    Использование:
        matcher = JobCandidateMatcher()
        matcher.build_vacancy_index(vacancies)
        results = matcher.find_best_matches(resume, top_k=5)
        print(matcher.generate_report(resume, top_k=5))
    """

    def __init__(
        self,
        sentence_model: str = "all-MiniLM-L6-v2",
        use_char_ngrams: bool = True,
        enable_keywords: bool = True,
        enable_ner: bool = True,
    ):
        print("=" * 60)
        print("Iconicompany Job-Candidate Matching System")
        print("=" * 60)

        # Модули
        self.keyword_extractor = KeywordExtractor() if enable_keywords else None
        self.ner_extractor = NERExtractor() if enable_ner else None
        self.embedding_model = EmbeddingModel(
            sentence_model_name=sentence_model,
            use_char_ngrams=use_char_ngrams,
        )
        self.vector_search = VectorSearch()
        self.compatibility_checker = CompatibilityChecker()

        # Состояние
        self.vacancies: dict[str | int, dict] = {}
        self.vacancy_enhancements: dict[str | int, dict] = {}

    # ------------------------------------------------------------------
    # Построение индекса
    # ------------------------------------------------------------------

    def build_vacancy_index(self, vacancies: list[dict]) -> None:
        """
        Строит индекс вакансий:
        1. Улучшает тексты ключевыми словами и сущностями
        2. Создаёт эмбеддинги
        3. Строит FAISS-индекс
        """
        print(f"\n📋 Индексация {len(vacancies)} вакансий...")

        # Сохраняем вакансии и строим улучшения
        for v in vacancies:
            vid = v.get("id", len(self.vacancies))
            self.vacancies[vid] = v
            self.vacancy_enhancements[vid] = self._enhance_vacancy_text(v)

        # Сначала строим словарь символьных n-грамм
        vacancy_texts = [
            self.embedding_model.prepare_vacancy_text(v) for v in vacancies
        ]
        self.embedding_model.build_char_vocab(vacancy_texts)

        # Создаём эмбеддинги с улучшенными текстами
        enhanced_texts = [
            self.vacancy_enhancements[vid]["enhanced_text"]
            for vid in [v.get("id", i) for i, v in enumerate(vacancies)]
        ]
        embeddings = self.embedding_model.encode_batch(enhanced_texts)

        # Строим поисковый индекс
        self.vector_search.build_index(vacancies, embeddings)

    def _enhance_vacancy_text(self, vacancy: dict) -> dict:
        """Улучшает текст вакансии ключевыми словами и сущностями."""
        base_text = self.embedding_model.prepare_vacancy_text(vacancy)
        enhanced = base_text
        keywords = []
        entities = []

        if self.keyword_extractor:
            keywords = self.keyword_extractor.extract_from_vacancy(vacancy)
            if keywords:
                enhanced += f" | Keywords: {', '.join(keywords)}"

        if self.ner_extractor:
            entities = self.ner_extractor.extract_entity_texts(base_text)
            if entities:
                enhanced += f" | Entities: {', '.join(entities)}"

        return {
            "base_text": base_text,
            "enhanced_text": enhanced,
            "keywords": keywords,
            "entities": entities,
        }

    # ------------------------------------------------------------------
    # Поиск и сопоставление
    # ------------------------------------------------------------------

    def find_best_matches(
        self,
        resume: dict,
        top_k: int = 5,
        detailed: bool = False,
    ) -> list[MatchResult]:
        """
        Ищет лучшие совпадения вакансий для данного резюме.

        Args:
            resume: Dict резюме.
            top_k: Количество лучших результатов.
            detailed: Включить полные детали совместимости.

        Returns:
            Список MatchResult, отсортированный по match_percentage по убыванию.
        """
        if self.vector_search.size == 0:
            raise ValueError("Индекс вакансий не построен. Вызовите build_vacancy_index() сначала.")

        # Улучшаем текст резюме
        resume_enhancement = self._enhance_resume_text(resume)

        # Создаём эмбеддинг резюме
        resume_embedding = self.embedding_model.encode(
            resume_enhancement["enhanced_text"]
        )

        # Векторный поиск
        vector_results = self.vector_search.search(resume_embedding, top_k)

        # Проверка совместимости для каждого результата
        results: list[MatchResult] = []
        for vacancy_id, l1_distance in vector_results:
            vacancy = self.vacancies.get(vacancy_id)
            if not vacancy:
                continue

            compatibility = self.compatibility_checker.check(
                vacancy,
                resume,
                vacancy_keywords=self.vacancy_enhancements[vacancy_id].get("keywords"),
                resume_keywords=resume_enhancement.get("keywords"),
                vacancy_entities=self.vacancy_enhancements[vacancy_id].get("entities"),
                resume_entities=resume_enhancement.get("entities"),
            )

            result = MatchResult(
                vacancy_id=vacancy_id,
                vacancy_position=vacancy.get("data", vacancy).get("position", "Unknown"),
                l1_distance=round(l1_distance, 4),
                match_percentage=compatibility.match_percentage,
                matched_skills=compatibility.matched_skills,
                missing_skills=compatibility.missing_skills,
                extra_candidate_skills=compatibility.extra_candidate_skills,
                experience_months=compatibility.key_skill_experience_months,
            )

            if detailed:
                result.compatibility_details = compatibility

            results.append(result)

        # Сортировка по проценту совпадения (по убыванию)
        results.sort(key=lambda r: r.match_percentage, reverse=True)
        return results

    def _enhance_resume_text(self, resume: dict) -> dict:
        """Улучшает текст резюме ключевыми словами и сущностями."""
        base_text = self.embedding_model.prepare_resume_text(resume)
        enhanced = base_text
        keywords = []
        entities = []

        if self.keyword_extractor:
            keywords = self.keyword_extractor.extract_from_resume(resume)
            if keywords:
                enhanced += f" | Keywords: {', '.join(keywords)}"

        if self.ner_extractor:
            entities = self.ner_extractor.extract_entity_texts(base_text)
            if entities:
                enhanced += f" | Entities: {', '.join(entities)}"

        return {
            "base_text": base_text,
            "enhanced_text": enhanced,
            "keywords": keywords,
            "entities": entities,
        }

    # ------------------------------------------------------------------
    # Генерация отчёта
    # ------------------------------------------------------------------

    def generate_report(self, resume: dict, top_k: int = 5) -> str:
        """
        Генерирует текстовый отчёт с подсветкой для рекрутера.

        Зелёный (✅) = совпавшие навыки
        Красный (❌) = отсутствующие навыки
        Синий (🔵) = дополнительные навыки кандидата
        """
        results = self.find_best_matches(resume, top_k, detailed=True)

        lines = [
            "=" * 70,
            "  ICONICOMPANY — ОТЧЁТ ПО ПОДБОРУ ВАКАНСИЙ ДЛЯ КАНДИДАТА",
            "=" * 70,
            "",
        ]

        for i, result in enumerate(results, 1):
            lines.extend([
                f"{'─' * 70}",
                f"  {i}. ВАКАНСИЯ: {result.vacancy_position}",
                f"     ID: {result.vacancy_id}",
                f"     L1 расстояние: {result.l1_distance}",
                f"     СОВПАДЕНИЕ: {result.match_percentage}%",
                f"",
            ])

            # Навыки с цветовой кодировкой
            if result.matched_skills:
                skills_str = ", ".join(result.matched_skills)
                lines.append(f"  ✅ Совпавшие навыки: {skills_str}")

            if result.missing_skills:
                missing_str = ", ".join(result.missing_skills)
                lines.append(f"  ❌ Отсутствующие навыки: {missing_str}")

            if result.extra_candidate_skills:
                extra_str = ", ".join(result.extra_candidate_skills)
                lines.append(f"  🔵 Дополнительные навыки кандидата: {extra_str}")

            # Опыт
            if result.experience_months:
                lines.append(f"  📅 Опыт по ключевым навыкам:")
                for skill, months in result.experience_months.items():
                    years = months // 12
                    remaining_months = months % 12
                    exp_str = f"{years} г. {remaining_months} мес." if years else f"{months} мес."
                    lines.append(f"     - {skill}: {exp_str}")

            # Детали совместимости
            if result.compatibility_details:
                comp = result.compatibility_details
                mand = comp.mandatory_requirements_met
                if not mand.met:
                    lines.append(f"  ⚠️  Обязательные требования НЕ выполнены:")
                    if mand.not_found:
                        lines.append(f"     Не найдено: {', '.join(mand.not_found)}")

                if comp.competency_match[0]:
                    lines.append(f"  💼 Компетенции: {', '.join(comp.competency_match[0])}")
                if comp.competency_match[1]:
                    lines.append(f"  ⚠️  Недостающие компетенции: {', '.join(comp.competency_match[1])}")

            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # JSON-вывод
    # ------------------------------------------------------------------

    def find_best_matches_json(
        self,
        resume: dict,
        top_k: int = 5,
        indent: int = 2,
    ) -> str:
        """Возвращает результаты сопоставления в формате JSON."""
        results = self.find_best_matches(resume, top_k, detailed=True)

        data = []
        for r in results:
            entry = {
                "vacancy_id": str(r.vacancy_id),
                "position": r.vacancy_position,
                "l1_distance": r.l1_distance,
                "match_percentage": r.match_percentage,
                "matched_skills": r.matched_skills,
                "missing_skills": r.missing_skills,
                "extra_candidate_skills": r.extra_candidate_skills,
                "experience_months": r.experience_months,
            }
            if r.compatibility_details:
                entry["mandatory_requirements"] = {
                    "met": r.compatibility_details.mandatory_requirements_met.met,
                    "compliance_rate": r.compatibility_details.mandatory_requirements_met.compliance_rate,
                    "not_found": r.compatibility_details.mandatory_requirements_met.not_found,
                }
            data.append(entry)

        return json.dumps(data, indent=indent, ensure_ascii=False)
