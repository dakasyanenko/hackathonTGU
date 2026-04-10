"""
Модуль проверки совместимости.

Проверяет соответствие кандидата вакансии по нескольким параметрам:
1. Совпадение навыков (совпавшие, отсутствующие, дополнительные)
2. Проверка обязательных требований
3. Расчёт длительности опыта по каждому навыку (в месяцах)
4. Оценка компетенций
5. Расчёт общего процента совпадения
"""

from __future__ import annotations

import json
import re
from datetime import datetime

from src.models import (
    CompatibilityResult,
    MandatoryMatch,
    SkillsMatch,
)


class CompatibilityChecker:
    """
    Проверяет, насколько хорошо резюме соответствует вакансии по нескольким параметрам.

    Формирует детальный отчёт совместимости с:
    - Совпавшими/отсутствующими/дополнительными навыками
    - Выполнением обязательных требований
    - Опытом по каждому навыку (месяцы)
    - Совпадением компетенций
    - Взвешенным общим процентом совпадения
    """

    # Веса для расчёта процента совпадения
    WEIGHT_SKILLS = 0.60
    WEIGHT_MANDATORY = 0.25
    WEIGHT_EXPERIENCE = 0.15

    # Ожидаемый максимум опыта для нормализации (5 лет)
    MAX_EXPECTED_MONTHS = 60

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def check(
        self,
        vacancy: dict,
        resume: dict,
        vacancy_keywords: list[str] | None = None,
        resume_keywords: list[str] | None = None,
        vacancy_entities: list[str] | None = None,
        resume_entities: list[str] | None = None,
    ) -> CompatibilityResult:
        """
        Полная проверка совместимости между вакансией и резюме.

        Args:
            vacancy: Dict вакансии.
            resume: Dict резюме.
            vacancy_keywords: Опционально — предизвлечённые ключевые слова вакансии.
            resume_keywords: Опционально — предизвлечённые ключевые слова резюме.
            vacancy_entities: Опционально — предизвлечённые сущности вакансии.
            resume_entities: Опционально — предизвлечённые сущности резюме.

        Returns:
            CompatibilityResult с детальной информацией о совпадении.
        """
        # 1. Совпадение навыков
        skills_match = self._check_skills(vacancy, resume)

        # 2. Обязательные требования
        mandatory_match = self._check_mandatory(vacancy, resume)

        # 3. Опыт по каждому ключевому навыку
        target_skills = vacancy.get("skills", [])
        if vacancy.get("data", {}).get("mandatoryRequirements"):
            req_skills = self._extract_skills_from_text(
                vacancy["data"]["mandatoryRequirements"]
            )
            target_skills = list(set(target_skills) | req_skills)

        experience_months = self._calculate_experience(resume, target_skills)

        # 4. Оценка компетенций
        competency_match = self._check_competencies(vacancy, resume)

        # 5. Общий процент совпадения
        match_percentage = self._calculate_match_percentage(
            skills_match, mandatory_match, experience_months
        )

        return CompatibilityResult(
            matched_skills=skills_match.matched,
            missing_skills=skills_match.missing,
            extra_candidate_skills=skills_match.extra,
            match_percentage=match_percentage,
            mandatory_requirements_met=mandatory_match,
            key_skill_experience_months=experience_months,
            competency_match=competency_match,
            total_vacancy_skills=skills_match.total,
        )

    # ------------------------------------------------------------------
    # Совпадение навыков
    # ------------------------------------------------------------------

    def _check_skills(self, vacancy: dict, resume: dict) -> SkillsMatch:
        """Сравнивает навыки вакансии и резюме."""
        vacancy_skills = self._extract_all_skills(vacancy)
        resume_skills = self._extract_all_skills(resume)

        matched = vacancy_skills.intersection(resume_skills)
        missing = vacancy_skills - resume_skills
        extra = resume_skills - vacancy_skills

        return SkillsMatch(
            matched=sorted(matched),
            missing=sorted(missing),
            extra=sorted(extra),
            total=len(vacancy_skills),
        )

    def _extract_all_skills(self, data: dict) -> set[str]:
        """Извлекает все навыки из dict вакансии или резюме."""
        skills: set[str] = set()

        # Явное поле skill_set / skills
        if data.get("skill_set"):
            skills.update(s.lower() for s in data["skill_set"])
        if data.get("skills"):
            skills.update(s.lower() for s in data["skills"])

        # Извлекаем из текстовых полей
        for key in ("description", "stack", "mandatoryRequirements",
                     "projectTasks", "additionalRequirements"):
            text = data.get(key, "")
            if not text:
                # Проверяем вложенный 'data'
                text = data.get("data", {}).get(key, "")
            if text:
                skills.update(self._extract_skills_from_text(text))

        # Извлекаем из записей опыта
        for exp in data.get("experience", []):
            if exp.get("stack"):
                skills.update(self._extract_skills_from_text(exp["stack"]))
            if exp.get("description"):
                skills.update(self._extract_skills_from_text(exp["description"]))

        return skills

    @staticmethod
    def _extract_skills_from_text(text: str) -> set[str]:
        """
        Извлекает токены, похожие на навыки, из произвольного текста.
        """
        skills: set[str] = set()

        # Разделяем по распространённым разделителям
        parts = re.split(r"[,;/|]+", text)
        for part in parts:
            part = part.strip()
            if 2 <= len(part) <= 40:
                skills.add(part.lower())

        # Также извлекаем капитализированные / camelCase токены
        tokens = re.findall(r'\b(?:[A-Z][a-z]+[A-Z]?[a-z]*|[a-z]{3,})\b', text)
        for token in tokens:
            if len(token) >= 3:
                skills.add(token.lower())

        return skills

    # ------------------------------------------------------------------
    # Обязательные требования
    # ------------------------------------------------------------------

    def _check_mandatory(self, vacancy: dict, resume: dict) -> MandatoryMatch:
        """Проверяет выполнение обязательных требований."""
        req_text = vacancy.get("data", {}).get("mandatoryRequirements", "")
        if not req_text:
            return MandatoryMatch(
                met=True, details="Обязательные требования не указаны"
            )

        resume_text = json.dumps(resume, ensure_ascii=False).lower()
        req_tokens = self._extract_skills_from_text(req_text)

        found = []
        not_found = []

        for req in req_tokens:
            if req in resume_text:
                found.append(req)
            else:
                not_found.append(req)

        compliance_rate = len(found) / len(req_tokens) if req_tokens else 1.0

        return MandatoryMatch(
            met=len(not_found) == 0,
            requirements=sorted(req_tokens),
            found=sorted(found),
            not_found=sorted(not_found),
            compliance_rate=compliance_rate,
        )

    # ------------------------------------------------------------------
    # Расчёт опыта
    # ------------------------------------------------------------------

    def _calculate_experience(
        self, resume: dict, target_skills: list[str]
    ) -> dict[str, int]:
        """
        Вычисляет суммарный опыт (в месяцах) по каждому целевому навыку.

        Суммирует опыт по ВСЕМ проектам, где упоминается навык.
        """
        experience: dict[str, int] = {}

        for skill in target_skills:
            if not skill:
                continue
            skill_lower = skill.lower()
            total_months = 0

            for exp in resume.get("experience", []):
                exp_text = json.dumps(exp, ensure_ascii=False).lower()

                if skill_lower in exp_text:
                    months = self._calculate_months_between(
                        exp.get("start", ""), exp.get("end", "")
                    )
                    total_months += max(0, months)

            if total_months > 0:
                experience[skill_lower] = total_months

        return experience

    @staticmethod
    def _calculate_months_between(start_str: str, end_str: str) -> int:
        """Вычисляет количество месяцев между двумя датами в формате YYYY-MM."""
        try:
            start = datetime.strptime(start_str, "%Y-%m")
            if not end_str or end_str.lower() in ("present", "now", "н.в.", "по н.в."):
                end = datetime.now()
            else:
                end = datetime.strptime(end_str, "%Y-%m")
            return max(0, (end.year - start.year) * 12 + (end.month - start.month))
        except (ValueError, TypeError):
            return 0

    # ------------------------------------------------------------------
    # Оценка компетенций
    # ------------------------------------------------------------------

    def _check_competencies(
        self, vacancy: dict, resume: dict
    ) -> tuple[list[str], list[str]]:
        """
        Оценивает компетенции:
        - Архитектурные паттерны (микросервисы, распределённые системы)
        - Методологии (Agile, Scrum, TDD)
        - Soft skills (лидерство, менторство)
        """
        competencies = [
            "microservice", "micro-service", "distributed", "architecture",
            "agile", "scrum", "kanban", "tdd", "bdd", "ci/cd", "devops",
            "testing", "code review", "mentoring", "leadership",
            "api design", "rest", "graphql",
            "database design", "optimization", "performance",
        ]

        vacancy_text = json.dumps(vacancy, ensure_ascii=False).lower()
        resume_text = json.dumps(resume, ensure_ascii=False).lower()

        matched = []
        missing = []

        for comp in competencies:
            if comp in vacancy_text:
                if comp in resume_text:
                    matched.append(comp)
                else:
                    missing.append(comp)

        return matched, missing

    # ------------------------------------------------------------------
    # Процент совпадения
    # ------------------------------------------------------------------

    def _calculate_match_percentage(
        self,
        skills_match: SkillsMatch,
        mandatory_match: MandatoryMatch,
        experience_months: dict[str, int],
    ) -> float:
        """
        Вычисляет взвешенный общий процент совпадения.

        Веса:
        - Навыки: 60%
        - Обязательные требования: 25%
        - Опыт: 15%
        """
        if skills_match.total == 0:
            return 0.0

        # Оценка навыков
        skills_score = len(skills_match.matched) / skills_match.total

        # Оценка обязательных требований
        mandatory_score = mandatory_match.compliance_rate

        # Оценка опыта (нормализация до MAX_EXPECTED_MONTHS)
        total_months = sum(experience_months.values())
        experience_score = min(total_months / self.MAX_EXPECTED_MONTHS, 1.0)

        # Взвешенная сумма
        percentage = (
            skills_score * self.WEIGHT_SKILLS
            + mandatory_score * self.WEIGHT_MANDATORY
            + experience_score * self.WEIGHT_EXPERIENCE
        ) * 100

        return round(percentage, 1)
