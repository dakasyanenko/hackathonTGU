"""
Основные типы данных для системы сопоставления вакансий и кандидатов.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ============================================================
# ТИПЫ ВАКАНСИЙ
# ============================================================

@dataclass
class VacancyData:
    """Структурированные данные вакансии."""
    position: str = ""
    industry: str = ""
    mandatory_requirements: str = ""
    project_tasks: str = ""
    experience_levels: str = ""
    description: str = ""


@dataclass
class Vacancy:
    """Полная запись вакансии."""
    id: str | int
    data: VacancyData
    skills: list[str] = field(default_factory=list)
    data_eng: str = ""
    additional_requirements: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


# ============================================================
# ТИПЫ РЕЗЮМЕ
# ============================================================

@dataclass
class Experience:
    """Одна запись опыта работы."""
    company: str = ""
    description: str = ""
    stack: str = ""
    start: str = ""   # YYYY-MM
    end: str = ""     # YYYY-MM


@dataclass
class Resume:
    """Полная запись резюме."""
    id: str | int
    skill_set: list[str] = field(default_factory=list)
    experience: list[Experience] = field(default_factory=list)
    education: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


# ============================================================
# ТИПЫ РЕЗУЛЬТАТОВ СОПОСТАВЛЕНИЯ
# ============================================================

@dataclass
class SkillsMatch:
    """Результат сравнения навыков."""
    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    total: int = 0


@dataclass
class MandatoryMatch:
    """Результат проверки обязательных требований."""
    met: bool = True
    requirements: list[str] = field(default_factory=list)
    found: list[str] = field(default_factory=list)
    not_found: list[str] = field(default_factory=list)
    compliance_rate: float = 1.0
    details: str = ""


@dataclass
class CompatibilityResult:
    """Полный результат проверки совместимости."""
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    extra_candidate_skills: list[str] = field(default_factory=list)
    match_percentage: float = 0.0
    mandatory_requirements_met: MandatoryMatch = field(default_factory=MandatoryMatch)
    key_skill_experience_months: dict[str, int] = field(default_factory=dict)
    competency_match: tuple[list[str], list[str]] = ( [], [] )
    total_vacancy_skills: int = 0


@dataclass
class MatchResult:
    """Результат сопоставления кандидата с одной вакансией."""
    vacancy_id: str | int
    vacancy_position: str
    l1_distance: float
    match_percentage: float
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    extra_candidate_skills: list[str] = field(default_factory=list)
    experience_months: dict[str, int] = field(default_factory=dict)
    compatibility_details: CompatibilityResult | None = None


@dataclass
class TextEnhancement:
    """Улучшенный текст с ключевыми словами и сущностями."""
    original_text: str
    summary: str
    keywords: list[str] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    enhanced_text: str = ""
