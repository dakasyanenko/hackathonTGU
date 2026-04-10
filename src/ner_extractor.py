"""
Модуль именованного распознавания сущностей (NER) с использованием spaCy.
Основан на исследовательской работе: улучшение текстов вакансий и резюме
с помощью сущностей (организации, продукты, технологии) повышает точность сопоставления.
"""

from __future__ import annotations

import re
from typing import Any

import spacy


# Шаблоны технологических сущностей для резервного извлечения
_TECH_PATTERNS = [
    r"\b(?:Python|Java(?:Script)?|TypeScript|C(?:\+\+|\#)?|Go|Rust|Kotlin|Swift|Scala|Ruby|PHP|Perl|Dart|R)\b",
    r"\b(?:PostgreSQL|MySQL|MongoDB|Redis|SQLite|Oracle|SQL\s+Server|MariaDB|Cassandra|DynamoDB|Couchbase)\b",
    r"\b(?:Docker|Kubernetes|K8s|Terraform|Ansible|Jenkins|GitLab|GitHub|Git|CI/CD|DevOps)\b",
    r"\b(?:React|Angular|Vue\.?js|Vue|Node\.?js|FastAPI|Flask|Django|Spring|Express|Laravel|Ruby on Rails)\b",
    r"\b(?:AWS|GCP|Azure|GCP|Cloud|S3|EC2|Lambda|CloudFront|Vercel|Netlify)\b",
    r"\b(?:Spark|Kafka|Airflow|Flink|Hadoop|Hive|Pig|Sqoop|Storm)\b",
    r"\b(?:TensorFlow|PyTorch|Keras|Scikit-?learn|Pandas|NumPy|SciPy|Matplotlib|Seaborn)\b",
    r"\b(?:REST|GraphQL|gRPC|SOAP|WebSocket|OAuth|JWT|SAML)\b",
    r"\b(?:Linux|MacOS|Windows|Unix|Ubuntu|CentOS|Debian|RedHat)\b",
    r"\b(?:Agile|Scrum|Kanban|Waterfall|TDD|BDD|DDD)\b",
]


class NERExtractor:
    """
    Извлекает именованные сущности из текста с помощью spaCy + regex-шаблонов технологий.

    Согласно статье, добавление сущностей (особенно ORG, PRODUCT)
    к суммаризованному тексту повышает точность ранжирования вакансий.
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"spaCy model '{model_name}' not found. Installing...")
            import subprocess
            subprocess.check_call(["python", "-m", "spacy", "download", model_name])
            self.nlp = spacy.load(model_name)

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def extract(self, text: str) -> list[dict[str, Any]]:
        """Извлекает именованные сущности из текста."""
        if not text or len(text.strip()) < 5:
            return []

        doc = self.nlp(text)

        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "type": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            })

        # Дополняем regex-шаблонами технологий
        tech_entities = self._extract_tech_entities(text)
        entities.extend(tech_entities)

        # Убираем дубликаты по тексту
        seen = set()
        unique_entities = []
        for ent in entities:
            if ent["text"] not in seen:
                seen.add(ent["text"])
                unique_entities.append(ent)

        return unique_entities

    def extract_entity_texts(self, text: str) -> list[str]:
        """Возвращает только тексты сущностей (удобная обёртка)."""
        return [e["text"] for e in self.extract(text)]

    def enhance_text(self, text: str) -> str:
        """
        Добавляет извлечённые сущности к исходному тексту.
        Используется для представления с ключевыми словами+сущностями из статьи.
        """
        entities = self.extract_entity_texts(text)
        if not entities:
            return text
        return f"{text} | Entities: {', '.join(entities)}"

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tech_entities(text: str) -> list[dict[str, Any]]:
        """Извлекает технологические сущности с помощью regex-шаблонов (резервный метод)."""
        entities = []
        for pattern in _TECH_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    "text": match.group(),
                    "type": "TECH",
                    "start": match.start(),
                    "end": match.end(),
                })
        return entities
