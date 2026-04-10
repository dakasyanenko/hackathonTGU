"""
Модуль суммаризации текста вакансий.

Extractive summarization: выбираем предложения с наибольшим
количеством технических терминов.

По статье Vanetik & Kogan (2023): суммаризация вакансий значительно
улучшает качество ранжирования.
"""

from __future__ import annotations

import re

# HR/legal boilerplate — предложения, которые нужно пропускать
_BOILERPLATE = [
    "equal opportunity", "affirmative action", "protected veteran",
    "disability accommodation", "background check", "drug screening",
    "copyright", "all rights reserved", "cybercoders",
    "looking forward to receiving", "not a fit for this position",
    "click the link at the bottom", "national general holdings",
    "national general insurance", "with us you can be extraordinary",
    "the 4es", "come join our team", "paid training",
    "wellness programs", "employee discount program",
    "on-site healthcare", "on-site fitness center", "subsidized parking",
    "company paid holidays", "generous time-off",
    "medical, dental, vision", "life and short",
    "401k w/ company match", "career advancement and development",
    "equal employment opportunity", "all qualified applicants",
    "consideration for employment", "protected characteristic",
    "veteran status", "national origin", "genetic information",
    "sexual orientation", "gender identity",
    "sponsor individuals for work visas",
    "work authorization", "federal background investigation",
    "security clearance", "u.s. citizens only",
    "accommodation in completing", "ngic main office",
    "benefits package", "health insurance", "dental insurance",
    "vision insurance", "disability insurance",
]


def summarize_vacancy(text: str, max_sentences: int = 6) -> str:
    """
    Extractive summarization: выбираем предложения с наибольшим
    количеством технических терминов и наименьшим boilerplate.

    Args:
        text: Полный текст вакансии.
        max_sentences: Максимум предложений в суммаризации.

    Returns:
        Суммаризованный текст.
    """
    if not text or len(text) < 100:
        return text

    sentences = re.split(r'(?<=[.!?])\s+', text)

    tech_keywords = {
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby',
        'go', 'rust', 'kotlin', 'swift', 'scala', 'php', 'perl', 'sql',
        'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'mssql',
        'react', 'angular', 'vue', 'node', 'jquery', 'bootstrap',
        'docker', 'kubernetes', 'aws', 'azure', 'linux', 'windows',
        'git', 'jenkins', 'rest', 'graphql', 'api', 'microservice',
        'spring', 'hibernate', 'django', 'flask', 'fastapi',
        'lamp', 'drupal', 'elasticsearch', 'kafka', 'rabbitmq',
        'terraform', 'ansible', 'ci/cd', 'agile', 'scrum',
        'backend', 'frontend', 'full-stack', 'developer', 'engineer',
        'framework', 'library', 'database', 'server', 'cloud',
        'testing', 'debugging', 'architecture', 'design pattern',
        'oop', 'mvc', 'restful', 'json', 'xml',
        'years', 'experience', 'required', 'minimum', 'skills',
        'responsibilities', 'develop', 'code', 'implement',
        'maintain', 'build', 'design', 'work with',
    }

    scored = []
    for sent in sentences:
        sent_lower = sent.lower().strip()

        # Пропускаем boilerplate
        if any(bp in sent_lower for bp in _BOILERPLATE):
            continue

        if len(sent_lower) < 15 or len(sent_lower) > 400:
            continue

        # Считаем tech слова
        words = set(re.findall(r'\b\w+\b', sent_lower))
        tech_count = len(words & tech_keywords)

        score = tech_count * 2.0

        # Штраф за HR
        if any(w in sent_lower for w in ['benefits', 'vacation', 'insurance',
                                          '401k', 'diversity', 'inclusion']):
            score -= 5

        scored.append((score, sent))

    scored.sort(key=lambda x: x[0], reverse=True)
    best = [s for _, s in scored[:max_sentences]]

    # Восстанавливаем порядок
    result = [s for s in sentences if s in best]
    return ' '.join(result)
