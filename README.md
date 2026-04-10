# Iconicompany ИИ-ассистент для более точного взаимодействия с заказчиками и агентствами

Система векторного поиска и проверки соответствия кандидатов вакансиям для платформы умного аутстаффинга **Iconicompany**.

## Обзор

Система решает задачу ранжирования вакансий относительно резюме кандидатов с использованием методов NLP, описанных в исследовательской работе:

> **Vanetik N., Kogan G.** Job Vacancy Ranking with Sentence Embeddings, Keywords, and Named Entities. *Information* 2023, 14(8), 468.

### Ключевые возможности

| Возможность | Описание |
|---|---|
| **Векторный поиск** | FAISS с L1 (Manhattan) расстоянием — лучший метод по результатам статьи |
| **Character n-grams** | Символьные n-граммы (1-3) + sentence embeddings — наилучшая точность |
| **KeyBERT** | Извлечение ключевых слов с MMR-фильтрацией для разнообразия |
| **spaCy NER** | Именованные сущности (ORG, PRODUCT, TECH) улучшают ранжирование |
| **Проверка навыков** | Сопоставление навыков vacancy ↔ resume с подсветкой совпадений |
| **Расчёт опыта** | Суммарный опыт (в месяцах) по каждому ключевому навыку |
| **Обязательные требования** | Проверка выполнения всех обязательных требований |
| **Компетенции** | Оценка soft skills и методологий (Agile, CI/CD, архитектура) |

## Установка

```bash
# Установка зависимостей
pip install -r requirements.txt

# Установка spaCy модели
python -m spacy download en_core_web_sm
```

### Зависимости

| Пакет | Назначение |
|---|---|
| `sentence-transformers` | Sentence embeddings (all-MiniLM-L6-v2) |
| `faiss-cpu` | Векторный поиск |
| `keybert` | Извлечение ключевых слов |
| `spacy` | Named Entity Recognition |
| `scikit-learn` | n-граммы TF-IDF векторизатор |
| `numpy` | Математические операции |

## Быстрый старт

```python
from src.matcher import JobCandidateMatcher

# Инициализация
matcher = JobCandidateMatcher(
    sentence_model="all-MiniLM-L6-v2",
    use_char_ngrams=True,
    enable_keywords=True,
    enable_ner=True,
)

# Индексация вакансий
matcher.build_vacancy_index(vacancies)

# Поиск для кандидата
results = matcher.find_best_matches(resume, top_k=5, detailed=True)

# Текстовый отчёт с подсветкой
print(matcher.generate_report(resume, top_k=5))

# JSON вывод
print(matcher.find_best_matches_json(resume, top_k=5))
```

## Запуск демо

```bash
python -m src.index
# или
python src/index.py
```

## Структура проекта

```
JobMatcher/
├── requirements.txt            
├── main.py                     
├── README.md                  
└── src/
    ├── __init__.py
    ├── types.py                # Типы данных (dataclass)
    ├── keyword_extractor.py    # KeyBERT извлечение ключевых слов
    ├── ner_extractor.py        # spaCy NER + regex
    ├── embedding_model.py      # эмбеддинги
    ├── vector_search.py        # FAISS с L1 distance
    ├── compatibility_checker.py # Проверка соответствия
    ├── matcher.py              # Основной пайплайн
    └── index.py                # Входной файл + демо
```

## Методология

### 1. Подготовка текста

**Вакансии:** суммаризация — оставляем только существенные поля (position, requirements, tasks, description). Удаляем boilerplate.

**Резюме:** используем полный текст — они уже достаточно краткие.

### 2. Улучшение текста

- **KeyBERT keywords** — извлекаем уни/би/три-граммы ключевые слова с MMR
- **spaCy NER** — добавляем именованные сущности (ORG, PRODUCT, TECH)

### 3. Векторное представление

- **Sentence embeddings** — all-MiniLM-L6-v2 (fallback) или nomic-embed-text
- **Character n-grams (1-3)** — TF-IDF, лучшая точность по статье
- **Конкатенация** обоих представлений

### 4. Поиск

- **L1 (Manhattan) расстояние**: Σ|x_resume[i] - x_vacancy[i]|
- Ранжирование по возрастанию (меньше = лучше)

### 5. Проверка соответствия

| Компонент | Вес | Описание |
|---|---|---|
| Навыки | 60% | matched / total vacancy skills |
| Обязательные требования | 25% | compliance rate |
| Опыт | 15% | normalized months / 60 |

## Подсветка для рекрутера

| Символ | Значение |
|---|---|
| ✅ | Совпавшие навыки |
| ❌ | Отсутствующие навыки |
| 🔵 | Дополнительные навыки кандидата |
| 📅 | Опыт по ключевым навыкам |
| ⚠️ | Невыполненные обязательные требования |
| 💼 | Совпавшие компетенции |

## API

### JobCandidateMatcher

| Метод | Описание |
|---|---|
| `build_vacancy_index(vacancies)` | Построить индекс вакансий |
| `find_best_matches(resume, top_k, detailed)` | Поиск лучших вакансий |
| `generate_report(resume, top_k)` | Текстовый отчёт с подсветкой |
| `find_best_matches_json(resume, top_k)` | JSON вывод |

### CompatibilityChecker

| Метод | Описание |
|---|---|
| `check(vacancy, resume, ...)` | Полная проверка соответствия |
| `_check_skills(vacancy, resume)` | Сравнение навыков |
| `_check_mandatory(vacancy, resume)` | Проверка обязательных требований |
| `_calculate_experience(resume, skills)` | Расчёт опыта по навыкам |
| `_check_competencies(vacancy, resume)` | Оценка компетенций |

## Конфигурация

```python
matcher = JobCandidateMatcher(
    sentence_model="all-MiniLM-L6-v2",  # или другая модель
    use_char_ngrams=True,               # character n-grams (лучшая точность)
    enable_keywords=True,               # KeyBERT
    enable_ner=True,                    # spaCy NER
)
```

## Команда

Д.А. Касьяненко

## Результат запуска с тестовыми данными из демо


======================================================================
  ICONICOMPANY — ОТЧЁТ ПО ПОДБОРУ ВАКАНСИЙ ДЛЯ КАНДИДАТА
======================================================================
Загрузка sentence-трансформера: all-MiniLM-L6-v2...
Размерность sentence-эмбеддинга: 384

📋 Индексация 5 вакансий...
Размер словаря символьных n-грамм: 1240
Построение индекса для 5 вакансий...
Индекс построен. Форма: (5, 1624)

======================================================================
  🔍 Кандидат res_1 (Python, PostgreSQL, Docker...)
======================================================================

  1. Python Developer
     Совпадение: 54.7%
     L1 расстояние: 15.7187
     ✅ Навыки: docker, fastapi, postgresql, python, redis
     ❌ Не хватает: 3+ years experience, agile, cd., ci, experience, years, интеграция с внешними сервисами, оптимизация api
     🔵 Дополнительно: backend, git, linux, микросервисы, разработка backend для платёжной системы, создание api для мобильного приложения
     📅 Опыт: postgresql: 29мес., docker: 29мес., python: 47мес., fastapi: 47мес., redis: 47мес.

  2. Full-Stack Developer
     Совпадение: 43.5%
     L1 расстояние: 20.4301
     ✅ Навыки: backend, fastapi, postgresql, python
     ❌ Не хватает: 2+ years, frontend, frontend на react., full, healthtech, react, stack, typescript, years
     🔵 Дополнительно: docker, git, linux, redis, микросервисы, разработка backend для платёжной системы, создание api для мобильного приложения
     📅 Опыт: postgresql: 29мес., python: 47мес., fastapi: 47мес.

  3. Frontend Developer
     Совпадение: 0.0%
     L1 расстояние: 26.3303
     ❌ Не хватает: 2+ years, css, javascript, react, redux, typescript, years, интеграция с rest api, оптимизация производительности, разработка ui для маркетплейса
     🔵 Дополнительно: backend, docker, fastapi, git, linux, postgresql, python, redis, микросервисы, разработка backend для платёжной системы, создание api для мобильного приложения

──────────────────────────────────────────────────────────────────────
  1. ВАКАНСИЯ: Python Developer
     ID: vac_1
     L1 расстояние: 15.7187
     СОВПАДЕНИЕ: 54.7%

  ✅ Совпавшие навыки: docker, fastapi, postgresql, python, redis
  ❌ Отсутствующие навыки: 3+ years experience, agile, cd., ci, experience, years, интеграция с внешними сервисами, оптимизация api
  🔵 Дополнительные навыки кандидата: backend, git, linux, микросервисы, разработка backend для платёжной системы, создание api для мобильного приложения
  📅 Опыт по ключевым навыкам:
     - postgresql: 2 г. 5 мес.
     - docker: 2 г. 5 мес.
     - python: 3 г. 11 мес.
     - fastapi: 3 г. 11 мес.
     - redis: 3 г. 11 мес.
  ⚠️  Обязательные требования НЕ выполнены:
     Не найдено: 3+ years experience, years
  ⚠️  Недостающие компетенции: microservice, agile, ci/cd

──────────────────────────────────────────────────────────────────────
  2. ВАКАНСИЯ: Full-Stack Developer
     ID: vac_5
     L1 расстояние: 20.4301
     СОВПАДЕНИЕ: 43.5%

  ✅ Совпавшие навыки: backend, fastapi, postgresql, python
  ❌ Отсутствующие навыки: 2+ years, frontend, frontend на react., full, healthtech, react, stack, typescript, years
  🔵 Дополнительные навыки кандидата: docker, git, linux, redis, микросервисы, разработка backend для платёжной системы, создание api для мобильного приложения
  📅 Опыт по ключевым навыкам:
     - postgresql: 2 г. 5 мес.
     - python: 3 г. 11 мес.
     - fastapi: 3 г. 11 мес.
  ⚠️  Обязательные требования НЕ выполнены:
     Не найдено: 2+ years, react, years

──────────────────────────────────────────────────────────────────────
  3. ВАКАНСИЯ: Data Engineer
     ID: vac_3
     L1 расстояние: 26.4787
     СОВПАДЕНИЕ: 27.1%

  ✅ Совпавшие навыки: python
  ❌ Отсутствующие навыки: airflow, apache, apache spark, data, etl pipelines, kafka, pipeline, pipelines, spark, sql, построение data pipeline для аналитики
  🔵 Дополнительные навыки кандидата: backend, docker, fastapi, git, linux, postgresql, redis, микросервисы, разработка backend для платёжной системы, создание api для мобильного приложения
  📅 Опыт по ключевым навыкам:
     - sql: 2 г. 5 мес.
     - python: 3 г. 11 мес.
  ⚠️  Обязательные требования НЕ выполнены:
     Не найдено: apache, apache spark, etl pipelines, pipelines, spark

──────────────────────────────────────────────────────────────────────
  4. ВАКАНСИЯ: DevOps Engineer
     ID: vac_4
     L1 расстояние: 27.6165
     СОВПАДЕНИЕ: 18.7%

  ✅ Совпавшие навыки: docker
  ❌ Отсутствующие навыки: 3+ years, aws, cd, cd пайплайнов., ci, jenkins, kubernetes, terraform, years, автоматизация ci, автоматизация деплоя, мониторинг инфраструктуры, оптимизация облачных ресурсов
  🔵 Дополнительные навыки кандидата: backend, fastapi, git, linux, postgresql, python, redis, микросервисы, разработка backend для платёжной системы, создание api для мобильного приложения
  📅 Опыт по ключевым навыкам:
     - docker: 2 г. 5 мес.
  ⚠️  Обязательные требования НЕ выполнены:
     Не найдено: 3+ years, cd, kubernetes, terraform, years
  ⚠️  Недостающие компетенции: ci/cd, devops

──────────────────────────────────────────────────────────────────────
  5. ВАКАНСИЯ: Frontend Developer
     ID: vac_2
     L1 расстояние: 26.3303
     СОВПАДЕНИЕ: 0.0%

  ❌ Отсутствующие навыки: 2+ years, css, javascript, react, redux, typescript, years, интеграция с rest api, оптимизация производительности, разработка ui для маркетплейса
  🔵 Дополнительные навыки кандидата: backend, docker, fastapi, git, linux, postgresql, python, redis, микросервисы, разработка backend для платёжной системы, создание api для мобильного приложения
  ⚠️  Обязательные требования НЕ выполнены:
     Не найдено: 2+ years, react, typescript, years
  ⚠️  Недостающие компетенции: rest

======================================================================

======================================================================
  🔍 Кандидат res_2 (React, TypeScript, JavaScript...)
======================================================================

  1. Frontend Developer
     Совпадение: 57.5%
     L1 расстояние: 17.53
     ✅ Навыки: css, javascript, react, redux, typescript
     ❌ Не хватает: 2+ years, years, интеграция с rest api, оптимизация производительности, разработка ui для маркетплейса
     🔵 Дополнительно: commerce, graphql, html, разработка spa для e-commerce клиентов
     📅 Опыт: typescript: 17мес., css: 35мес., react: 35мес., javascript: 18мес., redux: 17мес.

  2. Full-Stack Developer
     Совпадение: 27.2%
     L1 расстояние: 22.018
     ✅ Навыки: react, typescript
     ❌ Не хватает: 2+ years, backend, fastapi, frontend, frontend на react., full, healthtech, postgresql, python, stack, years
     🔵 Дополнительно: commerce, css, graphql, html, javascript, redux, разработка spa для e-commerce клиентов
     📅 Опыт: typescript: 17мес., react: 35мес.

  3. Python Developer
     Совпадение: 4.2%
     L1 расстояние: 27.2213
     ❌ Не хватает: 3+ years experience, agile, cd., ci, docker, experience, fastapi, postgresql, python, redis, years, интеграция с внешними сервисами, оптимизация api
     🔵 Дополнительно: commerce, css, graphql, html, javascript, react, redux, typescript, разработка spa для e-commerce клиентов

======================================================================
  🔍 Кандидат res_3 (Python, Apache Spark, SQL...)
======================================================================

  1. Data Engineer
     Совпадение: 67.9%
     L1 расстояние: 15.1751
     ✅ Навыки: airflow, apache, apache spark, kafka, python, spark, sql
     ❌ Не хватает: data, etl pipelines, pipeline, pipelines, построение data pipeline для аналитики
     🔵 Дополнительно: numpy, pandas, анализ данных и создание отчётов
     📅 Опыт: spark: 39мес., apache spark: 39мес., airflow: 39мес., sql: 56мес., python: 56мес., apache: 39мес., kafka: 39мес.

  2. Python Developer
     Совпадение: 26.9%
     L1 расстояние: 26.0836
     ✅ Навыки: python
     ❌ Не хватает: 3+ years experience, agile, cd., ci, docker, experience, fastapi, postgresql, redis, years, интеграция с внешними сервисами, оптимизация api
     🔵 Дополнительно: airflow, apache, apache spark, kafka, numpy, pandas, spark, sql, анализ данных и создание отчётов
     📅 Опыт: python: 56мес.

  3. Full-Stack Developer
     Совпадение: 23.6%
     L1 расстояние: 26.0095
     ✅ Навыки: python
     ❌ Не хватает: 2+ years, backend, fastapi, frontend, frontend на react., full, healthtech, postgresql, react, stack, typescript, years
     🔵 Дополнительно: airflow, apache, apache spark, kafka, numpy, pandas, spark, sql, анализ данных и создание отчётов
     📅 Опыт: python: 56мес.


📄 JSON OUTPUT:
[
  {
    "vacancy_id": "vac_1",
    "position": "Python Developer",
    "l1_distance": 15.7187,
    "match_percentage": 54.7,
    "matched_skills": [
      "docker",
      "fastapi",
      "postgresql",
      "python",
      "redis"
    ],
    "missing_skills": [
      "3+ years experience",
      "agile",
      "cd.",
      "ci",
      "experience",
      "years",
      "интеграция с внешними сервисами",
      "оптимизация api"
    ],
    "extra_candidate_skills": [
      "backend",
      "git",
      "linux",
      "микросервисы",
      "разработка backend для платёжной системы",
      "создание api для мобильного приложения"
    ],
    "experience_months": {
      "postgresql": 29,
      "docker": 29,
      "python": 47,
      "fastapi": 47,
      "redis": 47
    },
    "mandatory_requirements": {
      "met": false,
      "compliance_rate": 0.6666666666666666,
      "not_found": [
        "3+ years experience",
        "years"
      ]
    }
  },
  {
    "vacancy_id": "vac_5",
    "position": "Full-Stack Developer",
    "l1_distance": 20.4301,
    "match_percentage": 43.5,
    "matched_skills": [
      "backend",
      "fastapi",
      "postgresql",
      "python"
    ],
    "missing_skills": [
      "2+ years",
      "frontend",
      "frontend на react.",
      "full",
      "healthtech",
      "react",
      "stack",
      "typescript",
      "years"
    ],
    "extra_candidate_skills": [
      "docker",
      "git",
      "linux",
      "redis",
      "микросервисы",
      "разработка backend для платёжной системы",
      "создание api для мобильного приложения"
    ],
    "experience_months": {
      "postgresql": 29,
      "python": 47,
      "fastapi": 47
    },
    "mandatory_requirements": {
      "met": false,
      "compliance_rate": 0.4,
      "not_found": [
        "2+ years",
        "react",
        "years"
      ]
    }
  },
  {
    "vacancy_id": "vac_2",
    "position": "Frontend Developer",
    "l1_distance": 26.3303,
    "match_percentage": 0.0,
    "matched_skills": [],
    "missing_skills": [
      "2+ years",
      "css",
      "javascript",
      "react",
      "redux",
      "typescript",
      "years",
      "интеграция с rest api",
      "оптимизация производительности",
      "разработка ui для маркетплейса"
    ],
    "extra_candidate_skills": [
      "backend",
      "docker",
      "fastapi",
      "git",
      "linux",
      "postgresql",
      "python",
      "redis",
      "микросервисы",
      "разработка backend для платёжной системы",
      "создание api для мобильного приложения"
    ],
    "experience_months": {},
    "mandatory_requirements": {
      "met": false,
      "compliance_rate": 0.0,
      "not_found": [
        "2+ years",
        "react",
        "typescript",
        "years"
      ]
    }
  }
]
(base) kasyanenko@192 Хакатон % 
