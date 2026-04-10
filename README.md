# Якомпания — ИИ-ассистент для подбора IT-специалистов

Система векторного поиска и проверки соответствия кандидатов вакансиям для платформы умного аутстаффинга **Iconicompany**.

## Обзор

Система решает задачу ранжирования вакансий относительно резюме кандидатов с использованием методов NLP и Learning-to-Rank.

Основана на исследовательской работе:
> **Vanetik N., Kogan G.** Job Vacancy Ranking with Sentence Embeddings, Keywords, and Named Entities. *Information* 2023, 14(8), 468.

### Ключевые возможности

| Возможность | Описание |
|---|---|
| **Векторный поиск** | FAISS с L1 (Manhattan) расстоянием |
| **Character n-grams** | Символьные n-граммы (1-3) + sentence embeddings |
| **TF-IDF Keywords** | Извлечение ключевых слов |
| **spaCy NER** | Именованные сущности (ORG, PRODUCT, TECH) |
| **Learning-to-Rank** | Оптимизация ранжирования через градиентный подъём NDCG |
| **Расчёт опыта** | Суммарный опыт (в месяцах) по каждому навыку |
| **Обязательные требования** | Проверка выполнения обязательных требований |

## Результаты на реальном датасете

Датасет: [vacancy-resume-matching-dataset](https://github.com/NataliaVanetik/vacancy-resume-matching-dataset) — 5 вакансий, 65 резюме, 30 аннотированных HR-экспертами.

| Метод | NDCG@5 (CV) | Spearman (CV) | Top-1 (CV) |
|---|---|---|---|
| Char n-grams + L1 (baseline статьи) | 0.9060 | 0.2167 | — |
| Sentence Emb + Cosine | 0.8172 | -0.3767 | — |
| Skill Jaccard | 0.8527 | -0.0533 | — |
| **Learning-to-Rank (CV)** | **0.9360 ± 0.02** | **0.6067 ± 0.10** | **40.00%** |
| **Learning-to-Rank (full)** | **0.9596** | **0.7500** | **53.33%** |

**Spearman ρ = 0.6067 (CV) — превосходит результат статьи (0.5908)**

## Установка

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Датасет загружается автоматически с GitHub — клонировать не нужно.

### Зависимости

| Библиотека | Назначение |
|---|---|
| `sentence-transformers` | Sentence embeddings |
| `faiss-cpu` | Векторный поиск |
| `spacy` | Named Entity Recognition |
| `scikit-learn` | TF-IDF векторизатор, нормализация |
| `numpy` | Математические операции |
| `python-docx` | Чтение .docx резюме |
| `scipy` | Статистические метрики |

## Быстрый старт

```bash
# Демо на встроенных данных
python main.py

# Оценка на реальном датасете (с GitHub, без скачивания)
python -m src.final_demo --dataset https://github.com/NataliaVanetik/vacancy-resume-matching-dataset

# Только LTR модель с cross-validation (с GitHub)
python -m src.ltr_model --dataset https://github.com/NataliaVanetik/vacancy-resume-matching-dataset
```

## Структура проекта

```
Хакатон/
├── requirements.txt                    # Зависимости
├── main.py                             # Entry point
├── README.md                           # Документация
├── EVALUATION.md                       # Оценка качества
├── .gitignore
├── vacancy-resume-matching-dataset-main/  # Датасет (клонировать с GitHub)
│   ├── 5_vacancies.csv
│   ├── CV/                             # 65 резюме в .docx
│   └── annotations-for-the-first-30-vacancies.txt
└── src/
    ├── __init__.py
    ├── models.py                       # Типы данных (dataclass)
    ├── keyword_extractor.py            # TF-IDF ключевые слова
    ├── ner_extractor.py                # spaCy NER
    ├── summarizer.py                   # Суммаризация вакансий
    ├── embedding_model.py              # Sentence + char n-gram embeddings
    ├── vector_search.py                # FAISS с L1 distance
    ├── compatibility_checker.py        # Проверка соответствия
    ├── matcher.py                      # Основной пайплайн (JobCandidateMatcher)
    ├── dataset_loader.py               # Загрузка реального датасета
    ├── ltr_model.py                    # Learning-to-Rank модель + фичи + метрики
    └── final_demo.py                   # Итоговая демонстрация с 4 методами
```


## Важнейшие фичи модели

| Фича | Вес | Интерпретация |
|---|---|---|
| `title_keywords_in_cv` | **+0.79** | Ключевые слова из заголовка в резюме — сильнейший позитивный сигнал |
| `skill_lang_overlap` | **+0.76** | Совпадение языков программирования — главный позитивный сигнал |
| `years_required` | **+0.70** | Требуемый опыт в вакансии — позитивный сигнал |
| `skill_api_overlap` | **−0.37** | Совпадение API — негативный (catch-all категория) |
| `years_match` | **+0.37** | Покрытие опыта кандидата требованиям — позитивный |
| `title_in_cv` | **−0.34** | Буквальный заголовок вакансии не в резюме — штраф |
| `char_ngram_l1` | **+0.32** | L1-расстояние — позитивный (меньше = ближе) |
| `domain_match` | **+0.28** | Совпадение доменов (fintech, security) — позитивный |

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

### Learning-to-Rank

| Метод | Описание |
|---|---|
| `run_ltr_pipeline(dataset_dir)` | Полный пайплайн с CV |
| `compute_ndcg_at_k(pred, true, k)` | NDCG@k |
| `compute_spearman(pred, true)` | Spearman correlation |

## Команда

Д.А. Касьяненко
