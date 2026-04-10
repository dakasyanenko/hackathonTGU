"""
Точка входа с примером использования и демонстрацией.

Основан на исследовательской работе:
"Job Vacancy Ranking with Sentence Embeddings, Keywords, and Named Entities"
(Natalia Vanetik, Genady Kogan, Information 2023)

Оптимальная конфигурация по статье:
- Полный текст (резюме) + Суммаризованный текст (вакансии)
- Символьные n-граммы (1-3) + Sentence-эмбеддинги
- L1 (Манхэттенское) расстояние для ранжирования
- KeyBERT ключевые слова + spaCy NER для улучшения текста
"""

from src.matcher import JobCandidateMatcher


# ============================================================
# ТЕСТОВЫЕ ДАННЫЕ
# ============================================================

VACANCIES = [
    {
        "id": "vac_1",
        "data": {
            "position": "Python Developer",
            "industry": "FinTech",
            "mandatoryRequirements": "Python, PostgreSQL, Docker, 3+ years experience",
            "projectTasks": "Разработка микросервисов для платёжной системы, оптимизация API, интеграция с внешними сервисами",
            "experienceLevels": "Middle/Senior",
            "description": "Ищем разработчика Python для работы над высоконагруженной платёжной системой. Команда из 8 человек, Agile, CI/CD."
        },
        "skills": ["Python", "PostgreSQL", "Docker", "FastAPI", "Redis"],
        "dataEng": "Python Developer for FinTech payment system, microservices, high-load"
    },
    {
        "id": "vac_2",
        "data": {
            "position": "Frontend Developer",
            "industry": "E-commerce",
            "mandatoryRequirements": "React, TypeScript, 2+ years",
            "projectTasks": "Разработка UI для маркетплейса, интеграция с REST API, оптимизация производительности",
            "experienceLevels": "Junior+/Middle",
            "description": "Разработка интерфейса онлайн-магазина с миллионами пользователей."
        },
        "skills": ["React", "TypeScript", "JavaScript", "CSS", "Redux"],
        "dataEng": "Frontend Developer for e-commerce platform, React, TypeScript"
    },
    {
        "id": "vac_3",
        "data": {
            "position": "Data Engineer",
            "industry": "Banking",
            "mandatoryRequirements": "Python, Apache Spark, SQL, ETL pipelines",
            "projectTasks": "Построение data pipeline для аналитики, интеграция данных из различных источников",
            "experienceLevels": "Middle",
            "description": "Разработка и поддержка data pipeline для банковского сектора. Работа с большими данными."
        },
        "skills": ["Python", "Apache Spark", "SQL", "Airflow", "Kafka"],
        "dataEng": "Data Engineer for banking analytics, Spark, ETL"
    },
    {
        "id": "vac_4",
        "data": {
            "position": "DevOps Engineer",
            "industry": "Cloud Services",
            "mandatoryRequirements": "Kubernetes, Docker, CI/CD, Terraform, 3+ years",
            "projectTasks": "Автоматизация деплоя, мониторинг инфраструктуры, оптимизация облачных ресурсов",
            "experienceLevels": "Senior",
            "description": "Настройка и поддержка Kubernetes кластера, автоматизация CI/CD пайплайнов."
        },
        "skills": ["Kubernetes", "Docker", "Terraform", "AWS", "Jenkins"],
        "dataEng": "DevOps Engineer, Kubernetes, CI/CD automation"
    },
    {
        "id": "vac_5",
        "data": {
            "position": "Full-Stack Developer",
            "industry": "Startup",
            "mandatoryRequirements": "Python, React, PostgreSQL, 2+ years",
            "projectTasks": "Разработка full-stack MVP для стартапа в области healthtech",
            "experienceLevels": "Middle",
            "description": "Создание продукта с нуля: backend на FastAPI, frontend на React."
        },
        "skills": ["Python", "React", "PostgreSQL", "FastAPI", "TypeScript"],
        "dataEng": "Full-Stack Developer, Python + React, healthtech startup"
    },
]

RESUMES = [
    {
        "id": "res_1",
        "skill_set": ["Python", "PostgreSQL", "Docker", "FastAPI", "Git", "Linux"],
        "experience": [
            {
                "company": "TechCorp",
                "description": "Разработка backend для платёжной системы, микросервисы",
                "stack": "Python, PostgreSQL, Docker, FastAPI, Redis",
                "start": "2022-01",
                "end": "2024-06"
            },
            {
                "company": "StartupXYZ",
                "description": "Создание API для мобильного приложения",
                "stack": "Python, FastAPI, Redis, Git",
                "start": "2020-06",
                "end": "2021-12"
            }
        ],
        "education": "Computer Science, MSU"
    },
    {
        "id": "res_2",
        "skill_set": ["React", "TypeScript", "JavaScript", "CSS", "Redux", "HTML"],
        "experience": [
            {
                "company": "WebAgency",
                "description": "Разработка SPA для e-commerce клиентов",
                "stack": "React, TypeScript, Redux, CSS, GraphQL",
                "start": "2023-01",
                "end": "2024-06"
            },
            {
                "company": "Freelance",
                "description": "Фронтенд разработка для различных проектов",
                "stack": "React, JavaScript, HTML, CSS",
                "start": "2021-06",
                "end": "2022-12"
            }
        ],
        "education": "Web Development, HSE"
    },
    {
        "id": "res_3",
        "skill_set": ["Python", "Apache Spark", "SQL", "Pandas", "NumPy"],
        "experience": [
            {
                "company": "DataCorp",
                "description": "Построение ETL пайплайнов для аналитической платформы",
                "stack": "Python, Apache Spark, Airflow, SQL, Kafka",
                "start": "2021-03",
                "end": "2024-06"
            },
            {
                "company": "AnalyticsLab",
                "description": "Анализ данных и создание отчётов",
                "stack": "Python, Pandas, NumPy, SQL",
                "start": "2019-09",
                "end": "2021-02"
            }
        ],
        "education": "Data Science, MIPT"
    },
]


# ============================================================
# ДЕМОНСТРАЦИЯ
# ============================================================

def main():
    """Запускает полную демонстрацию сопоставления."""

    # Инициализация системы
    matcher = JobCandidateMatcher(
        sentence_model="all-MiniLM-L6-v2",
        use_char_ngrams=True,
        enable_keywords=True,
        enable_ner=True,
    )

    # Строим индекс вакансий
    matcher.build_vacancy_index(VACANCIES)

    # Тестируем каждое резюме
    for resume in RESUMES:
        resume_name = f"Кандидат {resume['id']} ({', '.join(resume.get('skill_set', [])[:3])}...)"
        print(f"\n{'=' * 70}")
        print(f"  🔍 {resume_name}")
        print(f"{'=' * 70}")

        # Поиск совпадений
        results = matcher.find_best_matches(resume, top_k=3, detailed=True)

        for i, result in enumerate(results, 1):
            print(f"\n  {i}. {result.vacancy_position}")
            print(f"     Совпадение: {result.match_percentage}%")
            print(f"     L1 расстояние: {result.l1_distance}")

            if result.matched_skills:
                print(f"     ✅ Навыки: {', '.join(result.matched_skills)}")
            if result.missing_skills:
                print(f"     ❌ Не хватает: {', '.join(result.missing_skills)}")
            if result.extra_candidate_skills:
                print(f"     🔵 Дополнительно: {', '.join(result.extra_candidate_skills)}")

            if result.experience_months:
                exp_parts = []
                for skill, months in result.experience_months.items():
                    exp_parts.append(f"{skill}: {months}мес.")
                print(f"     📅 Опыт: {', '.join(exp_parts)}")

        # Полный отчёт для первого резюме
        if resume["id"] == "res_1":
            print("\n")
            print(matcher.generate_report(resume, top_k=5))

    # Пример JSON-вывода
    print("\n\n📄 JSON OUTPUT:")
    print(matcher.find_best_matches_json(RESUMES[0], top_k=3))


if __name__ == "__main__":
    main()
