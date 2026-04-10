"""
Learning-to-Rank модуль для сопоставления вакансий и резюме.

Извлекает фичи для каждой пары CV-vacancy и обучает модель ранжирования,
оптимизированную для максимизации NDCG@5 и согласования с HR-аннотациями.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import docx
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ndcg_score


# ============================================================
# АННОТАЦИИ
# ============================================================

ANNOTATOR_1 = [
    [2,1,4,3,5],[1,2,3,4,5],[1,2,3,4,5],[3,1,2,4,5],[1,5,4,2,3],
    [3,2,1,4,5],[3,2,1,5,4],[2,4,3,1,5],[1,5,2,1,4],[3,2,1,4,5],
    [1,2,3,4,5],[1,2,3,4,5],[1,3,2,4,5],[1,2,3,4,5],[3,1,2,4,5],
    [3,1,2,4,5],[3,1,2,4,5],[1,2,5,3,4],[3,2,1,4,5],[3,2,1,4,5],
    [2,3,1,4,5],[1,2,3,5,4],[2,1,3,5,4],[1,2,3,5,4],[1,2,3,4,5],
    [2,1,3,4,5],[2,3,4,5,1],[2,4,3,2,5],[5,1,2,4,3],[2,1,4,3,5]
]

ANNOTATOR_2 = [
    [4,3,1,5,2],[2,4,3,1,5],[5,4,2,3,1],[1,3,2,4,5],[5,1,2,4,3],
    [1,3,2,4,5],[4,2,3,1,5],[2,4,3,1,5],[3,4,2,1,5],[4,1,2,5,3],
    [2,4,3,5,1],[4,3,2,1,5],[4,2,3,1,5],[3,4,2,1,5],[2,4,3,1,5],
    [3,2,4,1,5],[4,2,3,1,5],[4,2,5,3,1],[4,2,3,1,5],[1,5,2,4,3],
    [1,3,4,5,2],[4,1,3,2,5],[1,3,4,2,5],[1,4,3,5,2],[1,4,2,5,3],
    [1,5,2,4,3],[4,3,1,2,5],[1,4,2,3,5],[5,1,2,4,3],[1,2,3,4,5]
]


def get_consensus_rankings() -> list[list[int]]:
    """Вычисляет усреднённые (consensus) ранжирования."""
    rankings = []
    for r1, r2 in zip(ANNOTATOR_1, ANNOTATOR_2):
        avg = [(x + y) / 2 for x, y in zip(r1, r2)]
        sorted_idx = sorted(range(5), key=lambda i: avg[i])
        ranks = [0] * 5
        for rank, idx in enumerate(sorted_idx):
            ranks[idx] = rank + 1
        rankings.append(ranks)
    return rankings


# ============================================================
# ЗАГРУЗКА ДАТАСЕТА
# ============================================================

def load_vacancies(csv_path: str) -> list[dict[str, Any]]:
    """Загружает вакансии из CSV (файл или URL)."""
    import urllib.request
    import io

    if csv_path.startswith('http'):
        with urllib.request.urlopen(csv_path, timeout=30) as resp:
            csv_content = resp.read().decode('utf-8')
        reader = csv.DictReader(io.StringIO(csv_content))
        rows = list(reader)
    else:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    vacancies = []
    for row in rows:
        vacancies.append({
            'id': row['id'],
            'title': row.get('job_title', ''),
            'description': row.get('job_description', ''),
        })
    return vacancies


def load_resumes(cv_source: str) -> list[dict[str, Any]]:
    """Загружает резюме из .docx файлов (локально или с GitHub)."""
    from src.dataset_loader import load_all_resumes
    resumes_raw = load_all_resumes(cv_source)
    # Приводим к формату ltr_model
    result = []
    for r in resumes_raw:
        result.append({
            'id': r['id'].replace('cv_', ''),
            'text': r['text'],
        })
    return result


# ============================================================
# ИЗВЛЕЧЕНИЕ ТЕХНИЧЕСКИХ НАВЫКОВ
# ============================================================

# Известные технологии с категориями
TECH_SKILLS = {
    # Языки
    'python': 'lang', 'java': 'lang', 'javascript': 'lang', 'typescript': 'lang',
    'c++': 'lang', 'c#': 'lang', 'ruby': 'lang', 'go': 'lang', 'rust': 'lang',
    'kotlin': 'lang', 'swift': 'lang', 'scala': 'lang', 'php': 'lang',
    'perl': 'lang', 'clojure': 'lang', 'r': 'lang',
    # Базы данных
    'sql': 'db', 'mysql': 'db', 'postgresql': 'db', 'mongodb': 'db',
    'redis': 'db', 'oracle': 'db', 'mssql': 'db', 'mariadb': 'db',
    'nosql': 'db', 'cassandra': 'db', 'sqlite': 'db',
    # Фреймворки
    'react': 'fw', 'angular': 'fw', 'vue': 'fw', 'node': 'fw', 'node.js': 'fw',
    'jquery': 'fw', 'bootstrap': 'fw',
    'spring': 'fw', 'spring boot': 'fw', 'hibernate': 'fw', 'jpa': 'fw', 'jdbc': 'fw',
    'django': 'fw', 'flask': 'fw', 'fastapi': 'fw',
    'asp.net': 'fw', '.net': 'fw', 'wcf': 'fw', 'entity framework': 'fw',
    # Инфраструктура
    'docker': 'infra', 'kubernetes': 'infra', 'aws': 'infra', 'azure': 'infra',
    'gcp': 'infra', 'linux': 'infra', 'windows': 'infra',
    'jenkins': 'infra', 'git': 'infra', 'terraform': 'infra', 'ansible': 'infra',
    'ec2': 'infra', 's3': 'infra', 'lambda': 'infra', 'cloudfront': 'infra',
    # API/Протоколы
    'rest': 'api', 'graphql': 'api', 'api': 'api', 'json': 'api',
    'xml': 'api', 'soap': 'api', 'grpc': 'api',
    # Инструменты
    'maven': 'tool', 'gradle': 'tool', 'junit': 'tool', 'visual studio': 'tool',
    'tfs': 'tool', 'eclipse': 'tool', 'intellij': 'tool', 'intellij idea': 'tool',
    'swagger': 'tool', 'postman': 'tool',
    # Концепции
    'agile': 'concept', 'scrum': 'concept', 'tdd': 'concept', 'mvc': 'concept',
    'oop': 'concept', 'aop': 'concept', 'microservice': 'concept',
    'microservices': 'concept', 'ci/cd': 'concept',
    # Web
    'html': 'web', 'css': 'web', 'html5': 'web', 'css3': 'web',
    'lamp': 'web', 'drupal': 'web', 'elasticsearch': 'web',
    'view.js': 'web', 'hubspot': 'web', 'rabbitmq': 'web',
    # Security
    'pki': 'sec', 'ssl': 'sec', 'ipsec': 'sec', 'ipv4': 'sec', 'ipv6': 'sec',
    'cissp': 'sec', 'security+': 'sec', 'tcpdump': 'sec',
}


def extract_tech_skills(text: str) -> dict[str, str]:
    """Извлекает технические навыки из текста с категориями."""
    found = {}
    text_lower = text.lower()

    # Сортируем по длине (длинные фразы первыми) для корректного匹配
    for skill in sorted(TECH_SKILLS.keys(), key=len, reverse=True):
        if skill in text_lower:
            found[skill] = TECH_SKILLS[skill]

    return found


# ============================================================
# ИЗВЛЕЧЕНИЕ ФИЧЕЙ
# ============================================================

@dataclass
class PairFeatures:
    """Фичи для одной пары CV-vacancy."""
    # Skill features
    skill_overlap_count: int = 0
    skill_jaccard: float = 0.0
    skill_lang_overlap: int = 0
    skill_db_overlap: int = 0
    skill_fw_overlap: int = 0
    skill_infra_overlap: int = 0
    skill_api_overlap: int = 0
    skill_tool_overlap: int = 0
    skill_concept_overlap: int = 0
    skill_web_overlap: int = 0
    skill_sec_overlap: int = 0
    skill_recall: float = 0.0    # cv_skills ∩ vac_skills / vac_skills
    skill_precision: float = 0.0  # cv_skills ∩ vac_skills / cv_skills
    cv_skill_count: int = 0
    vac_skill_count: int = 0

    # Embedding features
    sentence_cosine: float = 0.0
    char_ngram_l1: float = 0.0
    char_ngram_cosine: float = 0.0

    # Text features
    text_overlap_ratio: float = 0.0
    title_in_cv: bool = False
    title_keywords_in_cv: int = 0

    # Experience features
    years_mentioned: float = 0.0
    years_required: float = 0.0
    years_match: float = 0.0

    # Level features
    level_match: float = 0.0
    seniority_match: float = 0.0

    # Domain features
    domain_match: float = 0.0


def extract_features(cv_text: str, vac_text: str, vac_title: str,
                     sent_model: SentenceTransformer,
                     char_vectorizer: TfidfVectorizer) -> PairFeatures:
    """Извлекает все фичи для пары CV-vacancy."""
    features = PairFeatures()

    # --- Skill features ---
    cv_skills = extract_tech_skills(cv_text)
    vac_skills = extract_tech_skills(vac_text)

    cv_skill_names = set(cv_skills.keys())
    vac_skill_names = set(vac_skills.keys())

    overlap = cv_skill_names & vac_skill_names
    features.skill_overlap_count = len(overlap)
    features.cv_skill_count = len(cv_skill_names)
    features.vac_skill_count = len(vac_skill_names)

    union = cv_skill_names | vac_skill_names
    features.skill_jaccard = len(overlap) / len(union) if union else 0.0

    overlap_categories = {}
    for skill in overlap:
        cat = vac_skills[skill]
        overlap_categories[cat] = overlap_categories.get(cat, 0) + 1

    features.skill_lang_overlap = overlap_categories.get('lang', 0)
    features.skill_db_overlap = overlap_categories.get('db', 0)
    features.skill_fw_overlap = overlap_categories.get('fw', 0)
    features.skill_infra_overlap = overlap_categories.get('infra', 0)
    features.skill_api_overlap = overlap_categories.get('api', 0)
    features.skill_tool_overlap = overlap_categories.get('tool', 0)
    features.skill_concept_overlap = overlap_categories.get('concept', 0)
    features.skill_web_overlap = overlap_categories.get('web', 0)
    features.skill_sec_overlap = overlap_categories.get('sec', 0)

    features.skill_recall = len(overlap) / len(vac_skill_names) if vac_skill_names else 0.0
    features.skill_precision = len(overlap) / len(cv_skill_names) if cv_skill_names else 0.0

    # --- Embedding features ---
    cv_sent_emb = sent_model.encode([cv_text], normalize_embeddings=True)[0]
    vac_sent_emb = sent_model.encode([vac_text], normalize_embeddings=True)[0]
    features.sentence_cosine = float(util.cos_sim(cv_sent_emb, vac_sent_emb)[0][0])

    cv_char = char_vectorizer.transform([cv_text]).toarray()[0]
    vac_char = char_vectorizer.transform([vac_text]).toarray()[0]
    features.char_ngram_l1 = float(np.sum(np.abs(cv_char - vac_char)))

    cv_char_norm = cv_char / (np.linalg.norm(cv_char) + 1e-8)
    vac_char_norm = vac_char / (np.linalg.norm(vac_char) + 1e-8)
    features.char_ngram_cosine = float(np.dot(cv_char_norm, vac_char_norm))

    # --- Text features ---
    cv_words = set(re.findall(r'\b\w{3,}\b', cv_text.lower()))
    vac_words = set(re.findall(r'\b\w{3,}\b', vac_text.lower()))
    text_overlap = cv_words & vac_words
    features.text_overlap_ratio = len(text_overlap) / len(vac_words) if vac_words else 0.0

    # Title match
    if vac_title:
        features.title_in_cv = vac_title.lower() in cv_text.lower()
        # Сколько ключевых слов из заголовка есть в резюме
        title_words = set(re.findall(r'\b\w{3,}\b', vac_title.lower()))
        features.title_keywords_in_cv = len(title_words & cv_words)

    # --- Experience features ---
    cv_year_matches = re.findall(r'(\d+)\+?\s*years?', cv_text.lower())
    features.years_mentioned = float(np.mean([int(y) for y in cv_year_matches])) if cv_year_matches else 0.0

    vac_year_matches = re.findall(r'(\d+)\+?\s*years?', vac_text.lower())
    features.years_required = float(np.mean([int(y) for y in vac_year_matches])) if vac_year_matches else 0.0

    # Years match: насколько опыт кандидата покрывает требования
    if features.years_required > 0:
        features.years_match = min(features.years_mentioned / features.years_required, 1.0)
    else:
        features.years_match = 0.5

    # --- Level features ---
    cv_level = _detect_seniority(cv_text)
    vac_level = _detect_seniority(vac_text)
    features.level_match = _level_similarity(cv_level, vac_level)

    # Seniority keywords match
    seniority_keywords = ['junior', 'mid', 'senior', 'lead', 'principal', 'staff', 'entry']
    cv_sen = sum(1 for w in seniority_keywords if w in cv_text.lower())
    vac_sen = sum(1 for w in seniority_keywords if w in vac_text.lower())
    features.seniority_match = 1.0 - abs(cv_sen - vac_sen) / max(len(seniority_keywords), 1)

    # --- Domain features ---
    features.domain_match = _domain_match_score(cv_text, vac_text)

    return features


def _detect_seniority(text: str) -> str:
    """Определяет уровень кандидата/вакансии."""
    text_lower = text.lower()

    if any(w in text_lower for w in ['senior', 'sr.', 'lead', 'principal', 'staff', '10+ years', '8+ years', '7+ years', '5+ years']):
        return 'senior'
    if any(w in text_lower for w in ['mid', 'middle', '3+ years', '4+ years', '5 years']):
        return 'mid'
    if any(w in text_lower for w in ['junior', 'jr.', 'entry', '1+ years', '2+ years', '0-2 years']):
        return 'junior'
    return 'unknown'


def _level_similarity(cv_level: str, vac_level: str) -> float:
    """Насколько уровни CV и вакансии совпадают."""
    levels = {'junior': 0, 'mid': 1, 'senior': 2, 'unknown': 1}
    diff = abs(levels.get(cv_level, 1) - levels.get(vac_level, 1))
    return max(0, 1.0 - diff * 0.5)


def _domain_match_score(cv_text: str, vac_text: str) -> float:
    """Проверяет совпадение доменов (fintech, security, healthcare и т.д.)."""
    domains = {
        'fintech': ['bank', 'finance', 'payment', 'trading', 'financial', 'fintech', 'credit', 'loan'],
        'security': ['cyber', 'security', 'defense', 'military', 'classified', 'clearance', 'pk'],
        'healthcare': ['health', 'medical', 'hospital', 'patient', 'clinical'],
        'ecommerce': ['e-commerce', 'ecommerce', 'retail', 'shop', 'marketplace'],
        'enterprise': ['enterprise', 'saas', 'b2b', 'crm', 'erp'],
        'startup': ['startup', 'early-stage', 'seed', 'series'],
        'gaming': ['game', 'gaming', 'unity', 'unreal'],
        'data': ['data science', 'analytics', 'big data', 'ml', 'machine learning'],
    }

    cv_lower = cv_text.lower()
    vac_lower = vac_text.lower()

    cv_domains = set()
    vac_domains = set()

    for domain, keywords in domains.items():
        if any(kw in cv_lower for kw in keywords):
            cv_domains.add(domain)
        if any(kw in vac_lower for kw in keywords):
            vac_domains.add(domain)

    if not vac_domains:
        return 0.5
    if not cv_domains:
        return 0.3

    overlap = cv_domains & vac_domains
    return len(overlap) / len(vac_domains)


def features_to_array(f: PairFeatures) -> np.ndarray:
    """Преобразует PairFeatures в numpy array."""
    return np.array([
        f.skill_overlap_count,
        f.skill_jaccard,
        f.skill_lang_overlap,
        f.skill_db_overlap,
        f.skill_fw_overlap,
        f.skill_infra_overlap,
        f.skill_api_overlap,
        f.skill_tool_overlap,
        f.skill_concept_overlap,
        f.skill_web_overlap,
        f.skill_sec_overlap,
        f.skill_recall,
        f.skill_precision,
        f.cv_skill_count,
        f.vac_skill_count,
        f.sentence_cosine,
        f.char_ngram_l1,
        f.char_ngram_cosine,
        f.text_overlap_ratio,
        float(f.title_in_cv),
        f.title_keywords_in_cv,
        f.years_mentioned,
        f.years_required,
        f.years_match,
        f.level_match,
        f.seniority_match,
        f.domain_match,
    ], dtype=np.float64)


# ============================================================
# МОДЕЛЬ РАНЖИРОВАНИЯ
# ============================================================

class RankModel:
    """
    Модель ранжирования: линейная комбинация фичей с обученными весами.

    score = w^T * x

    Веса обучаются через оптимизацию NDCG@5 на тренировочных данных.
    """

    def __init__(self):
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    def fit(self, X: np.ndarray, y_ranks: np.ndarray,
            groups: np.ndarray, n_iterations: int = 500,
            learning_rate: float = 0.01) -> None:
        """
        Обучает модель через градиентный подъём NDCG.

        Args:
            X: Матрица фичей (n_pairs, n_features).
            y_ranks: Ранги (n_pairs,) — 1 = лучший, 5 = худший.
            groups: Массив, указывающий к какой группе (CV) принадлежит пара.
            n_iterations: Число итераций градиентного спуска.
            learning_rate: Шаг обучения.
        """
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        unique_groups = np.unique(groups)

        for iteration in range(n_iterations):
            grad_w = np.zeros(n_features)
            grad_b = 0.0
            count = 0

            for group in unique_groups:
                mask = groups == group
                X_g = X[mask]
                y_g = y_ranks[mask]

                # Scores
                scores = X_g @ self.weights + self.bias

                # NDCG gradient approximation
                # Используем pairwise approach: для каждой пары (i,j) где rank_i < rank_j
                # хотим score_i > score_j
                n_items = len(scores)
                for i in range(n_items):
                    for j in range(i + 1, n_items):
                        if y_g[i] < y_g[j]:  # i должен быть выше j
                            diff = scores[i] - scores[j]
                            # Gradient of logistic loss
                            sigmoid = 1.0 / (1.0 + np.exp(diff))
                            grad_w += sigmoid * (X_g[i] - X_g[j])
                            grad_b += sigmoid * 2.0
                            count += 1
                        elif y_g[i] > y_g[j]:  # j должен быть выше i
                            diff = scores[j] - scores[i]
                            sigmoid = 1.0 / (1.0 + np.exp(diff))
                            grad_w += sigmoid * (X_g[j] - X_g[i])
                            grad_b += sigmoid * 2.0
                            count += 1

            if count > 0:
                grad_w /= count
                grad_b /= count

            # Update
            self.weights += learning_rate * grad_w
            self.bias += learning_rate * grad_b

            # L2 regularization
            self.weights *= 0.999

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказывает скоры для пар."""
        return X @ self.weights + self.bias

    def rank(self, X: np.ndarray) -> list[int]:
        """Возвращает ранги (1 = лучший)."""
        scores = self.predict(X)
        sorted_idx = np.argsort(-scores)  # По убыванию
        ranks = [0] * len(scores)
        for rank, idx in enumerate(sorted_idx):
            ranks[int(idx)] = rank + 1
        return ranks


# ============================================================
# МЕТРИКИ
# ============================================================

def compute_ndcg_at_k(predicted_ranks: list[list[int]],
                     true_ranks: list[list[int]],
                     k: int = 5) -> float:
    """
    Вычисляет NDCG@k.

    Args:
        predicted_ranks: Предсказанные ранги [[rank1, rank2, ..., rank5], ...]
        true_ranks: Истинные ранги
        k: Параметр k для NDCG

    Returns:
        Средний NDCG@k.
    """
    ndcgs = []
    for pred, true in zip(predicted_ranks, true_ranks):
        # Преобразуем ранги в релевантности (rank 1 -> релевантность 5, rank 5 -> релевантность 1)
        pred_rel = np.array([max(0, 6 - r) for r in pred]).reshape(1, -1)
        true_rel = np.array([max(0, 6 - r) for r in true]).reshape(1, -1)

        ndcg = ndcg_score(true_rel, pred_rel, k=min(k, len(pred)))
        ndcgs.append(ndcg)

    return float(np.mean(ndcgs))


def compute_spearman(predicted_ranks: list[list[int]],
                    true_ranks: list[list[int]]) -> float:
    """Вычисляет среднюю корреляцию Спирмена."""
    from scipy import stats

    rhos = []
    for pred, true in zip(predicted_ranks, true_ranks):
        if len(set(pred)) < len(pred):
            continue
        rho, _ = stats.spearmanr(pred, true)
        if not np.isnan(rho):
            rhos.append(rho)

    return float(np.mean(rhos)) if rhos else 0.0


# ============================================================
# ГЛАВНЫЙ ПЛАЙПЛАЙН
# ============================================================

def run_ltr_pipeline(dataset_dir: str, use_cv: bool = True) -> dict:
    """
    Полный пайплайн learning-to-rank.

    Args:
        dataset_dir: Путь к vacancy-resume-matching-dataset-main/
        use_cv: Использовать ли cross-validation.

    Returns:
        Словарь с метриками.
    """
    print("=" * 70)
    print("  LEARNING-TO-RANK ПЛАЙПЛАЙН")
    print("  Iconicompany Job-Candidate Matching System")
    print("=" * 70)

    # Загрузка данных
    # Определяем: это локальный путь или GitHub URL
    if dataset_dir.startswith('http'):
        # GitHub URL -> преобразуем в raw URL
        raw_base = dataset_dir.replace('github.com', 'raw.githubusercontent.com')
        if '/refs/heads/' not in raw_base:
            raw_base = raw_base + '/refs/heads/main'
        vacancies = load_vacancies(raw_base + '/5_vacancies.csv')
        resumes = load_resumes(raw_base + '/CV')
    else:
        # Локальный путь
        csv_path = Path(dataset_dir) / '5_vacancies.csv'
        cv_dir = Path(dataset_dir) / 'CV'
        vacancies = load_vacancies(str(csv_path))
        resumes = load_resumes(str(cv_dir))

    print(f"\n📊 Загружено: {len(vacancies)} вакансий, {len(resumes)} резюме")
    print(f"📊 Обучающая выборка: 30 аннотированных резюме")

    # Инициализация моделей
    print("\n🔄 Загрузка моделей...")
    sent_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Char n-gram vectorizer
    all_texts = [v['description'] for v in vacancies] + [r['text'] for r in resumes]
    char_vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(1, 3),
        max_features=15000,
        strip_accents='unicode',
        lowercase=True,
    )
    char_vectorizer.fit(all_texts)

    # Извлечение фичей для всех пар (30 резюме × 5 вакансий)
    print("\n🔧 Извлечение фичей для 30×5 = 150 пар...")

    consensus = get_consensus_rankings()

    all_X = []
    all_y = []  # Consensus ranks
    all_groups = []
    all_pairs_info = []

    for cv_idx in range(30):
        cv_text = resumes[cv_idx]['text']

        for vac_idx in range(5):
            vac = vacancies[vac_idx]
            features = extract_features(
                cv_text,
                vac['description'],
                vac['title'],
                sent_model,
                char_vectorizer,
            )

            all_X.append(features_to_array(features))
            all_y.append(consensus[cv_idx][vac_idx])
            all_groups.append(cv_idx)
            all_pairs_info.append({
                'cv_idx': cv_idx,
                'vac_idx': vac_idx,
                'cv_id': resumes[cv_idx]['id'],
                'vac_id': vac['id'],
            })

    X = np.array(all_X)
    y = np.array(all_y)
    groups = np.array(all_groups)

    print(f"   Матрица фичей: {X.shape}")

    # Нормализация фичей
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Обучение модели
    print("\n🎯 Обучение модели ранжирования...")
    model = RankModel()

    if use_cv:
        # 5-fold cross-validation
        print("   5-fold cross-validation...")
        from sklearn.model_selection import GroupKFold

        gkf = GroupKFold(n_splits=5)
        ndcg_scores = []
        spearman_scores = []
        top1_scores = []

        fold_preds = {}

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_scaled, y, groups)):
            X_train = X_scaled[train_idx]
            y_train = y[train_idx]
            groups_train = groups[train_idx]

            X_test = X_scaled[test_idx]
            y_test = y[test_idx]
            groups_test = groups[test_idx]

            # Обучаем модель
            fold_model = RankModel()
            fold_model.fit(X_train, y_train, groups_train, n_iterations=1000, learning_rate=0.05)

            # Предсказываем
            test_scores = fold_model.predict(X_test)

            # Группируем по CV
            unique_test_groups = np.unique(groups_test)
            pred_ranks = []
            true_ranks = []

            for g in unique_test_groups:
                g_mask = groups_test == g
                g_scores = test_scores[g_mask]
                g_true = y_test[g_mask]

                # Ранги из скоров
                sorted_idx = np.argsort(-g_scores)
                g_pred_ranks = [0] * len(g_scores)
                for rank, idx in enumerate(sorted_idx):
                    g_pred_ranks[int(idx)] = rank + 1

                pred_ranks.append(g_pred_ranks)
                true_ranks.append(list(g_true.astype(int)))

            ndcg = compute_ndcg_at_k(pred_ranks, true_ranks, k=5)
            spearman = compute_spearman(pred_ranks, true_ranks)

            # Top-1 accuracy
            top1 = 0
            for pr, tr in zip(pred_ranks, true_ranks):
                if pr.index(1) == tr.index(1):
                    top1 += 1
            top1_acc = top1 / len(pred_ranks) if pred_ranks else 0

            ndcg_scores.append(ndcg)
            spearman_scores.append(spearman)
            top1_scores.append(top1_acc)

            print(f"   Fold {fold_idx + 1}: NDCG@5={ndcg:.4f}, Spearman={spearman:.4f}, Top-1={top1_acc:.2%}")

        # Обучаем финальную модель на всех данных
        print("\n🎯 Финальная модель на всех данных...")
        model.fit(X_scaled, y, groups, n_iterations=2000, learning_rate=0.05)

        # Оцениваем на всех данных (для анализа)
        all_scores = model.predict(X_scaled)
        all_pred_ranks = []
        for g in range(30):
            g_mask = groups == g
            g_scores = all_scores[g_mask]
            sorted_idx = np.argsort(-g_scores)
            g_ranks = [0] * 5
            for rank, idx in enumerate(sorted_idx):
                g_ranks[int(idx)] = rank + 1
            all_pred_ranks.append(g_ranks)

        full_ndcg = compute_ndcg_at_k(all_pred_ranks, consensus, k=5)
        full_spearman = compute_spearman(all_pred_ranks, consensus)

        print(f"\n{'=' * 70}")
        print("  РЕЗУЛЬТАТЫ CROSS-VALIDATION")
        print(f"{'=' * 70}")
        print(f"\n  {'Метрика':<30} {'CV Mean':>10} {'CV Std':>10}")
        print(f"  {'─' * 52}")
        print(f"  {'NDCG@5':<30} {np.mean(ndcg_scores):>10.4f} {np.std(ndcg_scores):>10.4f}")
        print(f"  {'Spearman ρ':<30} {np.mean(spearman_scores):>10.4f} {np.std(spearman_scores):>10.4f}")
        print(f"  {'Top-1 Accuracy':<30} {np.mean(top1_scores):>10.4f} {np.std(top1_scores):>10.4f}")

        print(f"\n  {'=' * 70}")
        print("  СРАВНЕНИЕ С БАЗЛАЙНАМИ")
        print(f"{'=' * 70}")
        print(f"\n  {'Метод':<40} {'NDCG@5':>10} {'Spearman':>10}")
        print(f"  {'─' * 62}")
        print(f"  {'Char n-grams + L1 (статья baseline)':<40} {0.35:>10.4f} {0.59:>10.4f}")
        print(f"  {'Sentence Embeddings + L1':<40} {0.25:>10.4f} {-0.25:>10.4f}")
        print(f"  {'Skill Jaccard':<40} {0.30:>10.4f} {-0.08:>10.4f}")
        print(f"  {'Learning-to-Rank (наша, CV)':<40} {np.mean(ndcg_scores):>10.4f} {np.mean(spearman_scores):>10.4f}")
        print(f"  {'Learning-to-Rank (наша, full)':<40} {full_ndcg:>10.4f} {full_spearman:>10.4f}")

        # Feature importance
        print(f"\n{'=' * 70}")
        print("  ВАЖНОСТЬ ФИЧЕЙ")
        print(f"{'=' * 70}")
        feature_names = [
            'skill_overlap_count', 'skill_jaccard',
            'skill_lang_overlap', 'skill_db_overlap', 'skill_fw_overlap',
            'skill_infra_overlap', 'skill_api_overlap', 'skill_tool_overlap',
            'skill_concept_overlap', 'skill_web_overlap', 'skill_sec_overlap',
            'skill_recall', 'skill_precision',
            'cv_skill_count', 'vac_skill_count',
            'sentence_cosine', 'char_ngram_l1', 'char_ngram_cosine',
            'text_overlap_ratio', 'title_in_cv', 'title_keywords_in_cv',
            'years_mentioned', 'years_required', 'years_match',
            'level_match', 'seniority_match', 'domain_match'
        ]
        abs_weights = np.abs(model.weights)
        sorted_idx = np.argsort(-abs_weights)
        for i, idx in enumerate(sorted_idx):
            bar = '█' * int(abs(model.weights[idx]) * 20)
            print(f"  {feature_names[idx]:<25} {model.weights[idx]:>8.4f}  {bar}")

        return {
            'cv_ndcg_mean': float(np.mean(ndcg_scores)),
            'cv_ndcg_std': float(np.std(ndcg_scores)),
            'cv_spearman_mean': float(np.mean(spearman_scores)),
            'cv_spearman_std': float(np.std(spearman_scores)),
            'cv_top1_mean': float(np.mean(top1_scores)),
            'cv_top1_std': float(np.std(top1_scores)),
            'full_ndcg': full_ndcg,
            'full_spearman': full_spearman,
            'weights': model.weights.tolist(),
            'feature_names': feature_names,
        }
    else:
        # Без CV — обучаем на всех данных
        model.fit(X_scaled, y, groups, n_iterations=2000, learning_rate=0.05)

        all_scores = model.predict(X_scaled)
        all_pred_ranks = []
        for g in range(30):
            g_mask = groups == g
            g_scores = all_scores[g_mask]
            sorted_idx = np.argsort(-g_scores)
            g_ranks = [0] * 5
            for rank, idx in enumerate(sorted_idx):
                g_ranks[int(idx)] = rank + 1
            all_pred_ranks.append(g_ranks)

        ndcg = compute_ndcg_at_k(all_pred_ranks, consensus, k=5)
        spearman = compute_spearman(all_pred_ranks, consensus)

        print(f"\n  NDCG@5: {ndcg:.4f}")
        print(f"  Spearman: {spearman:.4f}")

        return {
            'ndcg': ndcg,
            'spearman': spearman,
            'weights': model.weights.tolist(),
        }


if __name__ == '__main__':
    import sys
    from pathlib import Path

    dataset_dir = None
    if '--dataset' in sys.argv:
        idx = sys.argv.index('--dataset')
        if idx + 1 < len(sys.argv):
            dataset_dir = sys.argv[idx + 1]

    if not dataset_dir:
        project_dir = Path(__file__).parent.parent
        possible_dir = project_dir / 'vacancy-resume-matching-dataset-main'
        if possible_dir.exists():
            dataset_dir = str(possible_dir)
        else:
            print("❌ Датасет не найден.")
            sys.exit(1)

    run_ltr_pipeline(dataset_dir)
