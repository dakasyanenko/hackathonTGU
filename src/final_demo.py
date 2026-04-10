"""
Итоговая демонстрация: Learning-to-Rank для Iconicompany.

Сравнивает 4 подхода:
1. Char n-grams TF-IDF + L1 (baseline из статьи)
2. Sentence Embeddings + L1
3. Skill Jaccard Overlap
4. Learning-to-Rank (наш метод)

Использование:
    python -m src.final_demo --dataset DIR
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

from src.ltr_model import (
    load_vacancies,
    load_resumes,
    get_consensus_rankings,
    extract_tech_skills,
    RankModel,
    compute_ndcg_at_k,
    compute_spearman,
    extract_features,
    features_to_array,
    ANNOTATOR_1,
    ANNOTATOR_2,
)


def baseline_char_ngrams(vacancies, resumes, char_vectorizer, n_cv=30):
    """Baseline: character n-grams + L1 distance."""
    vac_texts = [v['description'] for v in vacancies]
    vac_vectors = char_vectorizer.transform(vac_texts).toarray().astype(np.float32)

    consensus = get_consensus_rankings()
    pred_ranks = []

    for i in range(n_cv):
        res_vector = char_vectorizer.transform([resumes[i]['text']]).toarray()[0].astype(np.float32)
        diffs = np.abs(vac_vectors - res_vector)
        l1_dists = np.sum(diffs, axis=1)
        sorted_idx = np.argsort(l1_dists)
        ranks = [0] * 5
        for rank, idx in enumerate(sorted_idx):
            ranks[idx] = rank + 1
        pred_ranks.append(ranks)

    ndcg = compute_ndcg_at_k(pred_ranks, consensus, k=5)
    spearman = compute_spearman(pred_ranks, consensus)
    return ndcg, spearman, pred_ranks


def baseline_sentence_embeddings(vacancies, resumes, sent_model, n_cv=30):
    """Baseline: sentence embeddings + cosine similarity."""
    from sentence_transformers import util

    vac_texts = [v['description'] for v in vacancies]
    vac_embs = sent_model.encode(vac_texts, normalize_embeddings=True)

    consensus = get_consensus_rankings()
    pred_ranks = []

    for i in range(n_cv):
        res_emb = sent_model.encode([resumes[i]['text']], normalize_embeddings=True)[0]
        scores = [float(util.cos_sim(res_emb, vac_embs[j])[0][0]) for j in range(5)]
        sorted_idx = np.argsort([-s for s in scores])
        ranks = [0] * 5
        for rank, idx in enumerate(sorted_idx):
            ranks[idx] = rank + 1
        pred_ranks.append(ranks)

    ndcg = compute_ndcg_at_k(pred_ranks, consensus, k=5)
    spearman = compute_spearman(pred_ranks, consensus)
    return ndcg, spearman, pred_ranks


def baseline_skill_jaccard(vacancies, resumes, n_cv=30):
    """Baseline: skill Jaccard overlap."""
    consensus = get_consensus_rankings()
    pred_ranks = []

    vac_skills_list = []
    for v in vacancies:
        skills = set(extract_tech_skills(v['description']).keys())
        vac_skills_list.append(skills)

    for i in range(n_cv):
        cv_skills = set(extract_tech_skills(resumes[i]['text']).keys())

        scores = []
        for j, vac_skills in enumerate(vac_skills_list):
            if not vac_skills or not cv_skills:
                scores.append(0.0)
                continue
            inter = vac_skills & cv_skills
            union = vac_skills | cv_skills
            scores.append(len(inter) / len(union) if union else 0.0)

        sorted_idx = np.argsort([-s for s in scores])
        ranks = [0] * 5
        for rank, idx in enumerate(sorted_idx):
            ranks[idx] = rank + 1
        pred_ranks.append(ranks)

    ndcg = compute_ndcg_at_k(pred_ranks, consensus, k=5)
    spearman = compute_spearman(pred_ranks, consensus)
    return ndcg, spearman, pred_ranks


def run_final_demo(dataset_dir: str):
    """Запускает полную демонстрацию."""
    print("=" * 70)
    print("  ICONICOMPANY — ФИНАЛЬНАЯ ДЕМОНСТРАЦИЯ")
    print("  Learning-to-Rank для сопоставления вакансий и резюме")
    print("=" * 70)

    # Загрузка
    # GitHub URL или локальный путь
    if dataset_dir.startswith('http'):
        raw_base = dataset_dir.replace('github.com', 'raw.githubusercontent.com')
        if '/refs/heads/' not in raw_base:
            raw_base = raw_base + '/refs/heads/main'
        vacancies = load_vacancies(raw_base + '/5_vacancies.csv')
        resumes = load_resumes(raw_base + '/CV')
    else:
        csv_path = Path(dataset_dir) / '5_vacancies.csv'
        cv_dir_path = Path(dataset_dir) / 'CV'
        vacancies = load_vacancies(str(csv_path))
        resumes = load_resumes(str(cv_dir_path))
    consensus = get_consensus_rankings()

    print(f"\n📊 Датасет: {len(vacancies)} вакансий × {len(resumes)} резюме")
    print(f"📊 Оценка: первые {30} резюме с HR-аннотациями")
    print(f"📊 Всего пар для оценки: {30 * 5}")

    # Инициализация моделей
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sentence_transformers import SentenceTransformer

    sent_model = SentenceTransformer('all-MiniLM-L6-v2')

    all_texts = [v['description'] for v in vacancies] + [r['text'] for r in resumes]
    char_vectorizer = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(1, 3),
        max_features=15000, strip_accents='unicode', lowercase=True,
    )
    char_vectorizer.fit(all_texts)

    # 1. Char n-grams + L1
    print("\n" + "=" * 70)
    print("  1️⃣  CHAR N-GRAMS + L1 (baseline из статьи Vanetik & Kogan)")
    print("=" * 70)
    ndcg1, sp1, _ = baseline_char_ngrams(vacancies, resumes, char_vectorizer)
    print(f"  NDCG@5:    {ndcg1:.4f}")
    print(f"  Spearman:  {sp1:.4f}")

    # 2. Sentence Embeddings
    print("\n" + "=" * 70)
    print("  2️⃣  SENTENCE EMBEDDINGS + Cosine")
    print("=" * 70)
    ndcg2, sp2, _ = baseline_sentence_embeddings(vacancies, resumes, sent_model)
    print(f"  NDCG@5:    {ndcg2:.4f}")
    print(f"  Spearman:  {sp2:.4f}")

    # 3. Skill Jaccard
    print("\n" + "=" * 70)
    print("  3️⃣  SKILL JACCARD OVERLAP")
    print("=" * 70)
    ndcg3, sp3, _ = baseline_skill_jaccard(vacancies, resumes)
    print(f"  NDCG@5:    {ndcg3:.4f}")
    print(f"  Spearman:  {sp3:.4f}")

    # 4. Learning-to-Rank
    print("\n" + "=" * 70)
    print("  4️⃣  LEARNING-TO-RANK (наш метод)")
    print("=" * 70)

    from sklearn.preprocessing import StandardScaler

    all_X = []
    all_y = []
    all_groups = []

    for cv_idx in range(30):
        cv_text = resumes[cv_idx]['text']
        for vac_idx in range(5):
            vac = vacancies[vac_idx]
            features = extract_features(
                cv_text, vac['description'], vac['title'],
                sent_model, char_vectorizer,
            )
            all_X.append(features_to_array(features))
            all_y.append(consensus[cv_idx][vac_idx])
            all_groups.append(cv_idx)

    X = np.array(all_X)
    y = np.array(all_y)
    groups = np.array(all_groups)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5-fold CV
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)

    cv_ndcgs, cv_spearmans, cv_top1s = [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_scaled, y, groups)):
        fold_model = RankModel()
        fold_model.fit(
            X_scaled[train_idx], y[train_idx], groups[train_idx],
            n_iterations=1000, learning_rate=0.05
        )

        test_scores = fold_model.predict(X_scaled[test_idx])
        groups_test = groups[test_idx]
        unique_g = np.unique(groups_test)

        pred_ranks, true_ranks = [], []
        for g in unique_g:
            g_mask = groups_test == g
            g_scores = test_scores[g_mask]
            g_true = y[test_idx][g_mask]

            sorted_idx = np.argsort(-g_scores)
            g_ranks = [0] * len(g_scores)
            for rank, idx in enumerate(sorted_idx):
                g_ranks[int(idx)] = rank + 1

            pred_ranks.append(g_ranks)
            true_ranks.append(list(g_true.astype(int)))

        ndcg = compute_ndcg_at_k(pred_ranks, true_ranks, k=5)
        spearman = compute_spearman(pred_ranks, true_ranks)

        top1 = sum(1 for pr, tr in zip(pred_ranks, true_ranks) if pr.index(1) == tr.index(1))
        top1_acc = top1 / len(pred_ranks) if pred_ranks else 0

        cv_ndcgs.append(ndcg)
        cv_spearmans.append(spearman)
        cv_top1s.append(top1_acc)

    # Финальная модель
    final_model = RankModel()
    final_model.fit(X_scaled, y, groups, n_iterations=2000, learning_rate=0.05)

    all_scores = final_model.predict(X_scaled)
    all_pred_ranks = []
    for g in range(30):
        g_mask = groups == g
        g_scores = all_scores[g_mask]
        sorted_idx = np.argsort(-g_scores)
        g_ranks = [0] * 5
        for rank_pos, idx_val in enumerate(sorted_idx):
            g_ranks[int(idx_val)] = rank_pos + 1
        all_pred_ranks.append(g_ranks)

    full_ndcg = compute_ndcg_at_k(all_pred_ranks, consensus, k=5)
    full_spearman = compute_spearman(all_pred_ranks, consensus)

    full_top1 = sum(1 for pr, tr in zip(all_pred_ranks, consensus) if pr.index(1) == tr.index(1))
    full_top1_acc = full_top1 / 30

    print(f"  NDCG@5 (CV):        {np.mean(cv_ndcgs):.4f} ± {np.std(cv_ndcgs):.4f}")
    print(f"  Spearman (CV):      {np.mean(cv_spearmans):.4f} ± {np.std(cv_spearmans):.4f}")
    print(f"  Top-1 Acc (CV):     {np.mean(cv_top1s):.4f} ± {np.std(cv_top1s):.4f}")
    print(f"  NDCG@5 (full):      {full_ndcg:.4f}")
    print(f"  Spearman (full):    {full_spearman:.4f}")
    print(f"  Top-1 Acc (full):   {full_top1_acc:.2%}")

    # Итоговая таблица
    print(f"\n{'=' * 70}")
    print("  ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print(f"{'=' * 70}")
    print(f"\n  {'Метод':<35} {'NDCG@5':>10} {'Spearman':>10} {'Top-1':>10}")
    print(f"  {'─' * 67}")
    print(f"  {'Char n-grams + L1':<35} {ndcg1:>10.4f} {sp1:>10.4f} {'—':>10}")
    print(f"  {'Sentence Emb + Cosine':<35} {ndcg2:>10.4f} {sp2:>10.4f} {'—':>10}")
    print(f"  {'Skill Jaccard':<35} {ndcg3:>10.4f} {sp3:>10.4f} {'—':>10}")
    print(f"  {'Learning-to-Rank (CV)':<35} {np.mean(cv_ndcgs):>10.4f} {np.mean(cv_spearmans):>10.4f} {np.mean(cv_top1s):>10.4f}")
    print(f"  {'Learning-to-Rank (full)':<35} {full_ndcg:>10.4f} {full_spearman:>10.4f} {full_top1_acc:>10.2%}")

    # Превосходство
    improvement = (np.mean(cv_ndcgs) - max(ndcg1, ndcg2, ndcg3)) / max(ndcg1, ndcg2, ndcg3) * 100
    print(f"\n  🚀 Улучшение NDCG@5 vs лучший baseline: **{improvement:+.1f}%**")

    # Feature importance
    print(f"\n{'=' * 70}")
    print("  ТОП-10 ФИЧЕЙ МОДЕЛИ")
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
    abs_w = np.abs(final_model.weights)
    sorted_idx = np.argsort(-abs_w)
    for i in range(min(10, len(sorted_idx))):
        idx = sorted_idx[i]
        sign = '+' if final_model.weights[idx] >= 0 else '−'
        bar = '█' * int(abs(final_model.weights[idx]) * 30)
        print(f"  {sign} {feature_names[idx]:<25} {final_model.weights[idx]:>8.4f}  {bar}")

    # Примеры предсказаний
    print(f"\n{'=' * 70}")
    print("  ПРИМЕРЫ ПРЕДСКАЗАНИЙ (первые 5 резюме)")
    print(f"{'=' * 70}")

    vac_titles = [v['title'] for v in vacancies]

    for cv_idx in range(5):
        g_mask = groups == cv_idx
        g_scores = all_scores[g_mask]
        g_consensus = consensus[cv_idx]

        sorted_idx = np.argsort(-g_scores)
        pred_order = [vac_titles[i] for i in sorted_idx]
        true_best = vac_titles[g_consensus.index(1)]
        pred_best = pred_order[0]

        cv_skills = ', '.join(list(extract_tech_skills(resumes[cv_idx]['text']).keys())[:5])
        print(f"\n  CV {cv_idx + 1} ({resumes[cv_idx]['id']}): [{cv_skills}...]")
        print(f"    🎯 HR лучший: {true_best}")
        print(f"    🤖 Наш лучший: {pred_best}")
        match = '✅' if true_best == pred_best else '❌'
        print(f"    {match} {'Совпало' if true_best == pred_best else 'Не совпало'}")

    return {
        'ndcg_cv': np.mean(cv_ndcgs),
        'spearman_cv': np.mean(cv_spearmans),
        'top1_cv': np.mean(cv_top1s),
        'ndcg_full': full_ndcg,
        'spearman_full': full_spearman,
        'top1_full': full_top1_acc,
        'baseline_char_ndcg': ndcg1,
        'baseline_sent_ndcg': ndcg2,
        'baseline_skill_ndcg': ndcg3,
    }


def main():
    """Точка входа."""
    args = sys.argv[1:]
    dataset_dir = None

    if '--dataset' in args:
        idx = args.index('--dataset')
        if idx + 1 < len(args):
            dataset_dir = args[idx + 1]

    if not dataset_dir:
        project_dir = Path(__file__).parent.parent
        possible_dir = project_dir / 'vacancy-resume-matching-dataset-main'
        if possible_dir.exists():
            dataset_dir = str(possible_dir)
        else:
            print("❌ Датасет не найден. Укажите: --dataset DIR")
            return

    run_final_demo(dataset_dir)


if __name__ == '__main__':
    main()
