"""
src/analysis_utils.py
=====================
Validation analizleri (Adım 1-4: popülerlik kovaları, kazanç payları ve
gerçek katalog Coverage@10 ölçümleri) için yardımcı fonksiyonlar.

NOT: Mevcut src/evaluation.py, model dosyaları ve sözleşme dosyaları değiştirilmez.
Bu modül, src/evaluation.py içindeki evaluate() ile birebir aynı sıralama ve
tie-break mantığını (TIE_BREAK_SEED = 42, COLD_ITEM_PENALTY_DEFAULT = -1e9)
kullanarak kullanıcı bazlı sonuçları kaydeder ve keyfi kullanıcı alt kümeleri
(popülerlik kovaları, aktivite grupları vb.) üzerinde metrik hesaplar.
"""

from __future__ import annotations

import os
import time
import tracemalloc
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
from scipy.sparse import csr_matrix
from scipy.stats import rankdata

from src.data_loader import N_ITEMS, get_activity_groups
from src.evaluation import (
    COLD_ITEM_PENALTY_DEFAULT,
    K,
    N_ITEMS_TOTAL,
    TIE_BREAK_SEED,
    _compute_metrics,
)

N_CF_KNOWN_ITEMS = 25_367  # train'de en az 1 pozitif etkileşimi olan ürün sayısı


def evaluate_with_user_records(
    score_fn: Callable[[int, np.ndarray], np.ndarray],
    candidates: Dict[int, np.ndarray],
    cold_items: Optional[set] = None,
    user_activity: Optional[pd.Series] = None,
    cold_item_penalty: float = COLD_ITEM_PENALTY_DEFAULT,
) -> dict:
    """src/evaluation.py::evaluate() ile birebir aynı tie-break ve sıralama akışını
    çalıştırır; ek olarak kullanıcı bazlı hit_ranks ve top_k_per_user sözlüklerini
    döndürür. Böylece popülerlik kovaları gibi alt kümelerde yeniden skorlama
    yapmadan ve RNG sırasını bozmadan metrik hesaplanabilir.
    """
    hit_ranks: Dict[int, Optional[int]] = {}
    cold_users: set = set()
    top_k_per_user: Dict[int, List[int]] = {}

    rng = np.random.default_rng(TIE_BREAK_SEED)
    cold_items_list = list(cold_items) if cold_items is not None else None

    for uid, cands in candidates.items():
        pos_item = cands[0]
        if cold_items is not None and pos_item in cold_items:
            cold_users.add(uid)

        scores = score_fn(uid, cands).astype(np.float64)

        if cold_items_list is not None:
            cold_mask = np.isin(cands, cold_items_list)
            scores[cold_mask] = cold_item_penalty

        tiebreak_noise = rng.random(len(scores)) * 1e-10
        scores_with_noise = scores + tiebreak_noise
        sorted_indices = np.argsort(-scores_with_noise)

        rank = int(np.where(sorted_indices == 0)[0][0]) + 1
        hit_ranks[uid] = rank if rank <= K else None

        top_k_indices = sorted_indices[:K]
        top_k_per_user[uid] = cands[top_k_indices].tolist()

    overall_metrics = _compute_metrics(hit_ranks, top_k_per_user, K)
    cold_hit_ranks = {u: r for u, r in hit_ranks.items() if u in cold_users}
    cold_top_k = {u: t for u, t in top_k_per_user.items() if u in cold_users}
    cold_metrics = _compute_metrics(cold_hit_ranks, cold_top_k, K) if cold_users else {}

    activity_metrics = {}
    if user_activity is not None:
        groups = get_activity_groups(user_activity)
        for name, gset in groups.items():
            group_hit_ranks = {u: r for u, r in hit_ranks.items() if u in gset}
            group_top_k = {u: t for u, t in top_k_per_user.items() if u in gset}
            activity_metrics[name] = _compute_metrics(group_hit_ranks, group_top_k, K)

    return {
        "overall": overall_metrics,
        "cold_user": cold_metrics,
        "activity": activity_metrics,
        "hit_ranks": hit_ranks,
        "top_k_per_user": top_k_per_user,
    }


def compute_subset_metrics(
    user_records: dict,
    user_subset: Iterable[int],
    k: int = K,
) -> dict:
    """evaluate_with_user_records çıktısından belirli bir kullanıcı alt kümesi için
    HR@K, NDCG@K, Precision@K, Recall@K, Coverage@K ve toplam hit sayısını hesaplar.
    """
    subset_set = set(int(u) for u in user_subset)
    hit_ranks = user_records["hit_ranks"]
    top_k_per_user = user_records["top_k_per_user"]

    sub_hit_ranks = {u: r for u, r in hit_ranks.items() if u in subset_set}
    sub_top_k = {u: t for u, t in top_k_per_user.items() if u in subset_set}

    metrics = _compute_metrics(sub_hit_ranks, sub_top_k, k)
    metrics["hits"] = int(sum(1 for r in sub_hit_ranks.values() if r is not None))
    return metrics


def make_random_score_fn(seed: int = 42) -> Callable[[int, np.ndarray], np.ndarray]:
    """Aday havuzunda sabit tohumlu (seed=42) rastgele skor üreten fonksiyon döndürür.

    Her çağrıda (uid, seed) çiftinden deterministik bir RNG türetilir; böylece
    evaluate() ve evaluate_with_user_records() çağrılarında birebir aynı rastgele
    skorlar üretilir.
    """
    base_seed = int(seed)

    def _score_fn(uid: int, cands: np.ndarray) -> np.ndarray:
        user_rng = np.random.default_rng((base_seed, int(uid)))
        return user_rng.random(len(cands), dtype=np.float64)

    return _score_fn


def build_target_popularity_buckets(
    val_candidates: Dict[int, np.ndarray],
    item_pop: np.ndarray,
) -> Tuple[pd.DataFrame, List[dict]]:
    """Val hedef ürünlerini (val_candidates[uid][0]) train pozitif sayısına göre
    4 kovaya ayırır:
      1. 'soğuk' : pozitif sayısı == 0
      2. 'kuyruk': pozitif sayısı > 0 olan kullanıcıların alt 1/3 dilimi
      3. 'orta'  : pozitif sayısı > 0 olan kullanıcıların orta 1/3 dilimi
      4. 'baş'   : pozitif sayısı > 0 olan kullanıcıların üst 1/3 dilimi

    Tam sayı popülerlik değerlerindeki sınır beraberliklerinde:
      - Önce pd.qcut (veya yüzdelik sınırlar: 33.33% ve 66.67%) kontrol edilir.
      - Hem doğal tam sayı eşikli tercile (q1, q2 kuantil sınırlarıyla: pop <= q1,
        q1 < pop <= q2, pop > q2) hem de (target_pop, user_idx) sıralı tam eşit
        bölünme bilgileri hesaplanır.
      - Kuantil eşikleri (q1=23, q2=111) tam sayı sınırlarını koruduğu için aynı
        popülerlikteki ürünleri farklı kovalara bölmez ve üç kovada yaklaşık eşit
        kullanıcı sayısı (~1/3) sağlar.
    """
    rows = []
    for uid, cands in val_candidates.items():
        target_item = int(cands[0])
        pop = int(item_pop[target_item])
        rows.append({"user_idx": int(uid), "target_item_idx": target_item, "target_pop": pop})

    df = pd.DataFrame(rows)
    pos_mask = df["target_pop"] > 0
    pos_pops = df.loc[pos_mask, "target_pop"].to_numpy()

    # Tercile kuantil sınırları (pozitif hedefi olan kullanıcılar üzerinden)
    q1, q2 = np.quantile(pos_pops, [1.0 / 3.0, 2.0 / 3.0])
    q1_int = int(np.floor(q1))
    q2_int = int(np.floor(q2))

    buckets = np.empty(len(df), dtype=object)
    buckets[~pos_mask] = "soğuk"
    buckets[pos_mask & (df["target_pop"] <= q1_int)] = "kuyruk"
    buckets[pos_mask & (df["target_pop"] > q1_int) & (df["target_pop"] <= q2_int)] = "orta"
    buckets[pos_mask & (df["target_pop"] > q2_int)] = "baş"
    df["bucket"] = buckets

    n_total = len(df)
    bucket_order = ["soğuk", "kuyruk", "orta", "baş"]
    bucket_summaries = []

    for b_name in bucket_order:
        sub = df[df["bucket"] == b_name]
        n_u = len(sub)
        n_unique_items = int(sub["target_item_idx"].nunique())
        min_pop = int(sub["target_pop"].min()) if n_u > 0 else 0
        max_pop = int(sub["target_pop"].max()) if n_u > 0 else 0
        median_pop = float(sub["target_pop"].median()) if n_u > 0 else 0.0
        mean_pop = float(sub["target_pop"].mean()) if n_u > 0 else 0.0
        bucket_summaries.append(
            {
                "bucket": b_name,
                "n_users": n_u,
                "user_share_pct": (n_u / n_total) * 100.0,
                "n_unique_target_items": n_unique_items,
                "min_pop": min_pop,
                "max_pop": max_pop,
                "median_pop": median_pop,
                "mean_pop": mean_pop,
                "user_set": set(sub["user_idx"].tolist()),
            }
        )

    return df, bucket_summaries


def measure_catalog_coverage_topk(
    model_name: str,
    recommend_fn: Callable[[int], List[int]],
    sampled_uids: List[int],
    item_pop: np.ndarray,
    pool_coverage_pct: float,
    pool_coverage_n: int,
    k: int = K,
) -> dict:
    """Verilen modelin recommend_topk fonksiyonunu örneklem kullanıcılar üzerinde
    (mode='live', seen_matrix filtreli, tüm katalog = 25.610 ürün) çalıştırır ve
    gerçek katalog kapsama / popülerlik yoğunlaşma metriklerini ölçer.
    """
    process = psutil.Process(os.getpid())
    rss_before_mb = process.memory_info().rss / (1024 * 1024)
    peak_rss_mb = rss_before_mb

    tracemalloc.start()
    tracemalloc.reset_peak()

    n_items = len(item_pop)
    # Popülerlik yüzdelik sırası: en popüler ürünler ~100.0 yüzdelik dilimde
    pop_percentile = (rankdata(item_pop, method="average") / float(n_items)) * 100.0
    top100_popular_items = set(np.argsort(-item_pop)[:100].tolist())

    all_recommended: List[int] = []
    t0 = time.perf_counter()

    for idx, uid in enumerate(sampled_uids):
        recs = recommend_fn(int(uid))
        all_recommended.extend(recs[:k])
        if (idx + 1) % 2000 == 0 or (idx + 1) == len(sampled_uids):
            curr_rss = process.memory_info().rss / (1024 * 1024)
            if curr_rss > peak_rss_mb:
                peak_rss_mb = curr_rss

    elapsed_s = time.perf_counter() - t0
    _curr_traced, peak_traced_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_traced_mb = peak_traced_bytes / (1024 * 1024)

    rec_arr = np.asarray(all_recommended, dtype=np.int64)
    unique_items = np.unique(rec_arr)
    n_unique = int(len(unique_items))

    catalog_cov_all_pct = (n_unique / float(N_ITEMS_TOTAL)) * 100.0
    catalog_cov_known_pct = (n_unique / float(N_CF_KNOWN_ITEMS)) * 100.0

    in_top100_mask = np.isin(rec_arr, list(top100_popular_items))
    top100_share_pct = float(np.mean(in_top100_mask) * 100.0) if len(rec_arr) > 0 else 0.0
    mean_pop_pct_rank = float(np.mean(pop_percentile[rec_arr])) if len(rec_arr) > 0 else 0.0
    mean_raw_pop = float(np.mean(item_pop[rec_arr])) if len(rec_arr) > 0 else 0.0
    median_raw_pop = float(np.median(item_pop[rec_arr])) if len(rec_arr) > 0 else 0.0

    return {
        "model": model_name,
        "n_sample_users": len(sampled_uids),
        "total_rec_slots": int(len(rec_arr)),
        "unique_rec_items": n_unique,
        "catalog_cov_25610_pct": catalog_cov_all_pct,
        "catalog_cov_25367_known_pct": catalog_cov_known_pct,
        "top100_pop_item_share_pct": top100_share_pct,
        "mean_pop_percentile_rank": mean_pop_pct_rank,
        "mean_rec_item_pop": mean_raw_pop,
        "median_rec_item_pop": median_raw_pop,
        "pool_coverage_pct": float(pool_coverage_pct),
        "pool_coverage_n": int(pool_coverage_n),
        "elapsed_seconds": round(elapsed_s, 4),
        "ms_per_user": round((elapsed_s * 1000.0) / len(sampled_uids), 4),
        "rss_before_mb": round(rss_before_mb, 2),
        "peak_rss_mb": round(peak_rss_mb, 2),
        "tracemalloc_peak_mb": round(peak_traced_mb, 2),
    }
