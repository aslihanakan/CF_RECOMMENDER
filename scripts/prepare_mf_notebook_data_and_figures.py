"""
scripts/prepare_mf_notebook_data_and_figures.py
===============================================
GÖREV (Parça 4/4):
Tek süreçte (başka paralel iş çalışmıyorken, num_threads=1, BLAS=1):
  1. Veri yüklendikten hemen sonra (temiz bellek tabanında, Popularity ve Item-kNN ile
     birebir aynı protokolle) dondurulmuş ALS MF modelinin eğitim süresini ve 10.000
     validation kullanıcısı (seed=42, mode='live') üzerinde tam katalog recommend_topk
     süre/bellek (Başlangıç RSS, Tepe RSS, tracemalloc tepe) değerlerini ölçer.
  2. Ana karşılaştırma tablosu (results/mf_val_main_comparison.csv), popülerlik kovaları
     (results/mf_val_pop_buckets_4models.csv), yeni kullanıcı tercih formu sonuçları
     (results/mf_val_new_user_all_models.csv ve results/mf_val_new_user_paired_diffs.csv),
     5 örnek kullanıcı + 3 oyun tercih formu örneği ve figures/*.png grafiklerini üretir.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import tracemalloc
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis_utils import (
    N_CF_KNOWN_ITEMS,
    build_target_popularity_buckets,
    compute_subset_metrics,
    evaluate_with_user_records,
)
from src.contract import ContractAdapter
from src.data_loader import N_ITEMS, N_USERS, get_activity_groups, load_data
from src.evaluation import K, N_ITEMS_TOTAL
from src.models.item_knn import ItemKNNModel
from src.models.mf import ALSModel
from src.models.popularity import PopularityModel

FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"


def get_dir_size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def collect_recommended_items_and_timing(
    model_name: str,
    recommend_fn: Callable[[int], List[int]],
    sampled_uids: List[int],
    item_pop: np.ndarray,
    pool_coverage_pct: float,
    pool_coverage_n: int,
    k: int = K,
) -> Tuple[dict, np.ndarray]:
    """10.000 val kullanıcısı üzerinde tam katalog recommend_topk süre, bellek ve
    gerçek katalog kapsama metriklerini ölçer.
    """
    gc.collect()
    process = psutil.Process(os.getpid())
    rss_before_mb = process.memory_info().rss / (1024 * 1024)
    peak_rss_mb = rss_before_mb

    tracemalloc.start()
    tracemalloc.reset_peak()

    n_items = len(item_pop)
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

    stats = {
        "model": model_name,
        "n_sample_users": len(sampled_uids),
        "total_rec_slots": int(len(rec_arr)),
        "unique_rec_items": n_unique,
        "catalog_cov_25610_pct": round(catalog_cov_all_pct, 4),
        "catalog_cov_25367_known_pct": round(catalog_cov_known_pct, 4),
        "top100_pop_item_share_pct": round(top100_share_pct, 4),
        "mean_pop_percentile_rank": round(mean_pop_pct_rank, 4),
        "mean_rec_item_pop": round(mean_raw_pop, 2),
        "median_rec_item_pop": round(median_raw_pop, 2),
        "pool_coverage_pct": round(float(pool_coverage_pct), 4),
        "pool_coverage_n": int(pool_coverage_n),
        "elapsed_seconds": round(elapsed_s, 4),
        "ms_per_user": round((elapsed_s * 1000.0) / len(sampled_uids), 4),
        "rss_before_mb": round(rss_before_mb, 2),
        "peak_rss_mb": round(peak_rss_mb, 2),
        "tracemalloc_peak_mb": round(peak_traced_mb, 2),
    }
    return stats, rec_arr


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("PARÇA 4/4: TEK SÜREÇTE BELLEK/SÜRE, ÖRNEK ÖNERİ VE GRAFİK ÜRETİMİ")
    print("=" * 90)

    data = load_data(str(PROJECT_ROOT / "data"))
    interaction_matrix = data["interaction_matrix"]
    seen_matrix = data["seen_matrix"]
    val_candidates = data["val_candidates"]
    cold_items = data["cold_items"]
    user_activity = data["user_activity"]
    item_pop = np.asarray(interaction_matrix.sum(axis=0)).ravel()

    val_uids = list(val_candidates.keys())
    val_uids_set = set(val_uids)
    n_val = len(val_uids)
    uid_to_row = {uid: idx for idx, uid in enumerate(val_uids)}
    cands_2d = np.array([val_candidates[u] for u in val_uids], dtype=np.int64)
    targets_list = cands_2d[:, 0].tolist()

    rng_cov = np.random.default_rng(42)
    sampled_cov_uids = rng_cov.choice(val_uids, size=10000, replace=False).tolist()

    # ------------------------------------------------------------------
    # (d) TEK SÜREÇTE TEMİZ BELLEK TABANINDA EĞİTİM VE 10.000 KULLANICI SÜRE/BELLEK ÖLÇÜMÜ
    # ------------------------------------------------------------------
    print("\n[1] Tek süreçte (num_threads=1, BLAS=1) dondurulmuş MF eğitim süresi ve recommend_topk ölçülüyor...")
    gc.collect()
    with threadpool_limits(1, "blas"):
        mf_fresh = ALSModel(
            factors=32,
            regularization=0.01,
            alpha=150.0,
            iterations=50,
            random_state=42,
            num_threads=1,
            fold_in_method="ridge",
        )
        t0_fit = time.perf_counter()
        mf_fresh.fit(interaction_matrix, show_progress=False)
        single_proc_fit_seconds = time.perf_counter() - t0_fit

    mf_model = ALSModel.load(str(PROJECT_ROOT / "saved_models" / "mf"))
    assert np.array_equal(mf_fresh.user_factors, mf_model.user_factors)
    assert np.array_equal(mf_fresh.item_factors, mf_model.item_factors)
    del mf_fresh
    gc.collect()

    with threadpool_limits(1, "blas"):
        mf_cov_stats, mf_rec_items = collect_recommended_items_and_timing(
            model_name="ALS MF (f=32, α=150, it=50)",
            recommend_fn=lambda u: [
                i for i, _s in mf_model.recommend_topk(u, k=10, exclude_seen=True, seen_matrix=seen_matrix, mode="live")
            ],
            sampled_uids=sampled_cov_uids,
            item_pop=item_pop,
            pool_coverage_pct=89.7657,
            pool_coverage_n=22989,
            k=10,
        )

    print(
        f"    Tek süreç MF Eğitim Süresi : {single_proc_fit_seconds:.2f} sn\n"
        f"    10k recommend_topk Süresi  : {mf_cov_stats['elapsed_seconds']:.2f} sn ({mf_cov_stats['ms_per_user']:.3f} ms/kul)\n"
        f"    Başlangıç / Tepe RSS       : {mf_cov_stats['rss_before_mb']:.2f} MB / {mf_cov_stats['peak_rss_mb']:.2f} MB\n"
        f"    tracemalloc Tepe           : {mf_cov_stats['tracemalloc_peak_mb']:.2f} MB"
    )

    pop_disk_mb = get_dir_size_mb(PROJECT_ROOT / "saved_models" / "popularity.pkl")
    knn_disk_mb = get_dir_size_mb(PROJECT_ROOT / "saved_models" / "itemknn")
    mf_disk_mb = get_dir_size_mb(PROJECT_ROOT / "saved_models" / "mf")
    mf_factors_only_mb = (
        get_dir_size_mb(PROJECT_ROOT / "saved_models" / "mf" / "user_factors.npy")
        + get_dir_size_mb(PROJECT_ROOT / "saved_models" / "mf" / "item_factors.npy")
    )

    # Ana Tablo CSV'sini temiz tek-süreç süre/bellek değerleriyle güncelle
    df_main_table = pd.read_csv(RESULTS_DIR / "mf_val_main_comparison.csv")
    mf_mask = df_main_table["model"] == "ALS MF (f=32, α=150, it=50)"
    df_main_table.loc[mf_mask, "recommend_10k_seconds"] = round(float(mf_cov_stats["elapsed_seconds"]), 2)
    df_main_table.loc[mf_mask, "ms_per_user"] = round(float(mf_cov_stats["ms_per_user"]), 3)
    df_main_table.loc[mf_mask, "rss_before_mb"] = round(float(mf_cov_stats["rss_before_mb"]), 2)
    df_main_table.loc[mf_mask, "peak_rss_mb"] = round(float(mf_cov_stats["peak_rss_mb"]), 2)
    df_main_table.loc[mf_mask, "tracemalloc_peak_mb"] = round(float(mf_cov_stats["tracemalloc_peak_mb"]), 2)
    df_main_table.loc[mf_mask, "fit_seconds"] = round(float(single_proc_fit_seconds), 2)
    df_main_table.to_csv(RESULTS_DIR / "mf_val_main_comparison.csv", index=False)

    df_buckets_4m = pd.read_csv(RESULTS_DIR / "mf_val_pop_buckets_4models.csv")
    df_new_user = pd.read_csv(RESULTS_DIR / "mf_val_new_user_all_models.csv")
    df_paired = pd.read_csv(RESULTS_DIR / "mf_val_new_user_paired_diffs.csv")

    # Diğer modelleri yükle (kümülatif popülerlik eğrisi ve örnek öneriler için)
    pop_model = PopularityModel.load(str(PROJECT_ROOT / "saved_models" / "popularity.pkl"))
    knn_pw001 = ItemKNNModel.load(str(PROJECT_ROOT / "saved_models" / "itemknn"))
    knn_pw0 = ItemKNNModel.load(str(PROJECT_ROOT / "saved_models" / "itemknn"))
    knn_pw0.pop_weight = 0.0

    pop_rec_items_list = []
    knn0_rec_items_list = []
    knn001_rec_items_list = []
    for u in sampled_cov_uids:
        pop_rec_items_list.extend(
            pop_model.recommend_topk(
                u, interaction_matrix, k=10, exclude_seen=True, seen_matrix=seen_matrix, mode="live"
            ).tolist()
        )
        knn0_rec_items_list.extend(
            [i for i, _s in knn_pw0.recommend_topk(u, k=10, exclude_seen=True, seen_matrix=seen_matrix, mode="live")]
        )
        knn001_rec_items_list.extend(
            [i for i, _s in knn_pw001.recommend_topk(u, k=10, exclude_seen=True, seen_matrix=seen_matrix, mode="live")]
        )
    pop_rec_items = np.asarray(pop_rec_items_list, dtype=np.int64)
    knn0_rec_items = np.asarray(knn0_rec_items_list, dtype=np.int64)
    knn001_rec_items = np.asarray(knn001_rec_items_list, dtype=np.int64)

    # Soğuk ürün analizi
    cold_items_arr = np.array(sorted(cold_items), dtype=np.int64)
    cold_val_users = [u for u, c in val_candidates.items() if int(c[0]) in cold_items]
    cold_analysis = {
        "n_cold_items_total": len(cold_items),
        "n_cf_known_false_items_total": int(np.sum(~mf_model.item_cf_known)),
        "n_label0_only_items": int(np.sum(~mf_model.item_cf_known) - len(cold_items)),
        "n_val_users_with_cold_84_target": len(cold_val_users),
        "n_val_users_with_cf_known_false_243_target": int(df_buckets_4m.loc[0, "n_users"]),
        "mf_cold_84_factor_norm_sum": float(np.linalg.norm(mf_model.item_factors[cold_items_arr])),
        "mf_cold_84_cf_known_true_count": int(np.sum(mf_model.item_cf_known[cold_items_arr])),
        "mf_cold_114_users_hr_at_10": 0.0,
        "mf_cold_114_users_ndcg_at_10": 0.0,
        "mf_cold_263_bucket_hr_at_10": 0.0,
        "mf_cold_263_bucket_ndcg_at_10": 0.0,
    }

    # ------------------------------------------------------------------
    # (f) Örnek Öneriler: seed=42 ile 5 Val Kullanıcısı + 3 Oyun Tercih Formu
    # ------------------------------------------------------------------
    print("\n[2] Örnek öneriler (seed=42 ile 5 val kullanıcısı + 3 oyun tercih formu) hazırlanıyor...")
    items_df = pd.read_csv(
        PROJECT_ROOT / "data" / "items.csv",
        usecols=["item_idx", "parent_asin", "title_clean", "sub_category", "categories", "store"],
    ).set_index("item_idx")

    indptr = interaction_matrix.indptr
    indices = interaction_matrix.indices
    act_groups = get_activity_groups(user_activity)
    _df_b, bucket_summaries = build_target_popularity_buckets(val_candidates, item_pop)
    tail_user_set = next(b["user_set"] for b in bucket_summaries if b["bucket"] == "kuyruk")

    rng_ex = np.random.default_rng(42)
    low_pool = sorted(u for u in (act_groups["low"] & val_uids_set) if (indptr[u + 1] - indptr[u]) >= 2)
    mid_pool = sorted(u for u in (act_groups["medium"] & val_uids_set) if (indptr[u + 1] - indptr[u]) >= 4)
    high_pool = sorted(u for u in (act_groups["high"] & val_uids_set) if (indptr[u + 1] - indptr[u]) >= 7)
    tail_pool = sorted(u for u in (tail_user_set & act_groups["medium"] & val_uids_set) if (indptr[u + 1] - indptr[u]) >= 4)

    ex_u_low = int(rng_ex.choice(low_pool))
    ex_u_mid1 = int(rng_ex.choice(mid_pool))
    ex_u_tail = int(rng_ex.choice(tail_pool))
    ex_u_high1 = int(rng_ex.choice(high_pool))
    ex_u_high2 = int(rng_ex.choice([u for u in high_pool if u != ex_u_high1]))

    example_users_meta = [
        ("Kullanıcı 1 (Düşük Aktivite, =3)", ex_u_low),
        ("Kullanıcı 2 (Orta Aktivite, 4-6)", ex_u_mid1),
        ("Kullanıcı 3 (Orta Aktivite, Kuyruk Hedefli)", ex_u_tail),
        ("Kullanıcı 4 (Yüksek Aktivite, 7+)", ex_u_high1),
        ("Kullanıcı 5 (Yüksek Aktivite, 7+)", ex_u_high2),
    ]

    adapter_mf = ContractAdapter(
        model=mf_model,
        popularity_model=pop_model,
        seen_matrix=seen_matrix,
        interaction_matrix=interaction_matrix,
        n_users=N_USERS,
        n_items=N_ITEMS,
    )

    def _fmt_item(iid: int) -> dict:
        row_i = items_df.loc[int(iid)]
        return {
            "item_idx": int(iid),
            "title": str(row_i["title_clean"])[:75],
            "sub_category": str(row_i["sub_category"]),
            "categories": str(row_i["categories"])[:80],
            "store": str(row_i["store"]),
            "train_pos_pop": int(item_pop[int(iid)]),
        }

    examples_payload = []
    for role_label, uid in example_users_meta:
        u_liked = indices[indptr[uid] : indptr[uid + 1]].tolist()
        target_item = int(val_candidates[uid][0])

        mf_top10 = mf_model.recommend_topk(uid, k=10, exclude_seen=True, seen_matrix=seen_matrix, mode="live")
        knn_top10 = knn_pw001.recommend_topk(uid, k=10, exclude_seen=True, seen_matrix=seen_matrix, mode="live")

        # 100'lük val havuzunda hedef ürünün sırası
        u_cands = val_candidates[uid]
        s_mf_u = mf_model.score(uid, u_cands)
        s_knn_u = knn_pw001.score(uid, u_cands)
        rank_mf_pool = int(np.where(np.argsort(-s_mf_u) == 0)[0][0]) + 1
        rank_knn_pool = int(np.where(np.argsort(-s_knn_u) == 0)[0][0]) + 1

        examples_payload.append(
            {
                "label": role_label,
                "user_idx": uid,
                "total_train_interactions": int(user_activity.loc[uid]),
                "n_train_positives": len(u_liked),
                "val_target": _fmt_item(target_item),
                "val_target_rank_in_pool_mf": rank_mf_pool,
                "val_target_rank_in_pool_knn001": rank_knn_pool,
                "liked_items": [_fmt_item(i) for i in u_liked[:10]],
                "mf_top10_live": [
                    {**_fmt_item(i), "raw_score": round(float(s), 4)} for i, s in mf_top10
                ],
                "knn001_top10_live": [
                    {**_fmt_item(i), "raw_score": round(float(s), 4)} for i, s in knn_top10
                ],
            }
        )

    # 3 oyunluk tercih formu örneği
    form_seed_games = indices[indptr[ex_u_high1] : indptr[ex_u_high1 + 1]][:3].tolist()
    mf_form_top10 = mf_model.recommend_from_items(form_seed_games, k=10, method="ridge")
    knn_form_top10 = knn_pw001.recommend_from_items(form_seed_games, k=10)
    df_form_contract = adapter_mf.score_from_items(form_seed_games, val_candidates[ex_u_high1][:15])

    preference_form_example = {
        "source_user_idx": ex_u_high1,
        "selected_3_games": [_fmt_item(i) for i in form_seed_games],
        "mf_ridge_top10_catalog": [
            {**_fmt_item(i), "raw_score": round(float(s), 4)} for i, s in mf_form_top10
        ],
        "knn001_top10_catalog": [
            {**_fmt_item(i), "raw_score": round(float(s), 4)} for i, s in knn_form_top10
        ],
        "contract_adapter_top5": df_form_contract.sort_values("score", ascending=False).head(5).to_dict(orient="records"),
    }

    # ------------------------------------------------------------------
    # (g) GRAFİKLER (figures/*.png)
    # ------------------------------------------------------------------
    print("\n[3] Grafikler (figures/*.png) üretiliyor...")
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["font.size"] = 10

    # Grafik 1: Ana Metrik Çubukları
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    models_short = ["Popularity", "Item-kNN\n(pw=0)", "Item-kNN\n(pw=0.01)", "ALS MF\n(f=32, α=150)"]
    x = np.arange(len(models_short))
    w = 0.36
    hrs = df_main_table["hr_at_10"].to_numpy()
    ndcgs = df_main_table["ndcg_at_10"].to_numpy()
    b1 = axes[0].bar(x - w / 2, hrs, w, label="HR@10", color="#2563eb")
    b2 = axes[0].bar(x + w / 2, ndcgs, w, label="NDCG@10", color="#f59e0b")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models_short)
    axes[0].set_ylim(0, 0.80)
    axes[0].set_ylabel("Skor")
    axes[0].set_title("Tam Validation (N=75.799) HR@10 ve NDCG@10")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    for bar in list(b1) + list(b2):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.4f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    cat_covs = df_main_table["catalog_cov_at_10_pct"].to_numpy()
    top100_shares = df_main_table["top100_pop_share_pct"].to_numpy()
    b3 = axes[1].bar(x - w / 2, cat_covs, w, label="Gerçek Katalog Cov@10 (%)", color="#10b981")
    b4 = axes[1].bar(x + w / 2, top100_shares, w, label="İlk 100 Popüler Ürüne Pay (%)", color="#ef4444")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models_short)
    axes[1].set_ylim(0, 115)
    axes[1].set_ylabel("Yüzde (%)")
    axes[1].set_title("Gerçek Katalog Cov@10 ve İlk 100 Popüler Ürün Payı (N=10.000)")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    for bar in list(b3) + list(b4):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{bar.get_height():.2f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mf_01_main_metrics_bar.png", bbox_inches="tight")
    plt.close(fig)

    # Grafik 2: Popülerlik Kovası Bazlı HR@10 ve NDCG@10
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    b_labels = ["Soğuk\n(pop=0)", "Kuyruk\n(1..23)", "Orta\n(24..111)", "Baş\n(112..1901)"]
    xb = np.arange(len(b_labels))
    bw = 0.20
    colors = ["#64748b", "#10b981", "#f59e0b", "#2563eb"]
    cols_hr = ["pop_hr_at_10", "knn_pw0_hr_at_10", "knn_pw001_hr_at_10", "mf_hr_at_10"]
    cols_ndcg = ["pop_ndcg_at_10", "knn_pw0_ndcg_at_10", "knn_pw001_ndcg_at_10", "mf_ndcg_at_10"]
    m_names_leg = ["Popularity", "Item-kNN (pw=0)", "Item-kNN (pw=0.01)", "ALS MF (f=32, α=150)"]

    for idx_m, (col_h, col_n, m_lbl, clr) in enumerate(zip(cols_hr, cols_ndcg, m_names_leg, colors)):
        offset = (idx_m - 1.5) * bw
        axes[0].bar(xb + offset, df_buckets_4m[col_h].to_numpy(), bw, label=m_lbl, color=clr)
        axes[1].bar(xb + offset, df_buckets_4m[col_n].to_numpy(), bw, label=m_lbl, color=clr)

    axes[0].set_xticks(xb)
    axes[0].set_xticklabels(b_labels)
    axes[0].set_ylim(0, 1.12)
    axes[0].set_ylabel("HR@10")
    axes[0].set_title("Hedef Ürün Popülerlik Kovasına Göre HR@10")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].set_xticks(xb)
    axes[1].set_xticklabels(b_labels)
    axes[1].set_ylim(0, 1.00)
    axes[1].set_ylabel("NDCG@10")
    axes[1].set_title("Hedef Ürün Popülerlik Kovasına Göre NDCG@10")
    axes[1].legend(fontsize=8.5)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mf_02_pop_buckets_hr_ndcg.png", bbox_inches="tight")
    plt.close(fig)

    # Grafik 3: NDCG@10 vs Gerçek Katalog Coverage@10 Ödünleşimi
    df_trade = pd.read_csv(RESULTS_DIR / "mf_val_coverage_tradeoff.csv")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    pts = [
        ("Popularity (0.06%)", 0.06, 0.3369, "#64748b", "s"),
        ("Item-kNN pw=0 (50.85%)", 50.85, 0.4210, "#10b981", "^"),
        ("Item-kNN pw=0.01 (16.68%)", 16.68, 0.4520, "#f59e0b", "D"),
    ]
    for lbl, cx, cy, clr, mk in pts:
        ax.scatter(cx, cy, s=110, color=clr, marker=mk, label=lbl, zorder=4)
        ax.annotate(lbl, (cx, cy), xytext=(6, -4), textcoords="offset points", fontsize=8.5)

    als_cx = df_trade["catalog_cov_25610_pct"].to_numpy()
    als_cy = df_trade["val_ndcg_at_10"].to_numpy()
    ax.plot(als_cx, als_cy, "--", color="#2563eb", alpha=0.6, zorder=3)
    ax.scatter(als_cx, als_cy, s=110, color="#2563eb", marker="o", label="ALS Konfigürasyonları (a, b, c1-c3)", zorder=5)
    short_als_lbls = [
        "(a) ALS f=64, α=10 (4.88%)",
        "(b) ALS f=32, α=60, it=30 (5.98%)",
        "(c1) Kazanan ALS f=32, α=150, it=50 (8.26%)",
        "(c2) ALS f=32, reg=0.1, α=150 (8.34%)",
        "(c3) ALS f=32, α=150, it=30 (8.34%)",
    ]
    offsets_als = [(6, -10), (6, -8), (6, 5), (6, -12), (-145, -14)]
    for (cx, cy), s_lbl, off in zip(zip(als_cx, als_cy), short_als_lbls, offsets_als):
        ax.annotate(s_lbl, (cx, cy), xytext=off, textcoords="offset points", fontsize=8)

    ax.set_xlabel("Gerçek Katalog Coverage@10 (%, N=10.000, tüm katalog=25.610)")
    ax.set_ylabel("Tam Validation NDCG@10 (N=75.799)")
    ax.set_title("Validation NDCG@10 ve Gerçek Katalog Coverage@10 İlişkisi")
    ax.set_xlim(-2, 60)
    ax.set_ylim(0.32, 0.525)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mf_03_ndcg_vs_catalog_coverage.png", bbox_inches="tight")
    plt.close(fig)

    # Grafik 4: Önerilen Ürünlerin Popülerlik Yüzdeliğine Göre Kümülatif Payı (CDF)
    pop_percentile_all = (rankdata(item_pop, method="average") / float(len(item_pop))) * 100.0
    thresholds = np.linspace(50.0, 100.0, 201)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for lbl, arr_items, clr, ls in [
        ("Popularity", pop_rec_items, "#64748b", ":"),
        ("Item-kNN (pw=0.01)", knn001_rec_items, "#f59e0b", "-."),
        ("ALS MF (f=32, α=150, it=50)", mf_rec_items, "#2563eb", "-"),
        ("Item-kNN (pw=0)", knn0_rec_items, "#10b981", "--"),
    ]:
        pcts = pop_percentile_all[arr_items]
        cdf = np.searchsorted(np.sort(pcts), thresholds, side="right") / float(len(pcts)) * 100.0
        ax.plot(thresholds, cdf, label=lbl, color=clr, linewidth=2.2, linestyle=ls)

    ax.set_xlabel("Ürün Popülerlik Yüzdelik Dilimi (Eğitim Pozitif Sayısına Göre, %)")
    ax.set_ylabel("Kümülatif Öneri Payı (%)")
    ax.set_title("Önerilen Ürünlerin Popülerlik Yüzdeliğine Göre Kümülatif Dağılımı (N=10.000 x Top-10)")
    ax.set_xlim(50, 100)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mf_04_cumulative_pop_percentile.png", bbox_inches="tight")
    plt.close(fig)

    # Grafik 5: Grid Isı Haritası (alpha x factors, Stage 1-3 CSV'lerinden)
    df_s1 = pd.read_csv(RESULTS_DIR / "mf_val_grid_stage1.csv")
    df_s2 = pd.read_csv(RESULTS_DIR / "mf_val_grid_stage2.csv")
    df_s3 = pd.read_csv(RESULTS_DIR / "mf_val_grid_stage3.csv")
    df_grid_all = pd.concat([df_s1, df_s2, df_s3], ignore_index=True)

    piv_ndcg = df_grid_all.groupby(["factors", "alpha"])["ndcg_at_10"].max().unstack("alpha")
    piv_hr = df_grid_all.groupby(["factors", "alpha"])["hr_at_10"].max().unstack("alpha")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for ax_i, (piv, title_str, cmap_name) in enumerate(
        [
            (piv_ndcg, "Validation NDCG@10 (En İyi reg/iter)", "YlGnBu"),
            (piv_hr, "Validation HR@10 (En İyi reg/iter)", "YlOrRd"),
        ]
    ):
        im = axes[ax_i].imshow(piv.values, aspect="auto", cmap=cmap_name)
        axes[ax_i].set_xticks(np.arange(len(piv.columns)))
        axes[ax_i].set_xticklabels([f"{a:g}" for a in piv.columns])
        axes[ax_i].set_yticks(np.arange(len(piv.index)))
        axes[ax_i].set_yticklabels([f"{f}" for f in piv.index])
        axes[ax_i].set_xlabel("alpha (Güven Ağırlığı Katsayısı)")
        axes[ax_i].set_ylabel("factors (Gizli Faktör Boyutu)")
        axes[ax_i].set_title(title_str)
        for r_i in range(piv.shape[0]):
            for c_i in range(piv.shape[1]):
                val_v = piv.values[r_i, c_i]
                if np.isfinite(val_v):
                    axes[ax_i].text(
                        c_i,
                        r_i,
                        f"{val_v:.4f}",
                        ha="center",
                        va="center",
                        fontsize=8.5,
                        color="black" if val_v < np.nanmean(piv.values) else "white",
                        fontweight="bold",
                    )
        fig.colorbar(im, ax=axes[ax_i], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mf_05_grid_heatmap_alpha_factors.png", bbox_inches="tight")
    plt.close(fig)

    bundle = {
        "main_comparison_table": df_main_table.to_dict(orient="records"),
        "pop_buckets_4models": df_buckets_4m.to_dict(orient="records"),
        "cold_item_analysis": cold_analysis,
        "timing_and_memory_mf": {
            **mf_cov_stats,
            "single_process_fit_seconds": round(single_proc_fit_seconds, 2),
            "mf_total_disk_mb": round(mf_disk_mb, 2),
            "mf_factors_only_disk_mb": round(mf_factors_only_mb, 2),
            "itemknn_disk_mb": round(knn_disk_mb, 2),
            "popularity_disk_mb": round(pop_disk_mb, 2),
        },
        "new_user_all_models": df_new_user.to_dict(orient="records"),
        "new_user_paired_diffs": df_paired.to_dict(orient="records"),
        "example_users": examples_payload,
        "preference_form_example": preference_form_example,
    }
    bundle_path = RESULTS_DIR / "mf_notebook_bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[4] Tüm veriler ve grafikler hazırlandı: {bundle_path}")


if __name__ == "__main__":
    main()
