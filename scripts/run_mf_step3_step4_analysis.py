"""
scripts/run_mf_step3_step4_analysis.py
======================================
GÖREV (Parça 3/4) - Adım 3, Adım 4 ve Adım 5 (Model Kaydı):
  - Seçilen en iyi ALS konfigürasyonu:
      factors = 32, regularization = 0.01, alpha = 60.0, iterations = 30, seed = 42
  - Adım 3:
      a) Tam val metrikleri (HR@10, NDCG@10, Precision@10, Recall@10, Havuz Coverage@10,
         aktivite kırılımları)
      b) Popülerlik kovalarına göre HR@10 / NDCG@10 (soğuk, kuyruk, orta, baş) -
         src/analysis_utils.py ve TIE_BREAK_SEED=42 ile
      c) Gerçek katalog Coverage@10 (10.000 val kullanıcısı, seed=42, mode='live'),
         benzersiz önerilen ürün sayısı, en popüler 100 ürüne giden pay
      d) Stage 1 grid sonuçlarından alpha, regularization ve factors eğilim tabloları
  - Adım 4:
      Seçilen hiperparametrelerle fold_in_method='ridge' ve 'mean' karşılaştırması:
      val, seed=42, 5.000 kullanıcı, 3, 4 ve 5 oyun seçimi (ContractAdapter.score_from_items).
      Kullanıcı bazlı eşleştirilmiş standart hata (paired SE) ve t-istatistiği hesabı.
  - Adım 5 (Kayıt):
      En iyi modeli saved_models/mf/ altına kaydeder ve kaydet-yükle sonrası skorların
      birebir aynı olduğunu doğrular.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis_utils import (
    build_target_popularity_buckets,
    compute_subset_metrics,
    evaluate_with_user_records,
    measure_catalog_coverage_topk,
)
from src.contract import ContractAdapter
from src.data_loader import N_ITEMS, N_USERS, load_data
from src.evaluation import K, TIE_BREAK_SEED, evaluate
from src.models.mf import ALSModel
from src.models.popularity import PopularityModel

BEST_PARAMS = {
    "factors": 32,
    "regularization": 0.01,
    "alpha": 60.0,
    "iterations": 30,
    "random_state": 42,
    "num_threads": 1,
}


def compute_paired_stats(
    records_a: dict,
    records_b: dict,
    uids: List[int],
    k: int = K,
) -> Dict[str, float]:
    """İki modelin (A='ridge', B='mean') kullanıcı bazlı hit ve ndcg dizileri
    arasındaki eşleştirilmiş fark, standart hata (SE) ve z-skorunu hesaplar.
    """
    hr_a = np.array(
        [1.0 if records_a["hit_ranks"].get(u) is not None else 0.0 for u in uids],
        dtype=np.float64,
    )
    hr_b = np.array(
        [1.0 if records_b["hit_ranks"].get(u) is not None else 0.0 for u in uids],
        dtype=np.float64,
    )
    ndcg_a = np.array(
        [
            1.0 / np.log2(records_a["hit_ranks"][u] + 1)
            if records_a["hit_ranks"].get(u) is not None
            else 0.0
            for u in uids
        ],
        dtype=np.float64,
    )
    ndcg_b = np.array(
        [
            1.0 / np.log2(records_b["hit_ranks"][u] + 1)
            if records_b["hit_ranks"].get(u) is not None
            else 0.0
            for u in uids
        ],
        dtype=np.float64,
    )

    n = len(uids)
    diff_hr = hr_a - hr_b
    diff_ndcg = ndcg_a - ndcg_b

    mean_diff_hr = float(np.mean(diff_hr))
    se_diff_hr = float(np.std(diff_hr, ddof=1) / np.sqrt(n))
    z_hr = mean_diff_hr / se_diff_hr if se_diff_hr > 0 else 0.0

    mean_diff_ndcg = float(np.mean(diff_ndcg))
    se_diff_ndcg = float(np.std(diff_ndcg, ddof=1) / np.sqrt(n))
    z_ndcg = mean_diff_ndcg / se_diff_ndcg if se_diff_ndcg > 0 else 0.0

    return {
        "n_users": n,
        "ridge_hr_at_10": float(np.mean(hr_a)),
        "mean_hr_at_10": float(np.mean(hr_b)),
        "diff_hr_ridge_minus_mean": mean_diff_hr,
        "se_diff_hr": se_diff_hr,
        "z_stat_hr": z_hr,
        "ridge_ndcg_at_10": float(np.mean(ndcg_a)),
        "mean_ndcg_at_10": float(np.mean(ndcg_b)),
        "diff_ndcg_ridge_minus_mean": mean_diff_ndcg,
        "se_diff_ndcg": se_diff_ndcg,
        "z_stat_ndcg": z_ndcg,
    }


def main() -> None:
    print("=" * 90)
    print("ADIM 3, 4 VE 5: SEÇİLEN ALS KONFİGÜRASYONU ANALİZİ, FOLD-IN VE KAYIT")
    print("=" * 90)

    data = load_data(str(PROJECT_ROOT / "data"))
    interaction_matrix = data["interaction_matrix"]
    seen_matrix = data["seen_matrix"]
    val_candidates = data["val_candidates"]
    cold_items = data["cold_items"]
    user_activity = data["user_activity"]
    item_pop = np.asarray(interaction_matrix.sum(axis=0)).ravel()

    pop_model = PopularityModel.load(str(PROJECT_ROOT / "saved_models" / "popularity.pkl"))

    # ------------------------------------------------------------------
    # 1. Seçilen Konfigürasyonu Eğit
    # ------------------------------------------------------------------
    print(f"\n[1] Seçilen model eğitiliyor: {BEST_PARAMS}")
    with threadpool_limits(1, "blas"):
        als_model = ALSModel(**BEST_PARAMS)
        als_model.fit(interaction_matrix, show_progress=False)
    print(f"    Eğitim tamamlandı: {als_model.fit_seconds:.2f} sn")

    adapter = ContractAdapter(
        model=als_model,
        popularity_model=pop_model,
        seen_matrix=seen_matrix,
        interaction_matrix=interaction_matrix,
        n_users=N_USERS,
        n_items=N_ITEMS,
    )

    val_uids = list(val_candidates.keys())
    n_val = len(val_uids)
    uid_to_row = {uid: idx for idx, uid in enumerate(val_uids)}
    cands_2d = np.array([val_candidates[u] for u in val_uids], dtype=np.int64)
    targets_list = cands_2d[:, 0].tolist()

    # ------------------------------------------------------------------
    # ADIM 3a: Tam Val Metrikleri (ContractAdapter + evaluate_with_user_records)
    # ------------------------------------------------------------------
    print("\n[2] Adım 3a: Tam Validation (75.799 kullanıcı) ContractAdapter skorları hesaplanıyor...")
    t0_eval = time.perf_counter()
    adapter_score_matrix = np.empty((n_val, 100), dtype=np.float64)
    chunk_size = 15000
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for start_idx in range(0, n_val, chunk_size):
            end_idx = min(start_idx + chunk_size, n_val)
            df_chunk = adapter.score(
                users=val_uids[start_idx:end_idx],
                candidates=cands_2d[start_idx:end_idx],
                mode="offline",
                target_items=targets_list[start_idx:end_idx],
            )
            adapter_score_matrix[start_idx:end_idx, :] = (
                df_chunk["score"].to_numpy(dtype=np.float64).reshape(end_idx - start_idx, 100)
            )

    def adapter_score_fn(uid: int, _cands: np.ndarray) -> np.ndarray:
        return adapter_score_matrix[uid_to_row[uid]]

    val_records = evaluate_with_user_records(
        score_fn=adapter_score_fn,
        candidates=val_candidates,
        cold_items=cold_items,
        user_activity=user_activity,
    )
    eval_elapsed = time.perf_counter() - t0_eval
    ov = val_records["overall"]
    print(
        f"    Tam Val Sonucu ({eval_elapsed:.2f} sn): "
        f"HR@10={ov['hit_rate']:.4f}, NDCG@10={ov['ndcg']:.4f}, "
        f"Prec@10={ov['precision']:.4f}, Recall@10={ov['recall']:.4f}, "
        f"Pool Cov@10={ov['coverage_pct']:.2f}% ({ov['coverage_n']:,} ürün)"
    )

    # ------------------------------------------------------------------
    # ADIM 3b: Popülerlik Kovalarına Göre HR@10 ve NDCG@10
    # ------------------------------------------------------------------
    print("\n[3] Adım 3b: Popülerlik kovalarına göre HR@10 ve NDCG@10 hesaplanıyor...")
    _df_buckets, bucket_summaries = build_target_popularity_buckets(val_candidates, item_pop)
    bucket_results = []
    for b_info in bucket_summaries:
        b_name = b_info["bucket"]
        b_metrics = compute_subset_metrics(val_records, b_info["user_set"])
        row_b = {
            "bucket": b_name,
            "n_users": b_info["n_users"],
            "user_share_pct": round(b_info["user_share_pct"], 2),
            "min_pop": b_info["min_pop"],
            "max_pop": b_info["max_pop"],
            "hr_at_10": round(b_metrics["hit_rate"], 4),
            "ndcg_at_10": round(b_metrics["ndcg"], 4),
            "coverage_pct": round(b_metrics["coverage_pct"], 2),
            "hits": b_metrics["hits"],
        }
        bucket_results.append(row_b)
        print(
            f"    Kova '{b_name:6s}' (N={b_info['n_users']:6,d}, pop=[{b_info['min_pop']}..{b_info['max_pop']}]) -> "
            f"HR@10={b_metrics['hit_rate']:.4f}, NDCG@10={b_metrics['ndcg']:.4f}, Hits={b_metrics['hits']:,}"
        )

    # ------------------------------------------------------------------
    # ADIM 3c: Gerçek Katalog Coverage@10 (10.000 Val Kullanıcısı, seed=42, mode='live')
    # ------------------------------------------------------------------
    print("\n[4] Adım 3c: Gerçek katalog Coverage@10 (10.000 val kullanıcısı, seed=42, mode='live')...")
    rng_cov = np.random.default_rng(42)
    sampled_cov_uids = rng_cov.choice(val_uids, size=10000, replace=False).tolist()

    def als_recommend_live(uid: int) -> List[int]:
        recs = als_model.recommend_topk(
            uid=uid,
            k=10,
            exclude_seen=True,
            seen_matrix=seen_matrix,
            mode="live",
        )
        return [item_id for item_id, _score in recs]

    with threadpool_limits(1, "blas"):
        catalog_cov_stats = measure_catalog_coverage_topk(
            model_name="ALS (f=32, reg=0.01, alpha=60, iter=30)",
            recommend_fn=als_recommend_live,
            sampled_uids=sampled_cov_uids,
            item_pop=item_pop,
            pool_coverage_pct=ov["coverage_pct"],
            pool_coverage_n=ov["coverage_n"],
            k=10,
        )
    print(
        f"    Gerçek Katalog Coverage@10: {catalog_cov_stats['catalog_cov_25610_pct']:.2f}% "
        f"({catalog_cov_stats['unique_rec_items']:,} / 25.610 ürün) | "
        f"Bilinen Katalog: {catalog_cov_stats['catalog_cov_25367_known_pct']:.2f}% | "
        f"Top-100 Popüler Payı: {catalog_cov_stats['top100_pop_item_share_pct']:.2f}% | "
        f"Süre: {catalog_cov_stats['elapsed_seconds']:.2f} sn"
    )

    # ------------------------------------------------------------------
    # ADIM 3d: Stage 1 Grid Eğilim Tabloları (alpha, regularization, factors)
    # ------------------------------------------------------------------
    print("\n[5] Adım 3d: Stage 1 Grid eğilim özetleri hazırlanıyor...")
    df_s1 = pd.read_csv(PROJECT_ROOT / "results" / "mf_val_grid_stage1.csv")
    trend_by_alpha = (
        df_s1.groupby("alpha")[["hr_at_10", "ndcg_at_10", "pool_coverage_at_10_pct", "fit_seconds"]]
        .mean()
        .reset_index()
    )
    trend_by_factors = (
        df_s1.groupby("factors")[["hr_at_10", "ndcg_at_10", "pool_coverage_at_10_pct", "fit_seconds"]]
        .mean()
        .reset_index()
    )
    trend_by_reg = (
        df_s1.groupby("regularization")[["hr_at_10", "ndcg_at_10", "pool_coverage_at_10_pct", "fit_seconds"]]
        .mean()
        .reset_index()
    )
    trend_alpha_factors = (
        df_s1.groupby(["factors", "alpha"])[["hr_at_10", "ndcg_at_10", "pool_coverage_at_10_pct"]]
        .mean()
        .reset_index()
    )

    print("\n  [Alpha Eğilimi (Stage 1 Ortalaması)]:")
    print(trend_by_alpha.to_string(index=False))
    print("\n  [Factors Eğilimi (Stage 1 Ortalaması)]:")
    print(trend_by_factors.to_string(index=False))
    print("\n  [Regularization Eğilimi (Stage 1 Ortalaması)]:")
    print(trend_by_reg.to_string(index=False))

    # ------------------------------------------------------------------
    # ADIM 4: Fold-in Yöntemi Karşılaştırması (3, 4 ve 5 Oyun, N = 5.000, seed = 42)
    # ------------------------------------------------------------------
    print("\n[6] Adım 4: Seçilen hiperparametrelerle Fold-in ('ridge' vs 'mean') karşılaştırması (3, 4, 5 oyun)...")
    indptr = interaction_matrix.indptr
    indices = interaction_matrix.indices

    eligible_uids = [u for u in val_uids if (indptr[u + 1] - indptr[u]) >= 5]
    rng_sim = np.random.default_rng(42)
    sim_uids = rng_sim.choice(eligible_uids, size=5000, replace=False).tolist()
    sim_cands_dict = {u: val_candidates[u] for u in sim_uids}

    foldin_rows = []

    for n_games in (3, 4, 5):
        rng_games = np.random.default_rng(42)
        selected_games: Dict[int, np.ndarray] = {}
        for u in sim_uids:
            u_pos = indices[indptr[u] : indptr[u + 1]]
            selected_games[u] = rng_games.choice(u_pos, size=n_games, replace=False)

        records_by_method = {}
        elapsed_by_method = {}

        for method_name in ("ridge", "mean"):
            als_model.fold_in_method = method_name
            sim_scores: Dict[int, np.ndarray] = {}
            t0_m = time.perf_counter()
            with threadpool_limits(1, "blas"):
                for u in sim_uids:
                    cands = sim_cands_dict[u]
                    df_out = adapter.score_from_items(selected_games[u], cands)
                    scores_u = np.full(100, -1e9, dtype=np.float64)
                    if len(df_out) == 100:
                        scores_u[:] = df_out["score"].to_numpy(dtype=np.float64)
                    else:
                        item_to_s = dict(zip(df_out["item_idx"].tolist(), df_out["score"].tolist()))
                        for idx_c, cid in enumerate(cands):
                            scores_u[idx_c] = item_to_s.get(int(cid), -1e9)
                    sim_scores[u] = scores_u
            elapsed_m = time.perf_counter() - t0_m
            elapsed_by_method[method_name] = elapsed_m

            rec_m = evaluate_with_user_records(
                score_fn=lambda uid, _cands, d=sim_scores: d[uid],
                candidates=sim_cands_dict,
                cold_items=cold_items,
            )
            records_by_method[method_name] = rec_m

        paired = compute_paired_stats(
            records_by_method["ridge"],
            records_by_method["mean"],
            sim_uids,
        )
        paired["n_games"] = n_games
        paired["ridge_coverage_pct"] = round(records_by_method["ridge"]["overall"]["coverage_pct"], 2)
        paired["mean_coverage_pct"] = round(records_by_method["mean"]["overall"]["coverage_pct"], 2)
        paired["ridge_seconds"] = round(elapsed_by_method["ridge"], 2)
        paired["mean_seconds"] = round(elapsed_by_method["mean"], 2)
        foldin_rows.append(paired)

        print(
            f"  [{n_games} Oyun] "
            f"Ridge: HR@10={paired['ridge_hr_at_10']:.4f}, NDCG@10={paired['ridge_ndcg_at_10']:.4f}, Cov={paired['ridge_coverage_pct']:.2f}% ({paired['ridge_seconds']:.2f}s) | "
            f"Mean: HR@10={paired['mean_hr_at_10']:.4f}, NDCG@10={paired['mean_ndcg_at_10']:.4f}, Cov={paired['mean_coverage_pct']:.2f}% ({paired['mean_seconds']:.2f}s)"
        )
        print(
            f"           Fark (Ridge - Mean): "
            f"ΔNDCG@10 = {paired['diff_ndcg_ridge_minus_mean']:+.4f} ± {paired['se_diff_ndcg']:.4f} (z={paired['z_stat_ndcg']:+.2f}) | "
            f"ΔHR@10 = {paired['diff_hr_ridge_minus_mean']:+.4f} ± {paired['se_diff_hr']:.4f} (z={paired['z_stat_hr']:+.2f})"
        )

    df_foldin = pd.DataFrame(foldin_rows)
    df_foldin.to_csv(PROJECT_ROOT / "results" / "mf_val_foldin_comparison.csv", index=False)

    # Birincil ölçüt NDCG@10'a göre kazanan fold_in_method'u belirle (3 oyun simülasyonunda)
    row_3g = foldin_rows[0]
    chosen_fold_in = (
        "ridge" if row_3g["ridge_ndcg_at_10"] >= row_3g["mean_ndcg_at_10"] else "mean"
    )
    print(f"\n  -> Birincil ölçüt NDCG@10'a göre seçilen varsayılan fold_in_method: '{chosen_fold_in}'")

    # ------------------------------------------------------------------
    # ADIM 5: En İyi Modeli saved_models/mf/ Altına Kaydet ve Doğrula
    # ------------------------------------------------------------------
    als_model.fold_in_method = chosen_fold_in
    save_dir = PROJECT_ROOT / "saved_models" / "mf"
    als_model.save(str(save_dir))
    print(f"\n[7] Adım 5: En iyi model kaydedildi: {save_dir}")

    loaded_model = ALSModel.load(str(save_dir))
    sample_check_uids = val_uids[:200]
    max_score_diff = 0.0
    for u in sample_check_uids:
        c = val_candidates[u]
        s_orig = als_model.score(u, c)
        s_load = loaded_model.score(u, c)
        max_score_diff = max(max_score_diff, float(np.max(np.abs(s_orig - s_load))))

    s_fi_orig = als_model.score_from_items([0, 10, 20], val_candidates[val_uids[0]])
    s_fi_load = loaded_model.score_from_items([0, 10, 20], val_candidates[val_uids[0]])
    max_fi_diff = float(np.max(np.abs(s_fi_orig - s_fi_load)))
    print(
        f"    Kaydet-Yükle doğrulaması: max |Δscore| = {max_score_diff:.2e}, "
        f"max |Δscore_from_items| = {max_fi_diff:.2e}"
    )
    if max_score_diff > 0.0 or max_fi_diff > 0.0:
        raise RuntimeError("Kaydet-yükle sonrası skor farkı sıfır değil!")

    # Tüm analiz özetini JSON olarak kaydet
    summary_payload = {
        "selected_params": {
            **BEST_PARAMS,
            "fold_in_method": chosen_fold_in,
            "fit_seconds": round(float(als_model.fit_seconds), 3),
        },
        "step3a_overall_val": {
            "hr_at_10": round(float(ov["hit_rate"]), 6),
            "ndcg_at_10": round(float(ov["ndcg"]), 6),
            "precision_at_10": round(float(ov["precision"]), 6),
            "recall_at_10": round(float(ov["recall"]), 6),
            "pool_coverage_at_10_pct": round(float(ov["coverage_pct"]), 4),
            "pool_coverage_at_10_n": int(ov["coverage_n"]),
            "activity": val_records["activity"],
        },
        "step3b_popularity_buckets": bucket_results,
        "step3c_catalog_coverage": catalog_cov_stats,
        "step3d_stage1_trends": {
            "by_alpha": trend_by_alpha.to_dict(orient="records"),
            "by_factors": trend_by_factors.to_dict(orient="records"),
            "by_regularization": trend_by_reg.to_dict(orient="records"),
            "by_factors_and_alpha": trend_alpha_factors.to_dict(orient="records"),
        },
        "step4_foldin_comparison": foldin_rows,
    }
    out_json = PROJECT_ROOT / "results" / "mf_val_selected_analysis.json"
    out_json.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"    Analiz özeti kaydedildi: {out_json}")


if __name__ == "__main__":
    main()
