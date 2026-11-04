"""
scripts/run_mf_stage3_followup.py
=================================
GÖREV (Parça 3b) - Adım 2 ve Adım 3:
  1. Adım 2: Gerçek Katalog Coverage@10 Ödünleşim Tablosu (results/mf_val_coverage_tradeoff.csv)
     Konfigürasyonlar:
       (a) Varsayılan başlangıç : f=64, reg=0.10, alpha=10,  iter=15
       (b) Parça 3 kazananı     : f=32, reg=0.01, alpha=60,  iter=30
       (c1) Stage 3 #1 (Kazanan): f=32, reg=0.01, alpha=150, iter=50
       (c2) Stage 3 #2          : f=32, reg=0.10, alpha=150, iter=50
       (c3) Stage 3 #3          : f=32, reg=0.01, alpha=150, iter=30
     Protokol: 10.000 val kullanıcısı, seed=42, mode='live', seen_matrix filtreli.
  2. Adım 3: Yeni Kazanan (f=32, reg=0.01, alpha=150, iter=50) için:
       - Fold-in ('ridge' vs 'mean') karşılaştırması (3, 4, 5 oyun, 5.000 val kullanıcısı,
         seed=42, eşleştirilmiş SE ve z-istatistiği)
       - Popülerlik kovası kırılımı (soğuk, kuyruk, orta, baş)
       - saved_models/mf/ güncellemesi ve kaydet-yükle farkının 0 olduğunun doğrulanması.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_mf_step3_step4_analysis import compute_paired_stats
from src.analysis_utils import (
    build_target_popularity_buckets,
    compute_subset_metrics,
    evaluate_with_user_records,
    measure_catalog_coverage_topk,
)
from src.contract import ContractAdapter
from src.data_loader import N_ITEMS, N_USERS, load_data
from src.models.mf import ALSModel
from src.models.popularity import PopularityModel

NEW_WINNER_PARAMS = {
    "factors": 32,
    "regularization": 0.01,
    "alpha": 150.0,
    "iterations": 50,
    "random_state": 42,
    "num_threads": 1,
}


def main() -> None:
    print("=" * 95)
    print("PARÇA 3b - ADIM 2 (GERÇEK KATALOG COVERAGE TABLOSU) VE ADIM 3 (YENİ KAZANAN ANALİZ/KAYIT)")
    print("=" * 95)

    data = load_data(str(PROJECT_ROOT / "data"))
    interaction_matrix = data["interaction_matrix"]
    seen_matrix = data["seen_matrix"]
    val_candidates = data["val_candidates"]
    cold_items = data["cold_items"]
    user_activity = data["user_activity"]
    item_pop = np.asarray(interaction_matrix.sum(axis=0)).ravel()

    pop_model = PopularityModel.load(str(PROJECT_ROOT / "saved_models" / "popularity.pkl"))

    val_uids = list(val_candidates.keys())
    n_val = len(val_uids)
    uid_to_row = {uid: idx for idx, uid in enumerate(val_uids)}
    cands_2d = np.array([val_candidates[u] for u in val_uids], dtype=np.int64)
    targets_list = cands_2d[:, 0].tolist()

    # 10.000 val kullanıcısı (seed=42) - Parça 3 2c ile birebir aynı örneklem
    rng_cov = np.random.default_rng(42)
    sampled_cov_uids = rng_cov.choice(val_uids, size=10000, replace=False).tolist()

    # Stage 1, 2, 3 CSV'lerinden tam val metriklerini hazırla
    df_s1 = pd.read_csv(PROJECT_ROOT / "results" / "mf_val_grid_stage1.csv")
    df_s2 = pd.read_csv(PROJECT_ROOT / "results" / "mf_val_grid_stage2.csv")
    df_s3 = pd.read_csv(PROJECT_ROOT / "results" / "mf_val_grid_stage3.csv")
    df_all_grid = pd.concat([df_s1, df_s2, df_s3], ignore_index=True)

    def lookup_val_metrics(f: int, r: float, a: float, it: int) -> Dict[str, Any]:
        sub = df_all_grid[
            (df_all_grid["factors"] == f)
            & (np.isclose(df_all_grid["regularization"], r))
            & (np.isclose(df_all_grid["alpha"], a))
            & (df_all_grid["iterations"] == it)
        ]
        if len(sub) == 0:
            raise ValueError(f"Grid CSV içinde bulunamadı: f={f}, reg={r}, alpha={a}, iter={it}")
        row = sub.iloc[-1]
        return {
            "hr_at_10": float(row["hr_at_10"]),
            "ndcg_at_10": float(row["ndcg_at_10"]),
            "precision_at_10": float(row["precision_at_10"]),
            "recall_at_10": float(row["recall_at_10"]),
            "pool_coverage_at_10_pct": float(row["pool_coverage_at_10_pct"]),
            "pool_coverage_at_10_n": int(row["pool_coverage_at_10_n"]),
        }

    # ------------------------------------------------------------------
    # ADIM 2: Gerçek Katalog Coverage@10 (5 Konfigürasyon)
    # ------------------------------------------------------------------
    configs_to_measure = [
        {
            "config_id": "(a) Varsayılan Başlangıç",
            "label": "ALS (f=64, reg=0.1, alpha=10, iter=15)",
            "factors": 64,
            "regularization": 0.10,
            "alpha": 10.0,
            "iterations": 15,
        },
        {
            "config_id": "(b) Parça 3 Seçilen",
            "label": "ALS (f=32, reg=0.01, alpha=60, iter=30)",
            "factors": 32,
            "regularization": 0.01,
            "alpha": 60.0,
            "iterations": 30,
        },
        {
            "config_id": "(c1) Stage 3 #1 (Yeni Kazanan)",
            "label": "ALS (f=32, reg=0.01, alpha=150, iter=50)",
            "factors": 32,
            "regularization": 0.01,
            "alpha": 150.0,
            "iterations": 50,
        },
        {
            "config_id": "(c2) Stage 3 #2",
            "label": "ALS (f=32, reg=0.10, alpha=150, iter=50)",
            "factors": 32,
            "regularization": 0.10,
            "alpha": 150.0,
            "iterations": 50,
        },
        {
            "config_id": "(c3) Stage 3 #3",
            "label": "ALS (f=32, reg=0.01, alpha=150, iter=30)",
            "factors": 32,
            "regularization": 0.01,
            "alpha": 150.0,
            "iterations": 30,
        },
    ]

    tradeoff_rows = []
    winner_model: ALSModel | None = None

    print("\n[1] Adım 2: 5 Konfigürasyon için Gerçek Katalog Coverage@10 ölçülüyor (N=10.000, seed=42, mode='live')...")
    for cfg in configs_to_measure:
        f = cfg["factors"]
        r = cfg["regularization"]
        a = cfg["alpha"]
        it = cfg["iterations"]
        val_m = lookup_val_metrics(f, r, a, it)

        with threadpool_limits(1, "blas"):
            m = ALSModel(
                factors=f,
                regularization=r,
                alpha=a,
                iterations=it,
                random_state=42,
                num_threads=1,
            )
            m.fit(interaction_matrix, show_progress=False)

            def _rec_fn(uid: int, model_ref: ALSModel = m) -> List[int]:
                recs = model_ref.recommend_topk(
                    uid=uid,
                    k=10,
                    exclude_seen=True,
                    seen_matrix=seen_matrix,
                    mode="live",
                )
                return [item_id for item_id, _s in recs]

            cov_stats = measure_catalog_coverage_topk(
                model_name=cfg["label"],
                recommend_fn=_rec_fn,
                sampled_uids=sampled_cov_uids,
                item_pop=item_pop,
                pool_coverage_pct=val_m["pool_coverage_at_10_pct"],
                pool_coverage_n=val_m["pool_coverage_at_10_n"],
                k=10,
            )

        if cfg["config_id"].startswith("(c1)"):
            winner_model = m

        row_t = {
            "config_id": cfg["config_id"],
            "model_label": cfg["label"],
            "factors": f,
            "regularization": r,
            "alpha": a,
            "iterations": it,
            "val_hr_at_10": round(val_m["hr_at_10"], 6),
            "val_ndcg_at_10": round(val_m["ndcg_at_10"], 6),
            "pool_coverage_at_10_pct": round(val_m["pool_coverage_at_10_pct"], 4),
            "pool_coverage_at_10_n": val_m["pool_coverage_at_10_n"],
            "unique_rec_items": cov_stats["unique_rec_items"],
            "catalog_cov_25610_pct": round(cov_stats["catalog_cov_25610_pct"], 4),
            "catalog_cov_25367_known_pct": round(cov_stats["catalog_cov_25367_known_pct"], 4),
            "top100_pop_item_share_pct": round(cov_stats["top100_pop_item_share_pct"], 4),
            "mean_pop_percentile_rank": round(cov_stats["mean_pop_percentile_rank"], 4),
            "mean_rec_item_pop": round(cov_stats["mean_rec_item_pop"], 2),
            "median_rec_item_pop": round(cov_stats["median_rec_item_pop"], 2),
            "fit_seconds": round(float(m.fit_seconds), 2),
            "cov_eval_seconds": cov_stats["elapsed_seconds"],
        }
        tradeoff_rows.append(row_t)
        print(
            f"  {cfg['config_id']:<30} | Val NDCG@10={row_t['val_ndcg_at_10']:.4f}, HR@10={row_t['val_hr_at_10']:.4f} | "
            f"Benzersiz Ürün={row_t['unique_rec_items']:5,d} ({row_t['catalog_cov_25610_pct']:5.2f}%) | "
            f"Top-100 Payı={row_t['top100_pop_item_share_pct']:5.2f}% | "
            f"Ort. Pop Yüzdelik={row_t['mean_pop_percentile_rank']:5.2f}%"
        )

    df_tradeoff = pd.DataFrame(tradeoff_rows)
    tradeoff_csv = PROJECT_ROOT / "results" / "mf_val_coverage_tradeoff.csv"
    df_tradeoff.to_csv(tradeoff_csv, index=False)
    print(f"\n  -> Coverage-NDCG ödünleşim tablosu kaydedildi: {tradeoff_csv}")

    # ------------------------------------------------------------------
    # ADIM 3: Yeni Kazanan (f=32, reg=0.01, alpha=150, iter=50) Fold-in & Kayıt
    # ------------------------------------------------------------------
    assert winner_model is not None
    print("\n[2] Adım 3: Yeni Kazanan (f=32, reg=0.01, alpha=150, iter=50) için Fold-in karşılaştırması (3, 4, 5 oyun)...")

    adapter = ContractAdapter(
        model=winner_model,
        popularity_model=pop_model,
        seen_matrix=seen_matrix,
        interaction_matrix=interaction_matrix,
        n_users=N_USERS,
        n_items=N_ITEMS,
    )

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
            winner_model.fold_in_method = method_name
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

    chosen_fold_in = (
        "ridge"
        if foldin_rows[0]["ridge_ndcg_at_10"] >= foldin_rows[0]["mean_ndcg_at_10"]
        else "mean"
    )
    winner_model.fold_in_method = chosen_fold_in

    # Yeni kazananın popülerlik kovası ve aktivite kırılımını da güncelle
    print("\n[3] Yeni kazanan için tam val popülerlik kovası kırılımı hesaplanıyor...")
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

    val_records = evaluate_with_user_records(
        score_fn=lambda uid, _cands: adapter_score_matrix[uid_to_row[uid]],
        candidates=val_candidates,
        cold_items=cold_items,
        user_activity=user_activity,
    )
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
            f"    Kova '{b_name:6s}' (N={b_info['n_users']:6,d}) -> "
            f"HR@10={b_metrics['hit_rate']:.4f}, NDCG@10={b_metrics['ndcg']:.4f}, Hits={b_metrics['hits']:,}"
        )

    # Kayıt ve kaydet-yükle doğrulaması
    save_dir = PROJECT_ROOT / "saved_models" / "mf"
    winner_model.save(str(save_dir))
    print(f"\n[4] Yeni kazanan model kaydedildi: {save_dir}")

    loaded_model = ALSModel.load(str(save_dir))
    max_score_diff = 0.0
    for u in val_uids[:200]:
        c = val_candidates[u]
        s_orig = winner_model.score(u, c)
        s_load = loaded_model.score(u, c)
        max_score_diff = max(max_score_diff, float(np.max(np.abs(s_orig - s_load))))

    s_fi_orig = winner_model.score_from_items([0, 10, 20], val_candidates[val_uids[0]])
    s_fi_load = loaded_model.score_from_items([0, 10, 20], val_candidates[val_uids[0]])
    max_fi_diff = float(np.max(np.abs(s_fi_orig - s_fi_load)))
    print(
        f"    Kaydet-Yükle doğrulaması: max |Δscore| = {max_score_diff:.2e}, "
        f"max |Δscore_from_items| = {max_fi_diff:.2e}"
    )
    if max_score_diff > 0.0 or max_fi_diff > 0.0:
        raise RuntimeError("Kaydet-yükle sonrası skor farkı sıfır değil!")

    ov = val_records["overall"]
    summary_payload = {
        "selected_params": {
            **NEW_WINNER_PARAMS,
            "fold_in_method": chosen_fold_in,
            "fit_seconds": round(float(winner_model.fit_seconds), 3),
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
        "step2_coverage_tradeoff": tradeoff_rows,
        "step4_foldin_comparison": foldin_rows,
    }
    out_json = PROJECT_ROOT / "results" / "mf_val_selected_analysis.json"
    out_json.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"    Güncel analiz özeti kaydedildi: {out_json}")


if __name__ == "__main__":
    main()
