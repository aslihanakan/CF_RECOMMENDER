"""
scripts/run_mf_grid_stage1.py
=============================
GÖREV (Parça 3/4) - Adım 1: Kaba Grid Arama (Tam Validation, N = 75.799)

Kurallar:
  - Yalnızca validation seti kullanılır; test.csv'ye kesinlikle dokunulmaz.
  - factors in {32, 64, 128}, regularization in {0.01, 0.1, 1.0},
    alpha in {5, 10, 20, 40}, iterations = 15 (36 kombinasyon).
  - Sabit seed = 42, num_threads = 1, threadpool_limits(1, "blas").
  - Değerlendirme için ContractAdapter + src/evaluation.py::evaluate() kullanılır.
  - Her kombinasyon tamamlandığında results/mf_val_grid_stage1.csv dosyasına
    anında eklenir (kesilirse kaldığı yerden devam edebilir).
"""

from __future__ import annotations

import concurrent.futures
import csv
import itertools
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.contract import ContractAdapter
from src.data_loader import N_ITEMS, N_USERS, load_data
from src.evaluation import evaluate
from src.models.mf import ALSModel
from src.models.popularity import PopularityModel

STAGE1_CSV_PATH = PROJECT_ROOT / "results" / "mf_val_grid_stage1.csv"

CSV_COLUMNS = [
    "stage",
    "factors",
    "regularization",
    "alpha",
    "iterations",
    "seed",
    "hr_at_10",
    "ndcg_at_10",
    "precision_at_10",
    "recall_at_10",
    "pool_coverage_at_10_pct",
    "pool_coverage_at_10_n",
    "fit_seconds",
    "eval_seconds",
    "total_seconds",
]

# Worker süreçleri için paylaşılan salt-okunur nesneler
_WORKER_DATA: Dict[str, Any] = {}


def _init_worker() -> None:
    """Her paralel işçi sürecinde veri setini ve Popularity modelini bir kez yükler."""
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    warnings.filterwarnings("ignore", category=UserWarning)
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        old_stdout = sys.stdout
        try:
            sys.stdout = devnull
            data = load_data(str(PROJECT_ROOT / "data"))
            pop_model = PopularityModel.load(str(PROJECT_ROOT / "saved_models" / "popularity.pkl"))
        finally:
            sys.stdout = old_stdout

    val_candidates = data["val_candidates"]
    val_uids = list(val_candidates.keys())
    uid_to_row = {uid: idx for idx, uid in enumerate(val_uids)}
    cands_2d = np.array([val_candidates[u] for u in val_uids], dtype=np.int64)
    targets_list = cands_2d[:, 0].tolist()

    _WORKER_DATA["interaction_matrix"] = data["interaction_matrix"]
    _WORKER_DATA["seen_matrix"] = data["seen_matrix"]
    _WORKER_DATA["val_candidates"] = val_candidates
    _WORKER_DATA["cold_items"] = data["cold_items"]
    _WORKER_DATA["user_activity"] = data["user_activity"]
    _WORKER_DATA["pop_model"] = pop_model
    _WORKER_DATA["val_uids"] = val_uids
    _WORKER_DATA["uid_to_row"] = uid_to_row
    _WORKER_DATA["cands_2d"] = cands_2d
    _WORKER_DATA["targets_list"] = targets_list


def evaluate_single_combo(
    combo: Tuple[str, int, float, float, int]
) -> Dict[str, Any]:
    """Tek bir (stage, factors, regularization, alpha, iterations) kombinasyonunu
    eğitir ve ContractAdapter + evaluate() ile tam validation üzerinde değerlendirir.
    """
    stage_name, factors, reg, alpha, iterations = combo
    seed = 42

    interaction_matrix = _WORKER_DATA["interaction_matrix"]
    seen_matrix = _WORKER_DATA["seen_matrix"]
    val_candidates = _WORKER_DATA["val_candidates"]
    cold_items = _WORKER_DATA["cold_items"]
    user_activity = _WORKER_DATA["user_activity"]
    pop_model = _WORKER_DATA["pop_model"]
    val_uids = _WORKER_DATA["val_uids"]
    uid_to_row = _WORKER_DATA["uid_to_row"]
    cands_2d = _WORKER_DATA["cands_2d"]
    targets_list = _WORKER_DATA["targets_list"]

    with threadpool_limits(1, "blas"):
        t0_fit = time.perf_counter()
        model = ALSModel(
            factors=int(factors),
            regularization=float(reg),
            alpha=float(alpha),
            iterations=int(iterations),
            random_state=seed,
            num_threads=1,
        )
        model.fit(interaction_matrix, show_progress=False)
        fit_s = time.perf_counter() - t0_fit

        t0_eval = time.perf_counter()
        adapter = ContractAdapter(
            model=model,
            popularity_model=pop_model,
            seen_matrix=seen_matrix,
            interaction_matrix=interaction_matrix,
            n_users=N_USERS,
            n_items=N_ITEMS,
        )

        n_val = len(val_uids)
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

        eval_res = evaluate(
            score_fn=adapter_score_fn,
            candidates=val_candidates,
            cold_items=cold_items,
            user_activity=user_activity,
            verbose=False,
        )
        eval_s = time.perf_counter() - t0_eval

    ov = eval_res["overall"]
    return {
        "stage": stage_name,
        "factors": int(factors),
        "regularization": float(reg),
        "alpha": float(alpha),
        "iterations": int(iterations),
        "seed": seed,
        "hr_at_10": round(float(ov["hit_rate"]), 6),
        "ndcg_at_10": round(float(ov["ndcg"]), 6),
        "precision_at_10": round(float(ov["precision"]), 6),
        "recall_at_10": round(float(ov["recall"]), 6),
        "pool_coverage_at_10_pct": round(float(ov["coverage_pct"]), 4),
        "pool_coverage_at_10_n": int(ov["coverage_n"]),
        "fit_seconds": round(float(fit_s), 3),
        "eval_seconds": round(float(eval_s), 3),
        "total_seconds": round(float(fit_s + eval_s), 3),
    }


def load_completed_keys(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    keys = set()
    for _, r in df.iterrows():
        keys.add((int(r["factors"]), float(r["regularization"]), float(r["alpha"]), int(r["iterations"])))
    return keys


def append_row_to_csv(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def main() -> None:
    factors_list = [32, 64, 128]
    reg_list = [0.01, 0.1, 1.0]
    alpha_list = [5.0, 10.0, 20.0, 40.0]
    iterations_list = [15]

    all_combos: List[Tuple[str, int, float, float, int]] = [
        ("stage1", f, r, a, it)
        for f, r, a, it in itertools.product(factors_list, reg_list, alpha_list, iterations_list)
    ]

    completed = load_completed_keys(STAGE1_CSV_PATH)
    pending = [
        c for c in all_combos if (c[1], c[2], c[3], c[4]) not in completed
    ]

    n_workers = 4
    print("=" * 90)
    print(f"STAGE 1 GRID TARAMASI (Toplam: {len(all_combos)}, Tamamlanan: {len(completed)}, Kalan: {len(pending)})")
    print(f"Paralel işçi sayısı: {n_workers} (her süreçte num_threads=1, BLAS=1)")
    print("=" * 90)

    t0_all = time.perf_counter()
    if pending:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers, initializer=_init_worker
        ) as executor:
            future_to_combo = {
                executor.submit(evaluate_single_combo, combo): combo for combo in pending
            }
            done_count = len(completed)
            for future in concurrent.futures.as_completed(future_to_combo):
                row = future.result()
                append_row_to_csv(STAGE1_CSV_PATH, row)
                done_count += 1
                print(
                    f"[{done_count:2d}/{len(all_combos)}] "
                    f"f={row['factors']:3d}, reg={row['regularization']:5.2f}, "
                    f"alpha={row['alpha']:4.1f}, iter={row['iterations']:2d} | "
                    f"HR@10={row['hr_at_10']:.4f}, NDCG@10={row['ndcg_at_10']:.4f}, "
                    f"Cov@10={row['pool_coverage_at_10_pct']:5.2f}% | "
                    f"fit={row['fit_seconds']:5.1f}s, eval={row['eval_seconds']:5.1f}s",
                    flush=True,
                )

    elapsed_all = time.perf_counter() - t0_all
    df_res = pd.read_csv(STAGE1_CSV_PATH)
    df_sorted = df_res.sort_values(
        by=["ndcg_at_10", "hr_at_10", "factors"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    df_sorted.to_csv(STAGE1_CSV_PATH, index=False)

    print("\n" + "=" * 90)
    print(f"STAGE 1 TAMAMLANDI (Toplam duvar saati süresi: {elapsed_all:.1f} sn)")
    print("EN İYİ 10 KOMBİNASYON (NDCG@10 sırasıyla):")
    print("=" * 90)
    print(
        df_sorted[
            [
                "factors",
                "regularization",
                "alpha",
                "iterations",
                "hr_at_10",
                "ndcg_at_10",
                "pool_coverage_at_10_pct",
                "fit_seconds",
                "eval_seconds",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
