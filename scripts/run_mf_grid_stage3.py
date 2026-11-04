"""
scripts/run_mf_grid_stage3.py
=============================
GÖREV (Parça 3b) - Adım 1: Stage 3 Grid Kenar Genişletmesi (Tam Validation, N = 75.799)

Kurallar:
  - Yalnızca validation seti kullanılır; test.csv'ye kesinlikle dokunulmaz.
  - Mevcut stage1 ve stage2 CSV dosyalarına dokunulmaz.
  - factors in {32, 64}, alpha in {60, 100, 150}, reg in {0.01, 0.1},
    iterations in {30, 50} (24 kombinasyon).
  - Sabit seed = 42, num_threads = 1, threadpool_limits(1, "blas"), 4 paralel süreç.
  - Değerlendirme için ContractAdapter + src/evaluation.py::evaluate() kullanılır.
  - Her kombinasyondan sonra results/mf_val_grid_stage3.csv dosyasına eklenir.
"""

from __future__ import annotations

import concurrent.futures
import itertools
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_mf_grid_stage1 import (
    STAGE1_CSV_PATH,
    _init_worker,
    append_row_to_csv,
    evaluate_single_combo,
    load_completed_keys,
)

STAGE2_CSV_PATH = PROJECT_ROOT / "results" / "mf_val_grid_stage2.csv"
STAGE3_CSV_PATH = PROJECT_ROOT / "results" / "mf_val_grid_stage3.csv"


def main() -> None:
    factors_list = [32, 64]
    reg_list = [0.01, 0.1]
    alpha_list = [60.0, 100.0, 150.0]
    iterations_list = [30, 50]

    stage3_combos: List[Tuple[str, int, float, float, int]] = [
        ("stage3", f, r, a, it)
        for f, r, a, it in itertools.product(
            factors_list, reg_list, alpha_list, iterations_list
        )
    ]

    completed = load_completed_keys(STAGE3_CSV_PATH)
    pending = [
        c for c in stage3_combos if (c[1], c[2], c[3], c[4]) not in completed
    ]

    n_workers = 4
    print("=" * 90)
    print(
        f"STAGE 3 GRID TARAMASI (Toplam: {len(stage3_combos)}, "
        f"Tamamlanan: {len(completed)}, Kalan: {len(pending)})"
    )
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
                append_row_to_csv(STAGE3_CSV_PATH, row)
                done_count += 1
                print(
                    f"[{done_count:2d}/{len(stage3_combos)}] "
                    f"f={row['factors']:3d}, reg={row['regularization']:5.2f}, "
                    f"alpha={row['alpha']:5.1f}, iter={row['iterations']:2d} | "
                    f"HR@10={row['hr_at_10']:.4f}, NDCG@10={row['ndcg_at_10']:.4f}, "
                    f"Cov@10={row['pool_coverage_at_10_pct']:5.2f}% | "
                    f"fit={row['fit_seconds']:5.1f}s, eval={row['eval_seconds']:5.1f}s",
                    flush=True,
                )

    elapsed_all = time.perf_counter() - t0_all
    df_s3 = pd.read_csv(STAGE3_CSV_PATH)
    df_s3_sorted = df_s3.sort_values(
        by=["ndcg_at_10", "hr_at_10", "factors", "iterations", "fit_seconds"],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)
    df_s3_sorted.to_csv(STAGE3_CSV_PATH, index=False)

    print("\n" + "=" * 90)
    print(f"STAGE 3 TAMAMLANDI (Toplam duvar saati süresi: {elapsed_all:.1f} sn)")
    print("STAGE 3 TÜM SONUÇLAR (NDCG@10 sırasıyla):")
    print("=" * 90)
    print(
        df_s3_sorted[
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
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
