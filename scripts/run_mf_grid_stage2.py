"""
scripts/run_mf_grid_stage2.py
=============================
GÖREV (Parça 3/4) - Adım 2: İnce Grid Arama (Tam Validation, N = 75.799)

Stage 1 Bulgusu:
  En iyi 3 NDCG@10 kombinasyonu:
    1) factors=32, reg=0.01, alpha=40, iter=15 -> NDCG@10=0.477843, HR@10=0.662423
    2) factors=32, reg=0.10, alpha=40, iter=15 -> NDCG@10=0.477317, HR@10=0.662436
    3) factors=32, reg=1.00, alpha=40, iter=15 -> NDCG@10=0.477280, HR@10=0.662186

  Kenar (boundary) genişletmeleri:
    - factors=32 alt kenarda -> factors=16 eklendi (factors in {16, 32})
    - alpha=40 üst kenarda   -> alpha=60 ve ara değer alpha=30 eklendi (alpha in {30, 40, 60})
    - reg=0.01 alt kenarda   -> reg=0.001 eklendi (reg in {0.001, 0.01, 0.1})
    - iterations in {15, 30} denendi (toplam 12 kombinasyon).
"""

from __future__ import annotations

import concurrent.futures
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


def main() -> None:
    # En iyi bölge çevresinde kenar genişletmeli 12 kombinasyon (iterations in {15, 30})
    stage2_combos: List[Tuple[str, int, float, float, int]] = [
        ("stage2", 16, 0.01, 30.0, 15),
        ("stage2", 16, 0.01, 40.0, 15),
        ("stage2", 16, 0.01, 60.0, 15),
        ("stage2", 16, 0.01, 40.0, 30),
        ("stage2", 16, 0.01, 60.0, 30),
        ("stage2", 32, 0.01, 30.0, 15),
        ("stage2", 32, 0.01, 60.0, 15),
        ("stage2", 32, 0.001, 40.0, 15),
        ("stage2", 32, 0.01, 40.0, 30),
        ("stage2", 32, 0.01, 60.0, 30),
        ("stage2", 32, 0.10, 40.0, 30),
        ("stage2", 32, 0.10, 60.0, 15),
    ]

    completed = load_completed_keys(STAGE2_CSV_PATH)
    pending = [
        c for c in stage2_combos if (c[1], c[2], c[3], c[4]) not in completed
    ]

    n_workers = 4
    print("=" * 90)
    print(f"STAGE 2 İNCE GRID TARAMASI (Toplam: {len(stage2_combos)}, Tamamlanan: {len(completed)}, Kalan: {len(pending)})")
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
                append_row_to_csv(STAGE2_CSV_PATH, row)
                done_count += 1
                print(
                    f"[{done_count:2d}/{len(stage2_combos)}] "
                    f"f={row['factors']:3d}, reg={row['regularization']:6.3f}, "
                    f"alpha={row['alpha']:4.1f}, iter={row['iterations']:2d} | "
                    f"HR@10={row['hr_at_10']:.4f}, NDCG@10={row['ndcg_at_10']:.4f}, "
                    f"Cov@10={row['pool_coverage_at_10_pct']:5.2f}% | "
                    f"fit={row['fit_seconds']:5.1f}s, eval={row['eval_seconds']:5.1f}s",
                    flush=True,
                )

    elapsed_all = time.perf_counter() - t0_all
    df_s2 = pd.read_csv(STAGE2_CSV_PATH)
    df_s2_sorted = df_s2.sort_values(
        by=["ndcg_at_10", "hr_at_10", "factors", "iterations"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    df_s2_sorted.to_csv(STAGE2_CSV_PATH, index=False)

    print("\n" + "=" * 90)
    print(f"STAGE 2 TAMAMLANDI (Toplam duvar saati süresi: {elapsed_all:.1f} sn)")
    print("STAGE 2 TÜM SONUÇLAR (NDCG@10 sırasıyla):")
    print("=" * 90)
    print(
        df_s2_sorted[
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

    if STAGE1_CSV_PATH.exists():
        df_s1 = pd.read_csv(STAGE1_CSV_PATH)
        df_all = pd.concat([df_s1, df_s2_sorted], ignore_index=True)
        df_all_sorted = df_all.sort_values(
            by=["ndcg_at_10", "hr_at_10", "factors", "iterations"],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)
        print("\n" + "=" * 90)
        print("STAGE 1 + STAGE 2 BİRLEŞİK EN İYİ 10 KOMBİNASYON:")
        print("=" * 90)
        print(
            df_all_sorted[
                [
                    "stage",
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
