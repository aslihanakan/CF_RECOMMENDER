"""
scripts/check_adapter_regression.py
Adım 2 & Adım 3:
  - Tüm validation kullanıcıları (75.799 kullanıcı) için ContractAdapter'ı 5.000'lik
    dilimler halinde (mode="offline", target_items=val hedefleri) çalıştırır.
  - Süre (toplam ve dilim başına), tepe bellek kullanımı (psutil RSS ve tracemalloc),
    NaN/inf sayısı, [0, 1] aralık kontrolü ve satır sayısı (75.799 x 100) doğrulamasını yapar.
  - Adaptör çıktısını (user_idx, item_idx, score) tekrar [75.799 x 100] skor matrisine
    dönüştürür ve src/evaluation.py içindeki mevcut evaluate() fonksiyonunu değiştirmeden
    (aynı TIE_BREAK_SEED=42 ile) çağırarak referans Item-kNN sonuçlarıyla (Genel ve
    Aktivite kırılımı: Düşük/Orta/Yüksek) karşılaştırır.
  - Sonuçları results/adapter_val_check.json dosyasına kaydeder.
  - Test seti KESİNLİKLE kullanılmaz.
"""

import json
import os
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.contract import CONTRACT_COLUMNS, ContractAdapter
from src.data_loader import N_ITEMS, N_USERS, load_data
from src.evaluation import TIE_BREAK_SEED, evaluate
from src.models.item_knn import ItemKNNModel
from src.models.popularity import PopularityModel


def main() -> None:
    print("=" * 78)
    print("ADIM 2 & 3: VALIDATION REGRESYON, SÜRE, BELLEK VE SAĞLAMLIK KONTROLÜ")
    print("=" * 78)

    process = psutil.Process(os.getpid())
    tracemalloc.start()

    print("\n[1] Veri seti ve kayıtlı modeller yükleniyor (yalnızca train ve val kullanılır)...")
    data = load_data(str(PROJECT_ROOT / "data"))
    val_candidates = data["val_candidates"]
    cold_items = data["cold_items"]
    user_activity = data["user_activity"]

    pop_model = PopularityModel.load(str(PROJECT_ROOT / "saved_models" / "popularity.pkl"))
    item_knn = ItemKNNModel.load(str(PROJECT_ROOT / "saved_models" / "itemknn"))

    adapter = ContractAdapter(
        model=item_knn,
        popularity_model=pop_model,
        seen_matrix=data["seen_matrix"],
        interaction_matrix=data["interaction_matrix"],
        n_users=N_USERS,
        n_items=N_ITEMS,
    )

    # Validation kullanıcılarını val_candidates sırasıyla al (evaluate() ile birebir aynı sıra)
    val_uids = list(val_candidates.keys())
    n_val_users = len(val_uids)
    uid_to_row = {uid: idx for idx, uid in enumerate(val_uids)}

    # [n_val_users x 100] skor matrisi (adaptör çıktısından doldurulacak)
    adapter_score_matrix = np.empty((n_val_users, 100), dtype=np.float64)

    chunk_size = 5000
    chunk_stats = []
    total_rows = 0
    nan_count = 0
    inf_count = 0
    out_of_range_count = 0
    global_min_score = float("inf")
    global_max_score = float("-inf")
    total_cf_known_false_rows = 0

    rss_before_mb = process.memory_info().rss / (1024 * 1024)
    peak_rss_mb = rss_before_mb
    tracemalloc.reset_peak()

    print(
        f"\n[2] ContractAdapter tüm val kullanıcıları ({n_val_users:,}) için "
        f"{chunk_size:,}'lik dilimlerle çalıştırılıyor..."
    )
    t_start_total = time.perf_counter()

    for start_idx in range(0, n_val_users, chunk_size):
        end_idx = min(start_idx + chunk_size, n_val_users)
        chunk_uids = val_uids[start_idx:end_idx]
        chunk_cands_2d = np.array([val_candidates[u] for u in chunk_uids], dtype=np.int64)
        chunk_targets = chunk_cands_2d[:, 0].tolist()

        t0 = time.perf_counter()
        df_chunk = adapter.score(
            users=chunk_uids,
            candidates=chunk_cands_2d,
            mode="offline",
            target_items=chunk_targets,
        )
        elapsed_chunk = time.perf_counter() - t0

        curr_rss_mb = process.memory_info().rss / (1024 * 1024)
        if curr_rss_mb > peak_rss_mb:
            peak_rss_mb = curr_rss_mb

        # Sağlamlık kontrolleri
        n_rows_chunk = len(df_chunk)
        expected_rows_chunk = len(chunk_uids) * 100
        total_rows += n_rows_chunk

        scores = df_chunk["score"].to_numpy(dtype=np.float64)
        c_nan = int(np.isnan(scores).sum())
        c_inf = int(np.isinf(scores).sum())
        c_oor = int(np.sum((scores < 0.0) | (scores > 1.0)))
        c_false = int((~df_chunk["cf_known"].to_numpy()).sum())

        nan_count += c_nan
        inf_count += c_inf
        out_of_range_count += c_oor
        total_cf_known_false_rows += c_false

        c_min = float(scores.min()) if len(scores) > 0 else 0.0
        c_max = float(scores.max()) if len(scores) > 0 else 0.0
        if c_min < global_min_score:
            global_min_score = c_min
        if c_max > global_max_score:
            global_max_score = c_max

        # Adaptör çıktısını (user_idx, item_idx, score) -> [chunk_len x 100] matrise çevir
        if n_rows_chunk == expected_rows_chunk:
            items_2d = df_chunk["item_idx"].to_numpy().reshape(len(chunk_uids), 100)
            if not np.array_equal(items_2d, chunk_cands_2d):
                raise RuntimeError("Aday sırası uyuşmazlığı tespit edildi!")
            adapter_score_matrix[start_idx:end_idx, :] = scores.reshape(len(chunk_uids), 100)
        else:
            # Eğer herhangi bir satır eksilmiş olsaydı (user_idx, item_idx) eşlemesiyle yerleştir
            adapter_score_matrix[start_idx:end_idx, :] = -1e9
            for local_i, u in enumerate(chunk_uids):
                sub = df_chunk[df_chunk["user_idx"] == u]
                item_to_score = dict(zip(sub["item_idx"].tolist(), sub["score"].tolist()))
                for j, item_id in enumerate(chunk_cands_2d[local_i]):
                    adapter_score_matrix[start_idx + local_i, j] = item_to_score.get(int(item_id), -1e9)

        chunk_info = {
            "chunk_index": len(chunk_stats) + 1,
            "start_user_pos": start_idx,
            "end_user_pos": end_idx,
            "n_users": len(chunk_uids),
            "n_rows": n_rows_chunk,
            "expected_rows": expected_rows_chunk,
            "seconds": round(elapsed_chunk, 4),
            "ms_per_user": round((elapsed_chunk * 1000.0) / len(chunk_uids), 4),
            "rss_mb": round(curr_rss_mb, 2),
        }
        chunk_stats.append(chunk_info)
        print(
            f"  Dilim {chunk_info['chunk_index']:2d}: kullanıcı [{start_idx:5d}..{end_idx:5d}] | "
            f"satır={n_rows_chunk:7,d} | süre={elapsed_chunk:6.2f} sn "
            f"({chunk_info['ms_per_user']:.3f} ms/kul) | RSS={curr_rss_mb:.1f} MB"
        )

    total_adapter_seconds = time.perf_counter() - t_start_total
    _current_traced, peak_traced_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_traced_mb = peak_traced_bytes / (1024 * 1024)

    expected_total_rows = n_val_users * 100
    print("\n" + "-" * 78)
    print("[3] TÜM VALIDATION SAĞLAMLIK VE PERFORMANS ÖZETİ")
    print("-" * 78)
    print(f"  Toplam kullanıcı sayısı        : {n_val_users:,}")
    print(f"  Toplam üretilen satır sayısı   : {total_rows:,} (beklenen: {expected_total_rows:,}, sapma: {total_rows - expected_total_rows})")
    print(f"  Toplam süre                    : {total_adapter_seconds:.2f} sn (ortalama {total_adapter_seconds / len(chunk_stats):.2f} sn/dilim)")
    print(f"  Kullanıcı başına ortalama süre : {(total_adapter_seconds * 1000.0) / n_val_users:.4f} ms")
    print(f"  Başlangıç RSS Bellek           : {rss_before_mb:.2f} MB")
    print(f"  Tepe RSS Bellek (psutil)       : {peak_rss_mb:.2f} MB")
    print(f"  Tepe Python Tahsisi (tracemalloc): {peak_traced_mb:.2f} MB")
    print(f"  NaN sayısı                     : {nan_count}")
    print(f"  inf sayısı                     : {inf_count}")
    print(f"  [0, 1] dışı skor sayısı        : {out_of_range_count}")
    print(f"  Global min / max skor          : [{global_min_score:.6f}, {global_max_score:.6f}]")
    print(f"  Toplam cf_known=False satırı   : {total_cf_known_false_rows:,}")

    # ------------------------------------------------------------------
    # ADIM 2: evaluation.py ile Uçtan Uca Regresyon Karşılaştırması
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("[4] UÇTAN UCA REGRESYON KONTROLÜ (evaluation.evaluate ile)")
    print("-" * 78)

    # Referans: doğrudan item_knn.score ile evaluate()
    print("  Referans model (item_knn.score) değerlendiriliyor...")
    ref_eval = evaluate(
        score_fn=item_knn.score,
        candidates=val_candidates,
        cold_items=cold_items,
        user_activity=user_activity,
        verbose=False,
    )

    # Adaptör skor matrisi üzerinden evaluate() (cold_items cezası aktif - standart protokol)
    print("  Adaptör çıktısı (ContractAdapter -> [75799 x 100] matris) değerlendiriliyor...")

    def adapter_score_fn(uid: int, _cands: np.ndarray) -> np.ndarray:
        return adapter_score_matrix[uid_to_row[uid]]

    adapter_eval = evaluate(
        score_fn=adapter_score_fn,
        candidates=val_candidates,
        cold_items=cold_items,
        user_activity=user_activity,
        verbose=False,
    )

    # Ek kontrol: cold_items=None iken (saf adaptör [0, 1] çıktısı, cold_never_seen de 0.0 iken)
    adapter_eval_no_cold_mask = evaluate(
        score_fn=adapter_score_fn,
        candidates=val_candidates,
        cold_items=None,
        user_activity=user_activity,
        verbose=False,
    )

    # Karşılaştırma tablosu oluştur
    groups_order = [
        ("Genel (Overall)", "overall", None),
        ("Düşük Aktivite (=3)", "activity", "low"),
        ("Orta Aktivite (4-6)", "activity", "medium"),
        ("Yüksek Aktivite (7+)", "activity", "high"),
        ("Soğuk Pozitif", "cold_user", None),
    ]

    comparison_rows = []
    print("\n  " + "=" * 98)
    print(
        f"  {'Grup':<22} | {'N':>7} | {'Ref HR@10':>9} | {'Adp HR@10':>9} | {'ΔHR@10':>9} | "
        f"{'Ref NDCG':>9} | {'Adp NDCG':>9} | {'ΔNDCG@10':>9}"
    )
    print("  " + "-" * 98)

    max_abs_delta_hr = 0.0
    max_abs_delta_ndcg = 0.0

    for label, section, subkey in groups_order:
        ref_m = ref_eval[section] if subkey is None else ref_eval[section][subkey]
        adp_m = adapter_eval[section] if subkey is None else adapter_eval[section][subkey]
        adp_raw_m = (
            adapter_eval_no_cold_mask[section]
            if subkey is None
            else adapter_eval_no_cold_mask[section][subkey]
        )

        hr_ref = float(ref_m["hit_rate"])
        hr_adp = float(adp_m["hit_rate"])
        d_hr = hr_adp - hr_ref

        nd_ref = float(ref_m["ndcg"])
        nd_adp = float(adp_m["ndcg"])
        d_nd = nd_adp - nd_ref

        if section in ("overall", "activity"):
            max_abs_delta_hr = max(max_abs_delta_hr, abs(d_hr))
            max_abs_delta_ndcg = max(max_abs_delta_ndcg, abs(d_nd))

        print(
            f"  {label:<22} | {ref_m['n_users']:>7,d} | {hr_ref:>9.6f} | {hr_adp:>9.6f} | {d_hr:>+9.6f} | "
            f"{nd_ref:>9.6f} | {nd_adp:>9.6f} | {d_nd:>+9.6f}"
        )

        comparison_rows.append(
            {
                "group": label,
                "n_users": int(ref_m["n_users"]),
                "ref_hr_at_10": hr_ref,
                "adapter_hr_at_10": hr_adp,
                "delta_hr_at_10": d_hr,
                "ref_ndcg_at_10": nd_ref,
                "adapter_ndcg_at_10": nd_adp,
                "delta_ndcg_at_10": d_nd,
                "ref_coverage_pct": float(ref_m["coverage_pct"]),
                "adapter_coverage_pct": float(adp_m["coverage_pct"]),
                "adapter_no_cold_penalty_hr_at_10": float(adp_raw_m.get("hit_rate", 0.0)),
                "adapter_no_cold_penalty_ndcg_at_10": float(adp_raw_m.get("ndcg", 0.0)),
            }
        )

    print("  " + "=" * 98)

    overall_d_hr = abs(comparison_rows[0]["delta_hr_at_10"])
    overall_d_ndcg = abs(comparison_rows[0]["delta_ndcg_at_10"])
    threshold = 0.002
    regression_passed = (overall_d_hr <= threshold) and (overall_d_ndcg <= threshold)

    print(
        f"\n  Kabul Eşiği Kontrolü (|Δ| <= {threshold}): "
        f"|ΔHR@10| = {overall_d_hr:.6f}, |ΔNDCG@10| = {overall_d_ndcg:.6f} -> "
        f"{'GEÇTİ' if regression_passed else 'KALDI'}"
    )

    # JSON çıktısını kaydet
    out_json_path = PROJECT_ROOT / "results" / "adapter_val_check.json"
    report_data = {
        "split": "validation",
        "n_val_users": n_val_users,
        "chunk_size": chunk_size,
        "n_chunks": len(chunk_stats),
        "tie_break_seed": TIE_BREAK_SEED,
        "timing": {
            "total_seconds": round(total_adapter_seconds, 4),
            "avg_chunk_seconds": round(total_adapter_seconds / len(chunk_stats), 4),
            "ms_per_user": round((total_adapter_seconds * 1000.0) / n_val_users, 4),
            "chunks": chunk_stats,
        },
        "memory": {
            "rss_before_mb": round(rss_before_mb, 2),
            "peak_rss_mb": round(peak_rss_mb, 2),
            "tracemalloc_peak_mb": round(peak_traced_mb, 2),
        },
        "integrity_checks": {
            "total_rows": total_rows,
            "expected_rows": expected_total_rows,
            "row_deviation": total_rows - expected_total_rows,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "out_of_range_0_1_count": out_of_range_count,
            "global_min_score": global_min_score,
            "global_max_score": global_max_score,
            "total_cf_known_false_rows": total_cf_known_false_rows,
        },
        "regression": {
            "threshold": threshold,
            "passed": regression_passed,
            "overall_abs_delta_hr_at_10": overall_d_hr,
            "overall_abs_delta_ndcg_at_10": overall_d_ndcg,
            "max_group_abs_delta_hr_at_10": max_abs_delta_hr,
            "max_group_abs_delta_ndcg_at_10": max_abs_delta_ndcg,
            "comparison_table": comparison_rows,
        },
    }

    out_json_path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[5] Sonuçlar kaydedildi: {out_json_path}")

    if not regression_passed or nan_count > 0 or inf_count > 0 or out_of_range_count > 0 or total_rows != expected_total_rows:
        sys.exit(1)


if __name__ == "__main__":
    main()
