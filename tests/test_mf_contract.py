"""
tests/test_mf_contract.py
=========================
Parça 2/4: Implicit ALS (src/models/mf.py) modelinin ContractAdapter ile
entegrasyonu, tüm validation regresyonu, score_from_items (Yöntem A: Ridge Fold-in
vs Yöntem B: Mean Pseudo-user) 5.000 kullanıcı simülasyonu ve sözleşme birim testleri.

Çalıştırma:
    python tests/test_mf_contract.py
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.contract import CONTRACT_COLUMNS, ContractAdapter
from src.data_loader import N_ITEMS, N_USERS, load_data
from src.evaluation import evaluate
from src.filtering import FILTERED_SCORE
from src.models.mf import ALSModel, MFModel
from src.models.popularity import PopularityModel

PASS_COUNT = 0
FAIL_COUNT = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  [GECTI] {name}" + (f" -> {detail}" if detail else ""))
    else:
        FAIL_COUNT += 1
        print(f"  [KALDI] {name}" + (f" -> {detail}" if detail else ""))


def main() -> None:
    print("=" * 78)
    print("ALS MATRIX FACTORIZATION - CONTRACT ADAPTER VE SÖZLEŞME TESTLERİ")
    print("=" * 78)

    # ------------------------------------------------------------------
    # ADIM 0: Dosya konumu ve Parça 1 Validation sonucunun yeniden üretimi
    # ------------------------------------------------------------------
    print("\n--- ADIM 0: Dosya Konumu (src/models/mf.py) ve Parça 1 Sonuç Doğrulaması ---")
    old_mf_path = PROJECT_ROOT / "src" / "mf.py"
    new_mf_path = PROJECT_ROOT / "src" / "models" / "mf.py"
    check(
        not old_mf_path.exists() and new_mf_path.exists(),
        "src/mf.py kaldırıldı, model src/models/mf.py konumuna taşındı",
    )

    data = load_data(str(PROJECT_ROOT / "data"))
    interaction_matrix = data["interaction_matrix"]
    seen_matrix = data["seen_matrix"]
    val_candidates = data["val_candidates"]
    cold_items = data["cold_items"]
    user_activity = data["user_activity"]

    pop_model = PopularityModel.load(str(PROJECT_ROOT / "saved_models" / "popularity.pkl"))

    als_model = ALSModel(
        factors=64,
        regularization=0.1,
        alpha=10.0,
        iterations=15,
        random_state=42,
        num_threads=1,
    )
    als_model.fit(interaction_matrix, show_progress=False)

    raw_val_eval = evaluate(
        score_fn=als_model.score,
        candidates=val_candidates,
        cold_items=cold_items,
        user_activity=user_activity,
        verbose=False,
    )
    raw_hr = raw_val_eval["overall"]["hit_rate"]
    raw_ndcg = raw_val_eval["overall"]["ndcg"]

    check(
        abs(round(raw_hr, 4) - 0.6178) <= 1e-4 and abs(round(raw_ndcg, 4) - 0.4444) <= 1e-4,
        f"Parça 1 Val sonucu yeniden üretildi: HR@10={raw_hr:.6f} (0.6178), NDCG@10={raw_ndcg:.6f} (0.4444)",
        f"fit_s={als_model.fit_seconds:.2f} sn",
    )

    # ------------------------------------------------------------------
    # ADIM 1: ContractAdapter Bağlantısı ve Tüm Val (75.799 x 100) Regresyonu
    # ------------------------------------------------------------------
    print("\n--- ADIM 1: ContractAdapter Bağlantısı ve Tüm Validation (75.799 x 100) Kontrolü ---")
    adapter = ContractAdapter(
        model=als_model,
        popularity_model=pop_model,
        seen_matrix=seen_matrix,
        interaction_matrix=interaction_matrix,
        n_users=N_USERS,
        n_items=N_ITEMS,
    )

    val_uids = list(val_candidates.keys())
    n_val_users = len(val_uids)
    uid_to_row = {uid: idx for idx, uid in enumerate(val_uids)}
    adapter_score_matrix = np.empty((n_val_users, 100), dtype=np.float64)

    chunk_size = 10000
    total_rows = 0
    nan_count = 0
    inf_count = 0
    oor_count = 0
    global_min = float("inf")
    global_max = float("-inf")

    t0_adp = time.perf_counter()
    for start_idx in range(0, n_val_users, chunk_size):
        end_idx = min(start_idx + chunk_size, n_val_users)
        c_uids = val_uids[start_idx:end_idx]
        c_cands = np.array([val_candidates[u] for u in c_uids], dtype=np.int64)
        c_targets = c_cands[:, 0].tolist()

        df_chunk = adapter.score(
            users=c_uids,
            candidates=c_cands,
            mode="offline",
            target_items=c_targets,
        )
        total_rows += len(df_chunk)
        s_arr = df_chunk["score"].to_numpy(dtype=np.float64)
        nan_count += int(np.isnan(s_arr).sum())
        inf_count += int(np.isinf(s_arr).sum())
        oor_count += int(np.sum((s_arr < 0.0) | (s_arr > 1.0)))
        global_min = min(global_min, float(s_arr.min()))
        global_max = max(global_max, float(s_arr.max()))
        adapter_score_matrix[start_idx:end_idx, :] = s_arr.reshape(len(c_uids), 100)

    adp_elapsed = time.perf_counter() - t0_adp

    check(
        total_rows == n_val_users * 100,
        f"Tüm val için {total_rows:,} satır üretildi (75.799 x 100, süre={adp_elapsed:.2f} sn)",
    )
    check(
        nan_count == 0 and inf_count == 0 and oor_count == 0,
        f"Tüm val çıktısında NaN={nan_count}, inf={inf_count}, [0,1] dışı={oor_count} (min={global_min:.4f}, max={global_max:.4f})",
    )

    def adapter_score_fn(uid: int, _cands: np.ndarray) -> np.ndarray:
        return adapter_score_matrix[uid_to_row[uid]]

    adp_val_eval = evaluate(
        score_fn=adapter_score_fn,
        candidates=val_candidates,
        cold_items=cold_items,
        user_activity=user_activity,
        verbose=False,
    )
    adp_hr = adp_val_eval["overall"]["hit_rate"]
    adp_ndcg = adp_val_eval["overall"]["ndcg"]
    diff_hr = abs(adp_hr - raw_hr)
    diff_ndcg = abs(adp_ndcg - raw_ndcg)

    check(
        diff_hr <= 1e-6 and diff_ndcg <= 1e-6,
        f"Adaptör vs Ham ALS farkı <= 1e-6: |ΔHR@10|={diff_hr:.8f}, |ΔNDCG@10|={diff_ndcg:.8f}",
        f"Ham={raw_hr:.6f}/{raw_ndcg:.6f}, Adaptör={adp_hr:.6f}/{adp_ndcg:.6f}",
    )

    # ------------------------------------------------------------------
    # ADIM 2: score_from_items (A: Kapalı Form Ridge Fold-in vs B: Mean Pseudo-user)
    # ------------------------------------------------------------------
    print("\n--- ADIM 2: score_from_items Simülasyonu (N = 5.000 Val Kullanıcısı, seed=42, 3 Oyun) ---")
    indptr = interaction_matrix.indptr
    indices = interaction_matrix.indices
    user_pos_counts = np.diff(indptr)
    eligible_val_uids = sorted(uid for uid in val_candidates.keys() if user_pos_counts[uid] >= 5)

    rng = np.random.default_rng(42)
    sampled_uids = rng.choice(eligible_val_uids, size=5000, replace=False).tolist()
    sampled_candidates = {int(uid): val_candidates[int(uid)] for uid in sampled_uids}
    sample_uid_to_row = {int(uid): i for i, uid in enumerate(sampled_uids)}

    chosen_per_user = {}
    for uid in sampled_uids:
        u_pos = indices[indptr[uid] : indptr[uid + 1]]
        chosen_per_user[int(uid)] = rng.choice(u_pos, size=3, replace=False).astype(np.int64)

    # Yöntem (A): ridge (kapalı form fold-in)
    als_model.fold_in_method = "ridge"
    sfi_matrix_ridge = np.empty((5000, 100), dtype=np.float64)
    t0_ridge = time.perf_counter()
    for i, uid in enumerate(sampled_uids):
        df_r = adapter.score_from_items(chosen_per_user[uid], sampled_candidates[uid])
        sfi_matrix_ridge[i, :] = df_r["score"].to_numpy(dtype=np.float64)
    ridge_time = time.perf_counter() - t0_ridge

    # Yöntem (B): mean (item faktör ortalaması)
    als_model.fold_in_method = "mean"
    sfi_matrix_mean = np.empty((5000, 100), dtype=np.float64)
    t0_mean = time.perf_counter()
    for i, uid in enumerate(sampled_uids):
        df_m = adapter.score_from_items(chosen_per_user[uid], sampled_candidates[uid])
        sfi_matrix_mean[i, :] = df_m["score"].to_numpy(dtype=np.float64)
    mean_time = time.perf_counter() - t0_mean

    # Tam geçmişle ALS (aynı 5.000 kullanıcı)
    cands_5000_2d = np.array([sampled_candidates[u] for u in sampled_uids], dtype=np.int64)
    df_als_full_5000 = adapter.score(
        users=sampled_uids,
        candidates=cands_5000_2d,
        mode="offline",
        target_items=cands_5000_2d[:, 0].tolist(),
    )
    als_full_matrix_5000 = df_als_full_5000["score"].to_numpy(dtype=np.float64).reshape(5000, 100)

    eval_ridge = evaluate(
        score_fn=lambda u, _c: sfi_matrix_ridge[sample_uid_to_row[u]],
        candidates=sampled_candidates,
        cold_items=cold_items,
        verbose=False,
    )["overall"]
    eval_mean = evaluate(
        score_fn=lambda u, _c: sfi_matrix_mean[sample_uid_to_row[u]],
        candidates=sampled_candidates,
        cold_items=cold_items,
        verbose=False,
    )["overall"]
    eval_als_full = evaluate(
        score_fn=lambda u, _c: als_full_matrix_5000[sample_uid_to_row[u]],
        candidates=sampled_candidates,
        cold_items=cold_items,
        verbose=False,
    )["overall"]
    eval_pop_5000 = evaluate(
        score_fn=pop_model.score,
        candidates=sampled_candidates,
        cold_items=cold_items,
        verbose=False,
    )["overall"]

    # Birincil ölçüt NDCG@10'a göre varsayılan fold_in_method'u ayarla
    best_method = "ridge" if eval_ridge["ndcg"] >= eval_mean["ndcg"] else "mean"
    als_model.fold_in_method = best_method

    print("\n  " + "=" * 86)
    print(f"  {'Yöntem (N = 5.000 Val Kullanıcısı, 3 Oyun)':<46} | {'HR@10':>8} | {'NDCG@10':>8} | {'Cov@10':>8} | {'Süre':>7}")
    print("  " + "-" * 86)
    print(f"  {'Popularity (Referans Baseline)':<46} | {eval_pop_5000['hit_rate']:>8.4f} | {eval_pop_5000['ndcg']:>8.4f} | {eval_pop_5000['coverage_pct']:>7.2f}% | {'-':>7}")
    print(f"  {'ALS score_from_items (A) Kapalı Form Ridge':<46} | {eval_ridge['hit_rate']:>8.4f} | {eval_ridge['ndcg']:>8.4f} | {eval_ridge['coverage_pct']:>7.2f}% | {ridge_time:>5.2f}s")
    print(f"  {'ALS score_from_items (B) Mean Pseudo-user':<46} | {eval_mean['hit_rate']:>8.4f} | {eval_mean['ndcg']:>8.4f} | {eval_mean['coverage_pct']:>7.2f}% | {mean_time:>5.2f}s")
    print(f"  {'ALS score() (Tam Geçmiş - ort. 9.94 oyun)':<46} | {eval_als_full['hit_rate']:>8.4f} | {eval_als_full['ndcg']:>8.4f} | {eval_als_full['coverage_pct']:>7.2f}% | {'-':>7}")
    print(f"  {'Item-kNN score_from_items (3 oyun, Referans)':<46} | {0.6090:>8.4f} | {0.4175:>8.4f} | {'37.27%':>8} | {'-':>7}")
    print(f"  {'Item-kNN score() (Tam Geçmiş, Referans)':<46} | {0.6628:>8.4f} | {0.4732:>8.4f} | {'45.88%':>8} | {'-':>7}")
    print("  " + "=" * 86)
    print(f"  Seçilen varsayılan yöntem: fold_in_method={best_method!r}")

    check(
        eval_ridge["hit_rate"] > eval_pop_5000["hit_rate"]
        and eval_mean["hit_rate"] > eval_pop_5000["hit_rate"],
        "Her iki ALS score_from_items yöntemi (A ve B) Popularity baseline'ının belirgin üstünde",
    )

    # ------------------------------------------------------------------
    # ADIM 3: Sözleşme Birim ve Sınır Durum Testleri
    # ------------------------------------------------------------------
    print("\n--- ADIM 3: ALS + ContractAdapter Kapsamlı Sözleşme Testleri ---")

    # 3.1 İlk 500 kullanıcıda cf_known=True için min=0, max=1 ve argsort sıra korunumu
    uids_500 = val_uids[:500]
    cands_500 = np.array([val_candidates[u] for u in uids_500], dtype=np.int64)
    df_500 = adapter.score(
        users=uids_500,
        candidates=cands_500,
        mode="offline",
        target_items=cands_500[:, 0].tolist(),
    )
    min_max_ok = True
    order_ok = True
    for uid, grp in df_500.groupby("user_idx", sort=False):
        k_sub = grp[grp["cf_known"]]
        k_scores = k_sub["score"].to_numpy()
        k_items = k_sub["item_idx"].to_numpy()
        raw_s = als_model.score(int(uid), k_items)
        if np.max(raw_s) > np.min(raw_s):
            if not (np.isclose(k_scores.min(), 0.0, atol=1e-12) and np.isclose(k_scores.max(), 1.0, atol=1e-12)):
                min_max_ok = False
                break
            if not np.array_equal(np.argsort(-raw_s, kind="stable"), np.argsort(-k_scores, kind="stable")):
                order_ok = False
                break
    check(min_max_ok, "500 kullanıcıda cf_known=True adaylar için min=0.0 ve max=1.0")
    check(order_ok, "500 kullanıcıda cf_known=True adaylar için argsort sıra korunumu (negatif iç çarpımlar dahil)")

    # 3.2 cf_known=False maskesi (243 ürün score=0.0 ve normalizasyonu bozmuyor)
    unknown_cf_items = np.where(~adapter.item_cf_known)[0]
    check(len(unknown_cf_items) == 243, f"cf_known=False ürün sayısı = {len(unknown_cf_items)} (beklenen 243)")

    u0 = val_uids[0]
    known_10 = val_candidates[u0][:10]
    mixed_13 = np.concatenate([known_10, unknown_cf_items[:3]])
    df_k10 = adapter.score(users=[u0], candidates=known_10, mode="offline", target_items=int(known_10[0]))
    df_m13 = adapter.score(users=[u0], candidates=mixed_13, mode="offline", target_items=int(known_10[0]))
    check(
        bool(np.all(df_m13.loc[~df_m13["cf_known"], "score"] == 0.0))
        and np.allclose(df_k10["score"].to_numpy(), df_m13.loc[df_m13["cf_known"], "score"].to_numpy(), atol=1e-12),
        "cf_known=False ürünlerde score=0.0 ve cf_known=True normalizasyonunu hiç bozmuyor",
    )

    # 3.3 Bilinmeyen kullanıcı (user_idx=-1, user_idx >= N_USERS ve train'de pozitifi olmayan kullanıcı) -> Popularity fallback
    sample_cands = val_candidates[u0][:20]
    pop_raw_known = pop_model.score(0, sample_cands[adapter.item_cf_known[sample_cands]])
    pop_norm_known = (pop_raw_known - pop_raw_known.min()) / (pop_raw_known.max() - pop_raw_known.min())

    with warnings.catch_warnings(record=True) as w_neg:
        warnings.simplefilter("always")
        df_neg_u = adapter.score(users=[-1], candidates=sample_cands, mode="live")
    with warnings.catch_warnings(record=True) as w_oob:
        warnings.simplefilter("always")
        df_oob_u = adapter.score(users=[N_USERS + 10], candidates=sample_cands, mode="live")

    check(
        len(w_neg) >= 1
        and len(w_oob) >= 1
        and np.allclose(df_neg_u.loc[df_neg_u["cf_known"], "score"].to_numpy(), pop_norm_known, atol=1e-12)
        and np.allclose(df_oob_u.loc[df_oob_u["cf_known"], "score"].to_numpy(), pop_norm_known, atol=1e-12),
        "user_idx=-1 ve user_idx>=N_USERS durumlarında uyarı verildi ve Popularity fallback çalıştı",
    )

    # Doğrudan als_model.score(-1, ...) ve train'de pozitifi olmayan kullanıcı kontrolü
    with warnings.catch_warnings(record=True) as w_direct_u:
        warnings.simplefilter("always")
        direct_neg_scores = als_model.score(-1, sample_cands)
    check(
        len(w_direct_u) >= 1 and np.allclose(direct_neg_scores, pop_model.score(0, sample_cands), atol=1e-6),
        "ALSModel.score(user_idx=-1) doğrudan çağrıldığında da Popularity skorlarına düşüyor",
    )

    # 3.4 score_from_items: geçerli, boş, geçersiz ve cf_known=False ürün seçimi
    chosen_3 = sample_cands[:3].copy()
    df_sfi_valid = adapter.score_from_items(item_idxs=chosen_3, candidates=sample_cands)
    check(
        len(df_sfi_valid) == len(sample_cands) - 3
        and len(set(df_sfi_valid["item_idx"]).intersection(set(chosen_3))) == 0
        and bool(np.all(df_sfi_valid["user_idx"] == -1)),
        "score_from_items (3 geçerli oyun): seçilenler çıkarıldı, user_idx=-1, skorlar [0, 1]",
    )

    # Boş seçim ve geçersiz seçim -> Popularity fallback
    with warnings.catch_warnings(record=True) as w_empty_sfi:
        warnings.simplefilter("always")
        df_sfi_empty = adapter.score_from_items(item_idxs=[], candidates=sample_cands)
    with warnings.catch_warnings(record=True) as w_inv_sfi:
        warnings.simplefilter("always")
        df_sfi_inv = adapter.score_from_items(item_idxs=[-5, N_ITEMS + 99], candidates=sample_cands)
    check(
        len(w_empty_sfi) >= 1
        and len(w_inv_sfi) >= 1
        and np.allclose(df_sfi_empty.loc[df_sfi_empty["cf_known"], "score"].to_numpy(), pop_norm_known, atol=1e-12)
        and np.allclose(df_sfi_inv.loc[df_sfi_inv["cf_known"], "score"].to_numpy(), pop_norm_known, atol=1e-12),
        "score_from_items (boş ve geçersiz oyun seçimi): uyarı verildi ve Popularity'ye düşüldü",
    )

    # Sadece cf_known=False ürünler seçildiğinde -> Popularity fallback (Item-kNN ile birebir aynı davranış)
    cold_choice = unknown_cf_items[:3].tolist()
    with warnings.catch_warnings(record=True) as w_cold_sfi:
        warnings.simplefilter("always")
        df_sfi_cold = adapter.score_from_items(item_idxs=cold_choice, candidates=sample_cands)
    check(
        len(w_cold_sfi) >= 1
        and np.allclose(df_sfi_cold.loc[df_sfi_cold["cf_known"], "score"].to_numpy(), pop_norm_known, atol=1e-12),
        "score_from_items (yalnızca cf_known=False ürünler seçildiğinde): uyarı verildi ve Popularity'ye düşüldü",
    )

    # 3.5 Canlı modda bilinenlerin elenmesi, offline modda hedef ürünün korunması
    u0_seen = seen_matrix.indices[seen_matrix.indptr[u0] : seen_matrix.indptr[u0 + 1]]
    cands_with_seen = np.concatenate([u0_seen[:3], sample_cands[:5]])
    df_live = adapter.score(users=[u0], candidates=cands_with_seen, mode="live")
    df_off = adapter.score(users=[u0], candidates=cands_with_seen, mode="offline", target_items=int(u0_seen[0]))
    check(
        len(set(df_live["item_idx"]).intersection(set(u0_seen[:3]))) == 0
        and int(u0_seen[0]) in set(df_off["item_idx"]),
        "Canlı modda bilinen ürünler elendi; offline modda hedef ürün (seen içinde olsa bile) korundu",
    )

    # 3.6 Tüm skorlar eşitken (veya tek bilinen adayda) 0.0
    df_single = adapter.score(users=[u0], candidates=[int(sample_cands[0]), int(unknown_cf_items[0])], mode="live")
    check(
        bool(np.all(df_single["score"].to_numpy() == 0.0)),
        "Tek cf_known=True aday (veya eşit skor) durumunda çıktı skoru 0.0",
    )

    # 3.7 Aynı seed (42) ile iki eğitimin birebir aynı skoru üretmesi
    als_1 = ALSModel(factors=16, regularization=0.1, alpha=10.0, iterations=5, random_state=42, num_threads=1).fit(interaction_matrix)
    als_2 = ALSModel(factors=16, regularization=0.1, alpha=10.0, iterations=5, random_state=42, num_threads=1).fit(interaction_matrix)
    check(
        np.array_equal(als_1.user_factors, als_2.user_factors)
        and np.array_equal(als_1.item_factors, als_2.item_factors)
        and np.array_equal(als_1.score(u0, sample_cands), als_2.score(u0, sample_cands)),
        "Aynı seed (42) ile eğitilen iki ALS modeli bit-bazında birebir aynı faktör ve skorları üretti",
    )

    # 3.8 Kaydet-Yükle (save/load) sonrası aynı skor
    tmp_dir = tempfile.mkdtemp(prefix="als_test_")
    try:
        als_model.save(tmp_dir)
        loaded_als = ALSModel.load(tmp_dir)
        loaded_adapter = ContractAdapter(
            model=loaded_als,
            popularity_model=pop_model,
            seen_matrix=seen_matrix,
            interaction_matrix=interaction_matrix,
            n_users=N_USERS,
            n_items=N_ITEMS,
        )
        df_before = adapter.score(users=val_uids[:10], candidates=cands_500[:10], mode="offline", target_items=cands_500[:10, 0].tolist())
        df_after = loaded_adapter.score(users=val_uids[:10], candidates=cands_500[:10], mode="offline", target_items=cands_500[:10, 0].tolist())
        sfi_before = adapter.score_from_items(chosen_3, sample_cands)
        sfi_after = loaded_adapter.score_from_items(chosen_3, sample_cands)
        check(
            np.allclose(df_before["score"].to_numpy(), df_after["score"].to_numpy(), atol=1e-12)
            and np.allclose(sfi_before["score"].to_numpy(), sfi_after["score"].to_numpy(), atol=1e-12),
            "save() ve load() sonrası hem score() hem score_from_items() birebir aynı skorları üretti",
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # 3.9 Mevcut test paketlerinin (test_exclude_seen.py 39/39 ve test_contract.py 27/27) kontrolü
    print("\n--- Mevcut Test Paketlerinin Korunumu ---")
    p1 = subprocess.run(
        [sys.executable, "-X", "utf8", str(PROJECT_ROOT / "tests" / "test_exclude_seen.py")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    check(p1.returncode == 0 and "39/39" in p1.stdout, "tests/test_exclude_seen.py geçti (39/39)")

    p2 = subprocess.run(
        [sys.executable, "-X", "utf8", str(PROJECT_ROOT / "tests" / "test_contract.py")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    check(p2.returncode == 0 and "27 geçti, 0 kaldı" in p2.stdout, "tests/test_contract.py geçti (27/27)")

    print("\n" + "=" * 78)
    print(f"SONUÇ: {PASS_COUNT} geçti, {FAIL_COUNT} kaldı.")
    print("=" * 78)
    if FAIL_COUNT > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
