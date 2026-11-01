"""
tests/test_contract.py
ContractAdapter (src/contract.py) ortak sözleşme doğrulama betiği (düz Python, pytest gerektirmez).

Çalıştırma:
    python tests/test_contract.py
"""

import os
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.contract import CONTRACT_COLUMNS, ContractAdapter
from src.data_loader import N_ITEMS, N_USERS, load_data
from src.evaluation import evaluate
from src.models.item_knn import ItemKNNModel
from src.models.popularity import PopularityModel

PASS_COUNT = 0
FAIL_COUNT = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  [GECTI] {name}")
    else:
        FAIL_COUNT += 1
        msg = f"  [KALDI] {name}"
        if detail:
            msg += f" -> {detail}"
        print(msg)


class ConstantDummyModel:
    """Tüm adaylara sabit skor döndüren sahte model (Test 4 için)."""

    def __init__(self, const_val: float = 0.42):
        self.const_val = float(const_val)

    def score(self, user_idx: int, candidate_items: np.ndarray) -> np.ndarray:
        return np.full(len(candidate_items), self.const_val, dtype=np.float64)

    def score_from_items(self, liked_items: np.ndarray, candidate_items: np.ndarray) -> np.ndarray:
        return np.full(len(candidate_items), self.const_val, dtype=np.float64)


def main() -> None:
    print("=" * 72)
    print("HYBRID CONTRACT ADAPTER DOĞRULAMA TESTLERİ (tests/test_contract.py)")
    print("=" * 72)

    # Veri ve modelleri yükle
    print("\nVeri setleri ve kayıtlı modeller yükleniyor...")
    data = load_data(os.path.join(PROJECT_ROOT, "data"))
    n_users = N_USERS
    n_items = N_ITEMS

    interaction_matrix = data["interaction_matrix"]
    seen_matrix = data["seen_matrix"]
    val_candidates = data["val_candidates"]
    test_candidates = data["test_candidates"]

    pop_model = PopularityModel.load(os.path.join(PROJECT_ROOT, "saved_models", "popularity.pkl"))
    item_knn = ItemKNNModel.load(os.path.join(PROJECT_ROOT, "saved_models", "itemknn"))

    adapter = ContractAdapter(
        model=item_knn,
        popularity_model=pop_model,
        seen_matrix=seen_matrix,
        interaction_matrix=interaction_matrix,
        n_users=n_users,
        n_items=n_items,
    )

    # Val ilk 2000 kullanıcıyı hazırla
    val_uids = sorted(val_candidates.keys())[:2000]
    val_cands_2d = np.array([val_candidates[u] for u in val_uids], dtype=np.int64)
    val_targets = [int(val_candidates[u][0]) for u in val_uids]

    # ------------------------------------------------------------------
    # TEST 1: Val'de ilk 2000 kullanıcı için NaN/inf yok ve tüm skorlar [0, 1]
    # ------------------------------------------------------------------
    print("\n--- TEST 1: Val İlk 2000 Kullanıcı Çıktı Yapısı, NaN/inf ve [0, 1] Aralığı ---")
    df_val_2000 = adapter.score(
        users=val_uids,
        candidates=val_cands_2d,
        mode="offline",
        target_items=val_targets,
    )

    check(
        list(df_val_2000.columns) == CONTRACT_COLUMNS,
        "Sütun isimleri tam olarak ['user_idx', 'item_idx', 'score', 'cf_known']",
        f"Sütunlar: {list(df_val_2000.columns)}",
    )
    check(
        len(df_val_2000) == 2000 * 100,
        "2000 kullanıcı x 100 aday = 200,000 satır üretildi",
        f"Satır sayısı: {len(df_val_2000)}",
    )
    scores_arr = df_val_2000["score"].to_numpy()
    check(
        not np.isnan(scores_arr).any() and not np.isinf(scores_arr).any(),
        "Skorlarda hiçbir NaN veya inf değer yok",
    )
    check(
        bool(np.all((scores_arr >= 0.0) & (scores_arr <= 1.0))),
        f"Tüm skorlar [0.0, 1.0] aralığında (min={scores_arr.min():.6f}, max={scores_arr.max():.6f})",
    )

    # Ortak 1D candidates dizisi desteğini de doğrula
    shared_1d = val_cands_2d[0]
    df_shared = adapter.score(users=val_uids[:10], candidates=shared_1d, mode="live")
    check(
        len(df_shared) > 0 and list(df_shared.columns) == CONTRACT_COLUMNS,
        "Ortak 1D candidates dizisi tüm kullanıcılara başarıyla uygulandı",
    )

    # ------------------------------------------------------------------
    # TEST 2: Her kullanıcıda cf_known=True adaylar için min=0 ve max=1 (skorlar eşit değilse)
    # ------------------------------------------------------------------
    print("\n--- TEST 2: Kullanıcı Bazında cf_known=True Min=0.0 ve Max=1.0 Kontrolü ---")
    min_max_ok = True
    equal_score_users = 0
    bad_user_detail = ""

    for uid, group in df_val_2000.groupby("user_idx", sort=False):
        known_scores = group.loc[group["cf_known"], "score"].to_numpy()
        if len(known_scores) <= 1:
            if not np.all(known_scores == 0.0):
                min_max_ok = False
                bad_user_detail = f"user={uid}, tek aday skoru != 0.0"
                break
            continue
        # Ham skorları kontrol et: eşit mi değil mi?
        known_items = group.loc[group["cf_known"], "item_idx"].to_numpy()
        raw = item_knn.score(int(uid), known_items)
        if np.max(raw) > np.min(raw):
            if not (np.isclose(known_scores.min(), 0.0, atol=1e-12) and np.isclose(known_scores.max(), 1.0, atol=1e-12)):
                min_max_ok = False
                bad_user_detail = f"user={uid}, min={known_scores.min()}, max={known_scores.max()}"
                break
        else:
            equal_score_users += 1
            if not np.all(known_scores == 0.0):
                min_max_ok = False
                bad_user_detail = f"user={uid} (eşit ham skor), norm skorlar 0.0 değil"
                break

    check(
        min_max_ok,
        f"2000 kullanıcının tamamında cf_known=True için min=0.0, max=1.0 (eşit ham skorlu {equal_score_users} kullanıcıda 0.0)",
        bad_user_detail,
    )

    # ------------------------------------------------------------------
    # TEST 3: Sıra korunumu (cf_known=True adaylarda argsort eşitliği)
    # ------------------------------------------------------------------
    print("\n--- TEST 3: Sıra Korunumu (cf_known=True Adaylarda Ham ve Normalize Sıralama Eşitliği) ---")
    order_ok = True
    order_detail = ""
    for uid, group in df_val_2000.groupby("user_idx", sort=False):
        known_sub = group[group["cf_known"]]
        known_items = known_sub["item_idx"].to_numpy()
        norm_s = known_sub["score"].to_numpy()
        raw_s = item_knn.score(int(uid), known_items)

        if np.max(raw_s) > np.min(raw_s):
            order_raw = np.argsort(-raw_s, kind="stable")
            order_norm = np.argsort(-norm_s, kind="stable")
            if not np.array_equal(order_raw, order_norm):
                order_ok = False
                order_detail = f"user={uid} için argsort uyuşmazlığı"
                break

    check(
        order_ok,
        "2000 kullanıcının tamamında cf_known=True için argsort(-norm_score) == argsort(-raw_score)",
        order_detail,
    )

    # ------------------------------------------------------------------
    # TEST 4: Eşit skor durumu (sabit skor veren sahte model ile tüm çıktı 0.0)
    # ------------------------------------------------------------------
    print("\n--- TEST 4: Eşit Skor ve Tek Bilinen Aday Durumu ---")
    dummy_adapter = ContractAdapter(
        model=ConstantDummyModel(const_val=3.14),
        popularity_model=pop_model,
        seen_matrix=seen_matrix,
        interaction_matrix=interaction_matrix,
        n_users=n_users,
        n_items=n_items,
    )
    df_const = dummy_adapter.score(
        users=val_uids[:20],
        candidates=val_cands_2d[:20],
        mode="offline",
        target_items=val_targets[:20],
    )
    check(
        bool(np.all(df_const["score"].to_numpy() == 0.0)),
        "Sabit skor üreten modelde tüm çıktı skorları 0.0",
    )

    # Tek bilinen aday durumu (kullanıcının görmediği tek bir cf_known=True aday + cf_known=False adaylar)
    unknown_cf_items = np.where(~adapter.item_cf_known)[0]
    unseen_known_one = int(val_candidates[val_uids[0]][0])
    df_single_known = adapter.score(
        users=[val_uids[0]],
        candidates=[unseen_known_one, int(unknown_cf_items[0])],
        mode="live",
    )
    check(
        len(df_single_known) == 2
        and int(df_single_known["cf_known"].sum()) == 1
        and bool(np.all(df_single_known["score"].to_numpy() == 0.0)),
        "Tek cf_known=True aday olduğunda normalize skor 0.0",
    )

    # ------------------------------------------------------------------
    # TEST 5: cf_known=False ürünlerde score=0.0 ve normalizasyonu bozmuyor
    # ------------------------------------------------------------------
    print("\n--- TEST 5: cf_known=False Ürün Davranışı ve Normalizasyon İzolasyonu ---")
    check(
        len(unknown_cf_items) == 243,
        f"interaction_matrix sütun toplamı 0 olan ürün sayısı = {len(unknown_cf_items)} (beklenen: 243)",
    )

    test_uid = val_uids[0]
    # Bu kullanıcı için pozitif çeşitliliği olan 5 bilinen ürün + 3 cf_known=False ürün seçelim
    known_sample = val_candidates[test_uid][:10]
    cold_sample = unknown_cf_items[:3]
    mixed_cands = np.concatenate([known_sample, cold_sample])

    df_only_known = adapter.score(
        users=[test_uid],
        candidates=known_sample,
        mode="offline",
        target_items=int(known_sample[0]),
    )
    df_mixed = adapter.score(
        users=[test_uid],
        candidates=mixed_cands,
        mode="offline",
        target_items=int(known_sample[0]),
    )

    cold_rows = df_mixed[~df_mixed["cf_known"]]
    known_rows_in_mixed = df_mixed[df_mixed["cf_known"]]

    check(
        len(cold_rows) == len(cold_sample)
        and bool(np.all(cold_rows["score"].to_numpy() == 0.0)),
        "cf_known=False ürünlerin tamamında cf_known=False ve score=0.0",
    )
    check(
        np.allclose(
            df_only_known["score"].to_numpy(),
            known_rows_in_mixed["score"].to_numpy(),
            atol=1e-12,
        ),
        "cf_known=False ürünlerin eklenmesi cf_known=True ürünlerin normalize skorlarını hiç değiştirmedi",
    )

    # ------------------------------------------------------------------
    # TEST 6: Offline modda hedef ürün çıktıda var; live modda bilinen ürünler çıktıda yok
    # ------------------------------------------------------------------
    print("\n--- TEST 6: Live vs Offline Modda Bilinen Ürünlerin Çıkarılması ---")
    u6 = val_uids[0]
    u6_seen = seen_matrix.getrow(u6).indices
    u6_unseen = val_candidates[u6][:5]
    cands_with_seen = np.concatenate([u6_seen[:3], u6_unseen])

    # Live mod: u6_seen[:3] çıktıda olmamalı
    df_live = adapter.score(users=[u6], candidates=cands_with_seen, mode="live")
    live_items = set(df_live["item_idx"].tolist())
    check(
        len(live_items.intersection(set(u6_seen[:3].tolist()))) == 0
        and len(df_live) == len(u6_unseen),
        "Live modda kullanıcının seen_matrix'teki ürünleri adaylardan tamamen çıkarıldı",
    )

    # Offline mod: hedef ürün (ve sızıntı testi için seen içinden bir hedef ürün) çıktıda kalmalı
    offline_target_unseen = int(u6_unseen[0])
    df_offline = adapter.score(
        users=[u6],
        candidates=cands_with_seen,
        mode="offline",
        target_items=offline_target_unseen,
    )
    offline_items = set(df_offline["item_idx"].tolist())
    check(
        offline_target_unseen in offline_items
        and len(offline_items.intersection(set(u6_seen[:3].tolist()))) == 0,
        "Offline modda hedef ürün korundu, hedef olmayan seen ürünler çıkarıldı",
    )

    # Hedef ürün yanlışlıkla seen_matrix içindeyse bile offline modda korunmalı (uyarı vererek)
    leak_target = int(u6_seen[0])
    with warnings.catch_warnings(record=True) as w_leak:
        warnings.simplefilter("always")
        df_offline_leak = adapter.score(
            users=[u6],
            candidates=cands_with_seen,
            mode="offline",
            target_items=leak_target,
        )
    check(
        leak_target in set(df_offline_leak["item_idx"].tolist()) and len(w_leak) >= 1,
        "Offline modda hedef ürün seen_matrix içinde olsa dahi uyarı verilip çıktıda korundu",
    )

    # ------------------------------------------------------------------
    # TEST 7: Bilinmeyen kullanıcı, bilinmeyen ürün, boş aday listesi, negatif/geçersiz indeks
    # ------------------------------------------------------------------
    print("\n--- TEST 7: Sınır Durumlar (Bilinmeyen Kullanıcı/Ürün, Boş Aday, Negatif ve Tip Hataları) ---")
    # 7a. Bilinmeyen kullanıcı (n_users ve üstü) -> uyarı + Popularity fallback
    unknown_uid = n_users + 50
    sample_cands = val_candidates[val_uids[0]][:20]
    with warnings.catch_warnings(record=True) as w_user:
        warnings.simplefilter("always")
        df_unk_user = adapter.score(users=[unknown_uid], candidates=sample_cands, mode="live")

    expected_pop_raw = pop_model.score(0, sample_cands[adapter.item_cf_known[sample_cands]])
    expected_pop_norm = (expected_pop_raw - expected_pop_raw.min()) / (
        expected_pop_raw.max() - expected_pop_raw.min()
    )
    actual_unk_known = df_unk_user.loc[df_unk_user["cf_known"], "score"].to_numpy()
    check(
        len(w_user) >= 1
        and len(df_unk_user) == len(sample_cands)
        and np.allclose(actual_unk_known, expected_pop_norm, atol=1e-12),
        "Bilinmeyen kullanıcı (n_users üstü) hata fırlatmadı, uyarı verdi ve Popularity skorlarını normalize etti",
    )

    # 7b. Bilinmeyen ürün (n_items ve üstü) ve negatif ürün indeksi -> uyarı, cf_known=False, score=0.0
    cands_with_invalid = np.append(sample_cands[:5], [n_items + 10, -5])
    with warnings.catch_warnings(record=True) as w_item:
        warnings.simplefilter("always")
        df_unk_item = adapter.score(users=[val_uids[0]], candidates=cands_with_invalid, mode="live")

    inv_rows = df_unk_item[df_unk_item["item_idx"].isin([n_items + 10, -5])]
    check(
        len(w_item) >= 1
        and len(inv_rows) == 2
        and bool(np.all(~inv_rows["cf_known"]))
        and bool(np.all(inv_rows["score"] == 0.0)),
        "Bilinmeyen (n_items+) ve negatif (-5) ürün indekslerinde uyarı verildi, cf_known=False ve score=0.0 atandı",
    )

    # 7c. Negatif kullanıcı indeksi -> uyarı + Popularity fallback (seen_matrix[-1] okunmaz)
    with warnings.catch_warnings(record=True) as w_neg_u:
        warnings.simplefilter("always")
        df_neg_user = adapter.score(users=[-1], candidates=sample_cands, mode="live")
    check(
        len(w_neg_u) >= 1
        and len(df_neg_user) == len(sample_cands)
        and np.allclose(
            df_neg_user.loc[df_neg_user["cf_known"], "score"].to_numpy(),
            expected_pop_norm,
            atol=1e-12,
        ),
        "Negatif kullanıcı indeksinde (-1) uyarı verildi ve seen_matrix[-1] sızıntısı olmadan Popularity'ye düşüldü",
    )

    # 7d. Boş aday listesi -> boş DataFrame, doğru sütunlar
    df_empty_cands = adapter.score(users=[val_uids[0]], candidates=[], mode="live")
    df_empty_users = adapter.score(users=[], candidates=sample_cands, mode="live")
    check(
        isinstance(df_empty_cands, pd.DataFrame)
        and len(df_empty_cands) == 0
        and list(df_empty_cands.columns) == CONTRACT_COLUMNS
        and len(df_empty_users) == 0,
        "Boş aday veya boş kullanıcı listesinde çökmeden doğru sütunlu boş DataFrame döndürüldü",
    )

    # 7e. Geçersiz veri tipleri (float, str, bool) -> TypeError (filtering.py ile tutarlı)
    type_err_ok = True
    for bad_input in [[1.5, 2.0], ["a", "b"], [True, False]]:
        try:
            adapter.score(users=bad_input, candidates=sample_cands, mode="live")
            type_err_ok = False
        except TypeError:
            pass
        try:
            adapter.score(users=[0], candidates=bad_input, mode="live")
            type_err_ok = False
        except TypeError:
            pass
    check(
        type_err_ok,
        "Tam sayı olmayan (float, str, bool) kullanıcı/ürün girdilerinde TypeError fırlatıldı",
    )

    # ------------------------------------------------------------------
    # TEST 8: score_from_items senaryoları
    # ------------------------------------------------------------------
    print("\n--- TEST 8: score_from_items (3 Geçerli, 1 Geçerli + 2 Geçersiz, 0 Geçerli, Seçilenlerin Çıkarılması) ---")
    base_cands = val_candidates[val_uids[0]][:20].copy()
    chosen_3 = base_cands[:3].copy()  # ilk 3 aday seçilen oyun olsun

    # 8a. 3 geçerli oyun (seçilen oyunlar aday havuzunda da var -> çıktıdan çıkarılmalı)
    df_sfi_3 = adapter.score_from_items(item_idxs=chosen_3, candidates=base_cands)
    sfi_items_3 = set(df_sfi_3["item_idx"].tolist())
    check(
        len(df_sfi_3) == len(base_cands) - 3
        and len(sfi_items_3.intersection(set(chosen_3.tolist()))) == 0
        and bool(np.all(df_sfi_3["user_idx"] == -1)),
        "score_from_items (3 geçerli oyun): seçilen oyunlar çıktıdan çıkarıldı ve user_idx=-1 yazıldı",
    )
    sfi_known_3 = df_sfi_3.loc[df_sfi_3["cf_known"], "score"].to_numpy()
    check(
        np.isclose(sfi_known_3.min(), 0.0, atol=1e-12)
        and np.isclose(sfi_known_3.max(), 1.0, atol=1e-12),
        "score_from_items (3 geçerli oyun): cf_known=True skorları [0, 1] aralığına normalize edildi",
    )

    # 8b. 1 geçerli + 2 geçersiz oyun
    chosen_mixed = [int(chosen_3[0]), n_items + 999, -10]
    with warnings.catch_warnings(record=True) as w_sfi_mix:
        warnings.simplefilter("always")
        df_sfi_mix = adapter.score_from_items(item_idxs=chosen_mixed, candidates=base_cands)
    df_sfi_1 = adapter.score_from_items(item_idxs=[int(chosen_3[0])], candidates=base_cands)
    check(
        len(w_sfi_mix) >= 1
        and len(df_sfi_mix) == len(base_cands) - 1
        and int(chosen_3[0]) not in set(df_sfi_mix["item_idx"].tolist())
        and np.allclose(df_sfi_mix["score"].to_numpy(), df_sfi_1["score"].to_numpy(), atol=1e-12),
        "score_from_items (1 geçerli + 2 geçersiz oyun): geçersizler uyarılıp yok sayıldı, tek geçerli oyunla aynı sonucu üretti",
    )

    # 8c. Hiç geçerli oyun yok -> Popularity'ye düşüş + cf_known doldurulması
    cold_and_known_cands = np.concatenate([base_cands[:10], unknown_cf_items[:2]])
    with warnings.catch_warnings(record=True) as w_sfi_none:
        warnings.simplefilter("always")
        df_sfi_none = adapter.score_from_items(
            item_idxs=[n_items + 100, -50],
            candidates=cold_and_known_cands,
        )
    pop_raw_sub = pop_model.score(0, base_cands[:10][adapter.item_cf_known[base_cands[:10]]])
    pop_norm_sub = (pop_raw_sub - pop_raw_sub.min()) / (pop_raw_sub.max() - pop_raw_sub.min())
    check(
        len(w_sfi_none) >= 1
        and bool(np.all(df_sfi_none.loc[~df_sfi_none["cf_known"], "score"] == 0.0))
        and np.allclose(
            df_sfi_none.loc[df_sfi_none["cf_known"], "score"].to_numpy(),
            pop_norm_sub,
            atol=1e-12,
        ),
        "score_from_items (0 geçerli oyun): uyarı verilerek Popularity'ye düşüldü ve cf_known sütunu doğru dolduruldu",
    )

    # ------------------------------------------------------------------
    # TEST 9: Mevcut test_exclude_seen.py ve Item-kNN val HR/NDCG (0.6400 / 0.4520) kontrolü
    # ------------------------------------------------------------------
    print("\n--- TEST 9: Mevcut test_exclude_seen.py ve Item-kNN Val Metriklerinin Korunumu ---")
    res_proc = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "tests", "test_exclude_seen.py")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    check(
        res_proc.returncode == 0 and "39" in res_proc.stdout and "0" in res_proc.stdout,
        "tests/test_exclude_seen.py başarıyla geçti (39/39, returncode=0)",
    )

    val_metrics = evaluate(
        item_knn.score,
        val_candidates,
        cold_items=data["cold_items"],
        verbose=False,
    )
    hr10 = round(val_metrics["overall"]["hit_rate"], 4)
    ndcg10 = round(val_metrics["overall"]["ndcg"], 4)
    check(
        hr10 == 0.6400 and ndcg10 == 0.4520,
        f"Item-kNN Val metrikleri değişmedi: HR@10={hr10:.4f} (beklenen 0.6400), NDCG@10={ndcg10:.4f} (beklenen 0.4520)",
    )

    # ------------------------------------------------------------------
    # TEST 10: cf_known=False ürün sayılarının val ve test aday havuzlarında raporlanması
    # (Test seti için model skoru HESAPLANMAZ, yalnızca indeks sayımı yapılır)
    # ------------------------------------------------------------------
    print("\n--- TEST 10: Val ve Test Aday Havuzlarında cf_known=False Ürün Sayımı ---")
    total_catalog_unknown = int(np.sum(~adapter.item_cf_known))

    val_all_cands = np.concatenate(list(val_candidates.values()))
    val_unique_cands = np.unique(val_all_cands)
    val_total_false_slots = int(np.sum(~adapter.item_cf_known[val_all_cands]))
    val_unique_false_items = int(np.sum(~adapter.item_cf_known[val_unique_cands]))
    val_pos_targets = np.array([c[0] for c in val_candidates.values()], dtype=np.int64)
    val_pos_false = int(np.sum(~adapter.item_cf_known[val_pos_targets]))

    test_all_cands = np.concatenate(list(test_candidates.values()))
    test_unique_cands = np.unique(test_all_cands)
    test_total_false_slots = int(np.sum(~adapter.item_cf_known[test_all_cands]))
    test_unique_false_items = int(np.sum(~adapter.item_cf_known[test_unique_cands]))
    test_pos_targets = np.array([c[0] for c in test_candidates.values()], dtype=np.int64)
    test_pos_false = int(np.sum(~adapter.item_cf_known[test_pos_targets]))

    print(f"  * Katalog geneli (n_items={n_items}): cf_known=False ürün sayısı = {total_catalog_unknown}")
    print(
        f"  * VAL aday havuzu ({len(val_candidates)} kullanıcı, {len(val_all_cands)} toplam aday satırı, {len(val_unique_cands)} tekil ürün):\n"
        f"      - Toplam cf_known=False aday görünümü (satır sayısı): {val_total_false_slots}\n"
        f"      - Tekil (unique) cf_known=False ürün sayısı        : {val_unique_false_items}\n"
        f"      - Hedef (pozitif) ürünler içinde cf_known=False    : {val_pos_false}"
    )
    print(
        f"  * TEST aday havuzu ({len(test_candidates)} kullanıcı, {len(test_all_cands)} toplam aday satırı, {len(test_unique_cands)} tekil ürün):\n"
        f"      - Toplam cf_known=False aday görünümü (satır sayısı): {test_total_false_slots}\n"
        f"      - Tekil (unique) cf_known=False ürün sayısı        : {test_unique_false_items}\n"
        f"      - Hedef (pozitif) ürünler içinde cf_known=False    : {test_pos_false}"
    )
    check(
        total_catalog_unknown == 243,
        "Val ve Test aday havuzlarında cf_known=False sayımları model skoru hesaplanmadan raporlandı",
    )

    print("\n" + "=" * 72)
    print(f"SONUÇ: {PASS_COUNT} geçti, {FAIL_COUNT} kaldı.")
    print("=" * 72)
    if FAIL_COUNT > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
