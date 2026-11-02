"""
scripts/check_score_from_items.py
=================================
Sadece VALIDATION setini kullanarak ContractAdapter.score_from_items() yolunu
sayısal olarak doğrulayan betik (Test setine ve mevcut dosyalara dokunulmaz).

Adımlar:
  1. Validation aday havuzundaki kullanıcılardan, train'de en az 5 pozitif
     (label == 1) etkileşimi olanlar arasından sabit tohumla (SEED = 42)
     rastgele 5.000 kullanıcı seçilir.
  2. Her kullanıcı için train pozitiflerinden rastgele (sabit tohumla) 3 oyun
     "tercih formu seçimi" olarak alınır.
  3. Kullanıcının val_candidates havuzunda ContractAdapter.score_from_items(3_oyun, havuz)
     çağrılır; çıktıdan [5000 x 100] skor matrisi kurulur ve mevcut evaluate()
     fonksiyonu değiştirilmeden HR@10 ve NDCG@10 hesaplanır.
  4. Aynı 5.000 kullanıcı alt kümesi üzerinde tam geçmişli normal score() ve
     Popularity sonuçları da mevcut evaluate() ile hesaplanır.
  5. Üç yöntemin (score_from_items / score (tam geçmiş) / Popularity) HR@10 ve
     NDCG@10 karşılaştırma tablosu basılır.
  6. Seçilen 3 oyunun çıktıda yer almadığı (aday havuzuna eklendiğinde dahi
     çıkarıldığı) ve hedef ürünün (indeks 0) çıktıda korunduğu doğrulanır.
"""

from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.contract import ContractAdapter
from src.data_loader import N_ITEMS, N_USERS, load_data
from src.evaluation import evaluate
from src.filtering import FILTERED_SCORE
from src.models.item_knn import ItemKNNModel
from src.models.popularity import PopularityModel

SAMPLE_SEED = 42
N_SAMPLE_USERS = 5000
MIN_TRAIN_POS = 5
N_CHOSEN_ITEMS = 3


def main() -> None:
    print("=" * 78)
    print("VALIDATION ÜZERİNDE score_from_items SAYISAL DOĞRULAMA (N = 5.000)")
    print("=" * 78)

    # 1. Veri ve modelleri yükle (yalnızca train ve validation kullanılır)
    print("\n[1] Veri seti ve kayıtlı modeller yükleniyor...")
    data = load_data(str(PROJECT_ROOT / "data"))
    interaction_matrix = data["interaction_matrix"]
    seen_matrix = data["seen_matrix"]
    val_candidates = data["val_candidates"]
    cold_items = data["cold_items"]

    pop_model = PopularityModel.load(str(PROJECT_ROOT / "saved_models" / "popularity.pkl"))
    item_knn = ItemKNNModel.load(str(PROJECT_ROOT / "saved_models" / "itemknn"))

    adapter = ContractAdapter(
        model=item_knn,
        popularity_model=pop_model,
        seen_matrix=seen_matrix,
        interaction_matrix=interaction_matrix,
        n_users=N_USERS,
        n_items=N_ITEMS,
    )

    # 2. Train'de en az 5 pozitifi (label==1) olan val kullanıcılarını bul
    indptr = interaction_matrix.indptr
    indices = interaction_matrix.indices
    user_pos_counts = np.diff(indptr)

    eligible_val_uids = sorted(
        uid for uid in val_candidates.keys() if user_pos_counts[uid] >= MIN_TRAIN_POS
    )
    print(
        f"\n[2] Kullanıcı seçimi (SEED={SAMPLE_SEED}):\n"
        f"  Toplam val kullanıcısı                  : {len(val_candidates):,}\n"
        f"  Train'de >= {MIN_TRAIN_POS} pozitifi olan val kullanıcısı: {len(eligible_val_uids):,}"
    )

    rng = np.random.default_rng(SAMPLE_SEED)
    sampled_uids = rng.choice(eligible_val_uids, size=N_SAMPLE_USERS, replace=False).tolist()
    sampled_candidates = {int(uid): val_candidates[int(uid)] for uid in sampled_uids}
    uid_to_row = {int(uid): idx for idx, uid in enumerate(sampled_uids)}

    # Her kullanıcı için train pozitiflerinden 3 oyun seç (sabit seed'li rng akışıyla)
    chosen_per_user = {}
    for uid in sampled_uids:
        u_pos_items = indices[indptr[uid] : indptr[uid + 1]]
        chosen_3 = rng.choice(u_pos_items, size=N_CHOSEN_ITEMS, replace=False)
        chosen_per_user[int(uid)] = chosen_3.astype(np.int64)

    avg_pos_sampled = float(np.mean([user_pos_counts[u] for u in sampled_uids]))
    median_pos_sampled = float(np.median([user_pos_counts[u] for u in sampled_uids]))
    print(
        f"  Seçilen örneklem büyüklüğü              : {len(sampled_uids):,} kullanıcı\n"
        f"  Örneklem train pozitif sayısı (ort/med) : {avg_pos_sampled:.2f} / {median_pos_sampled:.1f}\n"
        f"  Tercih formu için seçilen oyun sayısı   : {N_CHOSEN_ITEMS} ürün / kullanıcı"
    )

    # 3. ContractAdapter.score_from_items ile [5000 x 100] skor matrisi kur
    #    ve Adım 6 kontrollerini (seçilen 3 oyun çıktıda yok, hedef ürün çıktıda var) yap
    print("\n[3] ContractAdapter.score_from_items() çalıştırılıyor ve kontroller yapılıyor...")
    sfi_score_matrix = np.full((N_SAMPLE_USERS, 100), FILTERED_SCORE, dtype=np.float64)

    chosen_absent_in_val_cands_ok = True
    target_present_in_val_cands_ok = True
    chosen_actively_removed_when_injected_ok = True

    t0 = time.perf_counter()
    for row_idx, uid in enumerate(sampled_uids):
        cands = sampled_candidates[uid]
        target_item = int(cands[0])
        chosen_3 = chosen_per_user[uid]

        df_sfi = adapter.score_from_items(item_idxs=chosen_3, candidates=cands)
        out_items = df_sfi["item_idx"].to_numpy()
        out_scores = df_sfi["score"].to_numpy(dtype=np.float64)

        # Kontrol 6a: Seçilen 3 oyun çıktıda bulunmamalı
        if np.isin(chosen_3, out_items).any():
            chosen_absent_in_val_cands_ok = False

        # Kontrol 6b: Kullanıcının hedef ürünü (cands[0]) çıktıda kalmalı
        if len(out_items) == 0 or out_items[0] != target_item:
            target_present_in_val_cands_ok = False

        # [N x 100] matris satırını doldur
        if len(out_items) == 100 and np.array_equal(out_items, cands):
            sfi_score_matrix[row_idx, :] = out_scores
        else:
            item_to_score = dict(zip(out_items.tolist(), out_scores.tolist()))
            for j, cid in enumerate(cands):
                sfi_score_matrix[row_idx, j] = item_to_score.get(int(cid), FILTERED_SCORE)

        # Ek Kontrol 6c: Seçilen 3 oyun aday havuzunun içine kasıtlı eklendiğinde de
        # çıktıdan aktif olarak çıkarıldığını ve hedef ürünün kaldığını doğrula (ilk 500 kullanıcıda)
        if row_idx < 500:
            injected_cands = np.concatenate([chosen_3, cands])
            df_inj = adapter.score_from_items(item_idxs=chosen_3, candidates=injected_cands)
            inj_items = df_inj["item_idx"].to_numpy()
            if (
                np.isin(chosen_3, inj_items).any()
                or target_item not in inj_items
                or len(inj_items) != 100
            ):
                chosen_actively_removed_removed_ok = False
                chosen_actively_removed_when_injected_ok = False

    sfi_elapsed = time.perf_counter() - t0
    print(f"  score_from_items tamamlandı ({sfi_elapsed:.2f} sn, {(sfi_elapsed * 1000) / N_SAMPLE_USERS:.3f} ms/kullanıcı).")
    print(
        f"  [Kontrol 6a] Seçilen 3 oyun çıktıda YOK (5.000/5.000 kullanıcı)          : "
        f"{'GEÇTİ' if chosen_absent_in_val_cands_ok else 'KALDI'}"
    )
    print(
        f"  [Kontrol 6b] Hedef ürün (indeks 0) çıktıda KORUNDU (5.000/5.000 kullanıcı): "
        f"{'GEÇTİ' if target_present_in_val_cands_ok else 'KALDI'}"
    )
    print(
        f"  [Kontrol 6c] Seçilen 3 oyun aday havuzuna eklendiğinde de ÇIKARILDI      : "
        f"{'GEÇTİ' if chosen_actively_removed_when_injected_ok else 'KALDI'}"
    )

    # 4. Aynı 5.000 kullanıcı için normal ContractAdapter.score() (tam geçmiş) matrisi kur
    print("\n[4] Aynı 5.000 kullanıcı için normal score() (tam geçmiş) ve Popularity hesaplanıyor...")
    cands_2d = np.array([sampled_candidates[u] for u in sampled_uids], dtype=np.int64)
    targets_list = cands_2d[:, 0].tolist()

    df_full = adapter.score(
        users=sampled_uids,
        candidates=cands_2d,
        mode="offline",
        target_items=targets_list,
    )
    full_score_matrix = df_full["score"].to_numpy(dtype=np.float64).reshape(N_SAMPLE_USERS, 100)

    # Mevcut evaluate() fonksiyonunu değiştirmeden her üç yöntemi değerlendir
    def sfi_score_fn(uid: int, _cands: np.ndarray) -> np.ndarray:
        return sfi_score_matrix[uid_to_row[uid]]

    def full_score_fn(uid: int, _cands: np.ndarray) -> np.ndarray:
        return full_score_matrix[uid_to_row[uid]]

    res_pop = evaluate(
        score_fn=pop_model.score,
        candidates=sampled_candidates,
        cold_items=cold_items,
        verbose=False,
    )
    res_sfi = evaluate(
        score_fn=sfi_score_fn,
        candidates=sampled_candidates,
        cold_items=cold_items,
        verbose=False,
    )
    res_full = evaluate(
        score_fn=full_score_fn,
        candidates=sampled_candidates,
        cold_items=cold_items,
        verbose=False,
    )
    # Referans kontrol: doğrudan item_knn.score (adaptörsüz) ile de birebir aynı mı?
    res_raw_knn = evaluate(
        score_fn=item_knn.score,
        candidates=sampled_candidates,
        cold_items=cold_items,
        verbose=False,
    )

    # 5. Karşılaştırma Tablosu
    pop_m = res_pop["overall"]
    sfi_m = res_sfi["overall"]
    full_m = res_full["overall"]
    raw_m = res_raw_knn["overall"]

    print("\n" + "=" * 78)
    print(f"KARŞILAŞTIRMA TABLOSU (Validation Örneklemi: N = {N_SAMPLE_USERS:,}, Train Pozitif >= {MIN_TRAIN_POS})")
    print("=" * 78)
    print(
        f"{'Yöntem':<38} | {'HR@10':>10} | {'NDCG@10':>10} | {'Prec@10':>9} | {'Cov@10 (%)':>10}"
    )
    print("-" * 78)
    print(
        f"{'1. Popularity (Baseline)':<38} | {pop_m['hit_rate']:>10.4f} | {pop_m['ndcg']:>10.4f} | "
        f"{pop_m['precision']:>9.4f} | {pop_m['coverage_pct']:>9.2f}%"
    )
    print(
        f"{'2. ContractAdapter.score_from_items (3 oyun)':<38} | {sfi_m['hit_rate']:>10.4f} | {sfi_m['ndcg']:>10.4f} | "
        f"{sfi_m['precision']:>9.4f} | {sfi_m['coverage_pct']:>9.2f}%"
    )
    print(
        f"{'3. ContractAdapter.score (Tam Geçmiş)':<38} | {full_m['hit_rate']:>10.4f} | {full_m['ndcg']:>10.4f} | "
        f"{full_m['precision']:>9.4f} | {full_m['coverage_pct']:>9.2f}%"
    )
    print(
        f"{'   (Referans: Ham ItemKNNModel.score)':<38} | {raw_m['hit_rate']:>10.4f} | {raw_m['ndcg']:>10.4f} | "
        f"{raw_m['precision']:>9.4f} | {raw_m['coverage_pct']:>9.2f}%"
    )
    print("=" * 78)

    # Beklenti kontrolü: Popularity < score_from_items < score (tam geçmiş)
    hr_order_ok = pop_m["hit_rate"] < sfi_m["hit_rate"] < full_m["hit_rate"]
    ndcg_order_ok = pop_m["ndcg"] < sfi_m["ndcg"] < full_m["ndcg"]

    print("\n[5] Beklenti ve Fark Analizi:")
    print(
        f"  * score_from_items vs Popularity   : "
        f"ΔHR@10 = {sfi_m['hit_rate'] - pop_m['hit_rate']:+.4f}, "
        f"ΔNDCG@10 = {sfi_m['ndcg'] - pop_m['ndcg']:+.4f}"
    )
    print(
        f"  * score_from_items vs Tam Geçmiş   : "
        f"ΔHR@10 = {sfi_m['hit_rate'] - full_m['hit_rate']:+.4f}, "
        f"ΔNDCG@10 = {sfi_m['ndcg'] - full_m['ndcg']:+.4f}"
    )
    print(
        f"  * Sıralama Beklentisi (Popularity < score_from_items < score): "
        f"{'SAĞLANDI (GEÇTİ)' if (hr_order_ok and ndcg_order_ok) else 'SAĞLANMADI (İNCELE)'}"
    )

    if not (
        chosen_absent_in_val_cands_ok
        and target_present_in_val_cands_ok
        and chosen_actively_removed_when_injected_ok
        and hr_order_ok
        and ndcg_order_ok
    ):
        sys.exit(1)


if __name__ == "__main__":
    main()
