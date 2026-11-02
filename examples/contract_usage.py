"""
examples/contract_usage.py
==========================
Hibrit (CF + Content-Based) entegrasyonu için Item-kNN modelinin ve
ContractAdapter sınıfının uçtan uca kullanım örneği.

Bu betik çalıştırıldığı dizinden bağımsızdır (tüm yollar proje kök dizinine
göre otomatik çözülür).

Çalıştırma (proje kökünden veya herhangi bir alt klasörden):
    python examples/contract_usage.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Proje kök dizinini bul ve Python modül arama yoluna ekle
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.contract import ContractAdapter
from src.data_loader import N_ITEMS, N_USERS, load_data
from src.models.item_knn import ItemKNNModel
from src.models.popularity import PopularityModel


def main() -> None:
    print("=" * 75)
    print("ORTAK MODEL SÖZLEŞMESİ (ContractAdapter) KULLANIM ÖRNEĞİ")
    print("=" * 75)

    # ------------------------------------------------------------------
    # 1. Veriyi ve Kayıtlı Modelleri Yükle
    # ------------------------------------------------------------------
    data_dir = PROJECT_ROOT / "data"
    itemknn_dir = PROJECT_ROOT / "saved_models" / "itemknn"
    popularity_path = PROJECT_ROOT / "saved_models" / "popularity.pkl"

    print("\n[1] Veri seti yükleniyor...")
    data = load_data(str(data_dir))

    print("\n[2] Kayıtlı Item-kNN ve Popularity modelleri yükleniyor...")
    item_knn = ItemKNNModel.load(str(itemknn_dir))
    pop_model = PopularityModel.load(str(popularity_path))
    print(f"  Yüklenen CF Modeli : {item_knn}")

    # ------------------------------------------------------------------
    # 2. ContractAdapter Kurulumu
    # ------------------------------------------------------------------
    adapter = ContractAdapter(
        model=item_knn,
        popularity_model=pop_model,
        seen_matrix=data["seen_matrix"],
        interaction_matrix=data["interaction_matrix"],
        n_users=N_USERS,
        n_items=N_ITEMS,
    )

    val_candidates = data["val_candidates"]
    sample_val_users = sorted(val_candidates.keys())[:5]

    # ------------------------------------------------------------------
    # 3. Senaryo A: Validation'dan 5 Kullanıcı İçin Offline Değerlendirme
    # ------------------------------------------------------------------
    # Offline modda her kullanıcının 100'lük aday havuzu (indeks 0 = pozitif hedef)
    # verilir ve target_items ile hedef ürünün filtrelenmemesi garanti altına alınır.
    print("\n" + "-" * 75)
    print("[Senaryo A] Validation'dan 5 Kullanıcı İçin Offline score() Çağrısı")
    print("-" * 75)
    cands_5_users = np.array([val_candidates[u] for u in sample_val_users], dtype=np.int64)
    targets_5_users = [int(val_candidates[u][0]) for u in sample_val_users]

    df_offline = adapter.score(
        users=sample_val_users,
        candidates=cands_5_users,
        mode="offline",
        target_items=targets_5_users,
    )
    print(f"  Üretilen toplam satır sayısı: {len(df_offline)} (5 kullanıcı x 100 aday)")
    print(f"  Sütunlar ve veri tipleri    : {df_offline.dtypes.to_dict()}")
    print("\n  İlk kullanıcının en yüksek skorlu 5 adayı:")
    first_u = sample_val_users[0]
    top5_offline = (
        df_offline[df_offline["user_idx"] == first_u]
        .sort_values("score", ascending=False)
        .head(5)
    )
    print(top5_offline.to_string(index=False))

    # ------------------------------------------------------------------
    # 4. Senaryo B: 1 Kullanıcı İçin Canlı (Live) Mod Çağrısı
    # ------------------------------------------------------------------
    # Canlı modda kullanıcının geçmişte etkileşimde bulunduğu (label 0 ve 1)
    # ürünler aday listesinden otomatik çıkarılır, kalanlar [0, 1]'e normalize edilir.
    print("\n" + "-" * 75)
    print("[Senaryo B] 1 Kullanıcı İçin Canlı (mode='live') score() Çağrısı")
    print("-" * 75)
    live_uid = sample_val_users[0]
    seen_row_start = data["seen_matrix"].indptr[live_uid]
    seen_row_end = data["seen_matrix"].indptr[live_uid + 1]
    user_seen_items = data["seen_matrix"].indices[seen_row_start:seen_row_end]

    # Aday listesine kasıtlı olarak kullanıcının bildiği 2 ürünü ve 10 yeni adayı ekleyelim
    live_test_cands = np.concatenate([user_seen_items[:2], val_candidates[live_uid][:10]])
    df_live = adapter.score(
        users=[live_uid],
        candidates=live_test_cands,
        mode="live",
    )
    print(f"  Girdi aday sayısı : {len(live_test_cands)} (2 bilinen + 10 bilinmeyen ürün)")
    print(f"  Çıktı satır sayısı: {len(df_live)} (bilinen 2 ürün otomatik çıkarıldı)")
    print("\n  Canlı mod çıktısı (en yüksek skorlu 5 aday):")
    print(df_live.sort_values("score", ascending=False).head(5).to_string(index=False))

    # ------------------------------------------------------------------
    # 5. Senaryo C: Yeni Kullanıcı (Tercih Formundan 3 Oyun ile score_from_items)
    # ------------------------------------------------------------------
    # Sistemde user_idx'i olmayan yeni kullanıcı 3 oyun seçtiğinde user_idx = -1 döner.
    # Seçilen oyunlar aday havuzundaysa çıktıdan çıkarılır.
    print("\n" + "-" * 75)
    print("[Senaryo C] Yeni Kullanıcı: 3 Seçilen Oyun ile score_from_items() Çağrısı")
    print("-" * 75)
    chosen_games = [0, 10, 25]  # Kullanıcının formdan seçtiği 3 geçerli oyun
    candidate_pool = np.concatenate([chosen_games, val_candidates[live_uid][:15]])

    df_new_user = adapter.score_from_items(
        item_idxs=chosen_games,
        candidates=candidate_pool,
    )
    print(f"  Seçilen oyunlar (item_idx): {chosen_games}")
    print(f"  Girdi aday sayısı         : {len(candidate_pool)} (3 seçilen + 15 aday)")
    print(f"  Çıktı satır sayısı        : {len(df_new_user)} (seçilen 3 oyun çıkarıldı)")
    print(f"  Çıktı user_idx değeri     : {df_new_user['user_idx'].unique().tolist()}")
    print("\n  Yeni kullanıcı için en yüksek skorlu 5 öneri adayı:")
    print(df_new_user.sort_values("score", ascending=False).head(5).to_string(index=False))
    print("=" * 75)


if __name__ == "__main__":
    main()
