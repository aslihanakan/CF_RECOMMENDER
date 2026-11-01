"""
src/evaluation.py
=================
Collaborative Filtering modellerini adil biçimde değerlendiren modül.

GENEL TASARIM:
    Tüm modeller aynı `evaluate()` fonksiyonundan geçer. Model sadece
    score_fn(user_idx, candidate_items) -> np.ndarray[float] arayüzünü
    uygulamak zorundadır; metrik hesaplama ve raporlama bu modülde yapılır.
    Böylece popülerlik, item-CF, MF ve ileride hybrid modeller için
    kodun kopyalanmasına gerek kalmaz.

METRİKLER (K=10):
    - HitRate@10  : Top-10'da pozitif varsa 1, yoksa 0. Kullanıcı ortalaması.
    - NDCG@10     : Pozitifin sırasını dikkate alan iskontolu kazanım.
    - Precision@10: Her kullanıcının top-10'undaki isabet oranı.
    - Recall@10   : Toplam pozitifin ne kadarı top-10'a girdi.
    - Coverage@10 : Tüm kullanıcıların MODELE GÖRE sıralanmış top-10
                    listelerinde görünen benzersiz ürün sayısı / toplam ürün
                    sayısı (N_ITEMS = 25.610). Yüzde olarak raporlanır.

ÖNEMLİ NOTLAR:
    1. Her kullanıcının değerlendirme setinde TEK pozitif ürünü vardır.
       Bu yüzden matematiksel olarak:
           Recall@10    = HitRate@10          (tek pozitif var, ya içindedir ya değil)
           Precision@10 = HitRate@10 / 10     (top-10'da en fazla 1 hit mümkün)

    2. 100'lük aday havuzunda Coverage@10 YANILTICIDIR. Rastgele model bile
       ~%100 coverage verir (havuzdaki ürünler kullanıcıdan kullanıcıya
       farklılaşır, her kullanıcıda farklı 10 ürün seçilir). Bu yüzden
       coverage SADECE katalog kapsaması olarak yorumlanmamalı; aday
       havuzundaki çeşitlilik ölçüsü olarak okunmalıdır.

    3. Soğuk ürünler (train'de HIÇBIR etkileşimi olmayan, label 0 dahil)
       için model skor üretemez. Varsayılan strateji: soğuk ürüne -1e9
       skoru atanır (önceki versiyon -inf kullanıyordu).

       NEDEN -1e9 (sonlu), -inf değil:
           Tie-break mekanizması "score + rng.random() * 1e-10" gürültüsü
           ekler. -inf + herhangi_bir_sayi = -inf olur; yani soğuk ürünler
           arasında bile sıralama deterministik kalır (aday listesinin
           orijinal sırasına göre). Sonlu -1e9 kullanıldığında tüm soğuk
           ürünler arası tie-break da düzgün rastgele çalışır.

EŞİT SKOR TIE-BREAK:
    Aynı skora sahip ürünlerin sıralaması sabit bir seed (42) ile rastgele
    belirlenir. Bu, aday listesinin sırasının metriği etkilememesini garanti
    eder; aynı zamanda farklı çalıştırmalar arasında tekrarlanabilirlik sağlar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Optional

# Metrik hesabında kullanılacak K değeri
K = 10

# Tie-break için sabit rastgele tohum
TIE_BREAK_SEED = 42

# Toplam katalog büyüklüğü (coverage paydası için)
N_ITEMS_TOTAL = 25_610

# Soğuk ürünlere atanan varsayılan ceza skoru.
# NEDEN -1e9 (sonlu), -inf değil:
#   tie-break gürültüsü "score + rng.random() * 1e-10" şeklinde çalışır.
#   -inf + küçük_sayı = -inf olduğundan, soğuk ürünler arası sıralama
#   aday listesinin orijinal sırasına göre belirlenirdi (gürültü işe yaramaz).
#   Sonlu -1e9 ile soğuk ürünler arası sıralama da rastgele ve adil çalışır.
COLD_ITEM_PENALTY_DEFAULT = -1e9


# ---------------------------------------------------------------------------
# Ana değerlendirme fonksiyonu
# ---------------------------------------------------------------------------

def evaluate(
    score_fn: Callable[[int, np.ndarray], np.ndarray],
    candidates: dict,
    cold_items: Optional[set] = None,
    user_activity: Optional[pd.Series] = None,
    cold_item_penalty: float = COLD_ITEM_PENALTY_DEFAULT,
    verbose: bool = True,
) -> dict:
    """
    NE YAPAR:
        Verilen score_fn'i tüm kullanıcılar üzerinde çalıştırır; HitRate,
        NDCG, Precision, Recall ve Coverage metriklerini hesaplar.
        İsteğe bağlı olarak soğuk ürün ve aktivite kırılımı raporlar.

    NEDEN TEK BİR FONKSİYON:
        Her model (popülerlik, item-CF, MF, hybrid) buradan geçer.
        Metrik kodu kopyalanmaz -> tutarlı karşılaştırma mümkün olur.

    PARAMETRELER:
        score_fn : callable
            score_fn(user_idx: int, candidate_items: np.ndarray) -> np.ndarray
            candidate_items ile aynı uzunlukta float skor dizisi döndürür.
            YÜKSEK SKOR = daha iyi öneri.

        candidates : dict {user_idx -> np.ndarray[100]}
            _build_candidates() çıktısı. Her dizinin ilk elemanı (indeks 0)
            pozitif ürün; geri kalan 99 eleman negatif.

        cold_items : set, optional
            Train'de hiç görülmemiş item_idx kümesi (label 0 dahil).
            Verilirse soğuk ürünlere cold_item_penalty skoru atanır.

        user_activity : pd.Series, optional
            {user_idx -> train toplam etkileşim sayısı}.
            Verilirse aktivite kırılımı (düşük/orta/yüksek) ayrı raporlanır.

        cold_item_penalty : float
            Soğuk ürünlere atanan skor. Varsayılan: -1e9.
            NEDEN -1e9 (sonlu, -inf değil): tie-break gürültüsü (1e-10 büyüklüğünde)
            -inf üzerinde çalışmaz (-inf + küçük = -inf). Sonlu ceza ile soğuk
            ürünler arası sıralama da düzgün rastgele belirlenir.

        verbose : bool
            True ise metrikleri konsola basar.

    DÖNDÜRÜR:
        {
            "overall"  : dict  - tüm kullanıcıların metrikleri
            "cold_user": dict  - pozitifi soğuk ürün olan kullanıcıların metrikleri
            "activity" : dict  - aktivite kırılımı metrikleri (isteğe bağlı)
        }
    """
    hit_ranks = {}              # {user_idx -> int rank ya da None}
    cold_users = set()          # pozitifi soğuk ürün olan kullanıcılar
    top_k_per_user = {}         # {user_idx -> list[item_idx]} — gerçek top-K listesi

    rng = np.random.default_rng(TIE_BREAK_SEED)  # tekrarlanabilir tie-break

    for uid, cands in candidates.items():
        pos_item = cands[0]    # pozitif ürün her zaman ilk sıradadır

        # --- soğuk ürün tespiti ---
        pos_is_cold = (cold_items is not None) and (pos_item in cold_items)
        if pos_is_cold:
            cold_users.add(uid)

        # --- skor hesapla ---
        scores = score_fn(uid, cands).astype(np.float64)

        # --- soğuk ürünlere ceza uygula ---
        if cold_items is not None:
            # Vektörleştirilmiş maske: aday dizisindeki soğuk ürün konumları
            cold_mask = np.isin(cands, list(cold_items))
            scores[cold_mask] = cold_item_penalty

        # --- tie-break: eşit skorlara rastgele küçük gürültü ekle ---
        # Gürültü büyüklüğü 1e-10; gerçek skor farkları çok daha büyük olur.
        # NEDEN rng (default_rng) değil np.random: seed kontrolü için.
        tiebreak_noise = rng.random(len(scores)) * 1e-10
        scores_with_noise = scores + tiebreak_noise

        # --- azalan sıra → indeks dizisi ---
        sorted_indices = np.argsort(-scores_with_noise)   # küçükten büyüğe → ters

        # --- pozitifin sıralamadaki yeri (candidates dizisindeki indeks 0) ---
        rank = int(np.where(sorted_indices == 0)[0][0]) + 1   # 1-tabanlı

        hit_ranks[uid] = rank if rank <= K else None

        # --- gerçek top-K ürün listesini kaydet (coverage için) ---
        top_k_indices = sorted_indices[:K]                 # top-K'nın candidates içindeki konumları
        top_k_per_user[uid] = cands[top_k_indices].tolist()  # gerçek item_idx'lere dönüştür

    # Metrikleri hesapla
    overall_metrics = _compute_metrics(hit_ranks, top_k_per_user, K)

    # Soğuk kullanıcı metrikleri
    cold_hit_ranks    = {u: r for u, r in hit_ranks.items() if u in cold_users}
    cold_top_k        = {u: t for u, t in top_k_per_user.items() if u in cold_users}
    cold_metrics = _compute_metrics(cold_hit_ranks, cold_top_k, K) if cold_users else {}

    # Aktivite kırılımı
    activity_metrics = {}
    if user_activity is not None:
        from src.data_loader import get_activity_groups  # döngüsel bağımlılığı önle
        groups = get_activity_groups(user_activity)
        for name, gset in groups.items():
            group_hit_ranks = {u: r for u, r in hit_ranks.items() if u in gset}
            group_top_k     = {u: t for u, t in top_k_per_user.items() if u in gset}
            activity_metrics[name] = _compute_metrics(group_hit_ranks, group_top_k, K)

    result = {
        "overall"  : overall_metrics,
        "cold_user": cold_metrics,
        "activity" : activity_metrics,
    }

    if verbose:
        _print_results(result, cold_users)

    return result


# ---------------------------------------------------------------------------
# Metrik hesaplama
# ---------------------------------------------------------------------------

def _compute_metrics(
    hit_ranks: dict,
    top_k_per_user: dict,
    k: int = K,
) -> dict:
    """
    NE YAPAR:
        hit_ranks ve top_k_per_user sözlüklerinden HitRate, NDCG, Precision,
        Recall ve Coverage metriklerini hesaplar.

    NEDEN AYRI FONKSİYON:
        Hem overall hem cold_user hem de aktivite grupları aynı hesaplamayı
        kullanır; kod tekrarını önler.

    PARAMETRELER:
        hit_ranks      : {user_idx -> int rank ya da None}
            Pozitifin sıralaması (1-tabanlı, K'dan büyükse None).
        top_k_per_user : {user_idx -> list[item_idx]}
            Her kullanıcı için modelin ürettiği gerçek top-K ürün listesi.
            Coverage bu listeden hesaplanır (candidates'in ilk k elemanı değil).
        k              : int
            Liste kesim noktası (varsayılan 10).

    DÖNDÜRÜR:
        {
            "hit_rate"    : float,
            "ndcg"        : float,
            "precision"   : float,
            "recall"      : float,
            "coverage_pct": float,   -- benzersiz ürün sayısı / N_ITEMS_TOTAL * 100
            "coverage_n"  : int,     -- benzersiz ürün sayısı (ham)
            "n_users"     : int
        }

    COVERAGE NOTU:
        Coverage, modelin top-K listelerindeki BENZERSIZ ürün sayısını
        toplam katalog büyüklüğüne (25.610) böler. Rastgele modelde
        her kullanıcıya farklı 10 rastgele ürün düşer, bu yüzden
        yeterince fazla kullanıcıda neredeyse tüm katalogu kapsar (~%100).
        Popülerlik modeli ise herkese aynı ~10 popüler ürünü onerdiginden
        coverage dusuk cikar (~%19.8).
    """
    n = len(hit_ranks)
    if n == 0:
        return {"hit_rate": 0.0, "ndcg": 0.0, "precision": 0.0,
                "recall": 0.0, "coverage_pct": 0.0, "coverage_n": 0, "n_users": 0}

    hits      = np.array([1 if r is not None else 0 for r in hit_ranks.values()], dtype=np.float64)
    ndcg_vals = np.array(
        [1.0 / np.log2(r + 1) if r is not None else 0.0 for r in hit_ranks.values()],
        dtype=np.float64
    )

    hit_rate  = float(hits.mean())
    ndcg      = float(ndcg_vals.mean())
    precision = hit_rate / k    # tek pozitif -> precision = hr/K
    recall    = hit_rate        # tek pozitif -> recall = hr

    # Coverage: modelin top-K listelerindeki benzersiz ürün sayısı / toplam katalog
    # top_k_per_user zaten evaluate() içinde dolduruldu; doğrudan kullan.
    all_top_items = set()
    for uid in hit_ranks:
        if uid in top_k_per_user:
            all_top_items.update(top_k_per_user[uid])  # gerçek top-K item_idx'leri

    coverage_n   = len(all_top_items)
    coverage_pct = coverage_n / N_ITEMS_TOTAL * 100.0  # yüzde

    return {
        "hit_rate"    : hit_rate,
        "ndcg"        : ndcg,
        "precision"   : precision,
        "recall"      : recall,
        "coverage_pct": coverage_pct,
        "coverage_n"  : coverage_n,
        "n_users"     : n,
    }


# ---------------------------------------------------------------------------
# Konsol çıktısı
# ---------------------------------------------------------------------------

def _print_results(result: dict, cold_users: set) -> None:
    """
    NE YAPAR:
        evaluate() sonuçlarını okunaklı tablo formatında basar.
        Coverage hem yüzde (katalog bazlı) hem ham sayı olarak gösterilir.
    """
    def _fmt(m: dict) -> str:
        if not m:
            return "  (veri yok)"
        return (
            f"  HR@10={m['hit_rate']:.4f}  NDCG@10={m['ndcg']:.4f}  "
            f"Prec@10={m['precision']:.4f}  Recall@10={m['recall']:.4f}  "
            f"Cov@10={m['coverage_pct']:.1f}%({m['coverage_n']:,})  N={m['n_users']:,}"
        )

    print("\n" + "=" * 80)
    print("DEGERLENDIRME SONUCLARI")
    print("=" * 80)
    print(f"[GENEL] {_fmt(result['overall'])}")

    if result["cold_user"]:
        print(f"[SOGUK] {_fmt(result['cold_user'])}  (soguk poz. {len(cold_users)} kullanici)")

    if result["activity"]:
        label_map = {"low": "Dusuk (=3)", "medium": "Orta (4-6)", "high": "Yuksek (7+)"}
        for key, m in result["activity"].items():
            print(f"[{label_map.get(key, key):12s}] {_fmt(m)}")

    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Doğrulama: if __name__ == "__main__"
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Bu blok evaluation.py'nin doğruluğunu iki basit model ile test eder:

    1. RASTGELE MODEL: Her adaya rastgele skor ver.
       Beklenen: HR@10 ~0.100, NDCG@10 ~0.046
       Coverage: ~%100 (her kullaniciya farkli 10 rastgele urun -> cok cesitli)

    2. POPULARLIK MODELI: Her adaya train'deki pozitif sayisini skor ver.
       Beklenen val : HR@10 ~0.549, NDCG@10 ~0.337
       Beklenen test: HR@10 ~0.507, NDCG@10 ~0.309
       Coverage: ~%19.8 (herkes ayni populer urunleri aliyor -> az cesit)

    Sayilar tutmazsa su alanlari kontrol et:
      - Aday havuzu (pozitif indeks 0 mi?)
      - label filtresi (sadece label==1 kullanicilar mi?)
      - Tie-break seed (sabit mi?)
      - Indeks eslemesi (user_idx/item_idx dogru mu?)
      - cold_item_penalty: -1e9 (sonlu) mi, -inf mi?
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from src.data_loader import load_data

    print("Veri yukleniyor...")
    data = load_data("data")

    mat           = data["interaction_matrix"]
    val_cands     = data["val_candidates"]
    test_cands    = data["test_candidates"]
    cold_items    = data["cold_items"]
    user_activity = data["user_activity"]

    # Populerlik skoru: her item'in train'deki pozitif etkilesim sayisi
    # mat.sum(axis=0) -> (1, N_ITEMS) matris; flatten ile 1-D diziye cevir
    item_pop = np.asarray(mat.sum(axis=0)).flatten()   # shape: (25610,)

    # ----------------------------------------------------------------
    # 1. Rastgele model
    # ----------------------------------------------------------------
    # Her kullaniciya uid-bazli deterministik rng -> tekrarlanabilir sonuc
    def random_score_fn(uid: int, cands: np.ndarray) -> np.ndarray:
        """Her adaya rastgele skor ver (uid seed'li -> tekrarlanabilir)."""
        return np.random.default_rng(uid).random(len(cands))

    print("\n--- RASTGELE MODEL (VAL) ---")
    res_rand_val = evaluate(
        score_fn=random_score_fn,
        candidates=val_cands,
        cold_items=cold_items,
        verbose=True,
    )

    print("\n--- RASTGELE MODEL (TEST) ---")
    res_rand_test = evaluate(
        score_fn=random_score_fn,
        candidates=test_cands,
        cold_items=cold_items,
        verbose=True,
    )

    # ----------------------------------------------------------------
    # 2. Popularlik modeli
    # ----------------------------------------------------------------
    def popularity_score_fn(uid: int, cands: np.ndarray) -> np.ndarray:
        """
        Her adaya train'deki pozitif etkilesim sayisini skor olarak ver.
        Populer urun -> yuksek skor -> listenin ustunde.
        Vektorlestirilmis indeksleme: dongu yok, hizli.
        """
        return item_pop[cands]

    print("\n--- POPULARLIK MODELI (VAL) ---")
    res_pop_val = evaluate(
        score_fn=popularity_score_fn,
        candidates=val_cands,
        cold_items=cold_items,
        user_activity=user_activity,
        verbose=True,
    )

    print("\n--- POPULARLIK MODELI (TEST) ---")
    res_pop_test = evaluate(
        score_fn=popularity_score_fn,
        candidates=test_cands,
        cold_items=cold_items,
        user_activity=user_activity,
        verbose=True,
    )

    # ----------------------------------------------------------------
    # 3. Referans karsilastirma ozeti
    # ----------------------------------------------------------------
    print("=" * 75)
    print("REFERANS KARSILASTIRMA")
    print("=" * 75)
    hdr = f"{'Model':<15} {'Split':<5} {'HR@10':>8} {'bek':>8} {'NDCG@10':>9} {'bek':>8} {'Cov%':>8} {'bek':>8}"
    print(hdr)
    print("-" * 75)

    rows = [
        ("Rastgele",   "val",  res_rand_val,  0.100, 0.046, 100.0),
        ("Rastgele",   "test", res_rand_test, 0.100, 0.046, 100.0),
        ("Popularlik", "val",  res_pop_val,   0.549, 0.337,  19.8),
        ("Popularlik", "test", res_pop_test,  0.507, 0.309,  19.8),
    ]
    for model, split, res, hr_exp, ndcg_exp, cov_exp in rows:
        m = res["overall"]
        print(f"{model:<15} {split:<5} "
              f"{m['hit_rate']:>8.4f} {hr_exp:>8.3f} "
              f"{m['ndcg']:>9.4f} {ndcg_exp:>8.3f} "
              f"{m['coverage_pct']:>8.1f} {cov_exp:>8.1f}")
    print("=" * 75)
