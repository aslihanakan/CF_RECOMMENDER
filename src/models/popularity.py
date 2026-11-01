"""
src/models/popularity.py
========================
Popülerlik tabanlı öneri modeli.

TEMEL MANTIK:
    Eğitimde her ürünün kaç kez beğenildiğini (label==1 sayısı) hesapla.
    Değerlendirmede aday ürünleri bu sayıya göre sırala; en çok beğenilen
    ürünü en üste koy.

NEDEN BAŞLANGIÇ NOKTASI OLARAK POPÜLERLİK:
    - Parametre yoktur, aşırı öğrenme (overfitting) yapmaz.
    - "Çoğunluk için ne iyi?" sorusunu doğrudan yanıtlar.
    - Diğer modeller (item-CF, MF) için taban çizgisi (baseline) oluşturur.
    - Hesaplama maliyeti O(|train|) fit, O(|cands|) score; çok hızlı.

ARAYÜZ (evaluation.py ile uyumluluk):
    evaluate(score_fn=model.score, ...) çağrısında score_fn imzası:
        score_fn(uid: int, cands: np.ndarray) -> np.ndarray
    PopularityModel.score() bu imzayı aynen uygular.

100'LÜK ADAY HAVUZUNDA FILTRE GEREKMEDİĞİ HAKKINDA:
    Tam katalog sıralamasında (recommend_topk) kullanıcının train'de
    etkileşimde bulunduğu TÜM ürünler (beğendiği VE beğenmediği; seen_matrix)
    listeden çıkarılır; çünkü bunları zaten biliyor.
    100'lük aday havuzunda bu filtreye GEREK YOKTUR:
      - 100 aday = 1 (val/test pozitifi) + 99 (eval_negatives'ten negatif)
      - Negatifler, kullanıcının train'de etkileşmediği ürünlerdir.
        (eval_negatives.csv veri hazırlama aşamasında bunu garanti eder.)
      - Dolayısıyla 100 adayın içinde kullanıcının train pozitifi yoktur;
        filtre gereksizdir ve accuracy metriklerini etkilemez.
"""

import numpy as np
import pandas as pd
import pickle
import warnings
from pathlib import Path
from scipy.sparse import csr_matrix
from typing import Optional

try:
    from src.filtering import (
        FILTERED_SCORE, apply_exclude_seen, validate_indices, validate_user,
    )
except ModuleNotFoundError:
    # Dosya doğrudan script olarak çalıştırıldığında (python src/models/popularity.py)
    # proje kökü henüz sys.path'te değildir; ekleyip tekrar dene.
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    from src.filtering import (
        FILTERED_SCORE, apply_exclude_seen, validate_indices, validate_user,
    )


class PopularityModel:
    """
    NE YAPAR:
        Her ürünün train'deki beğenilme sayısını (label==1) popülerlik
        skoru olarak kullanır. Skor kullanıcıdan bağımsızdır: herkes için
        aynı sıralama üretilir.

    ÖZELLIKLER:
        item_scores : np.ndarray, shape (N_ITEMS,)
            item_idx -> popülerlik skoru (label==1 etkileşim sayısı)
            fit() çağrısından önce None'dır.
        n_items : int
            Toplam ürün sayısı (fit sırasında matrisin sütun boyutundan alınır).
        n_users : int | None
            Toplam kullanıcı sayısı (fit sırasında matrisin satır boyutundan).
            Girdi doğrulamasında "bilinen kullanıcı" aralığı için kullanılır.
        is_fitted : bool
            fit() çağrıldıktan sonra True olur.
    """

    def __init__(self) -> None:
        self.item_scores: Optional[np.ndarray] = None
        self.n_items: int = 0
        self.n_users: Optional[int] = None
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    def fit(self, train_matrix: csr_matrix) -> "PopularityModel":
        """
        NE YAPAR:
            CSR etkileşim matrisinden her ürünün popülerlik skorunu hesaplar.
            Matris sadece label==1 satırlarından oluştuğu için sütun toplamı
            doğrudan "kaç kez beğenildi" anlamına gelir.

        NEDEN CSR SÜTUN TOPLAMI:
            mat.sum(axis=0) tüm kullanıcıların o ürüne verdiği 1.0 değerlerini
            toplar. Yoğun matris oluşturmadan O(nnz) zamanda hesaplanır.

        PARAMETRELER:
            train_matrix : csr_matrix, shape (N_USERS, N_ITEMS)
                load_data()["interaction_matrix"] — sadece label==1 etkileşimler.
                Değerler 1.0 (pozitif etkileşim) veya 0.0 (boş hücre).

        DÖNDÜRÜR:
            self  (zincirleme çağrı için: model.fit(mat).score(...))
        """
        # sum(axis=0): sütun toplamı → shape (1, N_ITEMS) matrix; düzleştir
        col_sums = np.asarray(train_matrix.sum(axis=0)).flatten()

        # float32 yeterli; bellek tasarrufu sağlar
        self.item_scores = col_sums.astype(np.float32)
        self.n_items     = train_matrix.shape[1]   # 25.610
        self.n_users     = train_matrix.shape[0]   # 94.745
        self.is_fitted   = True

        n_pos = int((self.item_scores > 0).sum())
        print(f"[PopularityModel] fit tamamlandi: {self.n_items:,} urun, "
              f"max_pop={int(self.item_scores.max())}, "
              f"pozitif_urun={n_pos:,}")
        return self

    # ------------------------------------------------------------------
    def score(self, uid: int, cands: np.ndarray) -> np.ndarray:
        """
        NE YAPAR:
            Verilen aday ürünlerin popülerlik skorlarını döndürür.
            Bu imza evaluation.evaluate(score_fn=model.score, ...) ile
            doğrudan uyumludur.

        NEDEN KULLANICI BAĞIMSIZ:
            Popülerlik modeli kullanıcı kişiselleştirmesi yapmaz; herkes için
            aynı sıralama üretilir. uid parametresi imza uyumu için alınır
            ama kullanılmaz.

        HAKKINDA — 100'LÜK HAVUZDA FİLTRE GEREKMEDİĞİ:
            100 adayın 99'u eval_negatives'ten gelir. Bu negatifler her
            kullanıcı için train'de etkileşilmemiş ürünlerdir; dolayısıyla
            "kullanıcının beğendiği ürünleri çıkar" mantığı burada
            uygulanmaz. (Tam katalog önerisi için recommend_topk() kullanın.)

        PARAMETRELER:
            uid   : int   — kullanıcı indeksi (kullanılmaz, imza için alınır)
            cands : np.ndarray[int], shape (100,)
                    item_idx dizisi (indeks 0 = pozitif, 1-99 = negatif)

        DÖNDÜRÜR:
            np.ndarray[float32], shape (100,)
                cands'daki her ürünün popülerlik skoru.
                Yüksek skor → daha üst sıra.

        GİRDİ DOĞRULAMASI:
            - Tam sayı olmayan uid/cands -> TypeError.
            - Bilinmeyen ürün (negatif veya >= n_items) -> hata yok, skor = -1e9.
              (Negatif indeks numpy'da sondan okurdu; artık okunmuyor.)
            - Bilinmeyen kullanıcı -> hata yok, uyarı basılır. Popülerlik zaten
              kullanıcıdan bağımsız olduğu için "popülerliğe geri dönüş"
              burada aynı skorları döndürmek demektir.
            - Boş cands -> boş dizi.
            Geçerli girdilerde sonuç önceki sürümle BİREBİR aynıdır.
        """
        assert self.is_fitted, "Önce fit() çağrın."

        cands_arr, valid = validate_indices(cands, self.n_items, name="item_idx")
        if cands_arr.size == 0:
            return np.empty(0, dtype=np.float32)

        if not validate_user(uid, self.n_users, warn=False):
            warnings.warn(
                f"Bilinmeyen user_idx={uid} (beklenen: 0 <= user_idx < {self.n_users}); "
                f"popülerlik sıralamasına geri dönülüyor.",
                stacklevel=2,
            )

        if valid.all():
            # Hızlı yol (normal durum): vektörleştirilmiş fancy indexing, O(|cands|)
            return self.item_scores[cands_arr]

        # Bilinmeyen ürünlere en düşük skor; geçerlilere normal popülerlik
        out = np.full(cands_arr.shape, FILTERED_SCORE, dtype=np.float32)
        out[valid] = self.item_scores[cands_arr[valid]]
        return out

    # ------------------------------------------------------------------
    def recommend_topk(
        self,
        uid: int,
        train_matrix: csr_matrix,
        k: int = 10,
        exclude_seen: bool = True,
        seen_matrix: Optional[csr_matrix] = None,
        mode: str = "live",
        target_items=None,
    ) -> np.ndarray:
        """
        NE YAPAR:
            Tam katalog üzerinde kullanıcı için en popüler k ürünü döndürür.
            Kullanıcının train'de BİLDİĞİ ürünler (beğendiği + beğenmediği)
            varsayılan olarak hariç tutulur.

        NEDEN FILTRE BURADA, score()'DA DEĞİL:
            Gerçek öneri senaryosunda kullanıcıya "zaten bildiği" ürünleri
            önermek istemeyiz. Bu filtreleme SADECE tam katalog sıralamasında
            gereklidir; 100'lük değerlendirme havuzunda değil (orada negatifler
            zaten train etkileşimlerinden farklıdır).

        PARAMETRELER:
            uid          : int         — kullanıcı indeksi
            train_matrix : csr_matrix  — etkileşim matrisi (imza uyumluluğu için)
            k            : int         — önerilecek ürün sayısı (varsayılan 10)
            exclude_seen : bool        — True ise bilinen ürünleri hariç tut
            seen_matrix  : csr_matrix  — load_data()["seen_matrix"] (label 0+1).
                                         exclude_seen=True iken ZORUNLUDUR;
                                         verilmezse ValueError fırlatılır.
            mode         : "live" | "offline" — bkz. filtering.apply_exclude_seen
            target_items : offline modda korunacak val/test hedef ürün(ler)i

        DAVRANIŞ (girdi doğrulaması):
            - exclude_seen=True ve seen_matrix=None -> ValueError.
            - Bilinmeyen/negatif uid -> hata yok; uyarı + saf popülerlik sıralaması
              (geçmiş olmadığı için filtre uygulanmaz).
            - k <= 0 -> boş dizi.
            - Filtrelenen ürünler (-1e9) listeye HİÇ girmez; k, filtre sonrası
              kalan ürün sayısından büyükse daha kısa liste döner.

        DÖNDÜRÜR:
            np.ndarray[int], shape (<=k,)
                Popülerliğe göre sıralanmış top-k item_idx dizisi.
        """
        assert self.is_fitted, "Önce fit() çağrın."

        k_arr = np.asarray(k)
        if k_arr.ndim != 0 or k_arr.dtype == bool or not np.issubdtype(k_arr.dtype, np.integer):
            raise TypeError(f"k tek bir tam sayı olmalı; gelen: {k!r}")
        if exclude_seen and seen_matrix is None:
            raise ValueError(
                "exclude_seen=True iken seen_matrix zorunludur "
                "(load_data()['seen_matrix'] verin)."
            )
        if k <= 0:
            return np.empty(0, dtype=np.int64)

        # Tam popülerlik vektörünü kopyala; görülmüş ürünleri maskeleyeceğiz
        scores = self.item_scores.copy()    # shape: (N_ITEMS,)

        if not validate_user(uid, self.n_users, warn=False):
            # Yeni/bilinmeyen kullanıcı: geçmişi yok -> saf popülerlik (fallback)
            warnings.warn(
                f"Bilinmeyen user_idx={uid} (beklenen: 0 <= user_idx < {self.n_users}); "
                f"popülerlik sıralamasına geri dönülüyor (filtre uygulanmadı).",
                stacklevel=2,
            )
        elif exclude_seen:
            # Görülen ürünler -1e9 alır (offline modda hedef korunur)
            scores = apply_exclude_seen(
                scores, np.arange(self.n_items), uid, seen_matrix,
                mode=mode, target_items=target_items,
            )

        # En yüksek skorlu ürünler (azalan sıra)
        # N_ITEMS = 25.610 → argsort yeterince hızlı (argpartition yerine)
        order = np.argsort(-scores)
        order = order[scores[order] > FILTERED_SCORE]   # filtrelenenleri listeden at
        return order[:k]

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """
        NE YAPAR:
            Modeli pickle formatında diske kaydeder.

        NEDEN PICKLE:
            PopularityModel yalnızca tek bir numpy dizisi içerir (item_scores).
            Pickle bu durumda en sade çözümdür; numpy.save'e göre
            ek meta veri (n_items, is_fitted) de tek seferde kaydedilir.

        PARAMETRELER:
            path : str  — kayıt yolu (örnek: "saved_models/popularity.pkl")
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)  # klasörü oluştur

        with open(save_path, "wb") as f:
            pickle.dump({
                "item_scores": self.item_scores,
                "n_items"    : self.n_items,
                "n_users"    : self.n_users,
                "is_fitted"  : self.is_fitted,
            }, f)

        print(f"[PopularityModel] Kaydedildi: {save_path}")

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str) -> "PopularityModel":
        """
        NE YAPAR:
            Daha önce save() ile kaydedilmiş modeli yükler ve
            PopularityModel nesnesi döndürür.

        NEDEN CLASSMETHOD:
            Nesne oluşturmadan önce yükleyebilmek için:
                model = PopularityModel.load("saved_models/popularity.pkl")
            Bu, scikit-learn geleneğiyle uyumludur.

        PARAMETRELER:
            path : str  — kayıtlı model dosyasının yolu

        DÖNDÜRÜR:
            PopularityModel  — fit edilmiş model nesnesi
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        model = cls()
        model.item_scores = state["item_scores"]
        model.n_items     = state["n_items"]
        model.n_users     = state.get("n_users")   # eski kayıtlarda yok -> None
        model.is_fitted   = state["is_fitted"]

        print(f"[PopularityModel] Yuklendi: {path}  "
              f"(n_items={model.n_items:,}, is_fitted={model.is_fitted})")
        return model

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        if not self.is_fitted:
            return "PopularityModel(fit edilmedi)"
        return (f"PopularityModel(n_items={self.n_items:,}, "
                f"max_pop={int(self.item_scores.max())}, "
                f"pos_items={int((self.item_scores > 0).sum()):,})")


# ---------------------------------------------------------------------------
# Test ve doğrulama
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    DOĞRULAMA HEDEFLERİ:
        PopularityModel.score() evaluation.evaluate() ile aynı sonucu vermeli.
        evaluation.py __main__'deki inline popularity_score_fn ile özdeş.

        Val:  HR@10 = 0.5488, NDCG@10 = 0.3369  (±1e-4 tolerans)
        Test: HR@10 = 0.5068, NDCG@10 = 0.3090  (±1e-4 tolerans)

    AKTIVITE KIRILIMI (TÜM train satırları — düzeltilmiş tanım):
        Gruplar: Düşük (=3), Orta (4-6), Yüksek (7+) toplam etkileşim.
        CSV çıktıları:
          results/popularity_activity_val.csv
          results/popularity_activity_test.csv
    """
    import sys, os
    # Proje kökünü (cf_recommender/) Python path'ine ekle
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    from src.data_loader import load_data, get_activity_groups
    from src.evaluation import evaluate

    # ----------------------------------------------------------------
    # 1. Veri yükle
    # ----------------------------------------------------------------
    print("Veri yukleniyor...")
    data = load_data("data")

    mat           = data["interaction_matrix"]   # CSR, sadece label==1
    val_cands     = data["val_candidates"]       # {uid -> ndarray[100]}
    test_cands    = data["test_candidates"]      # {uid -> ndarray[100]}
    cold_items    = data["cold_items"]           # set: train'de hiç görülmemiş
    user_activity = data["user_activity"]        # Series: uid -> total_count (0+1 dahil)

    # ----------------------------------------------------------------
    # 2. Fit → save → load (kayıt/yükleme testi)
    # ----------------------------------------------------------------
    model = PopularityModel()
    model.fit(mat)
    print(f"\nModel: {model}")

    save_path = "saved_models/popularity.pkl"
    model.save(save_path)
    model_loaded = PopularityModel.load(save_path)

    # Yüklenen modelin skorları orijinalle aynı mı?
    sample_cands  = list(test_cands.values())[0]       # herhangi bir kullanıcının adayları
    scores_orig   = model.score(0, sample_cands)
    scores_loaded = model_loaded.score(0, sample_cands)
    assert np.allclose(scores_orig, scores_loaded), \
        "HATA: save/load sonrasi skorlar eslemiyor!"
    print("ASSERT OK: save/load skoru eslesiyor.")

    # ----------------------------------------------------------------
    # 3. Değerlendirme — val ve test
    # ----------------------------------------------------------------
    print("\n--- POPULARLIK MODELI (VAL) ---")
    res_val = evaluate(
        score_fn=model.score,
        candidates=val_cands,
        cold_items=cold_items,
        user_activity=user_activity,
        verbose=True,
    )

    print("\n--- POPULARLIK MODELI (TEST) ---")
    res_test = evaluate(
        score_fn=model.score,
        candidates=test_cands,
        cold_items=cold_items,
        user_activity=user_activity,
        verbose=True,
    )

    # ----------------------------------------------------------------
    # 4. Referans doğrulama
    # ----------------------------------------------------------------
    TOLERANCE = 1e-4
    checks = [
        ("Val  HR@10",   res_val["overall"]["hit_rate"],  0.5488),
        ("Val  NDCG@10", res_val["overall"]["ndcg"],      0.3369),
        ("Test HR@10",   res_test["overall"]["hit_rate"], 0.5068),
        ("Test NDCG@10", res_test["overall"]["ndcg"],     0.3090),
    ]

    print("\nREFERANS DOGRULAMA:")
    all_ok = True
    for name, elde, beklenen in checks:
        diff = abs(elde - beklenen)
        ok   = diff <= TOLERANCE
        if not ok:
            all_ok = False
        status = "OK  " if ok else "HATA"
        print(f"  [{status}]  {name:<18}: elde={elde:.4f}  beklenen={beklenen:.4f}  fark={diff:.5f}")

    if all_ok:
        print("  => Tum metrikler referans degerleriyle eslesiyor.")
    else:
        print("  => UYARI: bazi metrikler eslesmiyor.")

    # ----------------------------------------------------------------
    # 5. Aktivite kırılımı tablosu → CSV kayıt + konsol çıktısı
    # ----------------------------------------------------------------
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    group_label = {
        "low"   : "Dusuk (=3)",
        "medium": "Orta (4-6)",
        "high"  : "Yuksek (7+)",
    }

    for split_name, res in [("val", res_val), ("test", res_test)]:
        rows = []

        def _row(grup: str, m: dict) -> dict:
            """Metrik sözlüğünden tek CSV satırı oluştur."""
            return {
                "grup"        : grup,
                "n_users"     : m["n_users"],
                "hr_at_10"    : round(m["hit_rate"],     4),
                "ndcg_at_10"  : round(m["ndcg"],         4),
                "prec_at_10"  : round(m["precision"],    4),
                "recall_at_10": round(m["recall"],        4),
                "coverage_pct": round(m["coverage_pct"], 2),
                "coverage_n"  : m["coverage_n"],
            }

        # Genel satır
        rows.append(_row("Genel", res["overall"]))

        # Soğuk kullanıcı satırı (var ise)
        if res["cold_user"]:
            rows.append(_row("Soguk pozitif", res["cold_user"]))

        # Aktivite grubu satırları
        for key, mg in res["activity"].items():
            rows.append(_row(group_label.get(key, key), mg))

        df_out   = pd.DataFrame(rows)
        csv_path = results_dir / f"popularity_activity_{split_name}.csv"
        df_out.to_csv(csv_path, index=False)

        print(f"\nAktivite kirilimi ({split_name.upper()}) — kaydedildi: {csv_path}")
        print(df_out.to_string(index=False))

    # ----------------------------------------------------------------
    # 6. recommend_topk örneği (uid=0)
    # ----------------------------------------------------------------
    print("\n--- recommend_topk ornegi (uid=0, k=10, bilinen urunler [label 0+1] filtreleniyor) ---")
    top10 = model.recommend_topk(uid=0, train_matrix=mat, k=10, exclude_seen=True,
                                 seen_matrix=data["seen_matrix"])
    print(f"  Top-10 item_idx : {top10.tolist()}")
    print(f"  Popularlik skoru: {model.item_scores[top10].tolist()}")
