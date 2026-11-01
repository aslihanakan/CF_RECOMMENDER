"""
src/data_loader.py
==================
Amazon Reviews 2023 (Video Games) veri setini öneri sistemi pipeline'ı için
yükleyen ve hazırlayan modül.

Veri yapısı (incelenerek doğrulandı, 2026-10-22):
  - train.csv  : 624.971 satır  | label dağılımı {1: 504.975, 0: 119.996}
  - val.csv    :  94.745 satır  | label dağılımı {1:  75.799, 0:  18.946}
  - test.csv   :  94.745 satır  | label dağılımı {1:  74.021, 0:  20.724}
  - user_map.csv: 94.745 kullanıcı (user_id -> user_idx)
  - item_map.csv: 25.610 ürün   (parent_asin -> item_idx)
  - eval_negatives.csv: her kullanıcı için 99 negatif item_idx (geniş format)

Sütunlar (train/val/test):
  user_idx, item_idx, rating, label, verified_purchase, timestamp,
  user_id, parent_asin

label == 1  -> kullanıcının beğendiği etkileşim (rating yüksek)
label == 0  -> kullanıcının beğenmediği etkileşim (rating düşük)

SOĞUK ÜRÜN TANIMI (önemli — iki farklı tanım ayırt edilir):
  cold_never_seen  : train.csv'de item_idx olarak HİÇ görünmeyen ürünler
                     (ne label==1 ne label==0 satırında). Bunlar için model
                     hiçbir etkileşim bilgisine sahip değildir.
                     -> Kullanılan tanım budur. (84 ürün)
  cold_no_positive : train'de label==0 olarak geçen ama hiç label==1 satırı
                     olmayan ürünler. Model bunları "negatif bağlamda" görmüştür.
                     -> Referans olarak ayrıca hesaplanır ama ceza uygulanmaz.
                     (243 ürün; cold_never_seen bu kümenin alt kümesidir)

  Test pozitiflerindeki dağılım (2026-10-22 itibarıyla):
    cold_never_seen  : 251 satır (kullanıcı), 78 tekil ürün
    cold_no_positive : 462 satır (kullanıcı), 184 tekil ürün
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from pathlib import Path


# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
N_USERS = 94_745   # user_idx aralığı: 0 .. 94744
N_ITEMS = 25_610   # item_idx aralığı: 0 .. 25609
N_NEGATIVES = 99   # eval_negatives.csv'deki negatif sayısı (kullanıcı başına)


def load_data(data_dir: str = "data") -> dict:
    """
    NE YAPAR:
        Projenin tüm veri dosyalarını okur; eğitim matrisini, aday havuzlarını,
        soğuk ürün kümesini ve aktivite kırılım bilgisini hazırlar.

    NEDEN BU FONKSİYON:
        Veri yükleme mantığını tek bir yerde toplamak, hem notebook'larda hem
        modellerde tutarlı veri kullanımını garanti eder. Her model aynı
        nesneleri alır → karşılaştırma adil olur.

    PARAMETRELER:
        data_dir : str
            CSV dosyalarının bulunduğu dizin (varsayılan: "data").

    DÖNDÜRÜR:
        dict ile aşağıdaki anahtarlar:
            "train_df"          : pd.DataFrame  – ham train verisi
            "val_df"            : pd.DataFrame  – ham val verisi
            "test_df"           : pd.DataFrame  – ham test verisi
            "interaction_matrix": csr_matrix (94745 x 25610)
                                  sadece label==1 train satırlarından
            "seen_matrix"       : csr_matrix (94745 x 25610)
                                  TÜM train satırlarından (label 0 ve 1);
                                  canlı öneride "bilinen ürünleri çıkar"
                                  filtresi için. Val/test İÇERMEZ.
            "val_candidates"    : dict {user_idx -> np.ndarray[100]}
                                  val pozitif item_idx + 99 negatif
            "test_candidates"   : dict {user_idx -> np.ndarray[100]}
                                  test pozitif item_idx + 99 negatif
            "cold_items"        : set  – train'de hiç görülmemiş item_idx'ler
            "user_activity"     : pd.Series (user_idx -> train pozitif sayısı)
    """
    data_dir = Path(data_dir)

    # -----------------------------------------------------------------------
    # 1. Ham CSV'leri oku
    # -----------------------------------------------------------------------
    # dtype='Int64' yerine int kullanıyoruz; idx sütunları zaten tam sayı.
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df   = pd.read_csv(data_dir / "val.csv")
    test_df  = pd.read_csv(data_dir / "test.csv")

    # -----------------------------------------------------------------------
    # 2. Seyrek eğitim matrisi (CSR) – sadece label == 1 satırlar
    # -----------------------------------------------------------------------
    # NEDEN SPARSE: 94745 * 25610 ≈ 2,4 milyar hücre → dense bellekte ~19 GB.
    # Sadece ~505k pozitif var; sparse CSR bu değerleri kompakt tutar.
    train_pos = train_df[train_df["label"] == 1]   # negatif etkileşimleri dışarıda bırak

    rows = train_pos["user_idx"].values             # satır indeksleri (kullanıcılar)
    cols = train_pos["item_idx"].values             # sütun indeksleri (ürünler)
    data = np.ones(len(train_pos), dtype=np.float32)  # her pozitif etkileşim = 1.0

    # Boyutu sabit ver: tüm kullanıcı/ürün uzayını kapsasın (train'de görülmeyenler dahil)
    interaction_matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(N_USERS, N_ITEMS),
        dtype=np.float32
    )

    # -----------------------------------------------------------------------
    # 2b. Görülmüş ürün matrisi (seen_matrix) – TÜM train satırları (label 0+1)
    # -----------------------------------------------------------------------
    # interaction_matrix model eğitimi içindir (sadece beğeniler).
    # seen_matrix ise filtre içindir: kullanıcının olumlu VEYA olumsuz
    # etkileşimde bulunduğu her ürün "bilinen" sayılır ve tekrar önerilmez.
    seen_matrix = build_seen_matrix(train_df, N_USERS, N_ITEMS)

    # -----------------------------------------------------------------------
    # 3. eval_negatives.csv → her kullanıcı için 99 negatif
    # -----------------------------------------------------------------------
    # eval_negatives: geniş format (wide) — 1 satır / kullanıcı, 99 negatif sütun.
    # NEDEN NUMPY DİZİSİ: Değerlendirme sırasında vektörleştirilmiş işlem için.
    neg_df = pd.read_csv(data_dir / "eval_negatives.csv")

    # Negatif sütun adları: neg_item_1 .. neg_item_99
    neg_cols = [c for c in neg_df.columns if c != "user_idx"]
    # Her satırı numpy dizisine çevir; shape (94745, 99)
    neg_matrix = neg_df[neg_cols].values  # int64 ndarray

    # user_idx → satır indeksi eşlemesi (user_idx 0-tabanlı ve sıralı, doğrudan kullanılabilir)
    neg_by_user = {
        int(neg_df["user_idx"].iloc[i]): neg_matrix[i]
        for i in range(len(neg_df))
    }

    # -----------------------------------------------------------------------
    # 4. Aday havuzları: pozitif + 99 negatif = 100 aday
    # -----------------------------------------------------------------------
    # NEDEN 100'LÜK HAVUZ: Hit-Rate@10 hesabında standart değerlendirme protokolü.
    # Kullanıcı başına tek pozitif (1 ürün beğendi); bu ürünün top-10'a girip
    # girmediğini test ediyoruz. 100 adayda rastgele model ~ %10 başarı beklenir.

    val_candidates  = _build_candidates(val_df,  neg_by_user, label_col="label")
    test_candidates = _build_candidates(test_df, neg_by_user, label_col="label")

    # -----------------------------------------------------------------------
    # 5. Soğuk ürünler: train'de item_idx olarak HİÇ görülmemiş ürünler
    # -----------------------------------------------------------------------
    # TANIM: cold_never_seen = train.csv'de (label==0 dahil) hiç geçmeyen ürünler.
    # Bu ürünler için model hiçbir bilgiye sahip değildir.
    # NOT: Sadece label==0'da görülen ama label==1'i olmayan "cold_no_positive"
    # tanımı farklıdır ve ayrıca hesaplanıp doğrulama çıktısında gösterilir.
    train_all_items = set(train_df["item_idx"].unique())  # label 0+1 dahil tüm train ürünleri
    train_pos_items = set(train_pos["item_idx"].unique()) # sadece label==1 train ürünleri
    all_items       = set(range(N_ITEMS))                 # tüm ürün uzayı

    # Kullanılan soğuk tanım: train'de hiç görülmemiş
    cold_items = all_items - train_all_items              # 84 ürün (doğrulandı)

    # Referans amaçlı: train'de label==1 olmayan (ceza uygulanmaz, sadece raporlanır)
    cold_no_positive = all_items - train_pos_items        # 243 ürün (cold_items bunun alt kümesi)

    # -----------------------------------------------------------------------
    # 6. Kullanıcı aktivitesi: train'deki TÜM etkileşim sayısı (label 0 + 1)
    # -----------------------------------------------------------------------
    # NEDEN TÜM TRAIN: Aktivite kırılımı kullanıcının genel etkileşim hacmini
    # ölçer; sadece pozitifler sayılsaydı 5-core filtrelemeden gelen ölçek
    # bozulurdu. "Aktif kullanıcı" = toplamda çok etkileşimi olan kullanıcı.
    # DÜZELTİLDİ: Önceki versiyon yalnızca label==1 satırları sayıyordu.
    user_activity = (
        train_df                          # TÜM train satırları (label 0 ve 1)
        .groupby("user_idx")["item_idx"]
        .count()
        .rename("total_count")            # "pos_count" değil "total_count"
    )

    # -----------------------------------------------------------------------
    # 7. Doğrulama çıktıları
    # -----------------------------------------------------------------------
    _print_validation_summary(
        interaction_matrix, val_df, test_df, cold_items, cold_no_positive
    )

    return {
        "train_df"          : train_df,
        "val_df"            : val_df,
        "test_df"           : test_df,
        "interaction_matrix": interaction_matrix,
        "seen_matrix"       : seen_matrix,         # TÜM train (label 0+1), filtre için
        "val_candidates"    : val_candidates,
        "test_candidates"   : test_candidates,
        "cold_items"        : cold_items,          # tanım: train'de hic gorunmeyen
        "cold_no_positive"  : cold_no_positive,    # tanım: train'de label==1 olmayan
        "user_activity"     : user_activity,
    }


# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

def build_seen_matrix(train_df: pd.DataFrame, n_users: int, n_items: int) -> csr_matrix:
    """
    NE YAPAR:
        Train setindeki TÜM bilinen kullanıcı-ürün etkileşimlerinden
        (label == 0 VE label == 1) ikili (0/1) seyrek CSR matris üretir.

    NEDEN AYRI BİR MATRİS (interaction_matrix yetmez mi?):
        interaction_matrix yalnızca beğenileri (label==1) içerir; model eğitimi
        için doğrudur. Ancak "kullanıcının zaten bildiği ürünleri önerme"
        filtresi için düşük puan verdiği (label==0) ürünler de bilinmelidir;
        aksi halde beğenmediği oyun ona tekrar önerilir.

    NEDEN SADECE TRAIN:
        Val/test etkileşimleri değerlendirmede "gizli cevap"tır. Onları
        seen_matrix'e koymak hem veri sızıntısı olur hem de offline
        değerlendirmede doğru cevabın filtrelenmesine yol açar. Bu fonksiyon
        bu yüzden YALNIZCA train_df alır.

    PARAMETRELER:
        train_df : pd.DataFrame  – user_idx, item_idx sütunlarını içeren train verisi
        n_users  : int           – matris satır sayısı (tüm kullanıcı uzayı)
        n_items  : int           – matris sütun sayısı (tüm ürün uzayı)

    DÖNDÜRÜR:
        csr_matrix, shape (n_users, n_items), değerler 1.0 (bilinen) / boş.
    """
    rows = train_df["user_idx"].to_numpy()
    cols = train_df["item_idx"].to_numpy()
    data = np.ones(len(train_df), dtype=np.float32)

    seen = csr_matrix((data, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)
    # Aynı çift iki kez geçerse CSR değerleri toplar (2.0 olur); ikili tut.
    seen.sum_duplicates()
    seen.data[:] = 1.0
    return seen


def _build_candidates(
    split_df: pd.DataFrame,
    neg_by_user: dict,
    label_col: str = "label"
) -> dict:
    """
    NE YAPAR:
        Bir split (val veya test) için her kullanıcıya ait 100'lük aday havuzu
        oluşturur: 1 pozitif ürün + 99 negatif ürün.

    NEDEN:
        Değerlendirme protokolü gereği her kullanıcı için aday sayısı sabittir.
        Pozitifi aday listenin başına koyuyoruz; sıralama modele bırakılır.

    PARAMETRELER:
        split_df    : val_df veya test_df
        neg_by_user : {user_idx -> ndarray[99]} sözlüğü (eval_negatives'ten)
        label_col   : pozitif/negatif etiket sütunu adı

    DÖNDÜRÜR:
        {user_idx -> np.ndarray[100]}
        Sadece label==1 olan (pozitif ürünü olan) kullanıcıları içerir.
        Pozitif ürün her zaman indeks 0'dadır (evaluation bunu bilir).
    """
    # Sadece pozitif etiketi olan kullanıcılar değerlendirmeye alınır
    pos_rows = split_df[split_df[label_col] == 1][["user_idx", "item_idx"]]

    candidates = {}
    for _, row in pos_rows.iterrows():
        uid = int(row["user_idx"])
        pos_item = int(row["item_idx"])

        if uid not in neg_by_user:
            # Bu kullanıcıya ait negatif yok; atla (veri tutarsızlığı)
            continue

        # [pozitif ürün] + [99 negatif ürün] → 100 elemanlı dizi
        cands = np.concatenate([[pos_item], neg_by_user[uid]])
        candidates[uid] = cands

    return candidates


def _print_validation_summary(
    interaction_matrix: csr_matrix,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cold_items: set,
    cold_no_positive: set,
) -> None:
    """
    NE YAPAR:
        Yükleme sonunda konsola doğrulama çıktısı basar; beklenen referans
        sayılarıyla karşılaştırma yapar. Soğuk ürün için iki tanımı
        (hiç görülmemiş vs label==1 olmayan) ayrı ayrı raporlar ve
        test pozitiflerindeki soğuk ürünleri hem satır (kullanıcı)
        hem tekil ürün sayısı olarak gösterir.

    NEDEN:
        Veri yükleme hatalarını (yanlış filtre, yanlış şekil, yanlış soğuk
        tanımı vb.) erken yakalamanın en hızlı yolu doğrulama çıktısıdır.
    """
    print("=" * 65)
    print("VERİ YÜKLENDİ — DOĞRULAMA ÖZETİ")
    print("=" * 65)

    # Matris şekli
    shape = interaction_matrix.shape
    nnz   = interaction_matrix.nnz
    print(f"  Etkileşim matrisi şekli  : {shape[0]} x {shape[1]}")
    print(f"  Beklenen                 : {N_USERS} x {N_ITEMS}")
    print(f"  Pozitif etkileşim (nnz)  : {nnz:,}")
    print(f"  Beklenen (yaklaşık)      : ~505.000")

    # Val / Test label==1 kullanıcı sayısı
    val_pos_users  = int((val_df["label"]  == 1).sum())
    test_pos_users = int((test_df["label"] == 1).sum())
    print(f"  Val  label==1 satır      : {val_pos_users:,}  (beklenen: 75.799)")
    print(f"  Test label==1 satır      : {test_pos_users:,}  (beklenen: 74.021)")

    # Soğuk ürün — iki tanım, test pozitiflerinde satır + tekil ürün ayrımı
    test_pos = test_df[test_df["label"] == 1]

    # Tanım 1: train'de hiç görülmemiş
    rows_cold_never  = test_pos[test_pos["item_idx"].isin(cold_items)]
    n_rows_never     = len(rows_cold_never)
    n_uniq_never     = rows_cold_never["item_idx"].nunique()

    # Tanım 2: train'de label==1 olmayan (cold_no_positive)
    rows_cold_nopoz  = test_pos[test_pos["item_idx"].isin(cold_no_positive)]
    n_rows_nopoz     = len(rows_cold_nopoz)
    n_uniq_nopoz     = rows_cold_nopoz["item_idx"].nunique()

    print(f"\n  SOGUK URUN (iki tanim karsilastirmasi):")
    print(f"    Tanim 1 — train'de HIC gorunmeyen  (KULLANILAN):")
    print(f"      Toplam soguk urun: {len(cold_items):,}")
    print(f"      Test pozitifinde : {n_rows_never:,} satir (kullanici)  |  {n_uniq_never:,} tekil urun")
    print(f"      Beklenen satir   : ~251")
    print(f"    Tanim 2 — train'de label==1 olmayan (referans):")
    print(f"      Toplam soguk urun: {len(cold_no_positive):,}")
    print(f"      Test pozitifinde : {n_rows_nopoz:,} satir (kullanici)  |  {n_uniq_nopoz:,} tekil urun")
    print("=" * 65)


def get_activity_groups(user_activity: pd.Series) -> dict:
    """
    NE YAPAR:
        Kullanıcıları train'deki TOPLAM etkileşim sayısına (label 0 + 1) göre
        üç gruba ayırır.

    NEDEN TOPLAM (label 0+1):
        Aktivite kırılımı kullanıcının genel etkileşim hacmini ölçer.
        "Aktif kullanıcı" = toplamda çok etkileşim kaydetmiş kullanıcı.
        Sadece pozitifler sayılsaydı, label==0 etkileşimleri fazla olan
        kullanıcılar yanlış gruba düşerdi.
        DÜZELTİLDİ: Önceki versiyon yalnızca label==1 satırları sayıyordu
        (user_activity = train_pos.groupby(...)).

    PARAMETRELER:
        user_activity : pd.Series (user_idx -> total_count)
            load_data() döndürdüğü "user_activity" anahtarı.
            Her kullanıcı için train'deki toplam etkileşim sayısı.

    DÖNDÜRÜR:
        {
            "low"    : set  -- tam olarak 3 toplam etkileşimi olan kullanıcılar
            "medium" : set  -- 4-6 toplam etkileşim
            "high"   : set  -- 7+ toplam etkileşim
        }

    KONTROL:
        Test'te label==1 olan kullanıcılar (74.021) bu üç grubun birleşimine
        eşit olmalıdır. load_data()'nın __main__ bloğu bunu assert ile doğrular.

    NOT:
        5-core filtreleme uygulandığı için her kullanıcının en az 5 toplam
        etkileşimi olması beklenir. Buna rağmen train/val/test bölünmesi
        nedeniyle bazı kullanıcıların train'deki etkileşim sayısı 3'e
        kadar düşebilir (val ve test'e en az 1'er satır gider).
    """
    low    = set(user_activity[user_activity == 3].index)
    medium = set(user_activity[(user_activity >= 4) & (user_activity <= 6)].index)
    high   = set(user_activity[user_activity >= 7].index)

    return {"low": low, "medium": medium, "high": high}


# ---------------------------------------------------------------------------
# Hızlı test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = load_data("data")
    mat  = data["interaction_matrix"]
    act  = data["user_activity"]
    grps = get_activity_groups(act)

    print(f"\nAktivite gruplari (TUM train etkilesimleri sayildi):")
    print(f"  Dusuk  (=3 etkilesim) : {len(grps['low']):,} kullanici")
    print(f"  Orta   (4-6 etkilesim): {len(grps['medium']):,} kullanici")
    print(f"  Yuksek (7+ etkilesim) : {len(grps['high']):,} kullanici")
    toplam = len(grps["low"]) + len(grps["medium"]) + len(grps["high"])
    print(f"  TOPLAM               : {toplam:,} (94.745 olmali)")

    # Test'te label==1 olan kullanicilar uclu gruplara esit olmalidir
    # (her kullanicinin test'te tam bir satiri var; label==1 olanlar degerlendirmeye girer)
    test_pos_users = set(data["test_df"][data["test_df"]["label"] == 1]["user_idx"].tolist())
    group_union = grps["low"] | grps["medium"] | grps["high"]
    # Aktivite gruplari TUM train kullanicilari kapsar (94.745).
    # Test pozitif kullanicilari (74.021) bu gruplarin alt kumesindedir.
    test_in_groups = test_pos_users & group_union
    assert len(test_in_groups) == len(test_pos_users), (
        f"Test pozitif kullanicilari ({len(test_pos_users)}) aktivite gruplarinin "
        f"disinda kaliyor! Gruplarla kesisim: {len(test_in_groups)}"
    )
    print(f"\n  ASSERT OK: test label==1 kullanicilarin tamami ({len(test_pos_users):,})")
    print(f"  aktivite gruplarinin birlesiminde yer aliyor.")

    print(f"\nVal  aday havuzu boyutu: {len(data['val_candidates']):,}")
    print(f"Test aday havuzu boyutu: {len(data['test_candidates']):,}")
    print(f"Soguk urun sayisi (hic gorunmemis): {len(data['cold_items']):,}")
    print(f"Soguk urun sayisi (label==1 yok)  : {len(data['cold_no_positive']):,}")
