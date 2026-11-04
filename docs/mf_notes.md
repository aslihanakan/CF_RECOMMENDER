# Implicit ALS Matrix Factorization (`ALSModel`) — Teknik Dokümantasyon ve Entegrasyon Notları

Bu doküman, Amazon Reviews 2023 (Video Games) İşbirlikçi Filtreleme (CF) katmanında geliştirilen **Implicit ALS Matrix Factorization (`src/models/mf.py`)** modelinin eğitimini, yüklenmesini, `ContractAdapter` üzerinden çağrılmasını, seçilen hiperparametrelerini, bilinen sınırlarını ve Hibrit (CF + Content-Based) katman için entegrasyon kurallarını açıklar.

---

## 1. Seçilen (Dondurulmuş) Hiperparametreler ve Validation Performansı

Validation seti (`N = 75.799` kullanıcı, 1 pozitif + 99 negatif aday havuzu) üzerinde 3 aşamalı ızgara araması (Stage 1: 36, Stage 2: 12, Stage 3: 24 kombinasyon; toplam 72 konfigürasyon) sonucunda seçilen ve `saved_models/mf/` altına kaydedilen dondurulmuş model parametreleri:

| Parametre | Değer | Açıklama |
| :--- | :--- | :--- |
| `factors` ($d$) | `32` | Gizli faktör (latent dimension) boyutu (`float32` matrisler) |
| `regularization` ($\lambda$) | `0.01` | $L_2$ düzenlileştirme katsayısı |
| `alpha` ($\alpha$) | `150.0` | Hu-Koren-Volinsky güven ağırlığı ölçeği ($c_{ui} = 1 + \alpha r_{ui}$) |
| `iterations` | `50` | ALS dönüşümlü en küçük kareler iterasyon sayısı |
| `random_state` | `42` | Sabit rastgelelik tohumu |
| `num_threads` | `1` | Deterministik tekrar üretilebilirlik için tek iş parçacığı (`BLAS=1`) |
| `fold_in_method` | `"ridge"` | Yeni kullanıcı (`score_from_items`) için analitik ALS ridge projeksiyonu |

### Validation Karşılaştırma Özeti (`N = 75.799` kullanıcı)

| Model | HR@10 | NDCG@10 | Precision@10 | Recall@10 | Havuz Cov@10 (`N=75.799`) | Gerçek Katalog Cov@10 (`10k` canlı kullanıcı) | İlk 100 Popüler Ürüne Pay |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Popularity** | 0.5488 | 0.3369 | 0.0549 | 0.5488 | %19.84 (5.082) | %0.06 (15 / 25.610) | %100.00 |
| **Item-kNN (`pw=0.0`)** | 0.5714 | 0.4210 | 0.0571 | 0.5714 | %99.67 (25.526) | %50.85 (13.022 / 25.610) | %28.68 |
| **Item-kNN (`pw=0.01`)** | 0.6400 | 0.4520 | 0.0640 | 0.6400 | %91.04 (23.316) | %16.68 (4.272 / 25.610) | %62.70 |
| **ALS MF (`f=32, α=150, it=50`)** | **0.6910** | **0.5029** | **0.0691** | **0.6910** | %89.77 (22.989) | %8.26 (2.116 / 25.610) | %32.17 |

---

## 2. Model Eğitimi, Kaydetme ve Yükleme

### 2.1. Sıfırdan Eğitim ve Kaydetme
`ALSModel`, `data_loader.load_data()` tarafından üretilen yalnızca pozitif etkileşimleri (`label == 1`, `rating >= 4`, toplam `504.975` etkileşim) içeren `interaction_matrix` (`94.745 × 25.610` seyrek CSR matrisi) ile eğitilir:

```python
from threadpoolctl import threadpool_limits
from src.data_loader import load_data
from src.models.mf import ALSModel

data = load_data("data")

# Deterministik eğitim için BLAS ve OpenMP tek iş parçacığına sabitlenir
with threadpool_limits(limits=1, user_api="blas"):
    mf_model = ALSModel(
        factors=32,
        regularization=0.01,
        alpha=150.0,
        iterations=50,
        random_state=42,
        num_threads=1,
        fold_in_method="ridge",
    )
    mf_model.fit(data["interaction_matrix"])

# Faktörleri (user_factors.npy, item_factors.npy), R.npz ve params.json'u kaydet
mf_model.save("saved_models/mf")
```

### 2.2. Kayıtlı Modeli Yükleme (`saved_models/mf/`)
Diskteki toplam boyutu **15.96 MB** olan (`user_factors.npy`: 11.57 MB, `item_factors.npy`: 3.13 MB, `R.npz`: 1.27 MB, `params.json`) dondurulmuş model doğrudan yüklenebilir:

```python
from src.models.mf import ALSModel

mf_model = ALSModel.load("saved_models/mf")
```

---

## 3. Kullanım Yöntemleri (`ContractAdapter` ve Doğrudan Çağrılar)

### 3.1. `ContractAdapter` ile Hibrit Katman Entegrasyonu (Önerilen)
Hibrit katman, `ItemKNNModel` ile birebir aynı sözleşmeyi (`ContractAdapter`) kullanarak `ALSModel`'den normalize edilmiş `[0, 1]` aralığında skorlar ve `cf_known` maskesi alır:

```python
from src.contract import ContractAdapter
from src.data_loader import N_ITEMS, N_USERS, load_data
from src.models.mf import ALSModel
from src.models.popularity import PopularityModel

data = load_data("data")
mf_model = ALSModel.load("saved_models/mf")
pop_model = PopularityModel.load("saved_models/popularity.pkl")

adapter = ContractAdapter(
    model=mf_model,
    popularity_model=pop_model,
    seen_matrix=data["seen_matrix"],
    interaction_matrix=data["interaction_matrix"],
    n_users=N_USERS,
    n_items=N_ITEMS,
)

# 1) Offline Değerlendirme (100'lük aday havuzu, hedef ürün indeks 0 korunur)
df_offline = adapter.score(
    users=[0, 1],
    candidates=[data["val_candidates"][0], data["val_candidates"][1]],
    mode="offline",
    target_items=[int(data["val_candidates"][0][0]), int(data["val_candidates"][1][0])],
)

# 2) Canlı (Live) Mod (Kullanıcının seen_matrix'teki tüm geçmiş ürünleri çıkarılır)
df_live = adapter.score(
    users=[0],
    candidates=[0, 10, 25, 100, 250],
    mode="live",
)

# 3) Yeni Kullanıcı / Tercih Formu (Seçilen oyunlardan ridge fold-in ile skor üretimi)
df_new = adapter.score_from_items(
    item_idxs=[0, 10, 25],
    candidates=[0, 10, 25, 100, 250, 500],
)
```

**Çıktı Şeması (`pd.DataFrame`):**
- `user_idx` (`int64`): Kullanıcı indeksi (`score_from_items` için `-1`).
- `item_idx` (`int64`): Aday ürün indeksi.
- `score` (`float64`): Kullanıcı bazında `[0.0, 1.0]` aralığına min-max normalize edilmiş skor (`cf_known == False` olan ürünlerde kesin olarak `0.0`).
- `cf_known` (`bool`): Ürünün eğitim (`train`) pozitif etkileşim matrisinde en az 1 pozitif etkileşimi olup olmadığı (`True` / `False`).

### 3.2. Doğrudan Tüm Katalog (`25.610` Ürün) Üzerinden Top-K Öneri (`recommend_topk`)
Canlı sistemde 100'lük aday havuzu olmadan doğrudan tüm katalogdan en yüksek skorlu $K$ ürünü almak için:

```python
# Mevcut kullanıcı için tüm katalogdan Top-10 [(item_idx, ham_skor), ...]
topk_existing = mf_model.recommend_topk(
    uid=0,
    k=10,
    exclude_seen=True,
    seen_matrix=data["seen_matrix"],
    mode="live",
)

# Yeni kullanıcı için seçtiği oyunlardan tüm katalogda Top-10 [(item_idx, ham_skor), ...]
topk_new_user = mf_model.recommend_from_items(
    liked_items=[0, 10, 25],
    k=10,
)
```

---

## 4. Bilinen Sınırlar ve Riskler

1. **Soğuk Ürünlerde (`cf_known=False`) Sıfır İsabet (`0 / 114` ve `0 / 263`):**
   - Eğitim setinde hiç görünmeyen **84 tamamen soğuk ürün** (`cold_items`, validation'da `114` hedef satırı) ve yalnızca `label == 0` (`rating < 4`) etkileşimi olup hiç pozitif etkileşimi olmayan **159 ürün** (toplam `243` ürün, validation'da `263` hedef satırı) için pozitif etkileşim matrisinde sütun toplamı sıfırdır.
   - `ALSModel.fit()` sırasında bu `243` ürünün faktör vektörleri sıfırlanır (`item_factors[i] = 0.0`, `float32` sayısal gürültü eşiği `norm < 1e-4` kontrolü dahil).
   - `ContractAdapter` bu ürünlere `cf_known=False` ve `score=0.0` atar; `evaluate()` ise soğuk ürünleri `-1e9` cezası ile listenin en sonuna iter. Sonuç olarak soğuk kovada (`pop=0`, `N=263`) `ALS MF` isabet oranı **HR@10 = 0.0000** (`0 / 263` hit) düzeyindedir.
2. **Baş (`pop >= 112`) Kovasında `Item-kNN (pw=0.01)`'in Gerisinde Kalması:**
   - Popülerlik kovası analizinde (`N = 75.799`), `ALS MF` kuyruk (`pop=1..23`, `N=25.320`) kovasında **HR@10 = 0.4654 / NDCG@10 = 0.2417** ile `Item-kNN (pw=0.01)`'i (`0.2231 / 0.1398`) **+24.23 puan HR@10** geçerken ve orta (`pop=24..111`, `N=25.114`) kovada **HR@10 = 0.7703 / NDCG@10 = 0.5619** ile `Item-kNN (pw=0.01)`'i (`0.7088 / 0.4022`) **+6.15 puan HR@10** aşarken; en popüler ürünlerin yer aldığı **baş (`pop=112..1.901`, `N=25.102`)** kovasında **HR@10 = 0.8463 / NDCG@10 = 0.7127** (`21.245` hit) elde ederek açık popülerlik önyargısı taşıyan `Item-kNN (pw=0.01)`'in (**HR@10 = 0.9985 / NDCG@10 = 0.8216**, `25.064` hit) **-15.22 puan HR@10** gerisinde kalmaktadır.
3. **Gerçek Katalog Kapsamı (`Coverage@10`) Ödünleşimi:**
   - `10.000` validation kullanıcısı (`seed=42`) için 25.610 ürünlük tam katalogdan `recommend_topk(k=10, mode="live")` çağrıldığında, `ALS MF` toplam **2.116 benzersiz ürün (%8.26 gerçek katalog kapsamı)** önermektedir (önerilerin **%32.17**'si en popüler 100 üründedir).
   - Bu kapsam, `Popularity` modelinden (**%0.06**, `15` ürün, ilk 100 payı **%100.00**) çok daha geniş olsa da `Item-kNN (pw=0.0)` (**%50.85**, `13.022` ürün) ve `Item-kNN (pw=0.01)` (**%16.68**, `4.272` ürün) modellerinin gerisindedir. Düşük boyutlu (`factors=32`) iç çarpım uzayı ve yüksek güven ağırlığı (`alpha=150`), genel doğruluk (`NDCG@10 = 0.5029`) sağlarken tam katalog sıralamasında yüksek normlu orta-üst popülerlikteki ürünlerde yoğunlaşmaya yol açar.

---

## 5. Hibrit (CF + Content-Based) Katman İçin Öneriler

1. **`cf_known=False` Ürünlerde Content-Based (CB) Devralması:**
   - `ContractAdapter` çıktısında `cf_known == False` olan `243` ürün için CF skoru `0.0` olarak sabitlenmiştir. Hibrit birleştirici (blender), `cf_known == False` gördüğü adaylarda CF ağırlığını `0.0`'a çekip karar ağırlığını tamamen Content-Based (metin/kategori/platform benzerliği) katmanına devretmelidir.
2. **Gerçek Katalog Kapsamını (`%8.26`) Genişletme Hipotezi:**
   - `ALS MF` tek başına tam katalogda `2.116` (%8.26) benzersiz ürün önermektedir; Content-Based katmanın ürün meta verileri üzerinden uzun kuyruk ürünlerine vereceği ek skor katkısının bu katalog kapsamını genişletebileceği bir **HİPOTEZ** olarak hibrit aşamada test edilmelidir.
3. **Yeni Kullanıcı (`score_from_items`) Entegrasyonu:**
   - Sistemde geçmişi olmayan yeni kullanıcılar 3, 4 veya 5 oyun seçtiğinde `ALS MF (ridge fold-in)` 5.000 kullanıcılık validation simülasyonunda 3 oyunda **HR@10 = 0.6624 / NDCG@10 = 0.4704** (5 oyunda **HR@10 = 0.7002 / NDCG@10 = 0.4992**) ile `Item-kNN (pw=0.01)` modelini 3 oyunda **+0.0512 ± 0.0074 HR@10** ($z = +6.92$) ve **+0.0571 ± 0.0053 NDCG@10** ($z = +10.76$) farkla anlamlı biçimde geçmektedir. Hibrit katmanda yeni kullanıcı soğuk başlangıç (preference form) akışında birincil CF skoru olarak `ALSModel.score_from_items` (`method="ridge"`) kullanılmalıdır.
