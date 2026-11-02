# cf_recommender — Amazon Reviews 2023 (Video Games) İşbirlikçi Filtreleme (CF) Katmanı

## 1. Proje Özeti ve Klasör Yapısı

Bu proje, **Amazon Reviews 2023 (Video Games)** veri seti üzerinde çalışan hibrit oyun öneri sisteminin **Collaborative Filtering (CF)** katmanını içerir. Temel (Popularity) ve komşuluk tabanlı (**Item-kNN**) modellerin eğitimini, sızıntısız değerlendirme protokolünü (`evaluate`), bilinen ürün filtrelemesini (`seen_matrix` / `apply_exclude_seen`) ve **Content-Based (CB)** model ile entegrasyonu sağlayan modelden bağımsız ortak sözleşme adaptörünü (`ContractAdapter`) kapsar.

### Klasör Yapısı

```text
cf_recommender/
├── data/                              # Veri dosyaları (.gitignore ile hariç tutulur)
│   ├── train.csv                      # Eğitim etkileşimleri (624.971 satır)
│   ├── val.csv                        # Doğrulama etkileşimleri (94.745 satır)
│   ├── test.csv                       # Test etkileşimleri (94.745 satır)
│   ├── eval_negatives.csv             # Kullanıcı başına 99 negatif aday (94.745 satır)
│   ├── user_map.csv                   # user_id <-> user_idx eşlemesi (94.745 kullanıcı)
│   ├── item_map.csv                   # parent_asin <-> item_idx eşlemesi (25.610 ürün)
│   └── items.csv                      # Ürün meta verileri
├── src/                               # Kaynak kod modülleri
│   ├── __init__.py
│   ├── data_loader.py                 # Veri yükleme, CSR matrisler ve aday havuzu inşası
│   ├── evaluation.py                  # Standart değerlendirme (HR@10, NDCG@10, Prec@10, Rec@10, Cov@10)
│   ├── filtering.py                   # Girdi doğrulama ve bilinen ürünleri çıkarma (apply_exclude_seen)
│   ├── contract.py                    # Hibrit entegrasyon için ortak sözleşme adaptörü (ContractAdapter)
│   ├── recommend.py                   # Yardımcı öneri arayüzü
│   └── models/
│       ├── __init__.py
│       ├── popularity.py              # Popülerlik baseline modeli (PopularityModel)
│       ├── item_knn.py                # Kosinüs + Shrinkage Item-kNN modeli (ItemKNNModel)
│       └── mf.py                      # Matrix Factorization modülü
├── saved_models/                      # Eğitilmiş model ağırlıkları (.gitignore ile hariç tutulur)
│   ├── popularity.pkl                 # Kayıtlı PopularityModel
│   └── itemknn/                       # Kayıtlı en iyi ItemKNNModel (W.npz, R.npz, params.json)
├── results/                           # Değerlendirme ve arama sonuçları
│   ├── popularity_activity_val.csv
│   ├── popularity_activity_test.csv
│   ├── itemknn_val_grid_stage1.csv
│   ├── itemknn_val_grid_stage2.csv
│   ├── itemknn_activity_val.csv
│   └── adapter_val_check.json         # ContractAdapter tüm validation regresyon ve performans raporu
├── notebooks/                         # Deney ve analiz defterleri
│   ├── 01_baseline.ipynb
│   ├── 02_item_based_cf.ipynb
│   ├── 03_matrix_factorization.ipynb
│   └── 04_comparison_examples.ipynb
├── tests/                             # Doğrulama test betikleri (pytest gerektirmez)
│   ├── test_exclude_seen.py           # seen_matrix, filtreleme ve sınır durum testleri (39 kontrol)
│   └── test_contract.py               # ContractAdapter ortak sözleşme testleri (27 kontrol)
├── scripts/                           # Doğrulama ve regresyon betikleri
│   └── check_adapter_regression.py    # Tüm validation üzerinde adaptör regresyon, süre ve bellek ölçümü
├── examples/                          # Entegrasyon ve kullanım örnekleri
│   └── contract_usage.py              # Taze süreçten model yükleme ve ContractAdapter kullanım örneği
├── requirements.txt                   # Temel paket bağımlılıkları
└── README.md
```

---

## 2. Kurulum ve Çalıştırma

### Ortam ve Paket Sürümleri (Gerçek Ortamdan)

- **Python Sürümü:** `Python 3.14.0`
- **Temel Paket Sürümleri (`.venv` ortamından):**
  - `numpy==2.5.4`
  - `pandas==3.0.6`
  - `scipy==1.18.1`
  - `scikit-learn==1.9.1`
  - `matplotlib==3.11.2`
  - `tqdm==4.70.1`
  - `jupyter==1.1.1`
  - `implicit==0.7.3`
  - `psutil==7.2.2`

Kurulum için:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Sabitler ve Tohum (Seed) Değerleri

- **`TIE_BREAK_SEED = 42`** (`src/evaluation.py`): Eşit skorlu adayların sıralamasında aday listesi sırasından kaynaklanan yanlılığı önlemek için `np.random.default_rng(42)` ile üretilen `1e-10` ölçekli deterministik gürültü tohumu.
- **`K = 10`** (`src/evaluation.py`): Top-K değerlendirme kesim noktası.
- **`COLD_ITEM_PENALTY_DEFAULT = -1e9`** (`src/evaluation.py`) ve **`FILTERED_SCORE = -1e9`** (`src/filtering.py`): Soğuk ve filtrelenen ürünlere atanan sonlu taban skor (`-np.inf` yerine sonlu `-1e9` kullanılarak tie-break mekanizmasının korunması sağlanır).

### Testlerin ve Betiklerin Çalıştırılması

Proje kök dizininden:

```bash
# 1. Bilinen ürün filtresi ve sınır durum testleri (39/39 kontrol)
.venv\Scripts\python.exe tests/test_exclude_seen.py

# 2. Ortak sözleşme adaptörü (ContractAdapter) doğrulama testleri (27/27 kontrol)
.venv\Scripts\python.exe tests/test_contract.py

# 3. Tüm validation seti (75.799 kullanıcı) uçtan uca regresyon, süre ve bellek kontrolü
.venv\Scripts\python.exe scripts/check_adapter_regression.py

# 4. Kayıtlı Item-kNN modelini yükleme ve ContractAdapter örnek kullanımı
.venv\Scripts\python.exe examples/contract_usage.py
```

---

## 3. Veri

- **Boyutlar:**
  - Kullanıcı sayısı (`N_USERS`): **94.745** (`user_idx`: `0 .. 94744`)
  - Ürün sayısı (`N_ITEMS`): **25.610** (`item_idx`: `0 .. 25609`)
- **Bölünmeler (Splits) ve Satır Sayıları:**
  - `train.csv`: **624.971** satır (`label == 1`: **504.975**, `label == 0`: **119.996**)
  - `val.csv`: **94.745** satır (`label == 1`: **75.799**, `label == 0`: **18.946**)
  - `test.csv`: **94.745** satır (`label == 1`: **74.021**, `label == 0`: **20.724**)
- **Etiket (`label`) Tanımı:** Kullanıcının ürüne verdiği puan `rating >= 4` ise pozitif etkileşim (`label = 1`), düşük puan ise negatif etkileşim (`label = 0`) olarak tanımlanmıştır.
- **`interaction_matrix` vs `seen_matrix` Farkı:**
  - `interaction_matrix` (`94.745 x 25.610` seyrek CSR): Sadece `train.csv` içindeki **`label == 1`** (beğeni) satırlarından oluşturulur (`nnz = 504.975`). CF modellerinin eğitiminde ve benzerlik hesaplarında kullanılır.
  - `seen_matrix` (`94.745 x 25.610` seyrek CSR): `train.csv` içindeki **tüm** satırlardan (`label == 0` ve `label == 1`, `nnz = 624.971`) oluşturulur. Canlı öneride kullanıcının daha önce gördüğü (beğendiği veya beğenmediği) hiçbir ürünün tekrar önerilmemesi için filtrelemede kullanılır. Veri sızıntısını önlemek amacıyla `val` ve `test` etkileşimlerini içermez.
- **100'lük Aday Havuzları (`val_candidates`, `test_candidates`):**
  - İlgili bölünmede `label == 1` etkileşimi bulunan her kullanıcı için (`val`: 75.799 kullanıcı, `test`: 74.021 kullanıcı) 100 elemanlı aday dizisi oluşturulur:
    - **İndeks `0`:** Kullanıcının gerçek pozitif (hedef) ürünü.
    - **İndeks `1 .. 99`:** `eval_negatives.csv` dosyasından gelen 99 negatif ürün.

---

## 4. Değerlendirme Protokolü

- **Veri Disiplini:** Tüm model mimarisi, hiperparametre (`k=2000`, `shrink=100.0`, `pop_weight=0.01`) ve hibrit birleştirme ağırlığı (`alpha`) kararları **SADECE validation** seti üzerinde verilir. Test seti hiçbir seçim veya ayar aşamasında kullanılmaz; yalnızca final karşılaştırmada bir kez çalıştırılır.
- **Metrikler (`K = 10`):**
  - **`HR@10` (`HitRate@10`):** Hedef pozitif ürün kullanıcının Top-10 listesine girdiyse `1`, girmediyse `0` alır; tüm değerlendirme kullanıcıları üzerinden ortalaması alınır.
  - **`NDCG@10`:** Hedef pozitif ürün Top-10 içinde `r` (`1 .. 10`) sırasındaysa `1 / log2(r + 1)`, değilse `0` alır; sıralama kalitesini ölçer.
  - **`Precision@10`:** Kullanıcı başına tek pozitif hedef ürün bulunduğu için `HR@10 / 10` değerine eşittir.
  - **`Recall@10`:** Kullanıcı başına tek pozitif hedef ürün bulunduğu için `HR@10` değerine eşittir.
  - **`Coverage@10`:** Mevcut tanımıyla, değerlendirilen tüm kullanıcıların modele göre sıralanmış **100'lük aday havuzu içindeki gerçek Top-10** listelerinde görünen benzersiz ürün sayısının toplam katalog büyüklüğüne (`25.610`) oranıdır. **Önemli Not:** Bu ölçüm 100'lük aday havuzuna bağlıdır; rastgele bir model bile her kullanıcıda farklı adaylar seçildiği için aday havuzu üzerinde yüksek kapsama üretir.

---

## 5. Ortak Model Sözleşmesi

1. Kimlik: Ortak anahtar item_idx (parent_asin <-> item_idx eşlemesi data_v1_final'dan gelir).
2. Arayüz: score(users, candidates) ve score_from_items(item_idxs, candidates). Çıktı: user_idx, item_idx, score, cf_known. Yeni kullanıcıda user_idx = -1.
3. Adaylar: val/test için hazır 100'lük havuzlar (1 pozitif + 99 negatif).
4. Skor: 0-1 aralığı. Kullanıcı başına aday havuzunda min-max; tüm skorlar eşitse normalize skor 0. Bilinen ürünler önce adaylardan çıkarılır, sonra skorlanır ve normalize edilir; maskelenmiş (-1e9) değerler normalizasyona girmez. Canlı sistemde aday havuzu ve normalizasyon backend aşamasında ayrıca tanımlanır.
5. cf_known: interaction_matrix'te hiç pozitif etkileşimi olmayan ürünler cf_known=False'tur; CF skoru bu ürünler için üretilmez (score=0.0). CF tarafı normalizasyonu sadece cf_known=True adaylar üzerinde yapar. Content-Based model ise her ürünü tanıdığı için normalizasyonunu TÜM adaylar (cf_known=False dahil) üzerinde yapar.
6. Seen filtresi: Canlı sistemde kullanıcının bildiği tüm ürünler (label 0 ve 1) filtrelenir. Offline değerlendirmede val/test hedef ürünü filtrelenmez.
7. Hybrid: cf_known=True ürünler için alpha * CF + (1 - alpha) * CB; cf_known=False ürünler için yalnızca CB skoru kullanılır. alpha 0-1 arasında 0,1 adımlarla sadece validation üzerinde seçilir.
8. Bilinmeyen durumlar: hiçbir modelin tanımadığı ürün en düşük skoru alır; bilinmeyen kullanıcıda Popularity'ye düşülür.
9. Yeni kullanıcı: tercih formundan 3-5 oyun seçerse Item-kNN ve Content-Based birlikte score_from_items ile öneri üretir; geçersiz oyunlar yok sayılır; geçerli oyun kalmazsa Popularity kullanılır.

---

## 6. Veri Denetimi Bulguları

- `train ∩ val`, `train ∩ test`, `val ∩ test` kullanıcı-ürün çifti kesişimi: **0 / 0 / 0**.
- Negatiflerin kullanıcının train'inde olması: **0**.
- Negatifler katalogdan düzgün (uniform) örneklenmiş: negatif ürünlerin ortalama/medyan popülerliği **19,7 / 7**, katalogla aynı.
- Test pozitiflerinin ortalama/medyan popülerliği **139,8 / 43** (pozitifler negatiflere göre yaklaşık 6-7 kat daha popüler); bu yüzden Popularity baseline yüksek görünür.
- `cf_known=False` ürün sayısı **243** (aday satırlarının yaklaşık **%0,95**'i, katalog payıyla aynı).

---

## 7. Bilinen Sınırlamalar

- **`Coverage@10` Metriğinin Aday Havuzuna Bağlılığı:** `src/evaluation.py` içerisindeki `Coverage@10` hesabı, 100'lük değerlendirme aday havuzundan seçilen Top-10 listeleri üzerinden yapılmaktadır; tüm katalog (`25.610` ürün) üzerinden bağımsız tam katalog kapsaması ölçümü henüz ayrı olarak yapılmamıştır.
