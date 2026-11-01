"""
tests/test_exclude_seen.py
==========================
seen_matrix / exclude_seen / girdi doğrulaması için doğrulama scripti.

Çalıştırma (proje kökünden):
    .venv\\Scripts\\python.exe tests\\test_exclude_seen.py

NOT: Ek bağımlılık (pytest) gerektirmez. Hiçbir dosyaya yazmaz.
Test seti burada SADECE "offline modda hedef filtrelenmez" kuralını ve
popülerlik referans sayılarının değişmediğini doğrulamak için okunur;
model seçimi / ayarı için kullanılmaz.
"""

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.data_loader import load_data, N_USERS, N_ITEMS          # noqa: E402
from src.evaluation import evaluate                                # noqa: E402
from src.filtering import (                                        # noqa: E402
    FILTERED_SCORE, apply_exclude_seen, validate_indices,
)
from src.models.popularity import PopularityModel                  # noqa: E402

RESULTS = []


def check(name, cond, detail=""):
    """Tek bir koşulu kaydet ve ekrana bas."""
    RESULTS.append((name, bool(cond)))
    print(f"  [{'GECTI' if cond else 'KALDI'}] {name}" + (f"  -> {detail}" if detail else ""))


def caught(fn, *args, **kwargs):
    """fn'i çalıştır; (sonuç, yakalanan uyarı mesajları) döndür."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = fn(*args, **kwargs)
    return out, [str(x.message) for x in w]


def raises(exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc as e:
        return str(e)
    return None


# ---------------------------------------------------------------------------
print("Veri yukleniyor...")
data = load_data("data")
train_df, val_df, test_df = data["train_df"], data["val_df"], data["test_df"]
mat, seen = data["interaction_matrix"], data["seen_matrix"]
val_c, test_c = data["val_candidates"], data["test_candidates"]

model = PopularityModel().fit(mat)
pop = model.item_scores

# Test kullanıcısı: train'de label==0 ürünü olan, val ve test'te pozitifi olan;
# label==0 ürünü EN POPÜLER olan kullanıcı (eski hatayı top-10'da görünür kılar).
neg = train_df[train_df["label"] == 0][["user_idx", "item_idx"]].copy()
neg = neg[neg["user_idx"].isin(val_c.keys()) & neg["user_idx"].isin(test_c.keys())]
neg["pop"] = pop[neg["item_idx"].to_numpy()]
row = neg.sort_values(["pop", "user_idx"], ascending=[False, True]).iloc[0]
U, J = int(row["user_idx"]), int(row["item_idx"])
print(f"\nSecilen kullanici U={U}, train label==0 urunu J={J} (populerlik={int(row['pop'])}, "
      f"global sira={int(np.where(np.argsort(-pop) == J)[0][0]) + 1})")

# ---------------------------------------------------------------------------
print("\n[1] Canli mod: label==0 urun oneride YOK")
old, w_old = caught(model.recommend_topk, U, mat, k=10, exclude_seen=True)   # eski davranis (seen_matrix yok)
new = model.recommend_topk(U, mat, k=10, exclude_seen=True, seen_matrix=seen, mode="live")
new_full = model.recommend_topk(U, mat, k=N_ITEMS, exclude_seen=True, seen_matrix=seen, mode="live")
user_seen = seen.indices[seen.indptr[U]:seen.indptr[U + 1]]
check("Eski davranis (sadece label==1) J'yi top-10'da oneriyordu (hata yeniden uretildi)", J in old,
      f"eski top10={old.tolist()}, uyari={'var' if w_old else 'yok'}")
check("Yeni canli mod: J top-10'da yok", J not in new, f"yeni top10={new.tolist()}")
check("Yeni canli mod: J tam katalog listesinde de yok", J not in new_full)
check("Yeni canli mod: kullanicinin bildigi HICBIR urun listede yok",
      not np.isin(new_full, user_seen).any(), f"bilinen urun sayisi={len(user_seen)}")
check("Tam liste boyu = katalog - bilinen", len(new_full) == N_ITEMS - len(user_seen))

# ---------------------------------------------------------------------------
print("\n[2] Offline mod: val/test hedefi filtrelenmiyor")
for split, cands_dict in [("val", val_c), ("test", test_c)]:
    cands = cands_dict[U]
    target = int(cands[0])
    base = model.score(U, cands)
    off = apply_exclude_seen(base, cands, U, seen, mode="offline", target_items=target)
    check(f"{split}: hedef ({target}) aday listesinde ve skoru -1e9 degil",
          target in cands and off[0] != FILTERED_SCORE and off[0] == base[0], f"skor={off[0]}")
    full = model.recommend_topk(U, mat, k=N_ITEMS, seen_matrix=seen, mode="offline", target_items=target)
    check(f"{split}: tam katalog offline listede hedef var, J yok", (target in full) and (J not in full))

# Güvencenin gerçekten çalıştığını göster: hedefi yapay olarak seen'e ekle (sızıntı simülasyonu)
target = int(val_c[U][0])
leaky = seen.tolil(copy=True); leaky[U, target] = 1.0; leaky = leaky.tocsr()
s = model.score(U, val_c[U])
off_leak, w_leak = caught(apply_exclude_seen, s, val_c[U], U, leaky, mode="offline", target_items=target)
live_leak = apply_exclude_seen(s, val_c[U], U, leaky, mode="live")
check("Sizinti simulasyonu: offline modda hedef YINE korunuyor + uyari", off_leak[0] != FILTERED_SCORE and len(w_leak) == 1,
      w_leak[0] if w_leak else "")
check("Sizinti simulasyonu: ayni durumda canli mod hedefi filtreliyor (fark kanitlandi)", live_leak[0] == FILTERED_SCORE)
msg = raises(ValueError, apply_exclude_seen, s, val_c[U], U, seen, mode="offline")
check("Offline modda target_items verilmezse acik hata", msg is not None, msg)

# ---------------------------------------------------------------------------
print("\n[3] seen_matrix ile val/test ciftleri kesismiyor")
check("seen_matrix nnz == train satir sayisi (label 0+1)", seen.nnz == len(train_df), f"{seen.nnz:,} / {len(train_df):,}")
t0 = train_df[train_df["label"] == 0]
check("Tum train label==0 ciftleri seen_matrix'te 1",
      np.asarray(seen[t0["user_idx"].to_numpy(), t0["item_idx"].to_numpy()]).ravel().min() == 1.0)
for split, df in [("val", val_df), ("test", test_df)]:
    hits = np.asarray(seen[df["user_idx"].to_numpy(), df["item_idx"].to_numpy()]).ravel()
    check(f"{split}: {len(df):,} ciftin (label 0+1) hicbiri seen_matrix'te degil", hits.sum() == 0,
          f"kesisim={int(hits.sum())}")

# ---------------------------------------------------------------------------
print("\n[4] Negatif indeksler sessizce sondan okunmuyor")
out, w = caught(model.score, 0, np.array([-1, 5]))
check("score(item=-1) -> -1e9 (son urunun skoru DEGIL) + uyari",
      out[0] == FILTERED_SCORE and out[1] == pop[5] and len(w) == 1,
      f"skor={out[0]}, son urun skoru={pop[-1]}, uyari='{w[0] if w else ''}'")
global_top = np.argsort(-pop)[:10]
last_user_top = model.recommend_topk(N_USERS - 1, mat, k=10, seen_matrix=seen)
neg_user_top, w = caught(model.recommend_topk, -1, mat, k=10, seen_matrix=seen)
check("recommend_topk(user=-1) -> uyari + saf populerlik (son kullanicinin filtresi DEGIL)",
      np.array_equal(neg_user_top, global_top) and len(w) == 1,
      f"son kullanici listesi farkli mi: {not np.array_equal(last_user_top, global_top)}")
unchanged = apply_exclude_seen(np.ones(3), np.array([1, 2, 3]), -1, seen)
check("apply_exclude_seen(uid=-1) son satiri okumuyor (hicbir sey filtrelenmez)", (unchanged == 1).all())
# Güçlü kanıt: SON kullanıcının gerçekten bildiği ürünleri aday yap.
# uid=-1 sondan okusaydı bunlar -1e9 olurdu; olmamalı.
last_seen = seen.indices[seen.indptr[N_USERS - 1]:seen.indptr[N_USERS]]
via_last = apply_exclude_seen(np.ones(len(last_seen)), last_seen, N_USERS - 1, seen)
via_neg = apply_exclude_seen(np.ones(len(last_seen)), last_seen, -1, seen)
check("Son kullanicinin bildigi urunler: uid=N-1 hepsini filtreler, uid=-1 HICBIRINI filtrelemez",
      (via_last == FILTERED_SCORE).all() and (via_neg == 1).all(), f"son kullanicinin bildigi urun={len(last_seen)}")
s_neg_user, w = caught(model.score, -1, last_seen)
check("score(user=-1) cokmez + uyari", len(w) == 1 and np.array_equal(s_neg_user, pop[last_seen]))
check("Tam sayi olmayan item_idx -> TypeError", raises(TypeError, validate_indices, np.array([1.5]), N_ITEMS) is not None,
      raises(TypeError, validate_indices, np.array([1.5]), N_ITEMS))
check("bool indeks -> TypeError", raises(TypeError, validate_indices, np.array([True]), N_ITEMS) is not None)
check("Tam sayi olmayan user_idx -> TypeError", raises(TypeError, model.score, 0.5, np.array([1])) is not None)

# ---------------------------------------------------------------------------
print("\n[5] Bilinmeyen kullanici / urun")
unk_top, w = caught(model.recommend_topk, N_USERS, mat, k=10, seen_matrix=seen)
check(f"Bilinmeyen kullanici ({N_USERS}) -> populerlik fallback + uyari",
      np.array_equal(unk_top, global_top) and len(w) == 1, w[0] if w else "")
c = val_c[U]
s_unk, w = caught(model.score, N_USERS, c)
check("score(bilinmeyen kullanici) -> cokmez, populerlik skorlari + uyari",
      np.array_equal(s_unk, model.score(U, c)) and len(w) == 1)
s_item, w = caught(model.score, 0, np.array([N_ITEMS, 10**9, 7]))
check(f"Bilinmeyen urun ({N_ITEMS}, 1e9) -> en dusuk skor (-1e9), gecerli urun normal",
      s_item[0] == FILTERED_SCORE and s_item[1] == FILTERED_SCORE and s_item[2] == pop[7], w[0] if w else "")

# ---------------------------------------------------------------------------
print("\n[6] Bos aday listesi")
check("score(0, []) -> bos dizi", model.score(0, []).size == 0)
check("score(0, np.array([], int)) -> bos dizi", model.score(0, np.array([], dtype=np.int64)).size == 0)
check("recommend_topk(k=0) -> bos dizi", model.recommend_topk(0, mat, k=0, seen_matrix=seen).size == 0)
check("apply_exclude_seen(bos) -> bos dizi", apply_exclude_seen(np.array([]), np.array([], dtype=np.int64), 0, seen).size == 0)

# ---------------------------------------------------------------------------
print("\n[7] Popularity baseline: once/sonra")
REF = {"val": (0.5488, 0.3369), "test": (0.5068, 0.3090)}
for split, cands_dict in [("val", val_c), ("test", test_c)]:
    res, w = caught(evaluate, model.score, cands_dict, cold_items=data["cold_items"], verbose=False)
    hr, nd = res["overall"]["hit_rate"], res["overall"]["ndcg"]
    check(f"{split}: HR@10={hr:.6f} (ref {REF[split][0]})  NDCG@10={nd:.6f} (ref {REF[split][1]})",
          abs(hr - REF[split][0]) <= 1e-4 and abs(nd - REF[split][1]) <= 1e-4 and not w,
          f"degerlendirme sirasinda uyari sayisi={len(w)}")

    # Ek: offline exclude_seen evaluate'e sarilsa bile sonuc degismez (negatiflerde train urunu yok)
    def wrapped(uid, cands):
        return apply_exclude_seen(model.score(uid, cands), cands, uid, seen, mode="offline", target_items=cands[0])
    res2 = evaluate(wrapped, cands_dict, cold_items=data["cold_items"], verbose=False)
    check(f"{split}: offline filtre eklenmis degerlendirme birebir ayni",
          res2["overall"]["hit_rate"] == hr and res2["overall"]["ndcg"] == nd)

# ---------------------------------------------------------------------------
n_ok = sum(ok for _, ok in RESULTS)
print(f"\nSONUC: {n_ok}/{len(RESULTS)} kontrol gecti.")
sys.exit(0 if n_ok == len(RESULTS) else 1)
