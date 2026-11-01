"""
src/filtering.py
================
Öneri filtreleme ve girdi doğrulama yardımcıları (modelden bağımsız).

İÇERİK:
    validate_indices   : user_idx / item_idx dizilerinin tip ve aralık kontrolü
    validate_user      : tek bir user_idx için kısa yol (bilinen kullanıcı mı?)
    apply_exclude_seen : "kullanıcının bildiği ürünleri çıkar" filtresi
                         (canlı mod / offline değerlendirme modu)

NEDEN AYRI MODÜL:
    Popülerlik, item-kNN, MF ve ileride hybrid model aynı kuralları
    kullanmalı. Kural tek yerde olursa modeller arasında tutarsızlık olmaz.
    Bu modül hiçbir modeli import etmez → döngüsel import riski yoktur.

NEDEN NEGATİF İNDEKS ÖZEL OLARAK YAKALANIYOR:
    numpy'da arr[-1] hata vermez, SON elemanı döndürür. user_idx=-1 ile
    son kullanıcının geçmişi, item_idx=-1 ile son ürünün skoru sessizce
    okunur. Bu yüzden indeksler kullanılmadan ÖNCE aralık kontrolünden geçer.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from src.evaluation import COLD_ITEM_PENALTY_DEFAULT

# Filtrelenen / bilinmeyen ürünlere verilen skor.
# evaluation.py'deki soğuk ürün cezasıyla AYNI sabit (-1e9, sonlu; -inf değil).
FILTERED_SCORE = COLD_ITEM_PENALTY_DEFAULT

VALID_MODES = ("live", "offline")


def validate_indices(
    values,
    n: Optional[int],
    name: str = "item_idx",
    warn: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    NE YAPAR:
        Bir indeks dizisini (veya tek değeri) 1-boyutlu int64 diziye çevirir ve
        her elemanın 0 <= değer < n aralığında olup olmadığını gösteren bir
        maske döndürür.

    DAVRANIŞ:
        - Tam sayı OLMAYAN girdi (float, bool, str ...) -> TypeError.
          Bu bir programlama hatasıdır; sessizce yuvarlamak yanlış sonuç verir.
        - Aralık dışı (negatif dahil) değer -> HATA FIRLATILMAZ; maskede False
          olur ve (warn=True ise) hangi değerin, hangi aralığın beklendiği
          yazan bir uyarı üretilir. Ne yapılacağına çağıran karar verir
          (bilinmeyen ürün -> en düşük skor, bilinmeyen kullanıcı -> fallback).
        - Boş girdi -> boş dizi ve boş maske (çökme yok).

    PARAMETRELER:
        values : int | array-like[int]
        n      : int | None  – geçerli aralığın üst sınırı (hariç).
                 None ise sadece negatiflik kontrol edilir.
        name   : str  – mesajlarda kullanılacak ad ("user_idx", "item_idx")
        warn   : bool – aralık dışı değer için uyarı basılsın mı

    DÖNDÜRÜR:
        (arr, valid_mask)
            arr        : np.ndarray[int64], shape (m,)
            valid_mask : np.ndarray[bool],  shape (m,)
    """
    arr = np.asarray(values)

    # Boş girdi: tip kontrolüne gerek yok (np.asarray([]) float64 döner)
    if arr.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=bool)

    # bool numpy'da tam sayı alt türüdür; True/False indeks olarak kabul edilmez
    if arr.dtype == bool or not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(
            f"{name} tam sayı olmalı; gelen tip: {arr.dtype} "
            f"(örnek değer: {arr.ravel()[0]!r})"
        )

    arr = arr.astype(np.int64, copy=False).ravel()

    valid = arr >= 0                       # negatifler: sondan okumayı engelle
    if n is not None:
        valid &= arr < n

    if warn and not valid.all():
        bad = arr[~valid]
        shown = ", ".join(str(v) for v in bad[:5]) + (" ..." if len(bad) > 5 else "")
        expected = f"0 <= {name} < {n}" if n is not None else f"{name} >= 0"
        warnings.warn(
            f"Geçersiz {name}: {len(bad)} değer aralık dışında [{shown}]. "
            f"Beklenen: {expected}.",
            stacklevel=3,
        )

    return arr, valid


def validate_user(uid, n_users: Optional[int], warn: bool = True) -> bool:
    """
    NE YAPAR:
        Tek bir user_idx'in bilinen (0 <= uid < n_users) bir kullanıcı olup
        olmadığını döndürür. Tam sayı değilse TypeError fırlatır.

    NEDEN bool DÖNDÜRÜYOR (hata değil):
        Bilinmeyen kullanıcı canlı sistemde normal bir durumdur (yeni kullanıcı).
        Çağıran taraf popülerlik sıralamasına geri döner (fallback).
    """
    arr = np.asarray(uid)
    if arr.ndim != 0:
        raise TypeError(f"user_idx tek bir tam sayı olmalı; gelen şekil: {arr.shape}")
    _, valid = validate_indices(arr, n_users, name="user_idx", warn=warn)
    return bool(valid[0])


def apply_exclude_seen(
    scores: np.ndarray,
    items: np.ndarray,
    uid: int,
    seen_matrix: csr_matrix,
    mode: str = "live",
    target_items=None,
) -> np.ndarray:
    """
    NE YAPAR:
        Kullanıcının train'de etkileşimde bulunduğu (seen_matrix'teki) ürünlerin
        skorunu FILTERED_SCORE (-1e9) yapar. Girdi dizisini DEĞİŞTİRMEZ; kopya döner.

    İKİ MOD:
        mode="live"    (canlı/production):
            seen_matrix'teki TÜM ürünler (label 0 ve 1) filtrelenir.
        mode="offline" (değerlendirme):
            Aynı filtre uygulanır, AMA target_items (o kullanıcının val/test
            hedef ürünü) ASLA filtrelenmez. Hedef filtrelenirse doğru cevabı
            kendimiz silmiş oluruz ve metrik yapay olarak düşer.
            Normalde hedef zaten seen_matrix'te olmamalıdır (seen_matrix sadece
            train'den üretilir). Eğer olursa bu bir veri sızıntısı işaretidir:
            hedef yine korunur ve uyarı basılır.

    PARAMETRELER:
        scores       : np.ndarray[float], shape (m,) – items ile hizalı skorlar
        items        : np.ndarray[int],   shape (m,) – skorların ait olduğu item_idx'ler
                       (tam katalog için np.arange(n_items), aday havuzu için cands)
        uid          : int – kullanıcı indeksi
        seen_matrix  : csr_matrix (n_users x n_items) – build_seen_matrix() çıktısı
        mode         : "live" | "offline"
        target_items : int | array-like[int] | None – offline modda ZORUNLU

    DÖNDÜRÜR:
        np.ndarray, scores ile aynı şekil/tip; filtrelenenler = -1e9.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode {VALID_MODES} değerlerinden biri olmalı; gelen: {mode!r}")
    if mode == "offline" and target_items is None:
        # Hedefi unutmak doğru cevabın sessizce silinmesine yol açabilir -> açık hata
        raise ValueError("offline modda target_items (val/test hedef ürünü) verilmelidir.")

    scores = np.array(scores, copy=True)
    items = np.asarray(items)
    if scores.shape != items.shape:
        raise ValueError(f"scores {scores.shape} ve items {items.shape} aynı şekilde olmalı.")
    if items.size == 0:
        return scores

    # Bilinmeyen kullanıcının geçmişi yoktur -> filtrelenecek ürün yok.
    # (Negatif uid ile seen_matrix'in SON satırını okumayı da engeller.)
    if not validate_user(uid, seen_matrix.shape[0], warn=False):
        return scores

    # CSR satırı: kullanıcının bildiği item_idx'ler (yoğun satır oluşturmadan)
    start, end = seen_matrix.indptr[uid], seen_matrix.indptr[uid + 1]
    seen_items = seen_matrix.indices[start:end]
    mask = np.isin(items, seen_items)

    if mode == "offline":
        # --- GÜVENCE: hedef ürün asla filtrelenmez ---
        protected = np.isin(items, np.atleast_1d(np.asarray(target_items)))
        if (mask & protected).any():
            warnings.warn(
                f"user_idx={uid}: hedef ürün seen_matrix'te bulundu (olası veri "
                f"sızıntısı). Offline modda hedef filtrelenmedi.",
                stacklevel=2,
            )
        mask &= ~protected

    scores[mask] = FILTERED_SCORE
    return scores
