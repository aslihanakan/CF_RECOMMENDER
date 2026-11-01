"""
src/contract.py
Hibrit (CF + Content-Based) entegrasyonu için modelden bağımsız ortak sözleşme adaptörü.

Mevcut CF modellerinin (ItemKNNModel, MatrixFactorization vb.) ham skor üreten
arayüzlerini ve evaluate() fonksiyonunu değiştirmeden, aday ürün skorlarını
ortak sözleşme tablosu formatına (user_idx, item_idx, score, cf_known) dönüştürür.
"""

from typing import Dict, Optional, Sequence, Union
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.filtering import (
    FILTERED_SCORE,
    apply_exclude_seen,
    validate_indices,
)

CONTRACT_COLUMNS = ["user_idx", "item_idx", "score", "cf_known"]


class ContractAdapter:
    """CF modellerini ortak Hibrit Sözleşme (Contract) arayüzüne bağlayan adaptör.

    Mevcut modellerin ``score()`` ve ``score_from_items()`` metodları ham skor
    döndürür. ``ContractAdapter`` ise:
      1. Girdi doğrulaması yapar (geçersiz tiplerde ``TypeError``, aralık dışı
         indekslerde uyarı).
      2. Kullanıcının geçmişte etkileşimde bulunduğu (seen) ürünleri aday
         listesinden çıkarır (``mode="offline"`` iken hedef ürün korunur).
      3. ``interaction_matrix`` içinde hiç pozitif etkileşimi olmayan (sütun
         toplamı 0) veya sistemde bulunmayan ürünleri ``cf_known=False`` ve
         ``score=0.0`` olarak işaretler; bu ürünleri CF skorlamasına ve
         normalizasyona dahil etmez.
      4. Yalnızca ``cf_known=True`` olan adaylar üzerinde kullanıcı bazında
         min-max [0, 1] normalizasyonu uygular.
      5. Bilinmeyen kullanıcılarda veya geçerli seçilen oyunun kalmadığı
         ``score_from_items`` çağrılarında uyarı vererek ``popularity_model``
         skorlarına düşer (fallback).
    """

    def __init__(
        self,
        model: object,
        popularity_model: object,
        seen_matrix: csr_matrix,
        interaction_matrix: csr_matrix,
        n_users: int,
        n_items: int,
    ) -> None:
        if seen_matrix is None:
            raise ValueError("seen_matrix None olamaz.")
        if interaction_matrix is None:
            raise ValueError("interaction_matrix None olamaz.")
        if int(n_users) <= 0 or int(n_items) <= 0:
            raise ValueError(f"n_users ({n_users}) ve n_items ({n_items}) pozitif olmalıdır.")

        self.model = model
        self.popularity_model = popularity_model
        self.seen_matrix = seen_matrix
        self.interaction_matrix = interaction_matrix
        self.n_users = int(n_users)
        self.n_items = int(n_items)

        # Ürün bazında pozitif etkileşim kontrolü (sütun toplamı > 0 ise cf_known=True)
        col_sums = np.asarray(self.interaction_matrix.sum(axis=0)).ravel()
        if len(col_sums) < self.n_items:
            padded = np.zeros(self.n_items, dtype=np.float64)
            padded[: len(col_sums)] = col_sums
            col_sums = padded
        self.item_cf_known: np.ndarray = (col_sums[: self.n_items] > 0).astype(bool)

    @staticmethod
    def _empty_df() -> pd.DataFrame:
        """Doğru sütun ve veri tiplerine sahip boş sözleşme DataFrame'i döndürür."""
        return pd.DataFrame(
            {
                "user_idx": pd.Series(dtype="int64"),
                "item_idx": pd.Series(dtype="int64"),
                "score": pd.Series(dtype="float64"),
                "cf_known": pd.Series(dtype="bool"),
            },
            columns=CONTRACT_COLUMNS,
        )

    @staticmethod
    def _normalize_known(raw_scores: np.ndarray) -> np.ndarray:
        """Sadece cf_known=True adayların ham skorlarını [0, 1] aralığına min-max normalize eder.

        Kurallar:
          - Maskelenmiş (FILTERED_SCORE / -1e9) değerler normalizasyona girmez.
          - Tek bilinen aday varsa veya tüm bilinen adayların skoru eşitse 0.0 döner.
          - Çıktıda NaN, inf veya [0, 1] dışı değer asla bulunmaz.
        """
        m = len(raw_scores)
        if m == 0:
            return np.empty(0, dtype=np.float64)
        if m == 1:
            return np.zeros(1, dtype=np.float64)

        scores = np.asarray(raw_scores, dtype=np.float64)
        # Güvenlik: olası FILTERED_SCORE veya NaN/inf sızıntılarına karşı kontrol
        valid_mask = np.isfinite(scores) & (scores > FILTERED_SCORE)
        if not np.all(valid_mask):
            out = np.zeros(m, dtype=np.float64)
            if np.sum(valid_mask) <= 1:
                return out
            sub = scores[valid_mask]
            s_min = float(np.min(sub))
            s_max = float(np.max(sub))
            if s_max <= s_min:
                return out
            norm_sub = (sub - s_min) / (s_max - s_min)
            norm_sub = np.clip(norm_sub, 0.0, 1.0)
            norm_sub[sub == s_min] = 0.0
            norm_sub[sub == s_max] = 1.0
            out[valid_mask] = norm_sub
            return out

        s_min = float(np.min(scores))
        s_max = float(np.max(scores))
        if s_max <= s_min:
            return np.zeros(m, dtype=np.float64)

        norm = (scores - s_min) / (s_max - s_min)
        norm = np.clip(norm, 0.0, 1.0)
        # Kayan nokta yuvarlama hatalarına karşı uçları kesinleştir
        norm[scores == s_min] = 0.0
        norm[scores == s_max] = 1.0
        return norm

    def _resolve_candidates_per_user(
        self,
        users_arr: np.ndarray,
        candidates: Union[Dict[int, Sequence[int]], Sequence[Sequence[int]], Sequence[int], np.ndarray],
    ) -> Optional[list]:
        """Kullanıcı dizisine karşılık gelen aday listelerini çözer.

        candidates:
          - dict ({user_idx: 1D_array})
          - 2D dizi / kullanıcı başına bir satır içeren liste (uzunluk == len(users_arr))
          - Tüm kullanıcılar için ortak 1D dizi
        Boş aday havuzu durumunda None döndürür.
        """
        if candidates is None:
            return None

        if isinstance(candidates, dict):
            if len(candidates) == 0:
                return None
            return [candidates.get(int(u), np.empty(0, dtype=np.int64)) for u in users_arr]

        if isinstance(candidates, np.ndarray):
            if candidates.size == 0:
                return None
            if candidates.ndim == 1:
                return [candidates for _ in range(len(users_arr))]
            if candidates.ndim == 2:
                if len(candidates) != len(users_arr):
                    raise ValueError(
                        f"2D candidates satır sayısı ({len(candidates)}) ile "
                        f"users uzunluğu ({len(users_arr)}) eşleşmiyor."
                    )
                return [candidates[i] for i in range(len(users_arr))]
            raise ValueError(f"candidates 1D veya 2D olmalıdır; alınan boyut: {candidates.ndim}D")

        # Liste / tuple / Sequence durumu
        if isinstance(candidates, (list, tuple)):
            if len(candidates) == 0:
                return None
            first = candidates[0]
            if isinstance(first, (list, tuple, np.ndarray)):
                if len(candidates) != len(users_arr):
                    raise ValueError(
                        f"2D candidates uzunluğu ({len(candidates)}) ile "
                        f"users uzunluğu ({len(users_arr)}) eşleşmiyor."
                    )
                return list(candidates)
            return [candidates for _ in range(len(users_arr))]

        # Diğer iterable veya skaler
        arr = np.asarray(candidates)
        if arr.size == 0:
            return None
        if arr.ndim <= 1:
            return [arr.ravel() for _ in range(len(users_arr))]
        if arr.ndim == 2:
            if len(arr) != len(users_arr):
                raise ValueError(
                    f"2D candidates satır sayısı ({len(arr)}) ile "
                    f"users uzunluğu ({len(users_arr)}) eşleşmiyor."
                )
            return [arr[i] for i in range(len(users_arr))]
        raise ValueError(f"Desteklenmeyen candidates boyutu: {arr.ndim}D")

    @staticmethod
    def _resolve_user_target(
        target_items: object,
        user_pos: int,
        uid: int,
        n_batch_users: int,
    ) -> object:
        """Offline modda ilgili kullanıcının hedef ürününü (target_items) çözer."""
        if target_items is None:
            return None
        if isinstance(target_items, dict):
            if uid not in target_items:
                raise ValueError(f"mode='offline' iken user_idx={uid} için target_items bulunamadı.")
            return target_items[uid]
        if isinstance(target_items, (list, tuple, np.ndarray)):
            if n_batch_users > 1 and len(target_items) == n_batch_users:
                return target_items[user_pos]
        return target_items

    def score(
        self,
        users: Union[int, Sequence[int], np.ndarray],
        candidates: Union[Dict[int, Sequence[int]], Sequence[Sequence[int]], Sequence[int], np.ndarray],
        mode: str = "live",
        target_items: Optional[object] = None,
    ) -> pd.DataFrame:
        """Kullanıcı(lar) ve aday ürünler için normalize edilmiş sözleşme tablosu üretir.

        Parameters
        ----------
        users : int veya 1D int dizisi
            Skorlanacak kullanıcı indeks(ler)i.
        candidates : 1D dizi, 2D dizi veya dict
            Tüm kullanıcılar için ortak 1D aday dizisi ya da kullanıcı başına
            bir aday satırı (2D dizi / ``val_candidates`` sözlüğü).
        mode : {"live", "offline"}, varsayılan="live"
            ``"live"`` modda kullanıcının tüm bilinen (seen) ürünleri adaylardan
            çıkarılır. ``"offline"`` modda ``target_items`` verilmeli ve hedef
            ürün asla adaylardan çıkarılmaz.
        target_items : int, Sequence[int] veya dict, opsiyonel
            ``mode="offline"`` iken korunacak hedef ürün(ler).

        Returns
        -------
        pd.DataFrame
            Sütunlar: ``["user_idx", "item_idx", "score", "cf_known"]``
        """
        if mode not in ("live", "offline"):
            raise ValueError(f"mode 'live' veya 'offline' olmalıdır; alınan: {mode!r}")
        if mode == "offline" and target_items is None:
            raise ValueError("mode='offline' iken target_items verilmelidir (None olamaz).")

        # 1. Kullanıcı girdi doğrulaması
        users_arr, valid_users_mask = validate_indices(
            users, self.n_users, name="user_idx", warn=True
        )
        if users_arr.size == 0:
            return self._empty_df()

        # Boş aday listesi kontrolü (girdi tipini de doğrulamak için)
        per_user_candidates = self._resolve_candidates_per_user(users_arr, candidates)
        if per_user_candidates is None:
            return self._empty_df()

        out_users = []
        out_items = []
        out_scores = []
        out_known = []

        n_batch = len(users_arr)

        for i in range(n_batch):
            uid = int(users_arr[i])
            is_valid_user = bool(valid_users_mask[i])
            raw_cands = per_user_candidates[i]

            # Aday girdi doğrulaması (tip + aralık kontrolü)
            u_cands, valid_item_mask = validate_indices(
                raw_cands, self.n_items, name="item_idx", warn=True
            )
            if u_cands.size == 0:
                continue

            # 2. Bilinen ürünleri adaylardan ÇIKAR (sadece geçerli kullanıcılar için)
            if is_valid_user:
                u_target = (
                    self._resolve_user_target(target_items, i, uid, n_batch)
                    if mode == "offline"
                    else None
                )
                dummy_scores = np.zeros(len(u_cands), dtype=np.float64)
                # apply_exclude_seen yeni bir dizi döndürür (in-place değildir);
                # geçersiz item_idx'ler varsa sadece geçerli dilime uygula
                if np.all(valid_item_mask):
                    dummy_scores = apply_exclude_seen(
                        dummy_scores,
                        u_cands,
                        uid,
                        self.seen_matrix,
                        mode=mode,
                        target_items=u_target,
                    )
                elif np.any(valid_item_mask):
                    valid_sub_scores = np.zeros(int(np.sum(valid_item_mask)), dtype=np.float64)
                    valid_sub_scores = apply_exclude_seen(
                        valid_sub_scores,
                        u_cands[valid_item_mask],
                        uid,
                        self.seen_matrix,
                        mode=mode,
                        target_items=u_target,
                    )
                    dummy_scores[valid_item_mask] = valid_sub_scores

                keep_mask = dummy_scores != FILTERED_SCORE
                u_cands = u_cands[keep_mask]
                valid_item_mask = valid_item_mask[keep_mask]

                if u_cands.size == 0:
                    continue
            else:
                # Bilinmeyen kullanıcı (uid < 0 veya uid >= n_users):
                # validate_indices zaten uyarı verdi; Popularity'ye düşülür.
                warnings.warn(
                    f"Bilinmeyen kullanıcı (user_idx={uid}); Popularity modeline düşülüyor.",
                    UserWarning,
                    stacklevel=2,
                )

            # 3. cf_known tespiti (geçersiz ürünler ve interaction_matrix sütun toplamı 0 olanlar False)
            cf_known = np.zeros(len(u_cands), dtype=bool)
            if np.any(valid_item_mask):
                cf_known[valid_item_mask] = self.item_cf_known[u_cands[valid_item_mask]]

            # 4. Modelle skorla (yalnızca cf_known=True ürünler) ve [0, 1] normalize et
            u_scores = np.zeros(len(u_cands), dtype=np.float64)
            if np.any(cf_known):
                known_cands = u_cands[cf_known]
                if is_valid_user:
                    raw_known_scores = np.asarray(
                        self.model.score(uid, known_cands), dtype=np.float64
                    )
                else:
                    # PopularityModel kullanıcıdan bağımsızdır; ikinci kez aralık dışı
                    # kullanıcı uyarısı üretmemek için geçerli 0 indeksiyle çağrılır.
                    raw_known_scores = np.asarray(
                        self.popularity_model.score(0, known_cands), dtype=np.float64
                    )
                u_scores[cf_known] = self._normalize_known(raw_known_scores)

            out_users.append(np.full(len(u_cands), uid, dtype=np.int64))
            out_items.append(u_cands.astype(np.int64, copy=False))
            out_scores.append(u_scores)
            out_known.append(cf_known)

        if not out_users:
            return self._empty_df()

        return pd.DataFrame(
            {
                "user_idx": np.concatenate(out_users),
                "item_idx": np.concatenate(out_items),
                "score": np.concatenate(out_scores),
                "cf_known": np.concatenate(out_known),
            },
            columns=CONTRACT_COLUMNS,
        )

    def score_from_items(
        self,
        item_idxs: Sequence[int],
        candidates: Sequence[int],
    ) -> pd.DataFrame:
        """Yeni (soğuk) kullanıcının seçtiği oyunlara göre adayları skorlar ve normalize eder.

        Not: Yeni kullanıcının sistemde kayıtlı bir kullanıcı kimliği olmadığı için
        çıktı tablosundaki ``user_idx`` sütununa ``-1`` yazılır.

        Kurallar:
          - ``item_idxs`` içindeki geçersiz / sistemde olmayan oyunlar yok sayılır
            ve uyarı verilir.
          - Seçilen geçerli oyunlar ``candidates`` listesinden çıkarılır (çıktıda yer almaz).
          - Geçerli seçilen oyun kalmazsa uyarı verilerek ``popularity_model``'e düşülür;
            ``cf_known`` sütunu yine aynı kurallarla doldurulur.
          - Skorlama için ``model.score_from_items(valid_item_idxs, known_candidates)``
            kullanılır ve yalnızca ``cf_known=True`` adaylar üzerinde [0, 1] min-max
            normalizasyonu yapılır.

        Parameters
        ----------
        item_idxs : Sequence[int]
            Yeni kullanıcının seçtiği (ör. 3-5 adet) oyun indeksleri.
        candidates : Sequence[int]
            Skorlanacak aday ürün indeksleri (1D dizi).

        Returns
        -------
        pd.DataFrame
            Sütunlar: ``["user_idx", "item_idx", "score", "cf_known"]`` (``user_idx = -1``).
        """
        # 1. Seçilen oyunları ve adayları doğrula
        liked_arr, valid_liked_mask = validate_indices(
            item_idxs, self.n_items, name="item_idx", warn=True
        )
        cands_arr, valid_cands_mask = validate_indices(
            candidates, self.n_items, name="item_idx", warn=True
        )

        if cands_arr.size == 0:
            return self._empty_df()

        valid_liked = liked_arr[valid_liked_mask]

        # 2. Seçilen geçerli oyunları adaylardan çıkar
        if valid_liked.size > 0:
            keep_mask = ~np.isin(cands_arr, valid_liked)
            cands_arr = cands_arr[keep_mask]
            valid_cands_mask = valid_cands_mask[keep_mask]
            if cands_arr.size == 0:
                return self._empty_df()

        # 3. cf_known sütununu belirle
        cf_known = np.zeros(len(cands_arr), dtype=bool)
        if np.any(valid_cands_mask):
            cf_known[valid_cands_mask] = self.item_cf_known[cands_arr[valid_cands_mask]]

        # 4. Skorla ve normalize et
        scores = np.zeros(len(cands_arr), dtype=np.float64)
        if np.any(cf_known):
            known_cands = cands_arr[cf_known]
            if valid_liked.size == 0:
                warnings.warn(
                    "score_from_items: Geçerli seçilen ürün kalmadı; Popularity modeline düşülüyor.",
                    UserWarning,
                    stacklevel=2,
                )
                raw_known_scores = np.asarray(
                    self.popularity_model.score(0, known_cands), dtype=np.float64
                )
            else:
                raw_known_scores = np.asarray(
                    self.model.score_from_items(valid_liked, known_cands), dtype=np.float64
                )
            scores[cf_known] = self._normalize_known(raw_known_scores)

        return pd.DataFrame(
            {
                "user_idx": np.full(len(cands_arr), -1, dtype=np.int64),
                "item_idx": cands_arr.astype(np.int64, copy=False),
                "score": scores,
                "cf_known": cf_known,
            },
            columns=CONTRACT_COLUMNS,
        )
