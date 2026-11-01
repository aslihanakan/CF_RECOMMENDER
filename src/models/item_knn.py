"""Item-Based Collaborative Filtering (kosinüs benzerliği, en yakın K komşu)."""
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp

from src.filtering import FILTERED_SCORE, apply_exclude_seen


class ItemKNNModel:
    """Ürün-ürün benzerliğine dayalı öneri modeli.

    Skor: kullanıcının beğendiği ürünlerden, aday ürünün en yakın K komşuları arasında
    olanların benzerlik toplamı.
    """

    def __init__(self, k: int = 1000, shrink: float = 10.0, pop_weight: float = 0.0, chunk_size: int = 1000) -> None:
        self.k = k                    # her ürün için tutulan komşu sayısı
        self.shrink = shrink          # ortak kullanıcısı az çiftlerin benzerliğini küçültür (0 = kapalı)
        self.pop_weight = pop_weight  # skora eklenen küçük popülerlik payı (CF skoru 0 ise sıralamayı belirler)
        self.chunk_size = chunk_size  # benzerlik hesabında aynı anda işlenen ürün sayısı
        self.pop_norm = None          # ürün popülerliği / en yüksek popülerlik
        self.W = None                 # (ürün x ürün) seyrek komşuluk matrisi: W[j, i] = benzerlik(j, i)
        self.train_matrix = None      # (kullanıcı x ürün) CSR, label==1
        self.fit_seconds = None

    # ------------------------------------------------------------------ eğitim
    def fit(self, train_matrix: sp.csr_matrix) -> "ItemKNNModel":
        t0 = time.time()
        R = train_matrix.tocsr().astype(np.float32)
        self.train_matrix = R
        n_items = R.shape[1]
        self._set_popularity()

        X = R.T.tocsr()                                        # ürün x kullanıcı
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        inv = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
        Xn = sp.diags(inv.astype(np.float32)) @ X              # satır başına birim uzunluk
        XnT = Xn.T.tocsr()
        XT = X.T.tocsr()

        rows, cols, vals = [], [], []
        for start in range(0, n_items, self.chunk_size):
            end = min(start + self.chunk_size, n_items)
            block = (Xn[start:end] @ XnT).toarray()            # (chunk x n_items) kosinüs
            if self.shrink > 0:
                co = (X[start:end] @ XT).toarray()             # ortak kullanıcı sayısı
                block *= co / (co + self.shrink)
            block[np.arange(end - start), np.arange(start, end)] = 0.0   # kendisiyle benzerlik yok
            kk = min(self.k, n_items - 1)
            top = np.argpartition(-block, kk - 1, axis=1)[:, :kk]
            top_vals = np.take_along_axis(block, top, axis=1)
            keep = top_vals > 0
            r = np.repeat(np.arange(start, end), kk).reshape(-1, kk)
            rows.append(r[keep]); cols.append(top[keep]); vals.append(top_vals[keep])

        self.W = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_items, n_items), dtype=np.float32,
        )
        self.fit_seconds = time.time() - t0
        return self

    # ------------------------------------------------------------------ skorlama
    def _set_popularity(self) -> None:
        pop = np.asarray(self.train_matrix.sum(axis=0)).ravel()
        self.pop_norm = (pop / pop.max()).astype(np.float32)

    def _all_scores(self, liked_items) -> np.ndarray:
        """Tüm ürünler için skor: beğenilen ürünlerin komşu listelerinin toplamı (+ küçük popülerlik payı)."""
        liked = np.asarray(liked_items, dtype=np.int64)
        if len(liked) == 0:
            scores = np.zeros(self.W.shape[0], dtype=np.float64)
        else:
            scores = np.asarray(self.W[liked].sum(axis=0), dtype=np.float64).ravel()
        if self.pop_weight > 0:
            scores = scores + self.pop_weight * self.pop_norm
        return scores

    def _liked_of(self, uid: int) -> np.ndarray:
        R = self.train_matrix
        return R.indices[R.indptr[uid]:R.indptr[uid + 1]]

    def score(self, uid: int, cands: np.ndarray) -> np.ndarray:
        """Bir kullanıcı için aday ürünlerin skorları (değerlendirme fonksiyonunun beklediği arayüz)."""
        return self._all_scores(self._liked_of(uid))[cands]

    def score_from_items(self, liked_items, cands: np.ndarray) -> np.ndarray:
        """Eğitimde olmayan (yeni) kullanıcı: sadece beğendiği ürün listesinden skor."""
        return self._all_scores(liked_items)[cands]

    # ------------------------------------------------------------------ öneri
    def _topk(self, scores: np.ndarray, k: int, exclude) -> list:
        scores = scores.copy()
        if exclude is not None and len(exclude) > 0:
            scores[np.asarray(exclude, dtype=np.int64)] = FILTERED_SCORE
        k = min(k, len(scores))
        if k <= 0:
            return []
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx if scores[i] > FILTERED_SCORE]

    def recommend_topk(
        self,
        uid: int,
        k: int = 10,
        exclude_seen: bool = True,
        seen_matrix: Optional[sp.csr_matrix] = None,
        mode: str = "live",
        target_items=None,
        extra_exclude=None,
    ) -> list:
        """Tüm ürünler arasından Top-K. Döner: [(item_idx, skor), ...]."""
        if exclude_seen and seen_matrix is None:
            raise ValueError(
                "exclude_seen=True iken seen_matrix zorunludur "
                "(load_data()['seen_matrix'] verin)."
            )
        liked = self._liked_of(uid)
        scores = self._all_scores(liked)
        if exclude_seen:
            scores = apply_exclude_seen(
                scores, np.arange(self.W.shape[0]), uid, seen_matrix,
                mode=mode, target_items=target_items,
            )
        return self._topk(scores, k, extra_exclude)

    def recommend_from_items(self, liked_items, k: int = 10) -> list:
        """Yeni kullanıcı (tercih formu): seçtiği oyunlardan Top-K, seçtikleri hariç."""
        return self._topk(self._all_scores(liked_items), k, liked_items)

    # ------------------------------------------------------------------ kaydet / yükle
    def save(self, path: str) -> None:
        """path bir klasördür: W.npz, R.npz ve params.json yazılır."""
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        sp.save_npz(folder / "W.npz", self.W)
        sp.save_npz(folder / "R.npz", self.train_matrix)
        (folder / "params.json").write_text(
            json.dumps({"k": self.k, "shrink": self.shrink, "pop_weight": self.pop_weight, "fit_seconds": self.fit_seconds}), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "ItemKNNModel":
        folder = Path(path)
        params = json.loads((folder / "params.json").read_text(encoding="utf-8"))
        model = cls(k=params["k"], shrink=params["shrink"], pop_weight=params.get("pop_weight", 0.0))
        model.W = sp.load_npz(folder / "W.npz").tocsr()
        model.train_matrix = sp.load_npz(folder / "R.npz").tocsr()
        model._set_popularity()
        model.fit_seconds = params.get("fit_seconds")
        return model

    def __repr__(self) -> str:
        nnz = None if self.W is None else self.W.nnz
        return f"ItemKNNModel(k={self.k}, shrink={self.shrink}, pop_weight={self.pop_weight}, W_nnz={nnz})"

