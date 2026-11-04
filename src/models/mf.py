"""
src/models/mf.py
================
Implicit ALS (Alternating Least Squares - Hu, Koren & Volinsky, 2008) Matrix
Factorization öneri modeli.

Özellikler:
  - Yalnızca train setindeki pozitif (label == 1) etkileşimlerden oluşan
    seyrek CSR matrisi (kullanıcı x ürün: 94.745 x 25.610) ile eğitilir.
  - Yoğun (dense) matris oluşturulmaz; girdi CSR formatında korunur.
  - implicit (0.7.3) CPU ALS çözücüsü kullanılır; OpenBLAS iç iş parçacığı
    havuzu 1'e sabitlenerek (threadpool_limits(1, "blas")) deterministik ve
    hızlı çalışma sağlanır.
  - Güven (confidence) ağırlığı: C_ui = 1 + alpha * R_ui.
  - Yeni kullanıcı (score_from_items) için iki yöntem desteklenir:
      (A) "ridge" (fold-in): (Y^T Y + alpha * Y_u^T Y_u + reg * I)^{-1} (1 + alpha) sum(y_i)
      (B) "mean"  (pseudo-user): seçilen ürünlerin item faktörlerinin ortalaması (mean(y_i))
"""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path
from typing import Optional, Sequence

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import implicit.cpu.als as cpu_als
import numpy as np
import scipy.sparse as sp
from threadpoolctl import threadpool_limits

try:
    from src.filtering import (
        FILTERED_SCORE,
        apply_exclude_seen,
        validate_indices,
        validate_user,
    )
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.filtering import (
        FILTERED_SCORE,
        apply_exclude_seen,
        validate_indices,
        validate_user,
    )

VALID_FOLD_IN_METHODS = ("ridge", "mean")

# Validation hiperparametre taraması (Parça 3/4) sonucunda seçilen en iyi sabitler:
DEFAULT_FACTORS = 32
DEFAULT_REGULARIZATION = 0.01
DEFAULT_ALPHA = 60.0
DEFAULT_ITERATIONS = 30
DEFAULT_RANDOM_STATE = 42
DEFAULT_NUM_THREADS = 1
DEFAULT_FOLD_IN_METHOD = "ridge"


class ALSModel:
    """Implicit Alternating Least Squares (ALS) Matrix Factorization modeli.

    Varsayılan parametreler (Validation grid taraması en iyi konfigürasyonu):
      - factors        = 32
      - regularization = 0.01
      - alpha          = 60.0  (confidence = 1 + alpha * r_ui)
      - iterations     = 30
      - random_state   = 42
      - num_threads    = 1     (CPU üzerinde deterministik çalışma)
      - fold_in_method = "ridge" ("ridge" veya "mean"; seçilen parametrelerle
                                   5.000 val simülasyonunda "ridge" hem NDCG@10
                                   (0.4538 vs 0.4004) hem HR@10 (0.6466 vs 0.6062)
                                   bazında istatistiksel olarak anlamlı üstündür)
    """

    def __init__(
        self,
        factors: int = DEFAULT_FACTORS,
        regularization: float = DEFAULT_REGULARIZATION,
        alpha: float = DEFAULT_ALPHA,
        iterations: int = DEFAULT_ITERATIONS,
        random_state: int = DEFAULT_RANDOM_STATE,
        num_threads: int = DEFAULT_NUM_THREADS,
        fold_in_method: str = DEFAULT_FOLD_IN_METHOD,
    ) -> None:
        if fold_in_method not in VALID_FOLD_IN_METHODS:
            raise ValueError(
                f"fold_in_method {VALID_FOLD_IN_METHODS} değerlerinden biri olmalı; "
                f"gelen: {fold_in_method!r}"
            )
        self.factors = int(factors)
        self.regularization = float(regularization)
        self.alpha = float(alpha)
        self.iterations = int(iterations)
        self.random_state = int(random_state)
        self.num_threads = int(num_threads)
        self.fold_in_method = str(fold_in_method)

        self._als: Optional[cpu_als.AlternatingLeastSquares] = None
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self._YtY: Optional[np.ndarray] = None
        self.train_matrix: Optional[sp.csr_matrix] = None
        self.item_pop_scores: Optional[np.ndarray] = None
        self.item_cf_known: Optional[np.ndarray] = None
        self.user_has_pos: Optional[np.ndarray] = None
        self.n_users: int = 0
        self.n_items: int = 0
        self.fit_seconds: Optional[float] = None

    def _init_metadata_from_matrix(self) -> None:
        """Eğitim matrisinden popülerlik fallback vektörünü ve cf_known maskelerini hazırlar."""
        if self.train_matrix is None:
            return
        pop = np.asarray(self.train_matrix.sum(axis=0)).ravel().astype(np.float64)
        self.item_pop_scores = pop
        self.item_cf_known = (pop > 0).astype(bool)
        user_pos = np.diff(self.train_matrix.indptr)
        self.user_has_pos = (user_pos > 0).astype(bool)

    def _prepare_yty(self) -> None:
        """Kapalı form fold-in için (factors x factors) boyutlu Y^T Y matrisini önbelleğe alır."""
        if self.item_factors is not None:
            Y64 = self.item_factors.astype(np.float64)
            self._YtY = Y64.T @ Y64

    def fit(self, train_matrix: sp.csr_matrix, show_progress: bool = False) -> "ALSModel":
        """Kullanıcı x ürün (N_USERS x N_ITEMS) seyrek CSR matrisi üzerinde ALS eğitir."""
        if not sp.isspmatrix_csr(train_matrix):
            R = train_matrix.tocsr().astype(np.float32)
        else:
            R = train_matrix.astype(np.float32, copy=False)

        self.train_matrix = R
        self.n_users, self.n_items = R.shape
        self._init_metadata_from_matrix()

        t0 = time.perf_counter()
        with threadpool_limits(1, "blas"):
            als_solver = cpu_als.AlternatingLeastSquares(
                factors=self.factors,
                regularization=self.regularization,
                alpha=self.alpha,
                iterations=self.iterations,
                random_state=self.random_state,
                num_threads=self.num_threads,
                dtype=np.float32,
            )
            als_solver.fit(R, show_progress=show_progress)

        self.fit_seconds = time.perf_counter() - t0
        self._als = als_solver
        self.user_factors = np.asarray(als_solver.user_factors, dtype=np.float32)
        self.item_factors = np.asarray(als_solver.item_factors, dtype=np.float32)
        # Pozitif etkileşimi olmayan veya izole bileşenlerde CG çözücüsünün float32
        # makine epsilonu (~1e-7) düzeyinde bıraktığı sayısal kalıntıları sıfırla;
        # böylece min-max normalizasyonunda ~1e-13 gürültünün [0, 1]'e büyütülmesi önlenir.
        if self.user_has_pos is not None:
            self.user_factors[~self.user_has_pos] = 0.0
        u_norms = np.linalg.norm(self.user_factors, axis=1)
        self.user_factors[u_norms < 1e-4] = 0.0
        if self.item_cf_known is not None:
            self.item_factors[~self.item_cf_known] = 0.0
        i_norms = np.linalg.norm(self.item_factors, axis=1)
        self.item_factors[i_norms < 1e-4] = 0.0
        self._prepare_yty()
        return self

    def score(self, uid: int, cands: Sequence[int]) -> np.ndarray:
        """Bir kullanıcı için aday ürünlerin ham skorlarını döndürür.

        Kurallar:
          - Bilinmeyen kullanıcı (uid < 0 veya uid >= n_users, örn. user_idx=-1)
            için uyarı verilir ve Popularity skorlarına düşülür.
          - Geçersiz ürün indekslerine FILTERED_SCORE (-1e9) atanır.
          - Bilinen kullanıcı ve geçerli ürünlerde <user_factors[uid], item_factors[cand]>
            iç çarpımı döndürülür.
        """
        cands_arr, valid_items = validate_indices(cands, self.n_items, name="item_idx")
        if cands_arr.size == 0:
            return np.empty(0, dtype=np.float64)

        is_known_uid = validate_user(uid, self.n_users, warn=False)

        if not is_known_uid:
            warnings.warn(
                f"Bilinmeyen kullanıcı (user_idx={uid}); "
                f"popülerlik sıralamasına geri dönülüyor.",
                UserWarning,
                stacklevel=2,
            )
            out = np.full(cands_arr.shape, FILTERED_SCORE, dtype=np.float64)
            if self.item_pop_scores is not None:
                out[valid_items] = self.item_pop_scores[cands_arr[valid_items]]
            else:
                out[valid_items] = np.linalg.norm(
                    self.item_factors[cands_arr[valid_items]], axis=1
                )
            return out

        u_vec = self.user_factors[int(uid)]
        if valid_items.all():
            v_mat = self.item_factors[cands_arr]
            return np.dot(v_mat, u_vec).astype(np.float64)

        out = np.full(cands_arr.shape, FILTERED_SCORE, dtype=np.float64)
        if valid_items.any():
            v_mat = self.item_factors[cands_arr[valid_items]]
            out[valid_items] = np.dot(v_mat, u_vec).astype(np.float64)
        return out

    def _compute_new_user_vector(
        self,
        valid_known_liked: np.ndarray,
        method: Optional[str] = None,
    ) -> np.ndarray:
        """Seçilen geçerli ve cf_known=True ürünlerden yeni kullanıcı faktör vektörünü hesaplar.

        Yöntem (A) - "ridge" (kapalı form ALS fold-in):
            u = (Y^T Y + alpha * Y_u^T Y_u + lambda * I)^{-1} (1 + alpha) * sum_{i in u} y_i
        Yöntem (B) - "mean" (pseudo-user item faktör ortalaması):
            u = (1 / |u|) * sum_{i in u} y_i
        """
        chosen_method = self.fold_in_method if method is None else str(method)
        if chosen_method not in VALID_FOLD_IN_METHODS:
            raise ValueError(
                f"method {VALID_FOLD_IN_METHODS} değerlerinden biri olmalı; gelen: {chosen_method!r}"
            )

        Y_u = self.item_factors[valid_known_liked].astype(np.float64)
        if chosen_method == "mean":
            return Y_u.mean(axis=0)

        # Yöntem (A): Kapalı form ridge / fold-in
        if self._YtY is None:
            self._prepare_yty()
        A = (
            self._YtY
            + self.alpha * (Y_u.T @ Y_u)
            + self.regularization * np.eye(self.factors, dtype=np.float64)
        )
        b = (1.0 + self.alpha) * Y_u.sum(axis=0)
        return np.linalg.solve(A, b)

    def score_from_items(
        self,
        liked_items: Sequence[int],
        cands: Sequence[int],
        method: Optional[str] = None,
    ) -> np.ndarray:
        """Eğitimde olmayan (yeni) kullanıcı için seçtiği oyunlardan aday ürün skorlarını üretir.

        Kurallar:
          - liked_items içindeki geçersiz (< 0 veya >= n_items) veya cf_known=False
            (eğitimde hiç pozitif etkileşimi olmayan) ürünler uyarı verilerek yok sayılır.
          - Geriye hiç geçerli cf_known=True ürün kalmazsa uyarı verilir ve Popularity
            skorlarına düşülür.
        """
        cands_arr, valid_cands = validate_indices(cands, self.n_items, name="item_idx")
        if cands_arr.size == 0:
            return np.empty(0, dtype=np.float64)

        liked_arr, valid_liked = validate_indices(liked_items, self.n_items, name="item_idx")
        valid_liked_items = liked_arr[valid_liked]

        # cf_known=True filtresi (eğitimde pozitif etkileşimi olmayan ürünler CF vektörü üretemez)
        if self.item_cf_known is not None and valid_liked_items.size > 0:
            known_mask = self.item_cf_known[valid_liked_items]
            if not known_mask.all():
                warnings.warn(
                    f"score_from_items: Seçilen ürünler içinde {int((~known_mask).sum())} adet "
                    f"cf_known=False ürün var ve yok sayıldı.",
                    UserWarning,
                    stacklevel=2,
                )
            valid_known_liked = np.unique(valid_liked_items[known_mask])
        else:
            valid_known_liked = np.unique(valid_liked_items)

        # Hiç geçerli cf_known=True ürün kalmadıysa Popularity fallback
        if valid_known_liked.size == 0:
            warnings.warn(
                "score_from_items: Geçerli cf_known=True seçilen ürün kalmadı; "
                "popülerlik sıralamasına geri dönülüyor.",
                UserWarning,
                stacklevel=2,
            )
            out = np.full(cands_arr.shape, FILTERED_SCORE, dtype=np.float64)
            if self.item_pop_scores is not None:
                out[valid_cands] = self.item_pop_scores[cands_arr[valid_cands]]
            return out

        u_vec = self._compute_new_user_vector(valid_known_liked, method=method)

        if valid_cands.all():
            v_mat = self.item_factors[cands_arr].astype(np.float64)
            return np.dot(v_mat, u_vec)

        out = np.full(cands_arr.shape, FILTERED_SCORE, dtype=np.float64)
        if valid_cands.any():
            v_mat = self.item_factors[cands_arr[valid_cands]].astype(np.float64)
            out[valid_cands] = np.dot(v_mat, u_vec)
        return out

    def _topk(self, scores: np.ndarray, k: int, exclude: Optional[Sequence[int]] = None) -> list:
        scores = scores.copy()
        if exclude is not None and len(exclude) > 0:
            scores[np.asarray(exclude, dtype=np.int64)] = FILTERED_SCORE
        k = min(int(k), len(scores))
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
        """Tüm katalog (N_ITEMS) arasından Top-K öneri üretir: [(item_idx, score), ...]."""
        if exclude_seen and seen_matrix is None:
            raise ValueError(
                "exclude_seen=True iken seen_matrix zorunludur "
                "(load_data()['seen_matrix'] verin)."
            )
        all_items = np.arange(self.n_items, dtype=np.int64)
        scores = self.score(uid, all_items)
        if exclude_seen and validate_user(uid, self.n_users, warn=False):
            scores = apply_exclude_seen(
                scores,
                all_items,
                int(uid),
                seen_matrix,
                mode=mode,
                target_items=target_items,
            )
        return self._topk(scores, k, extra_exclude)

    def recommend_from_items(
        self,
        liked_items: Sequence[int],
        k: int = 10,
        method: Optional[str] = None,
    ) -> list:
        """Yeni kullanıcı (tercih formu): seçtiği oyunlardan Top-K, seçtikleri hariç."""
        all_items = np.arange(self.n_items, dtype=np.int64)
        scores = self.score_from_items(liked_items, all_items, method=method)
        liked_arr, valid_liked = validate_indices(liked_items, self.n_items, name="item_idx", warn=False)
        return self._topk(scores, k, liked_arr[valid_liked])

    def save(self, path: str) -> None:
        """Model faktörlerini (user_factors.npy, item_factors.npy), R.npz ve params.json'u kaydeder."""
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "user_factors.npy", self.user_factors)
        np.save(folder / "item_factors.npy", self.item_factors)
        if self.train_matrix is not None:
            sp.save_npz(folder / "R.npz", self.train_matrix)
        params = {
            "factors": self.factors,
            "regularization": self.regularization,
            "alpha": self.alpha,
            "iterations": self.iterations,
            "random_state": self.random_state,
            "num_threads": self.num_threads,
            "fold_in_method": self.fold_in_method,
            "n_users": self.n_users,
            "n_items": self.n_items,
            "fit_seconds": self.fit_seconds,
        }
        (folder / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "ALSModel":
        """Kayıtlı ALS modelini klasörden yükler."""
        folder = Path(path)
        params = json.loads((folder / "params.json").read_text(encoding="utf-8"))
        model = cls(
            factors=params["factors"],
            regularization=params["regularization"],
            alpha=params["alpha"],
            iterations=params["iterations"],
            random_state=params.get("random_state", DEFAULT_RANDOM_STATE),
            num_threads=params.get("num_threads", DEFAULT_NUM_THREADS),
            fold_in_method=params.get("fold_in_method", DEFAULT_FOLD_IN_METHOD),
        )
        model.user_factors = np.load(folder / "user_factors.npy")
        model.item_factors = np.load(folder / "item_factors.npy")
        model.n_users, model.factors = model.user_factors.shape
        model.n_items = model.item_factors.shape[0]
        model.fit_seconds = params.get("fit_seconds")
        model._prepare_yty()
        r_path = folder / "R.npz"
        if r_path.exists():
            model.train_matrix = sp.load_npz(r_path).tocsr()
            model._init_metadata_from_matrix()
        return model

    def __repr__(self) -> str:
        return (
            f"ALSModel(factors={self.factors}, reg={self.regularization}, "
            f"alpha={self.alpha}, iterations={self.iterations}, "
            f"seed={self.random_state}, threads={self.num_threads}, "
            f"fold_in={self.fold_in_method!r})"
        )


MFModel = ALSModel


if __name__ == "__main__":
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.data_loader import load_data
    from src.evaluation import evaluate

    print("=" * 75)
    print("IMPLICIT ALS BASELINE EĞİTİMİ VE VALIDATION DEĞERLENDİRMESİ (src/models/mf.py)")
    print("=" * 75)

    data = load_data(str(PROJECT_ROOT / "data"))
    interaction_matrix = data["interaction_matrix"]
    val_candidates = data["val_candidates"]
    cold_items = data["cold_items"]
    user_activity = data["user_activity"]

    model = ALSModel(
        factors=64,
        regularization=0.1,
        alpha=10.0,
        iterations=15,
        random_state=42,
        num_threads=1,
    )
    print(f"\n[1] Model eğitiliyor: {model}")
    model.fit(interaction_matrix, show_progress=False)
    print(
        f"  Eğitim süresi       : {model.fit_seconds:.2f} sn\n"
        f"  user_factors şekli  : {model.user_factors.shape} ({model.user_factors.dtype})\n"
        f"  item_factors şekli  : {model.item_factors.shape} ({model.item_factors.dtype})"
    )

    print("\n[2] Validation setinde evaluate() çalıştırılıyor (N = 75.799)...")
    t_eval = time.perf_counter()
    val_res = evaluate(
        score_fn=model.score,
        candidates=val_candidates,
        cold_items=cold_items,
        user_activity=user_activity,
        verbose=True,
    )
    eval_seconds = time.perf_counter() - t_eval
    print(f"Değerlendirme süresi: {eval_seconds:.2f} sn")
