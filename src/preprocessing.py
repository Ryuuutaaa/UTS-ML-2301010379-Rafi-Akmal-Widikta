"""Komponen pra-pemrosesan untuk UTS Machine Learning (Jalur A - Klasifikasi).

Notebook memuat definisi kelas ini secara langsung agar dapat dijalankan di Google Colab
maupun Jupyter. Berkas ini disediakan sebagai rujukan modul: isinya identik dengan yang
didefinisikan di notebook, sehingga dapat dipakai ulang di luar notebook.

Semua proses belajar parameter (kuantil) dilakukan di dalam ``fit`` sehingga aman
dipakai di dalam Pipeline tanpa kebocoran data.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array


class Winsorizer(BaseEstimator, TransformerMixin):
    """Memotong nilai ekstrem pada persentil yang ditentukan.

    Berbeda dari aturan IQR, batas dihitung dari persentil sehingga kolom yang
    mayoritas bernilai nol (IQR = 0) tetap mempertahankan variasi aslinya.
    Transformer ini memproses seluruh kolom yang diberikan sekaligus.

    Parameters
    ----------
    lower, upper : float
        Persentil batas bawah dan atas (0 <= lower < upper <= 1).
    """

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = check_array(X, dtype=float)
        self.lower_ = np.quantile(X, self.lower, axis=0)
        self.upper_ = np.quantile(X, self.upper, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(check_array(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class IQRClipper(BaseEstimator, TransformerMixin):
    """Pemotongan outlier memakai batas kuartil.

    Disediakan hanya sebagai pembanding: pada dataset ini pemotongan IQR menghapus
    seluruh variasi ``PageValues`` (IQR = 0) dan menjatuhkan F1 dari 0,588 ke 0,015.
    """

    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = check_array(X, dtype=float)
        q1 = np.quantile(X, 0.25, axis=0)
        q3 = np.quantile(X, 0.75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(check_array(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)
