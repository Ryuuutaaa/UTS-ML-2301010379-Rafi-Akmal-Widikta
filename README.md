# UTS Machine Learning — Prediksi Niat Pembelian Pengunjung E-Commerce

Proyek Ujian Tengah Semester mata kuliah Machine Learning, dikerjakan mengikuti alur
**CRISP-DM** pada **Jalur A (Klasifikasi)**.

## Identitas

| | |
| --- | --- |
| **NIM** | 2301010379 |
| **Nama** | Rafi Akmal Widikta |
| **Kelas** | Informatika XX |
| **Mata Kuliah** | Machine Learning |
| **Dosen** | Dr. I Gede Totok Suryawan, S.Kom., M.T. |

> **`SEED = 379`** adalah tiga digit terakhir NIM (2301010379) dan dipakai di seluruh proses: pembagian
> data latih–uji, 5-fold cross-validation, dan model SVM. Notebook sudah dijalankan penuh (**Run All**)
> sehingga seluruh keluaran dan angka pada narasi konsisten dengan `SEED` tersebut.

## Ringkasan Hasil

| Metrik | Baseline (DummyClassifier) | SVM final | Selisih |
| --- | --- | --- | --- |
| Akurasi data uji | 0,8435 | **0,8685** | +0,0250 |
| Presisi kelas Membeli | 0,0000 | **0,5611** | +0,5611 |
| Recall kelas Membeli | 0,0000 | **0,7330** | +0,7330 |
| **F1 kelas Membeli** | **0,0000** | **0,6356** | **+0,6356** |
| F1 cross-validation 5-fold | 0,0000 | 0,6594 ± 0,0062 | +0,6594 |
| Overfitting (akurasi latih − uji) | +0,0002 | +0,0090 | seimbang |
| AUC (kurva ROC) | 0,5000 | **0,8982** | +0,3982 |

**Confusion matrix SVM** (data uji, 2.441 sesi): TN = 1.840, FP = 219, FN = 102, TP = 280.

**Model terpilih:** SVM **kernel linear**, `C=1`, `class_weight='balanced'`, `random_state=SEED`.

> Hasil tuning memilih **kernel linear** (F1 CV 0,6594) yang praktis setara dengan RBF (0,6589) —
> selisih 0,0005, jauh di bawah simpangan baku antar-fold. Pilihan diterima apa adanya dari CV,
> tanpa memaksakan kernel yang terdengar lebih canggih.

## Dataset

| Keterangan | Isi |
| --- | --- |
| Judul | Online Shoppers Purchasing Intention Dataset |
| Sumber (Kaggle) | https://www.kaggle.com/datasets/satyajeetkumarraj/online-shoppers-purchasing-intention-dataset |
| Sumber asli (otoritatif) | UCI Machine Learning Repository, dataset ID 468 — https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset |
| Tanggal akses | 24 November 2026 |
| Lisensi | **CC BY 4.0** (Creative Commons Attribution 4.0 International) |
| DOI dataset | 10.24432/C5F88Q |
| Ukuran | 12.330 baris × 18 kolom (17 fitur + 1 target) |
| Berkas lokal | `data/online_shoppers_intention.csv` |

**Catatan verifikasi lisensi.** Lisensi **CC BY 4.0** dipastikan langsung pada halaman resmi UCI (sumber
asli dataset), yang menyatakan: *"This dataset is licensed under a Creative Commons Attribution 4.0
International (CC BY 4.0) license."* Tautan Kaggle yang dipakai di atas juga mencantumkan
*"Attribution 4.0 International (CC BY 4.0)"* secara eksplisit dan berkasnya **identik** dengan versi UCI
(diverifikasi dengan pencocokan hash). Sebagian unggahan ulang dataset ini di Kaggle hanya memberi label
lisensi "Other (specified in description)", sehingga sengaja tidak dipakai sebagai rujukan utama.

**Atribusi (wajib menurut lisensi CC BY 4.0):**

> Sakar, C.O., Polat, S.O., Katircioglu, M., & Kastro, Y. (2019). *Real-time prediction of online
> shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks.*
> Neural Computing and Applications, 31(10), 6893–6908.
> Dataset tersedia di UCI ML Repository (ID 468) dan Kaggle.

### Kepatuhan UU No. 27 Tahun 2022 tentang Pelindungan Data Pribadi

Dataset ini **tidak memuat data pribadi**: tidak ada nama, NIK, alamat, nomor telepon, maupun surel.
Setiap baris adalah satu sesi anonim dari pengguna berbeda dalam rentang satu tahun. Kolom
`OperatingSystems`, `Browser`, `Region`, dan `TrafficType` sudah dikodekan sebagai angka **tanpa
pemetaan** ke merek, perangkat, atau wilayah sebenarnya, sehingga tidak dapat dilacak ke individu
mana pun. Tidak diperlukan anonimisasi tambahan.

## Struktur Repositori

```
UTS-ML-2301010379-Rafi-Akmal-Widikta/
├── README.md
├── requirements.txt
├── data/
│   └── online_shoppers_intention.csv
├── notebook/
│   └── UTS_ML_2301010379_Rafi_Akmal_Widikta.ipynb   # sudah dijalankan penuh (Run All)
├── reports/
│   ├── fig01_distribusi_target.png
│   ├── fig02_distribusi_fitur.png
│   ├── fig03_fitur_vs_target.png
│   ├── fig04_korelasi.png
│   ├── fig05_outlier.png
│   ├── fig06_tuning_knn.png
│   ├── fig07_tuning_svm.png
│   ├── fig08_confusion_roc.png
│   ├── fig09_feature_importance.png
│   └── slide_presentasi.md
└── src/
    └── preprocessing.py           # transformer kustom (Winsorizer, IQRClipper)
```

## Cara Menjalankan

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab notebook/UTS_ML_2301010379_Rafi_Akmal_Widikta.ipynb
```

Notebook mencari dataset pada `../data/online_shoppers_intention.csv` relatif terhadap folder
`notebook/`. Jika dijalankan di Google Colab, unggah dataset ke `/content/` lalu sesuaikan satu baris
`pd.read_csv` pada bagian awal, dan sesuaikan path `savefig` pada `../reports/`.

## Alur CRISP-DM

| Tahap | Isi | Keputusan utama |
| --- | --- | --- |
| 1. Pemahaman masalah | Tujuan, pengguna, sumber data, deskripsi 18 kolom | Jalur A; kolom kode diperlakukan kategorikal, bukan numerik |
| 2. EDA | 5 grafik + interpretasi tiap grafik | F1 sebagai metrik utama; `PageValues` & `Month` dipertahankan |
| 3. Pra-pemrosesan | Duplikat, `June`→`Jun`, winsorizing 1–99%, scaling, one-hot | **Menolak IQR** karena merusak model (F1 0,596 → 0,009) |
| 4. Pemodelan | Baseline, tuning KNN (9 nilai k), tuning SVM (48 kombinasi), Naive Bayes | SVM linear + `class_weight='balanced'` |
| 5. Evaluasi | Confusion matrix, presisi/recall/F1, perhitungan manual, ROC | Tidak ada overfitting (selisih latih−uji +0,0090) |
| 6. Kesimpulan | Kelayakan, rekomendasi, keterbatasan, saran perbaikan | Model layak dipakai, dengan 6 keterbatasan tercatat |

## Temuan Penting (bahan presentasi)

1. **IQR merusak model.** Pemotongan outlier dengan aturan IQR menurunkan F1 dari **0,596 menjadi
   0,009**, karena `PageValues` memiliki Q1 = Q3 = 0 sehingga semua nilai bukan-nolnya terpangkas
   menjadi nol. Solusi yang dipakai: winsorizing persentil 1–99 (**F1 0,6085**).
2. **Akurasi menyesatkan.** Baseline yang selalu menebak "Tidak Membeli" mendapat akurasi **84,35%**
   tetapi F1 **0,0000**. Karena itu seluruh pemilihan model memakai F1.
3. **`class_weight='balanced'` adalah kunci.** Recall naik **+22,8 poin** dengan biaya presisi
   −12,8 poin dan akurasi −1,8 poin; F1 bersihnya naik **+5,3 poin**.
4. **Linear dan RBF setara.** F1 CV 0,6594 vs 0,6589 — selisih 0,0005, jauh di bawah simpangan baku
   antar-fold (0,006–0,010). Pemenang menurut CV adalah **linear**, dan itu diterima apa adanya.
5. **Kategori langka membuat bobot tidak stabil.** `TrafficType=16` berbobot tertinggi (+2,05) tetapi
   hanya muncul pada **3 baris**. Karena itu rekomendasi bisnis tidak didasarkan pada kolom kode.
6. **Naive Bayes salah model, bukan bug.** Akurasi uji hanya 0,2589 karena 65 dari 75 fitur hasil
   one-hot bersifat biner, sehingga asumsi distribusi Gaussian dilanggar.

## Pernyataan Penggunaan AI

Sesuai ketentuan soal poin 8, berikut pernyataan jujur mengenai penggunaan asisten AI.

**Alat yang dipakai:** Command Code (agen pengkodean berbasis model bahasa besar).

**Bagian yang dibantu AI:**

1. **Penelusuran kandidat dataset** — membandingkan kelayakan beberapa dataset publik (jumlah baris,
   keseimbangan kelas, lisensi, dan sinyal prediktif) sebelum memilih dataset final.
2. **Penulisan kerangka kode** — struktur `Pipeline`, `ColumnTransformer`, `GridSearchCV`, dan kode
   visualisasi.
3. **Penyusunan draf narasi** — teks interpretasi grafik dan bagian kesimpulan.
4. **Penyuntingan bahasa** — perbaikan kejelasan kalimat.

**Bagaimana hasilnya saya periksa:**

1. **Seluruh notebook dijalankan ulang dari awal (Restart & Run All)** dan setiap keluaran angka
   dicocokkan dengan narasi pada sel "Keputusan saya".
2. **Perhitungan manual metrik** (akurasi, presisi, recall, F1) dikerjakan sendiri dari angka confusion
   matrix, lalu dibandingkan dengan keluaran `classification_report`. Hasilnya cocok hingga selisih
   pembulatan.
3. **Keputusan teknis kunci diuji sendiri, bukan diterima begitu saja.** Contoh paling penting: saran
   pemotongan outlier dengan IQR diuji dan hasilnya **merusak model** (F1 0,596 → 0,009) karena
   `PageValues` memiliki IQR = 0. Penggantinya winsorizing persentil 1–99, dan buktinya dilaporkan
   pada Tahap 3.
4. **Setiap parameter dipahami maknanya**: `C` mengatur toleransi margin, `gamma` mengatur jangkauan
   pengaruh tiap titik, `class_weight='balanced'` memberi bobot berbanding terbalik dengan frekuensi
   kelas, dan `stratify=y` menjaga proporsi kelas saat pembagian data.
5. **Dipastikan tidak ada kebocoran data**: scaler dan encoder hanya di-*fit* di dalam `Pipeline` pada
   data latih, dan data uji baru dibuka sekali pada Tahap 5.
6. **Kesalahan pada versi awal diperbaiki, bukan disembunyikan.** Dua bug ditemukan saat pengujian:
   transformer winsorizing yang membuang 8 kolom numerik, dan `groupby` yang menghilangkan kelompok
   `class_weight=None`. Keduanya diperbaiki dan seluruh angka dihitung ulang.

Saya siap menjelaskan dan mengubah bagian mana pun dari kode ini saat presentasi.

## Keterbatasan

1. Setiap baris adalah sesi anonim — pengunjung yang kembali tidak dapat dilacak.
2. Kolom kode tidak memiliki kamus, sehingga rekomendasi berbasis peramban/wilayah tidak dapat
   ditindaklanjuti secara operasional; koefisiennya juga tidak stabil karena banyak kategori langka.
3. Data bersifat historis satu tahun dan tidak memuat harga, stok, atau kampanye iklan.
4. `ExitRates`–`BounceRates` berkorelasi 0,913 dan `ProductRelated`–`ProductRelated_Duration` 0,861,
   sehingga interpretasi bobot tidak sepenuhnya bersih.
5. Recall 0,7330 berarti **26,7% pembeli tetap terlewat**.
6. Naive Bayes tidak layak dipakai karena asumsi distribusi Gaussian dilanggar oleh 65 fitur biner.

## Saran Perbaikan

1. Bandingkan `class_weight='balanced'` dengan **SMOTE**, atau turunkan ambang keputusan ke sekitar 0,4
   dan laporkan kurva presisi-recall.
2. Rekayasa fitur: `ProductRelated_Duration / ProductRelated` (detik per halaman produk).
3. Gabungkan kategori langka pada `TrafficType` dan `Browser` (mis. semua kategori < 50 baris menjadi
   "Lainnya") agar koefisien stabil.
4. Coba model berbasis pohon. Random Forest sudah diuji: F1 CV **0,6861** (lebih tinggi dari SVM
   0,6594), tetapi akurasi latih **0,9994** vs uji 0,8873 — **overfit ekstrem**, belum layak dipilih
   tanpa penyetelan kedalaman.
5. Validasi lebih ketat dengan `RepeatedStratifiedKFold` (5-fold × 3 pengulangan).
6. Kalibrasi probabilitas dengan `CalibratedClassifierCV` agar skor `decision_function` dapat dibaca
   sebagai peluang sesungguhnya — SVM tidak menyediakan `predict_proba` secara bawaan.
7. Untuk Naive Bayes, gunakan `BernoulliNB` atau `CategoricalNB` yang memang dirancang untuk fitur
   biner/kategorikal.
