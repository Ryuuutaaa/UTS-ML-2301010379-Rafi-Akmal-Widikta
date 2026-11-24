# Slide Presentasi UTS Machine Learning (±5 menit)

Kerangka ini disusun mengikuti batas waktu demo 5 menit ditambah 5 menit tanya jawab.
Setiap poin menyebut angka konkret agar mudah dipertahankan saat ditanya.

---

## Slide 1 — Judul & Masalah (30 detik)

**Prediksi Niat Pembelian Pengunjung E-Commerce**

- Dataset: Online Shoppers Purchasing Intention (UCI ID 468, CC BY 4.0)
- 12.330 sesi kunjungan, 17 fitur, 1 target (`Revenue`)
- **Pertanyaan:** apakah sesi ini akan berakhir dengan pembelian?
- **Pengguna:** tim pemasaran digital — menentukan siapa yang layak diberi voucher

> Kalimat kunci: "Kami ingin memprediksi **sebelum** sesi berakhir, sehingga intervensi masih mungkin
> dilakukan."

---

## Slide 2 — Data & Tantangan (40 detik)

Angka yang wajib disebut:

| Fakta | Angka |
| --- | --- |
| Ketidakseimbangan kelas | 84,4% tidak membeli : 15,6% membeli (**5,46 : 1**) |
| Missing value | **0** |
| Duplikat | **125 baris** (dihapus) |
| Kategori tidak konsisten | `'June'` vs `'Jun'` (**288 baris**) → diseragamkan |
| Korelasi tertinggi antar fitur | `ExitRates`–`BounceRates` = **0,913** |

> Kalimat kunci: "Karena kelasnya 5,46 : 1, **akurasi tidak bisa dipakai** — model yang selalu menebak
> 'tidak membeli' sudah dapat 84,5%."

---

## Slide 3 — Pra-pemrosesan: keputusan yang paling sering ditanya (60 detik)

**Alur:** bagi data (`stratify=y`, SEED) → Pipeline → latih

| Kelompok fitur | Perlakuan |
| --- | --- |
| 6 kontinu (durasi, rasio, PageValues) | imputer median → **winsorizing 1–99%** → StandardScaler |
| 4 diskrit (jumlah halaman) | imputer median → StandardScaler |
| 7 kategorikal (`Month`, `VisitorType`, `Weekend`, 4 kolom kode) | imputer modus → OneHotEncoder |

**Temuan utama — mengapa IQR ditolak:**

| Strategi outlier | CV-F1 |
| --- | --- |
| Tanpa penanganan | 0,5959 |
| **IQR clip 1,5×** | **0,0090** ← model rusak |
| **Winsorizer 1–99%** | **0,6085** ← dipilih |

Alasannya: `PageValues` punya Q1 = Q3 = 0, sehingga batas IQR = 0 dan **semua nilai bukan-nol
terpangkas**. Padahal `PageValues` adalah fitur terkuat (rata-rata 27,27 pada pembeli vs 2,00 pada
bukan pembeli).

> Kalimat kunci: "Saya **tidak mengikuti** anjuran IQR begitu saja — saya ujinya, hasilnya F1 jatuh dari
> 0,596 ke 0,009, jadi saya ganti dengan winsorizing."

---

## Slide 4 — Pemodelan (50 detik)

Semua tuning memakai **5-fold StratifiedKFold pada data latih saja**.

| Model | F1 CV |
| --- | --- |
| DummyClassifier (baseline) | **0,0000** |
| Naive Bayes | 0,2934 |
| KNN (k=5, terbaik dari 9 nilai) | 0,5210 |
| **SVM linear, C=1, balanced** | **0,6594** |

Dua hal yang jujur harus disebut:

1. **Linear vs RBF setara** (0,6594 vs 0,6589) — selisih 0,0005, di bawah simpangan baku 0,006–0,010.
   Pemenang menurut CV adalah **linear**, dan itu diterima apa adanya.
2. **`class_weight='balanced'` menaikkan F1 sebesar 0,035** — seluruh konfigurasi teratas memakainya.

---

## Slide 5 — Evaluasi (60 detik)

**Confusion matrix (data uji, 2.441 sesi):**

|  | Prediksi Tidak | Prediksi Membeli |
| --- | --- | --- |
| **Aktual Tidak** | TN = 1.840 | FP = 219 |
| **Aktual Membeli** | FN = 102 | TP = 280 |

**Perhitungan manual yang harus bisa diucapkan:**

- Presisi = 280 / (280+219) = 280/499 = **0,5611**
- Recall = 280 / (280+102) = 280/382 = **0,7330**
- F1 = 2(280) / (2·280 + 219 + 102) = 560/881 = **0,6356**
- Akurasi = (280+1840)/2441 = **0,8685**

**Uji overfitting:**

| Model | Akurasi latih | Akurasi uji | Selisih |
| --- | --- | --- | --- |
| KNN (k=5) | 0,9080 | 0,8656 | **+0,0424** ← overfit |
| **SVM** | 0,8775 | 0,8685 | **+0,0090** ← sehat |

> Kalimat kunci: "Recall 0,7330 artinya **hampir 3 dari 4 pembeli tertangkap**; presisi 0,5611 artinya
> **hampir separuh penerima voucher tidak jadi membeli**."

---

## Slide 6 — Kesimpulan & Rekomendasi (40 detik)

**Layak dipakai**, dua alasan berangka:

1. F1 naik dari **0,0000 → 0,6356** (+0,6356)
2. Selisih latih–uji hanya **+0,0090** (KNN +0,0424)

**Rekomendasi:**

1. Pakai **peringkat skor**, bukan label biner (AUC = 0,8982)
2. Geser anggaran promo ke **November–Desember** (konversi 25,4% vs 1,6% Februari)
3. **Jangan** bertindak atas kolom kode — `TrafficType=16` berbobot tertinggi tapi hanya **3 baris**

---

## Antisipasi Tanya Jawab

**"Kenapa bukan akurasi?"**
Karena baseline selalu menebak "tidak membeli" sudah dapat 84,35% dengan F1 = 0. Akurasi tinggi bisa
menyesatkan pada data tidak seimbang.

**"Kenapa `random_state` dipakai di mana-mana?"**
Agar hasil dapat direproduksi. Nilainya tiga digit terakhir NIM, disimpan sekali sebagai `SEED`.

**"Di mana letak pencegahan kebocoran data?"**
Data dibagi **sebelum** scaler dibuat, dan semua pra-pemrosesan ada di dalam `Pipeline` sehingga
`fit` hanya menyentuh data latih. Data uji dibuka sekali di Tahap 5.

**"Coba ganti k jadi 15, apa yang terjadi?"**
F1 CV turun dari 0,5210 (k=5) ke 0,5151 (k=15), karena semakin banyak tetangga mayoritas ikut memilih
sehingga model makin jarang memprediksi "Membeli".

**"Coba naikkan C jadi 100?"**
Pada kernel linear F1 CV nyaris tidak berubah (0,6594 pada C=1 vs 0,6577 pada C=100). C terlalu besar
membuat margin terlalu sempit sehingga model lebih mudah overfit, tanpa manfaat yang sepadan.

**"Kenapa `class_weight='balanced'` bikin akurasi turun tapi F1 naik?"**
Karena model jadi lebih berani memprediksi "Membeli": recall naik 22,8 poin sementara presisi turun
12,8 poin. Kenaikan recall lebih besar, sehingga F1 bersihnya naik 5,3 poin.

**"Kalau saya hapus `PageValues`, apa efeknya?"**
F1 akan turun tajam karena `PageValues` adalah fitur terkuat (bobot +1,82; rata-rata 27,27 vs 2,00
antar kelas). Bisa diuji langsung di notebook dengan `drop(columns=["PageValues"])`.

**"Kenapa Naive Bayes jelek sekali?"**
Bukan bug: 65 dari 75 fitur hasil one-hot bersifat biner, sehingga asumsi distribusi Gaussian
dilanggar. Solusinya `BernoulliNB` atau `CategoricalNB`.
