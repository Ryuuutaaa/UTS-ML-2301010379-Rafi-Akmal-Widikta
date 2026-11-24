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
| Ketidakseimbangan kelas | 84,5% tidak membeli : 15,5% membeli (**5,46 : 1**) |
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
| Tanpa penanganan | 0,5876 |
| **IQR clip 1,5×** | **0,0152** ← model rusak |
| **Winsorizer 1–99%** | **0,5960** ← dipilih |

Alasannya: `PageValues` punya Q1 = Q3 = 0, sehingga batas IQR = 0 dan **semua nilai bukan-nol
terpangkas**. Padahal `PageValues` adalah fitur terkuat (rata-rata 27,26 pada pembeli vs 1,98 pada
bukan pembeli).

> Kalimat kunci: "Saya **tidak mengikuti** anjuran IQR begitu saja — saya ujinya, hasilnya F1 jatuh dari
> 0,588 ke 0,015, jadi saya ganti dengan winsorizing."

---

## Slide 4 — Pemodelan (50 detik)

Semua tuning memakai **5-fold StratifiedKFold pada data latih saja**.

| Model | F1 CV |
| --- | --- |
| DummyClassifier (baseline) | **0,0000** |
| Naive Bayes | 0,2939 |
| KNN (k=3, terbaik dari 9 nilai) | 0,5280 |
| **SVM RBF, C=10, gamma=0.01, balanced** | **0,6574** |

Dua hal yang jujur harus disebut:

1. **RBF vs linear hampir setara** (0,6574 vs 0,6569) — selisih 0,0005, di bawah simpangan baku 0,008.
2. **`class_weight='balanced'` menaikkan F1 sebesar 0,040** — seluruh 10 konfigurasi teratas memakainya.

---

## Slide 5 — Evaluasi (60 detik)

**Confusion matrix (data uji, 2.441 sesi):**

|  | Prediksi Tidak | Prediksi Membeli |
| --- | --- | --- |
| **Aktual Tidak** | TN = 1.837 | FP = 222 |
| **Aktual Membeli** | FN = 91 | TP = 291 |

**Perhitungan manual yang harus bisa diucapkan:**

- Presisi = 291 / (291+222) = 291/513 = **0,5673**
- Recall = 291 / (291+91) = 291/382 = **0,7618**
- F1 = 2(291) / (2·291 + 222 + 91) = 582/895 = **0,6503**
- Akurasi = (291+1837)/2441 = **0,8718**

**Uji overfitting:**

| Model | Akurasi latih | Akurasi uji | Selisih |
| --- | --- | --- | --- |
| KNN (k=3) | 0,9217 | 0,8632 | **+0,0585** ← overfit |
| **SVM** | 0,8838 | 0,8718 | **+0,0120** ← sehat |

> Kalimat kunci: "Recall 0,7618 artinya **3 dari 4 pembeli tertangkap**; presisi 0,5673 artinya
> **2 dari 5 penerima voucher tidak jadi membeli**."

---

## Slide 6 — Kesimpulan & Rekomendasi (40 detik)

**Layak dipakai**, dua alasan berangka:

1. F1 naik dari **0,0000 → 0,6503** (+0,6503)
2. Selisih latih–uji hanya **+0,0120** (KNN +0,0585)

**Rekomendasi:**

1. Pakai **peringkat skor**, bukan label biner (AUC = 0,9108)
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
F1 CV turun dari 0,5280 (k=3) ke 0,5053 (k=15), karena semakin banyak tetangga mayoritas ikut memilih
sehingga model makin jarang memprediksi "Membeli".

**"Coba naikkan C jadi 100?"**
Pada RBF F1 CV turun dari 0,6574 (C=10) menjadi 0,6538 (C=100) dengan `balanced`. C terlalu besar
membuat margin terlalu sempit sehingga model lebih mudah overfit.

**"Kenapa `class_weight='balanced'` bikin akurasi turun tapi F1 naik?"**
Karena model jadi lebih berani memprediksi "Membeli": recall naik 23,04 poin sementara presisi turun
16,03 poin. Kenaikan recall lebih besar, sehingga F1 bersihnya naik 3,61 poin.

**"Kalau saya hapus `PageValues`, apa efeknya?"**
F1 akan turun tajam karena `PageValues` adalah fitur terkuat (bobot +1,81; rata-rata 27,26 vs 1,98
antar kelas). Bisa diuji langsung di notebook dengan `drop(columns=["PageValues"])`.

**"Kenapa Naive Bayes jelek sekali?"**
Bukan bug: 65 dari 75 fitur hasil one-hot bersifat biner, sehingga asumsi distribusi Gaussian
dilanggar. Solusinya `BernoulliNB` atau `CategoricalNB`.
