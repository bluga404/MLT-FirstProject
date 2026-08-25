# Coffee Sales Forecasting & Customer Purchase Pattern Analysis

> Proyek Machine Learning — Dicoding Academy

**Nama:** Walker Valentinus Simanjuntak
**Email:** walkervalentinussimanjuntak@gmail.com
**ID Dicoding:** walkervs

---

## Business Understanding

Proyek ini menganalisis pola penjualan kopi dari sebuah *vending machine* dan membangun model peramalan (*forecasting*) penjualan harian menggunakan dataset **Coffee Sales** dari Kaggle (Isaienkov, 2025).

### Problem Statements
1. Bagaimana tren dan pola penjualan kopi berdasarkan waktu?
2. Siapa pelanggan paling aktif dan produk apa yang paling sering dibeli?
3. Model mana yang memberikan error terkecil dalam memprediksi penjualan harian berikutnya?

### Solution Statements
- **Solusi 1:** Membandingkan 5 model peramalan (KNN, Random Forest, AdaBoost, Simple Linear Regression, Fuzzy Time Series)
- **Solusi 2:** Melakukan hyperparameter tuning pada model baseline terbaik (AdaBoost)

---

## Data Understanding

| Kolom | Tipe | Deskripsi |
|---|---|---|
| `date` | Date | Tanggal transaksi |
| `datetime` | Datetime | Tanggal dan waktu transaksi |
| `cash_type` | Categorical | Metode pembayaran (card / cash) |
| `card` | Identifier | Identitas pelanggan (anonymized) |
| `money` | Numerical | Nilai transaksi |
| `coffee_name` | Categorical | Nama produk kopi |

- **Jumlah baris:** 3.636
- **Rentang tanggal:** 1 Maret 2024 — 23 Maret 2025
- **Missing values:** 89 pada kolom `card` (wajar, karena transaksi tunai)

---

## Exploratory Data Analysis (EDA)

Temuan utama dari EDA:
- Penjualan harian berfluktuasi dengan tren yang dapat diprediksi
- **Peak hour:** Jam 10 pagi
- Produk terlaris: **Latte** dan **Americano**
- Mayoritas pembayaran menggunakan kartu (~77%)
- Pelanggan paling aktif: 129 transaksi dengan total pengeluaran 3.785,92

---

## Model Development

### 5 Model yang Dibandingkan

| Model | RMSE | MAE | Keterangan |
|---|---|---|---|
| KNN (default) | 232,79 | 182,29 | Baseline ML |
| Random Forest (default) | 273,43 | 214,51 | Baseline ML |
| AdaBoost (default) | 224,64 | 170,22 | Baseline ML (terbaik) |
| **Simple Linear Regression** | **200,24** | **155,42** | **Model Terbaik** |
| Fuzzy Time Series (Chen) | 215,74 | 171,92 | Pendekatan statistik |

### Hyperparameter Tuning

Model baseline terbaik (**AdaBoost**, RMSE 224,64) di-tuning menggunakan `GridSearchCV` dengan `TimeSeriesSplit`.

| Parameter | Default | Tuned |
|---|---|---|
| `n_estimators` | 50 | 50 |
| `learning_rate` | 1.0 | 0.1 |
| `loss` | 'linear' | 'linear' |

**Hasil:** Tuning tidak meningkatkan performa — RMSE justru naik dari 224,64 menjadi 231,95. Parameter default AdaBoost sudah optimal untuk dataset ini.

---

## Kesimpulan

### Model Terbaik: Simple Linear Regression (RMSE 200,24)

Bertentangan dengan intuisi populer, model **Simple Linear Regression** (SLR) mengungguli seluruh model ML yang lebih kompleks. Temuan kunci:

1. **SLR (RMSE 200,24)** adalah model terbaik, mengalahkan AdaBoost (224,64) dan model lainnya
2. **Fitur `lag_1`** (penjualan hari sebelumnya) adalah prediktor paling penting — hubungan antara penjualan kemarin dan hari ini bersifat **dominan linear**
3. **Hyperparameter tuning tidak selalu meningkatkan performa** — AdaBoost default sudah optimal

### Business Insights
- Pola penjualan harian dapat diprediksi dengan model sederhana
- Pelanggan loyal memiliki pola pembelian yang konsisten dan dapat diidentifikasi
- Untuk dataset ini, pendekatan sederhana (SLR) lebih baik daripada model kompleks

### Keterbatasan
- Dataset relatif singkat (< 1 tahun)
- Tidak memperhitungkan faktor eksternal (cuaca, event, harga)

---

## Cara Menjalankan

```bash
# Install dependencies
pip install -r requirements.txt

# Jalankan notebook
jupyter notebook notebook_v3.ipynb
```

---

## Struktur Proyek

```
MLT-FirstProject/
├── notebook_v3.ipynb          # Notebook utama (EDA + Modeling)
├── submission.py              # Python script version
├── ReportTemplate.md          # Laporan lengkap
├── CoffeeSales/
│   └── index_1.csv            # Dataset
└── submission/                # File submission
    ├── notebook_v3.ipynb
    ├── submission.py
    ├── ReportTemplate.md
    ├── REVIEWER_NOTE.md       # Catatan perubahan untuk reviewer
    └── images/                # Visualisasi
```

---

## Referensi

1. Isaienkov, Y. (2025). Coffee Sales [Dataset]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/11159944
2. Sales Prediction using Linear Regression. ResearchGate
3. Penerapan Fuzzy Time Series dan Simple Linear Regression. IJODAS, 6(3). https://doi.org/10.56705/ijodas.v6i3.368
