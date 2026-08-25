# Catatan Perubahan untuk Reviewer

Berikut adalah ringkasan perubahan yang telah dilakukan berdasarkan feedback reviewer:

---

## 1. Kriteria 8 — Model Development: Hyperparameter Tuning

**Feedback:**
> Laporan menyebutkan bahwa tuning dilakukan pada "model baseline terbaik (Random Forest)". Keterangan ini tidak sesuai dengan hasil pada notebook. Random Forest justru merupakan baseline dengan error tertinggi (RMSE 273,43), sedangkan AdaBoost memiliki RMSE terendah di antara baseline (224,64).

**Perubahan yang dilakukan:**

| Aspek | Sebelum | Sesudah |
|---|---|---|
| Model yang di-tuning | Random Forest (RMSE 273,43 — tertinggi) | **AdaBoost** (RMSE 224,64 — terendah di antara baseline ML) |
| Parameter yang diuji | `n_estimators`, `max_depth`, `min_samples_split` | `n_estimators`, `learning_rate`, `loss` |
| Hasil tuning | RMSE 273,43 → 252,22 | RMSE 224,64 → 231,95 |

**File yang diubah:**
- `notebook_v3.ipynb` — Section 9 (markdown + code), Section 11, Section 12
- `submission.py` — Bagian tuning dari Random Forest ke AdaBoost
- `ReportTemplate.md` — Solution Statements, Hyperparameter Tuning section, Evaluation table, Solusi 2 impact

**Catatan penting:**
Hasil tuning menunjukkan bahwa **parameter default AdaBoost sudah optimal** untuk dataset ini. Tuning justru meningkatkan RMSE dari 224,64 menjadi 231,95 (~3,3% lebih buruk). Hal ini dijelasarkan secara jujur dalam laporan.

---

## 2. Kriteria 10 — Struktur Laporan: Gambar Tidak Dimuat

**Feedback:**
> Seluruh gambar pada laporan tidak dapat dimuat ketika dibuka menggunakan Markdown Viewer. Folder images tidak disertakan dalam submission.

**Perubahan yang dilakukan:**

| Aspek | Sebelum | Sesudah |
|---|---|---|
| `submission.zip` | Hanya berisi 3 file (notebook, .py, .md) | **Berisi 14 file** (3 file + `images/` folder dengan 10 gambar) |

**Struktur submission.zip yang baru:**
```
submission.zip
├── notebook_v3.ipynb
├── submission.py
├── ReportTemplate.md
└── images/
    ├── image_0.png  (Tren Penjualan Harian)
    ├── image_1.png  (Pola Temporal)
    ├── image_2.png  (Distribusi Produk)
    ├── image_3.png  (Metode Pembayaran)
    ├── image_4.png  (Top Pelanggan)
    ├── image_5.png  (Waktu Pembelian)
    ├── image_6.png  (Perbandingan RMSE & MAE)
    ├── image_7.png  (Visualisasi Prediksi)
    ├── image_8.png  (Analisis Residual)
    └── image_9.png  (Feature Importance)
```

---

## Ringkasan File yang Berubah

| File | Perubahan Utama |
|---|---|
| `notebook_v3.ipynb` | Tuning RF → AdaBoost, output di-re-run |
| `submission/notebook_v3.ipynb` | Sama dengan notebook utama |
| `submission/submission.py` | Tuning RF → AdaBoost |
| `ReportTemplate.md` | Narasi + tabel disesuaikan dengan hasil aktual |
| `submission/ReportTemplate.md` | Sama dengan report utama |
| `submission.zip` | Ditambahkan folder `images/` |

---

**Commit:** `13406cb` — *Fix: Change hyperparameter tuning from Random Forest to AdaBoost (best baseline)*
