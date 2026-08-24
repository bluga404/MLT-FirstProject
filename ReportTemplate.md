# Laporan Proyek Machine Learning - [Nama Anda]

## Domain Proyek

Penjualan kopi merupakan salah satu jenis transaksi yang memiliki pola pembelian yang dapat berubah berdasarkan waktu, jenis produk, serta karakteristik pelanggan. Pada bisnis *vending machine*, data transaksi yang tercatat secara berkala dapat dimanfaatkan untuk memahami pola penjualan dan perilaku pembelian pelanggan. Kemampuan untuk memprediksi penjualan dapat membantu bisnis dalam mengambil keputusan berdasarkan data historis. Tanpa pemahaman terhadap pola penjualan, bisnis dapat mengalami kesulitan dalam menentukan target penjualan, memperkirakan kebutuhan persediaan, dan mengalokasikan sumber daya secara optimal.

Pada *vending machine*, pola transaksi juga dapat memberikan informasi mengenai produk yang paling banyak diminati serta waktu ketika permintaan cenderung meningkat atau menurun. Oleh karena itu, analisis terhadap data penjualan historis diperlukan untuk mengidentifikasi pola tersebut dan menghasilkan informasi yang dapat mendukung proses pengambilan keputusan.

Penelitian mengenai prediksi penjualan menunjukkan bahwa data historis dapat dimanfaatkan untuk memperkirakan nilai penjualan pada periode berikutnya. Salah satu pendekatan yang dapat digunakan adalah regresi linear dan berbagai model *Machine Learning* yang mampu menangkap pola kompleks. Pendekatan lain seperti *Fuzzy Time Series* (FTS) juga terbukti handal dalam peramalan deret waktu.

### Referensi:
- Isaienkov, Y. (2025). Coffee Sales [Dataset]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/11159944
- *Sales Prediction using Linear Regression*. ResearchGate. https://www.researchgate.net/publication/376738143_Sales_Prediction_using_Linear_Regression
- *Penerapan Fuzzy Time Series dan Simple Linear Regression*. International Journal of Data and Operational Systems, 6(3). https://doi.org/10.56705/ijodas.v6i3.368

## Business Understanding

### Problem Statements

- **Pernyataan Masalah 1**: Bagaimana tren penjualan kopi harian dari *vending machine* berdasarkan waktu, dan apa saja produk yang paling diminati?
- **Pernyataan Masalah 2**: Siapa saja pelanggan (berdasarkan identitas kartu anonim) dengan total transaksi dan pengeluaran tertinggi, serta bagaimana pola pembelian mereka?
- **Pernyataan Masalah 3**: Model *machine learning* atau pendekatan statistik manakah yang memberikan tingkat error terkecil dalam memprediksi total penjualan harian berikutnya?

### Goals

- **Tujuan 1**: Memahami tren penjualan historis dan mengekstraksi pola temporal (jam, hari, bulan) dari data transaksi melalui *Exploratory Data Analysis* (EDA).
- **Tujuan 2**: Mengidentifikasi dan menganalisis perilaku pelanggan paling aktif untuk mendapatkan wawasan bisnis terkait preferensi pelanggan setia.
- **Tujuan 3**: Membangun, melatih, dan mengevaluasi lima model peramalan (KNN, Random Forest, AdaBoost, Simple Linear Regression, dan Fuzzy Time Series) guna menentukan pendekatan terbaik secara empiris dalam memprediksi target penjualan harian.

### Solution Statements
Untuk mencapai tujuan yang telah diuraikan, proyek ini menggunakan 2 solusi utama:
- **Solusi 1 (Komparasi 5 Algoritma)**: Membandingkan tiga algoritma *baseline Machine Learning* (KNN, Random Forest, AdaBoost) dengan dua pendekatan berbasis statistik/logika (Simple Linear Regression, Fuzzy Time Series). Kinerja setiap model diukur dengan metrik evaluasi yang sebanding (RMSE dan MAE), di samping MSE dan MAPE.
- **Solusi 2 (Hyperparameter Tuning)**: Melakukan optimasi pada model *baseline* terbaik (Random Forest) menggunakan `GridSearchCV` dengan strategi validasi `TimeSeriesSplit` agar terhindar dari *temporal data leakage*. Tujuannya adalah untuk meningkatkan akurasi (mengurangi RMSE) dibandingkan model dengan parameter *default*.

## Data Understanding

Data yang digunakan dalam proyek ini adalah dataset **Coffee Sales** yang bersumber dari [Kaggle](https://doi.org/10.34740/KAGGLE/DSV/11159944). Dataset ini merepresentasikan log transaksi *vending machine* kopi dengan jumlah sampel sebanyak **3.636 baris** setelah pembersihan. Transaksi tercatat pada rentang tanggal **1 Maret 2024 hingga 23 Maret 2025**.

### Variabel-variabel pada dataset adalah sebagai berikut:
- **`date`** (Date): Tanggal terjadinya transaksi.
- **`datetime`** (Datetime): Waktu spesifik dan tanggal transaksi.
- **`cash_type`** (Categorical): Metode pembayaran yang digunakan, yaitu `card` (kartu) atau `cash` (tunai).
- **`card`** (Categorical): Identifier unik pelanggan (dianonimkan), hanya tersedia untuk pembayaran via kartu.
- **`money`** (Numerical): Nilai atau besaran uang yang dibayarkan.
- **`coffee_name`** (Categorical): Nama varian kopi yang dibeli pelanggan (misal: *Latte*, *Americano*).

### Exploratory Data Analysis (EDA)

Untuk memahami lebih lanjut tentang kondisi data, berikut adalah visualisasi hasil observasi data:

**1. Tren Penjualan Harian**
![Tren Penjualan Harian](images/image_0.png)
Total pendapatan dari *vending machine* berfluktuasi dari hari ke hari dengan rentang nilai tertentu. Ini mengonfirmasi bahwa data memiliki variansi berdasarkan deret waktu yang perlu dipelajari.

**2. Pola Temporal Penjualan**
![Pola Temporal](images/image_1.png)
Sebaran waktu menunjukkan lonjakan transaksi pada rentang jam sibuk (terutama jam 10 pagi). Distribusi berdasarkan hari dalam seminggu (*Day of Week*) menunjukkan performa yang cukup stabil di tengah minggu.

**3. Distribusi Produk Kopi**
![Distribusi Produk](images/image_2.png)
Varian kopi seperti *Latte* dan *Americano* mendominasi baik dalam hal kuantitas maupun kontribusi pendapatan, sementara menu khusus tertentu berkontribusi lebih sedikit.

**4. Metode Pembayaran**
![Metode Pembayaran](images/image_3.png)
Mayoritas pelanggan lebih suka menggunakan metode pembayaran nirsentuh (Kartu) dibandingkan Uang Tunai.

**5. Top Pelanggan & Waktu Pembelian**
![Top Pelanggan](images/image_4.png)
![Waktu Pembelian Pelanggan Teratas](images/image_5.png)
Analisis terhadap atribut `card` mengungkap adanya sekelompok pelanggan loyal dengan frekuensi transaksi dan pengeluaran jauh melampaui rata-rata. Pelanggan paling teratas cenderung melakukan transaksi di jam-jam konsisten.

## Data Preparation

Tahap persiapan data merupakan hal yang sangat krusial dalam deret waktu (*time series*). Berikut adalah tahapan yang telah diterapkan:

1. **Pembersihan Data (*Data Cleaning*)**
   - **Tindakan:** Mengubah tipe data tanggal menjadi datetime dan memeriksa *missing values*. Kolom `card` memiliki *missing values*, namun ini wajar untuk transaksi kas, sehingga tidak di-*drop* dari *raw data*. Tidak ditemukan nilai duplikat yang mengganggu.

2. **Agregasi Harian (*Time Aggregation*)**
   - **Tindakan:** Mengelompokkan transaksi per tanggal (`date`) untuk menjumlahkan `money`. Hari yang tidak memiliki transaksi diisi dengan `0`.
   - **Alasan:** Tujuan peramalan ini adalah mengetahui *daily target sales* (Target Penjualan Harian). Transaksi level individual tidak mengandung cukup informasi sekuesial secara mandiri.

3. **Rekayasa Fitur (*Feature Engineering*)**
   - **Tindakan:** Membuat atribut `lag_1` (penjualan hari sebelumnya), `lag_7` (penjualan seminggu sebelumnya), `day_of_week` (hari dalam 1 pekan), `month`, dan indikator `is_weekend`. Baris pertama yang *null* akibat pergeseran data (*shift*) kemudian dihapus.
   - **Alasan:** Model tradisional dan ML dasar tidak mengenali urutan baris. Dengan mengubahnya menjadi format *lag*, model diajarkan "informasi kemarin" dan "informasi minggu lalu" agar dapat mendeteksi pola autokorelasi berulang.

4. **Pemisahan Data secara Kronologis (*Chronological Data Splitting*)**
   - **Tindakan:** Memecah data menjadi 80% *Train* dan 20% *Test* secara kronologis. Fungsi *shuffle* dimatikan.
   - **Alasan:** Mencegah terjadinya *temporal data leakage*. Apabila diacak, kita "menggunakan informasi masa depan untuk memprediksi masa lalu". Evaluasi harus meniru kondisi nyata: melatih model dengan data lama dan mengetes pada data terbaru.

5. **Standarisasi Fitur (*Standard Scaling*)**
   - **Tindakan:** Menggunakan `StandardScaler` dari scikit-learn. Scaler **hanya di-fit pada data latih**, lalu mentransformasi data latih dan uji.
   - **Alasan:** Fitur jarak berdekatan (seperti KNN) sangat rentan pada variabel dengan skala nilai yang berbeda. Scaling menormalisasi jangkauan skala setiap variabel.

## Modeling

Proyek ini mengeksplorasi lima buah model, mencakup pendekatan Machine Learning dan Statistik/Logika:

1. **K-Nearest Neighbors (KNN)**
   - Algoritma berbasis jarak yang mengklasifikasikan atau memprediksi berdasarkan *k* contoh fitur historis paling mirip. Menggunakan nilai `k=5`.
   - *Kelebihan:* Sangat mudah dipahami, bebas asumsi terdistribusi (non-parametrik).
   - *Kekurangan:* Tidak mampu melakukan ekstrapolasi tren linier ke masa depan. 

2. **Random Forest Regressor**
   - Sebuah model ensambel *bagging* berbasi sekumpulan pohon keputusan.
   - *Kelebihan:* Sangat handal (*robust*) tanpa *scaling*, otomatis menyeleksi fitur berharga, dan rentan *overfitting*.
   - *Kekurangan:* Mirip dengan KNN, tidak dapat memprediksi nilai di luar rentang ekstrim dari data latihnya.

3. **AdaBoost Regressor**
   - Model ensambel *boosting* yang secara sekuensial mengurangi porsi nilai error model lemah sebelumnya.
   - *Kelebihan:* Umumnya memiliki *bias* yang rendah karena mengoreksi kesalahan tahapan sebelumnya secara iteratif.
   - *Kekurangan:* Sangat rentan terhadap gangguan *outlier*.

4. **Simple Linear Regression (SLR)**
   - Pendekatan klasik menggunakan persamaan linier garis lurus yang berpegang pada satu variabel utama. Dalam kasus ini, kita menggunakan `lag_1`.
   - *Kelebihan:* Menghasilkan interpretasi fungsi secara langsung (misal: "kenaikan 1 unit X berkontribusi N unit terhadap y").
   - *Kekurangan:* Gagal memprediksi apabila ada hubungan bergejolak yang rumit dan sangat non-linier.

5. **Fuzzy Time Series (FTS) - Chen**
   - Memetakan interval nilai-nilai *real* dan logika himpunan fuzzy.
   - *Kelebihan:* Mudah ditangani tanpa perlu menyeleksi beragam jenis hiperparameter, ideal untuk ketidakpastian.
   - *Kekurangan:* Pembagian partisi dan relasi *universe of discourse* harus diekstrapolasi sedemikian rupa, rentan akurasi kasar bila rentang interval disetel terlalu sempit/lebar.

### Hyperparameter Tuning (Proses Improvement)
Untuk model **Random Forest** (yang cukup menjanjikan), dilakukan optimasi untuk mengurangi *error*. Pencarian grid mendalam (*Grid Search CV*) dilakukan. Untuk menjaga keamanan deret waktu, diatur agar teknik validasi menggunakan iterasi maju `TimeSeriesSplit` dengan 3 lipatan.

- **Parameter yang diuji:** `n_estimators` (50, 100, 200), `max_depth` (None, 5, 10), dan `min_samples_split` (2, 5).
- **Hasil:** Model terbaik ditemukan pada kombinasi `n_estimators=50`, `max_depth=5`, `min_samples_split=5`.
- Peningkatan model (RMSE) ini kemudian dicatat dalam klasemen akhir di bagian evaluasi.

## Evaluation

Model dievaluasi menggunakan metrik yang relevan dengan tipe regresinya, di samping merangkum matrik pembanding.
- **MSE (Mean Squared Error)**: Rata-rata dari nilai kuadrat error. Menghukum kesalahan besar secara drastis. Formula: $MSE = \frac{1}{n} \Sigma_{i=1}^n(y_i - \hat{y}_i)^2$
- **MAPE (Mean Absolute Percentage Error)**: Persentase rata-rata simpangan kesalahan. Sangat intuitif bagi pengambil keputusan bisnis. Namun dikalkulasikan di luar saat $y=0$. Formula: $MAPE = \frac{1}{n} \Sigma_{i=1}^n \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100$
- **RMSE (Root Mean Squared Error)**: Metrik standar utama universal kita yang mengakar nilai MSE agar kembali sejajar dengan satuan target (*total sales*). Formula: $RMSE = \sqrt{MSE}$
- **MAE (Mean Absolute Error)**: Simpangan mutlak universal. Formula: $MAE = \frac{1}{n} \Sigma_{i=1}^n|y_i - \hat{y}_i|$

### Tabel Hasil Evaluasi (Data Uji)

| Model | MSE | MAPE (%) | RMSE | MAE |
|---|---|---|---|---|
| KNN (default) | 54.191,74 | - | 232,79 | 182,29 |
| Random Forest (default) | 74.766,12 | - | 273,43 | 214,51 |
| AdaBoost (default) | 50.465,32 | - | 224,64 | 170,22 |
| **Simple Linear Regression** | - | **56.31%** | **200,24** | **155,42** |
| Fuzzy Time Series (Chen) | - | 60.60% | 215,74 | 171,92 |
| Tuned Random Forest | 63.612,78 | - | 252,22 | 197,97 |

![Perbandingan RMSE dan MAE](images/image_6.png)
![Visualisasi Prediksi Deret Waktu](images/image_7.png)
![Analisis Residual](images/image_8.png)
![Feature Importance](images/image_9.png)

### Penjelasan Hasil Proyek

*   **Penyelesaian Goals**: Tujuan bisnis telah berhasil diraih, dengan mengidentifikasi pelanggan spesifik sekaligus memberikan prediksi jitu. 
*   **Pemilihan Model**: Bertentangan dengan intuisi populer, algoritma tradisional klasik yaitu **Simple Linear Regression** muncul sebagai primadona dalam menyelesaikan masalah *Coffee Sales Forecasting*. Model tersebut mencatat kesalahan RMSE terendah (*margin error* simpangan rata-rata sebesar ~200 uang lokal) mengungguli algoritma ansambel modern yang lebih kompleks.
*   **Wawasan Prediktif**: Melalui penelusuran bobot model Random Forest dan Koefisien Regresi SLR, ditemukan bahwa nilai total sales pada **1 hari sebelumnya** (`lag_1`) menjadi fondasi prediktor penentu terpenting bagi sebuah peramalan pendapatan *vending machine* di masa depan.
