## Project Domain

### Latar Belakang

Penjualan kopi merupakan salah satu jenis transaksi yang memiliki pola pembelian yang dapat berubah berdasarkan waktu, jenis produk, serta karakteristik pelanggan. Pada bisnis vending machine, data transaksi yang tercatat secara berkala dapat dimanfaatkan untuk memahami pola penjualan dan perilaku pembelian pelanggan. Informasi tersebut dapat menjadi dasar bagi bisnis untuk menentukan strategi penjualan, mengatur persediaan, dan mengalokasikan sumber daya secara lebih efektif.

Dataset yang digunakan dalam proyek ini adalah **Coffee Sales**, yang berisi catatan transaksi penjualan kopi dari sebuah vending machine. Dataset tersebut dapat digunakan untuk mengeksplorasi pola penjualan berdasarkan waktu, produk, dan pelanggan. Dengan memanfaatkan data historis tersebut, proyek ini berfokus pada analisis pola penjualan dan pengembangan model untuk memprediksi penjualan pada periode berikutnya.

Permasalahan yang ingin dikaji adalah bagaimana data transaksi historis dapat digunakan untuk memahami tren penjualan serta memprediksi performa penjualan di masa mendatang. Selain itu, analisis dilakukan untuk memahami pola pembelian pelanggan tertentu sehingga dapat memberikan gambaran mengenai preferensi pelanggan terhadap produk kopi.

### Mengapa dan Bagaimana Masalah Ini Harus Diselesaikan

#### Mengapa

Kemampuan untuk memprediksi penjualan dapat membantu bisnis dalam mengambil keputusan berdasarkan data historis. Tanpa pemahaman terhadap pola penjualan, bisnis dapat mengalami kesulitan dalam menentukan target penjualan, memperkirakan kebutuhan persediaan, dan mengalokasikan sumber daya secara optimal.

Pada vending machine, pola transaksi juga dapat memberikan informasi mengenai produk yang paling banyak diminati serta waktu ketika permintaan cenderung meningkat atau menurun. Oleh karena itu, analisis terhadap data penjualan historis diperlukan untuk mengidentifikasi pola tersebut dan menghasilkan informasi yang dapat mendukung proses pengambilan keputusan.

#### Bagaimana

Masalah tersebut akan diselesaikan melalui analisis data penjualan historis dengan beberapa pendekatan. Tahap awal dilakukan melalui **Time Series Exploratory Data Analysis (EDA)** untuk memahami tren, pola penjualan, distribusi transaksi, serta perubahan volume penjualan berdasarkan waktu.

Selanjutnya, beberapa algoritma *machine learning* akan digunakan sebagai **baseline model**, yaitu **K-Nearest Neighbors (KNN), Random Forest, dan AdaBoost**. Ketiga model tersebut digunakan berdasarkan pendekatan yang terdapat pada materi dan kode pembelajaran yang menjadi acuan proyek. Performa model baseline akan dievaluasi menggunakan **Mean Squared Error (MSE)**.

Sebagai eksperimen tambahan, proyek ini juga akan mengeksplorasi penerapan **Fuzzy Time Series (FTS)** dan **Simple Linear Regression (SLR)** untuk melakukan forecasting penjualan. Pendekatan tersebut digunakan berdasarkan penelitian terkait forecasting yang menerapkan Fuzzy Time Series dan Simple Linear Regression pada data penjualan. Performa kedua pendekatan tambahan tersebut akan dievaluasi menggunakan **Mean Absolute Percentage Error (MAPE)**.

Hasil dari beberapa pendekatan tersebut kemudian dibandingkan untuk mengetahui pendekatan yang memberikan performa prediksi yang lebih baik terhadap data penjualan kopi. Selain forecasting, analisis terhadap transaksi pelanggan tertentu dilakukan untuk mengidentifikasi pola pembelian dan produk yang sering dibeli oleh pelanggan.

Dengan demikian, proyek ini tidak hanya berfokus pada prediksi nilai penjualan, tetapi juga pada pemahaman terhadap pola historis dan perilaku pembelian yang dapat digunakan sebagai dasar pengambilan keputusan berbasis data.

### Hasil Riset dan Referensi Terkait

Penelitian mengenai prediksi penjualan menunjukkan bahwa data historis dapat dimanfaatkan untuk memperkirakan nilai penjualan pada periode berikutnya. Salah satu pendekatan yang dapat digunakan adalah **Linear Regression**, yang memungkinkan hubungan antara variabel independen dan penjualan sebagai variabel target untuk dianalisis dan digunakan dalam proses prediksi.

Selain pendekatan regresi, **Fuzzy Time Series** dapat digunakan untuk melakukan forecasting pada data yang memiliki pola berdasarkan waktu. Penelitian terkait yang menjadi referensi tambahan dalam proyek ini membahas penerapan **Fuzzy Time Series dan Simple Linear Regression** untuk melakukan prediksi, sehingga pendekatan tersebut digunakan sebagai salah satu eksperimen tambahan dalam proyek Coffee Sales.

Di sisi lain, KNN, Random Forest, dan AdaBoost digunakan sebagai baseline berdasarkan pendekatan yang diberikan dalam materi pembelajaran. Penggunaan beberapa algoritma memungkinkan proyek untuk membandingkan pendekatan *machine learning* yang berbeda dalam memprediksi penjualan.

Dataset yang digunakan dalam proyek ini adalah dataset **Coffee Sales** yang dikembangkan oleh Yaroslav Isaienkov. Dataset tersebut berisi catatan transaksi penjualan kopi dari vending machine dan dapat digunakan untuk menganalisis pola pembelian, tren penjualan, serta preferensi pelanggan.

### Referensi

* Isaienkov, Y. (2025). *Coffee Sales* [Dataset]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/11159944
* *Sales Prediction using Linear Regression*. ResearchGate. https://www.researchgate.net/publication/376738143_Sales_Prediction_using_Linear_Regression
* *[Penelitian terkait Fuzzy Time Series dan Simple Linear Regression]*. International Journal of Data and Operational Systems, 6(3). https://doi.org/10.56705/ijodas.v6i3.368