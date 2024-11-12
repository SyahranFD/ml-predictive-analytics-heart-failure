# Laporan Proyek Machine Learning - Predictive Analytics Heart Failure - Syahran Fadhil Dafanindra

## Domain Proyek
Domain yang dipilih dalam proyek ini adalah kesehatan, khususnya pada gagal jantung.

### Latar Belakang
Gagal jantung merupakan masalah epidemi global yang angka kejadiannya semakin meningkat baik dalam hal prevalensi, morbiditas dan mortalitas [[1]](https://jos.unsoed.ac.id/index.php/jchd/article/view/5511).
Gagal jantung merupakan penyebab kematian utama, merenggut 1,9 juta jiwa per tahun menurut WHO [[2]](https://wellness.journalpress.id/index.php?journal=wellness&page=article&op=view&path[]=410).
Gagal jantung adalah suatu keadaan ketika jantung tidak mampu mempertahankan sirkulasi yang cukup untuk kebutuhan tubuh, gagal jantung terjadi karena kondisi jantung yang terlalu lemah dalam memompa darah keseluruh tubuh untuk memenuhi kebutuhan oksigen dan nutrisi [[3]](https://ejournal.itka.ac.id/index.php/aohj/article/view/100).

Gagal jantung akut adalah kondisi medis yang dapat mengancam nyawa, ditandai dengan perburukan gejala dan tanda gagal jantung. Kondisi ini sering kali disebabkan oleh sindrom koroner akut, khususnya subset ST elevation myocardial infarction (STEMI). Untuk menurunkan risiko morbiditas, re-hospitalisasi, dan mortalitas, diperlukan diagnosis yang cepat dan tata laksana yang tepat [[4]](https://cdkjournal.com/index.php/cdk/article/view/336).
Dalam upaya untuk meningkatkan diagnosis dan pengelolaan gagal jantung, penggunaan algoritma machine learning menjadi solusi yang menjanjikan. 
Algoritma ini dapat memprediksi kemungkinan seseorang terkena gagal jantung berdasarkan data klinis yang tersedia.
Hal ini tidak hanya mempercepat proses diagnosis tetapi juga memungkinkan intervensi yang lebih dini dan tepat sasaran. 
Penggunaan model prediktif ini diharapkan dapat membantu tenaga medis dalam membuat keputusan klinis yang lebih baik dan meningkatkan hasil perawatan pasien.

## Business Understanding

### Problem Statements
1. Apa saja faktor-faktor klinis yang berpengaruh dalam memprediksi kemungkinan seseorang mengalami gagal jantung?
2. Bagaimana cara mengembangkan model machine learning yang dapat memprediksi risiko gagal jantung pada pasien dengan akurasi yang tinggi berdasarkan data klinis yang tersedia?
3. Seberapa efektif penggunaan algoritma machine learning dalam membantu proses diagnosis awal gagal jantung?

### Goals
1. Mengidentifikasi faktor-faktor klinis yang berpengaruh dalam memprediksi kemungkinan seseorang mengalami gagal jantung.
2. Mengembangkan model machine learning yang dapat memprediksi risiko gagal jantung pada pasien dengan akurasi yang tinggi.
3. Membandingkan dan mengevaluasi performa minimal 2 algoritma machine learning dalam memprediksi gagal jantung untuk menemukan pendekatan yang paling efektif dalam mendukung keputusan klinis.

### Solution Statements
1. Mengembangkan model machine learning dengan membandingkan dua algoritma klasifikasi yang berbeda:
   - K-Nearest Neighbors (KNN) sebagai baseline model karena kemampuannya dalam melakukan klasifikasi berdasarkan kemiripan karakteristik dengan data terdekat.
   - Random Forest sebagai model pembanding karena kemampuannya menangani data non-linear dan menghasilkan feature importance.
2. Mengukur performa model menggunakan berbagai metrik evaluasi:
   - Accuracy: untuk mengukur tingkat ketepatan prediksi secara keseluruhan.
   - Precision dan Recall: untuk mengevaluasi kemampuan model dalam mengidentifikasi kasus positif gagal jantung.
   - F1-Score: untuk mengukur trade-off antara Precision dan Recall.

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah Heart Failure Prediction Dataset yang tersedia di Kaggle.
Dataset tersebut dapat diunduh pada tautan berikut, https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/.

### Informasi Dataset:
1. Jumlah total data: 918 data.
2. Jumlah fitur: 12 fitur.
3. Jenis fitur: 6 fitur kategorikal, 5 fitur numerik, dan 1 fitur target.
4. Jenis file: CSV.
5. Ukuran file: 12 KB.

### Variabel-variabel pada Heart Failure Prediction Dataset adalah sebagai berikut:
1. Age: usia pasien [tahun]
2. Sex: jenis kelamin pasien [M: Male, F: Female]
3. ChestPainType: tipe nyeri dada :
   - TA: Typical Angina
   - ATA: Atypical Angina
   - NAP: Non-Anginal Pain
   - ASY: Asymptomatic
4. RestingBP: tekanan darah istirahat [mm Hg]
5. Cholesterol: kolesterol serum [mm/dl]
6. FastingBS: gula darah puasa [1: jika FastingBS > 120 mg/dl, 0: sebaliknya]
7. RestingECG: hasil elektrokardiogram istirahat:
   - Normal: Normal
   - ST: memiliki kelainan gelombang ST-T
   - LVH: menunjukkan kemungkinan atau pasti hipertrofi ventrikel kiri
8. MaxHR: detak jantung maksimum yang dicapai [Numeric value between 60 and 202]
9. ExerciseAngina: angina yang diinduksi oleh olahraga [Y: Yes, N: No]
10. Oldpeak: depresi ST yang diinduksi oleh olahraga relatif terhadap istirahat [Numeric value measured in depression]
11. ST_Slope: kemiringan segmen ST latihan puncak:
    - Up: upsloping
    - Flat: flat
    - Down: downsloping
12. HeartDisease: kelas output [1: heart disease, 0: Normal]

### Mengecek Missing Value
Setelah melakukan pemeriksaan terhadap missing value menggunakan fungsi `isnull().sum()`, hasil menunjukkan bahwa dataset yang digunakan memiliki kualitas data yang baik dari segi kelengkapan data. 
Tidak ditemukan adanya missing value pada seluruh fitur yang ada, baik fitur numerik maupun kategorikal. 
Hal ini mengindikasikan bahwa data telah terkumpul dengan lengkap dan tidak memerlukan penanganan khusus untuk missing value pada tahap data preparation.
```python
# Mengecek missing value pada dataset
df.isnull().sum()
```
<br>![img_missing_value](https://i.ibb.co.com/3182NNb/Missng-value-crop.png)

### Visualisasi Data
Setelah melakukan visualisasi pada data, dapat terlihat bahwa terdapat anomali pada fitur Cholesterol dan RestingBP. 
Kedua fitur tersebut memiliki beberapa nilai 0 yang secara medis tidak valid, karena setiap orang pasti memiliki kadar kolesterol dalam darah dan tidak mungkin memiliki tekanan darah 0 mmHg. 
Anomali ini dapat dilihat jelas pada histogram distribusi kedua fitur tersebut, di mana terdapat beberapa data yang bernilai 0 yang akan perlu ditangani pada tahap data preparation untuk menghindari bias dalam model.

<br>![img_cholesterol_histogram](https://i.ibb.co.com/wLzx9tY/cholesterol-histogram.png)
<br>![img_resting_bp_histogram](https://i.ibb.co.com/7b4NHx8/resting-bp-histogram.png)

## Data Preparation
Dalam proyek ini, dilakukan beberapa tahapan data preparation untuk memastikan kualitas data yang akan digunakan dalam pembuatan model machine learning. 
Berikut adalah tahapan-tahapan yang dilakukan:

1. Penanganan Nilai Tidak Valid
   - Mengganti nilai 0 pada fitur RestingBP dan Cholesterol menjadi NaN (Not a Number), kemudian mengisi nilai NaN tersebut dengan nilai median dari masing-masing fitur.
   - Penggunaan median untuk mengisi missing values dipilih karena Median lebih robust terhadap outlier dibandingkan mean.
   - Alasan: 
     - Nilai 0 pada kedua fitur tersebut tidak valid secara medis karena tidak mungkin seseorang memiliki tekanan darah 0 mmHg atau kadar kolesterol 0 mm/dl.

2. Penanganan Outlier
   - Melakukan visualisasi menggunakan boxplot untuk mendeteksi keberadaan outlier pada setiap fitur numerik. Berikut adalah contoh visualisasi boxplot untuk fitur Cholesterol:
   <br>![img_boxplot_cholesterol](https://i.ibb.co.com/pnTkK2q/outlier-boxplot-cholesterol.png)
   - Setelah teridentifikasi adanya outlier, dilakukan penanganan menggunakan metode IQR.
   - Proses penanganan:
     - Menghitung Q1 (kuartil pertama) dan Q3 (kuartil ketiga).
     - Menentukan IQR = Q3 - Q1.
     - Menetapkan batas bawah = Q1 - 1.5 * IQR.
     - Menetapkan batas atas = Q3 + 1.5 * IQR.
     - Mengganti nilai yang lebih kecil dari batas bawah dengan nilai batas bawah
     - Mengganti nilai yang lebih besar dari batas atas dengan nilai batas atas
   - Alasan:
     - Menghindari bias dalam model.
     - Meningkatkan akurasi prediksi.
     - Memastikan model lebih robust.

3. Pemisahan Dataset
   - Membagi dataset menjadi data latih (80%) dan data uji (20%).
   - Menggunakan random_state=123 untuk memastikan reproducibility.
   - Alasan:
     - Mengevaluasi performa model secara objektif.
     - Menghindari overfitting.
     - Memastikan model dapat melakukan generalisasi dengan baik.

4. One-Hot Encoding
   - Melakukan encoding pada fitur kategorikal (Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope).
   - Proses encoding:
     - Setiap nilai unik dalam fitur kategorikal akan diubah menjadi kolom baru.
     - Nilai 1 diberikan jika data termasuk kategori tersebut.
     - Nilai 0 diberikan jika data tidak termasuk kategori tersebut.
     - Pada akhirnya, fitur kategorikal asli akan dihapus.
   - Alasan:
     - Model machine learning hanya dapat memproses data numerik.
     - Mencegah model menginterpretasikan hubungan ordinal yang tidak ada.
     - Mempertahankan informasi kategorikal tanpa kehilangan makna.
     - Menghindari bias dalam pembobotan fitur.

5. Standarisasi Fitur Numerik
   - Menggunakan StandardScaler untuk menstandarisasi fitur numerik.
   - Proses:
     - Mengubah skala data sehingga memiliki mean = 0.
     - Mengubah skala data sehingga memiliki standar deviasi = 1.
    - Alasan:
      - Menyamakan skala antar fitur numerik.
      - Meningkatkan konvergensi dalam proses training.
      - Menghindari dominasi fitur dengan skala besar.
      - Meningkatkan performa model machine learning.

## Modeling
Pada tahap modeling, dua algoritma machine learning yang digunakan untuk memprediksi risiko gagal jantung adalah K-Nearest Neighbors (KNN) dan Random Forest. 
Berikut adalah penjelasan detail mengenai tahapan, parameter yang digunakan, serta kelebihan dan kekurangan dari masing-masing algoritma.

### K-Nearest Neighbors (KNN)
1. Tahapan dan Parameter:
   - Inisialisasi Model: Model KNN diinisialisasi dengan parameter n_neighbors=10, yang berarti model akan mempertimbangkan 10 tetangga terdekat untuk klasifikasi.
   - Pelatihan Model: Model dilatih menggunakan data latih X_train dan label y_train.
   - Prediksi: Setelah model dilatih, digunakan untuk memprediksi data uji X_test.

2. Kelebihan:
   - Sederhana dan Mudah Dipahami: KNN mudah dipahami dan diimplementasikan.
   - Tidak Memerlukan Pelatihan Ekstensif: Model tidak memerlukan proses pelatihan yang kompleks.

3. Kekurangan:
   - Sensitif terhadap Outlier: Kinerja model dapat terpengaruh oleh keberadaan outlier dalam data.
   - Waktu Prediksi yang Lama: Proses prediksi dapat menjadi lambat pada dataset besar karena harus menghitung jarak ke semua titik data.

### Random Forest
1. Tahapan dan Parameter:
   - Inisialisasi Model: Model Random Forest diinisialisasi dengan parameter:
     - n_estimators=50: jumlah trees (pohon) di forest.
     - max_depth=16: kedalaman atau panjang pohon. Ia merupakan ukuran seberapa banyak pohon dapat membelah (splitting).
     - random_state=55: digunakan untuk mengontrol random number generator yang digunakan.
     - n_jobs=-1: jumlah job (pekerjaan) yang digunakan secara paralel. n_jobs=-1 artinya semua proses berjalan secara paralel.
   - Pelatihan Model: Model dilatih menggunakan data latih X_train dan label y_train.
   - Prediksi: Setelah model dilatih, digunakan untuk memprediksi data uji X_test.

2. Kelebihan:
   - Kemampuan Menangani Data Non-linear: Random Forest dapat menangani hubungan non-linear antara fitur.
   - Feature Importance: Dapat memberikan informasi tentang pentingnya setiap fitur dalam proses klasifikasi.

3. Kekurangan:
   - Kompleksitas Model Lebih Tinggi: Lebih sulit untuk diinterpretasikan dibandingkan model sederhana seperti KNN.
   - Memerlukan Lebih Banyak Waktu untuk Pelatihan: Proses pelatihan bisa lebih lama karena melibatkan banyak pohon keputusan.

### Pemilihan Model Terbaik
Setelah melakukan pelatihan dan prediksi, akurasi dari kedua model adalah sebagai berikut:

- Akurasi KNN: 0.8533
- Akurasi Random Forest: 0.8641

Berdasarkan hasil akurasi, Random Forest dipilih sebagai model terbaik karena menunjukkan akurasi yang lebih tinggi dibandingkan KNN. 
Ini menunjukkan bahwa Random Forest lebih efektif dalam memprediksi risiko gagal jantung berdasarkan dataset yang tersedia.

## Evaluation
Pada tahap evaluasi, metrik yang digunakan untuk mengukur kinerja model adalah Accuracy, Precision, Recall, dan F1-Score. 
Berikut adalah penjelasan mengenai masing-masing metrik serta hasil evaluasi dari kedua model.

### Metrik Evaluasi
1. Accuracy
   - Mengukur proporsi prediksi yang benar dari total prediksi.
   - Rumus: <br>![img_rumus_precision](https://i.ibb.co.com/Dk7xDsc/rumus-accuracy.png)
   - Dimana:
     - TP = True Positives
     - TN = True Negatives
     - FP = False Positives
     - FN = False Negatives

2. Precision
   - Mengukur proporsi prediksi positif yang benar dari total prediksi positif.
   - Rumus: <br>![img_rumus_precision](https://i.ibb.co.com/0jmCvFW/rumus-precision.png)
   - Dimana:
     - TP = True Positives
     - FP = False Positives

3. Recall
   - Mengukur proporsi prediksi positif yang benar dari total kasus positif.
   - Rumus: <br>![img_rumus_recall](https://i.ibb.co.com/NZ8z5rp/rumus-recall.png)
   - Dimana:
     - TP = True Positives
     - FN = False Negatives

4. F1-Score
   - Mengukur trade-off antara Precision dan Recall.
   - Rumus: <br>![img_rumus_f1_score](https://i.ibb.co.com/XJLRMHN/rumus-f1-score.png)

### Hasil Evaluasi
1. Classification Report untuk KNN:

| Kelas | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.86      | 0.80   | 0.83     | 81      |
| 1     | 0.85      | 0.89   | 0.87     | 103     |
| **Akurasi** |           |        | 0.85     | 184     |
| **Macro Avg** | 0.85      | 0.85   | 0.85     | 184     |
| **Weighted Avg** | 0.85      | 0.85   | 0.85     | 184     |

2. Classification Report untuk Random Forest:

| Kelas | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.90      | 0.78   | 0.83     | 81      |
| 1     | 0.84      | 0.93   | 0.88     | 103     |
| **Akurasi** |           |        | 0.86     | 184     |
| **Macro Avg** | 0.87      | 0.85   | 0.86     | 184     |
| **Weighted Avg** | 0.87      | 0.86   | 0.86     | 184     |

## Dampak Model Terhadap Business Understanding
1. Problem Statements:
   - "Apa saja faktor-faktor klinis yang berpengaruh dalam memprediksi kemungkinan seseorang mengalami gagal jantung?"
     - Model Random Forest berhasil mengidentifikasi faktor-faktor klinis penting melalui feature importance.
     - Memberikan wawasan tentang kontribusi relatif setiap variabel dalam prediksi gagal jantung.
   - "Bagaimana cara mengembangkan model machine learning yang dapat memprediksi risiko gagal jantung pada pasien dengan akurasi yang tinggi?"
     - Model Random Forest mencapai akurasi 86.41%.
     - Menunjukkan kemampuan yang baik dalam membedakan kasus positif dan negatif.
   - "Seberapa efektif penggunaan algoritma machine learning dalam membantu proses diagnosis awal gagal jantung?"
     - Model menunjukkan efektivitas tinggi dengan F1-Score 0.88 untuk kasus positif.
     - Memberikan alat bantu yang reliable untuk screening awal.

2. Goals:
   - "Mengidentifikasi faktor-faktor klinis yang berpengaruh"
     - Tercapai melalui analisis feature importance dari Random Forest.
     - Memberikan pemahaman mendalam tentang kontribusi setiap variabel.
   - "Mengembangkan model dengan akurasi tinggi"
     - Tercapai dengan akurasi 86.41% pada Random Forest.
     - Model menunjukkan keseimbangan baik antara precision dan recall.
   - "Membandingkan dan mengevaluasi performa minimal 2 algoritma"
     - Tercapai dengan membandingkan KNN (85.33%) dan Random Forest (86.41%).
     - Evaluasi komprehensif menggunakan berbagai metrik.

3. Solution Statements:
   - "Mengembangkan model dengan membandingkan dua algoritma"
     - Berhasil mengimplementasikan dan membandingkan KNN dan Random Forest.
     - Random Forest menunjukkan performa lebih baik secara konsisten.
   - "Mengukur performa model dengan berbagai metrik"
     - Berhasil mengevaluasi model menggunakan accuracy, precision, recall, dan F1-score.
     - Memberikan pemahaman menyeluruh tentang kinerja model.

## Referensi


Nurhayati, N., Andari, F.N., Ferasinta, F., & Oktarianita, O. (2023). Upaya Peningkatan Aktifitas Fisik Melalui Latihan the Six-minute Walk Pada Masyarakat Penderita Gagal Jantung. _Jurnal of Community Health Development_.

Hidayaturahmah, R. (2024). Kajian Potensi Interaksi Obat Pada Pasien Gagal Jantung Di Bangsal Rawat Inap Di Rumah Sakit Umum DR. H. Abdul Moeloek. _Wellness And Healthy Magazine_.

Sari, S.K., Ismansyah, I., & Andrianur, F. (2023). Hubungan Dukungan Keluarga dengan Tingkat Kekambuhan Pasien Gagal Jantung di RSD dr. H. Soemarno Sosroatmodjo Tanjung Selor, Kalimantan Utara. _Aspiration of Health Journal_.

Nency, C., Surya, M.K., & Kurnia, A. (2023). Gagal Jantung Akut sebagai Komplikasi Sindrom Koroner Akut. _Cermin Dunia Kedokteran_.








