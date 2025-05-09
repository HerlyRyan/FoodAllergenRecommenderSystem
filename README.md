# Proyek Klasifikasi Kandungan Alergen pada Produk Makanan

## Project Overview

Proyek ini bertujuan membangun model machine learning untuk memprediksi apakah suatu produk makanan mengandung alergen berdasarkan bahan-bahan dan karakteristiknya. Dataset yang digunakan adalah `food_ingredients_and_allergens.csv`, berisi daftar produk makanan dengan informasi bahan utama, pemanis, jenis lemak/minyak, bumbu, alergen yang diketahui, serta label prediksi kandungan alergen.

### Latar Belakang

Makanan merupakan suatu kebutuhan utama manusia untuk menjalani kehidupan, namun ada beberapa kasus dimana kita harus menghindari beberapa product makanan yang mengandung alergen yang sama. Hal ini yang mendorong saya untuk melakukan pembuatan model machine learning untuk memberikan makanan yang harus dihindari sesuai dari product makanan yang kita cari

### Hasil Riset Terkait

1. Falahah, Falahah, and Rita Komalasari. "Model Rekomendasi Makanan Menggunakan Content-Based dan Collaborative Filtering." Prosiding SISFOTEK 8.1 (2024): 673-678.
2. Maheswara, Anak Agung Gde Agastya, et al. "Pengembangan Aplikasi Deteksi Allergen pada Makanan Menggunakan Convolutional Neural Network Berbasis Android." Jurnal Pengembangan Teknologi Informasi dan Ilmu Komputer 8.6 (2024).

---

## Bussiness Understanding

### Problem Statements

Berdasarkan latar belakang di atas, rincian masalahnya adalah sebagai berikut:

- Model _Machine Learning_ apa yang cocok untuk menyelesaikan permasalahan tersebut?
- Bagaimana cara menentukan hasil rekomendasi suatu model _Machine Learning_ dapat dikatakan baik?

### Goals

Untuk menjawab pertanyaan di atas, maka akan dijabarkan sebagai berikut:

- Model yang cocok untuk menyelesaikan masalah tersebut adalah model yang berbasis dengan konten atau biasa disebut _Content-Based Filtering_.
- Melakukan evaluasi terhadap metrik dari model _Machine Learning_ tersebut.

---

## Deskripsi Dataset

Berikut merupakan sumber dari dataset yang saya pakai: [Food Allergens]('https://www.kaggle.com/datasets/uom190346a/food-ingredients-and-allergens/data')

### Deskripsi Variabel

Dataset memiliki kolom sebagai berikut:

| Nama Kolom      | Deskripsi                                                                                | Tipe Data     |
| --------------- | ---------------------------------------------------------------------------------------- | ------------- |
| Food Product    | Nama produk makanan                                                                      | Kategorikal   |
| Main Ingredient | Bahan utama produk                                                                       | Kategorikal   |
| Sweetener       | Jenis pemanis (jika ada)                                                                 | Kategorikal   |
| Fat/Oil         | Jenis lemak atau minyak yang digunakan                                                   | Kategorikal   |
| Seasoning       | Bumbu atau rempah yang ditambahkan                                                       | Kategorikal   |
| Allergens       | Alergen yang terkandung (bisa lebih dari satu)                                           | Teks (daftar) |
| Prediction      | Label target: "Contains" (mengandung alergen) atau "Does not contain" (tidak mengandung) | Kategorikal   |

Produk makanan beragam, mulai dari kue, sup, salad, produk susu, daging, hingga minuman dan makanan penutup.

### Exploratory Data Analysis - Univariate Analysis

- Terdapat missing value pada beberepa kolom
- Terdapat data duplicated berjumlah 90 data
- Dari hasil analisa data alergen, alergen yang palin banyak adalah 'Dairy' yang berjumlah: 262

---

## Data Preparation

Berikut merupakan tahapan-tahapan dalam melakukan data preparation:

- _Menangani Missing Value_  
  Proses ini dilakukan untuk menangani data yang memiliki nilai kosong/_null_ dengan nilai mode (nilai yang sering keluar) yang akan diisi menggunakan fungsi [_fillna()_](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html) dengan tujuan agar mudah untuk diproses.
- _Menghapus Data Duplikat_  
  Proses ini dilakukan dengan menggunakan fungsi [_drop_duplicates()_](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html) agar tidak ada data yang memiliki nilai sama untuk mencegah kekeliruan.
- _Melakukan Vektorisasi dengan TF-IDF_  
  Pada tahap ini data yang telah disiapkan dikonversi menjadi bentuk vektor menggunakan fungsi [tfidfvectorizer()](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) dari library sklearn untuk mengidentifikasi korelasi antara judul film dengan kategori genrenya.
- _Mengukur tingkat kesamaan dengan [Cosine Similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)_  
  Setelah data dikonversi menjadi bentuk vektor, selanjutnya ukur tingkat kesamaan antara dua vektor dan menentukan apakah kedua vektor tersebut menunjuk ke arah yang sama. Semakin kecil sudut cosinus, semakin besar nilai cosine similarity.

## Modeling

Setelah data selesai disiapkan, proses selanjutnya adalah membuat model adapun tahap-tahapnya diantaranya sebagai berikut:

- _Membuat Fungsi food_avoids()_
  Tahap terakhir dari proses modeling adalah membuat fungsi untuk mendapatkan hasil _top-N avoids_, kali ini fungsinya dinamakan _food_avoids()_. Cara kerja dari fungsi ini yaitu menggunakan fungsi [argpartition](https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html) untuk mengambil sejumlah nilai k tertinggi dari similarity data (dalam kasus ini: dataframe **cosine_sim_df**). Kemudian mengambil data dari bobot (tingkat kesamaan) tertinggi ke terendah. Data ini lalu dimasukkan ke dalam variabel closest. Berikutnya menghapus food*product yang dicari menggunakan fungsi [drop()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html) agar tidak muncul dalam daftar rekomendasi.
  Penjelasan parameter dari fungsi \_food_avoids()* adalah sebagai berikut:
  - food_product : Nama makanan (index kemiripan dataframe) (str)
  - similarity_data : Kesamaan dataframe simetrik dengan nama makanan sebagai indeks dan kolom (object)
  - items : Mengandung kedua nama dan fitur lainnya yang digunakan untuk mendefinisikan kemiripan (object)
  - k : Banyaknya jumlah makanan yang harus dihindari (int)

### Result

Setelah model selesai dibuat, panggil model untuk menampilkan hasil makanan yang dihindari, sebagai contoh kita gunakan nama makanan _Mushroom Risotto_ untuk menguji model.

| index | Food Product     | Main Ingredient | Sweetener | Fat/Oil | Seasoning                     | Allergens | Prediction |
| ----- | ---------------- | --------------- | --------- | ------- | ----------------------------- | --------- | ---------- |
| 33    | Mushroom Risotto | Mushrooms       | Sugar     | Butter  | Arborio rice, Parmesan cheese | Dairy     | Contains   |

Dapat terlihat pada Tabel diatas bahwa nama makanan _Mushroom Risotto_ merupakan makanan dengan Allergens Dairy. Selanjutnya kita lihat makanan yang harus dihindari yang sesuai dengan Allergens yang sama dengan nama makanan tersebut.

| index | Food Product      | Allergens |
| ----- | ----------------- | --------- |
| 0     | Chicken Shawarma  | Dairy     |
| 1     | Chicken Shawarma  | Dairy     |
| 2     | Chicken Fajitas   | Dairy     |
| 3     | Mango Salsa       | Dairy     |
| 4     | Mango Salsa       | Dairy     |
| 5     | Lentil Curry      | Dairy     |
| 6     | Green Smoothie    | Dairy     |
| 7     | Sausage Pizza     | Dairy     |
| 8     | Beef and Broccoli | Dairy     |
| 9     | Beef Burritos     | Dairy     |

Seperti terlihat pada Tabel diatas, model berhasil menampilkan nama makanan berdasarkan Allergens-nya.

---

## Evaluation

Karena model yang digunakan untuk proyek kali ini adalah **_Content-Based Filtering_**, maka metrik yang cocok untuk digunakan adalah _Precision@K_. Secara matematis dapat dirumuskan sebagai berikut:  
![img](https://dicoding-web-img.sgp1.cdn.digitaloceanspaces.com/original/academy/dos:819311f78d87da1e0fd8660171fa58e620211012160253.png)

Berdasarkan hasil yang telah ditampilkan pada bagian [_Result_](#result) dapat disimpulkan bahwa dari 10 judul makanan yang harus dihindari, ada 10 makanan yang relevan oleh karena itu nilai _Precision@K_ dari model ini adalah 10/10 atau 100%.

---

## Conclution and Recommendation

- Model machine learning ini efektif untuk mengklasifikasi produk makanan berdasarkan kandungan alergen dari bahan-bahannya.
- Pemrosesan dan ekstraksi fitur dari kolom alergen sangat menentukan performa model.
- Disarankan penggunaan model ini sebagai alat bantu bagi konsumen dan produsen untuk meningkatkan kesadaran alergen.
- Pengembangan selanjutnya meliputi:
  - Penambahan fitur bahan yang lebih detail.
  - Perluasan kategori alergen untuk granularitas lebih tinggi.
  - Pembuatan antarmuka pengguna untuk prediksi alergen secara real-time.

---
