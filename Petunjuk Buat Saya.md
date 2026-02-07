# Panduan Pengembang (Author Guideline)

Dokumen ini dibuat untuk membantu pengembang memahami struktur proyek, alur data, dan cara melakukan debugging pada aplikasi **Thesis-GUI**.

## 1. Struktur Proyek (Terbaru)

Proyek ini telah direfaktor untuk menyederhanakan struktur direktori.

```
Thesis-GUI/
├── backend/                   # Logika bisnis utama (Backend)
│   ├── data/                  # Tempat penyimpanan data sementara/cache
│   ├── services.py            # LAYANAN TERPADU (Data, Geo, Model, Stats)
│   ├── settings.py            # Konfigurasi (path, settings)
│   └── utils.py               # Fungsi bantuan (Validators, Helpers)
├── Geodata Jawa Tengah/       # File Shapefile (.shp) peta dasar
├── Model/                     # File Model Machine Learning (.pkl)
├── streamlit_app.py           # Entry point aplikasi (Frontend Streamlit)
└── requirements.txt           # Daftar library Python yang dibutuhkan
```

## 2. Komponen Utama

### A. Frontend (`streamlit_app.py`)
File ini mengatur tampilan antarmuka (UI). Tidak boleh ada logika bisnis yang rumit di sini. Frontend hanya memanggil fungsi dari `backend.services`.
*   **Layout**: Menggunakan **3 Halaman** (Beranda, Import & Exploration, Prediction).
*   **Navigasi**: Menggunakan `streamlit-option-menu`.

### B. Backend (`backend/services.py`)
Semua logika layanan digabung dalam satu file `services.py` untuk kemudahan akses:
1.  **`DataService`**: Manipulasi data, load file, dan auto-merge.
2.  **`GeoService`**: Operasi peta, Shapefile, dan GeoJSON.
3.  **`ModelService`**: Load model ML dan prediksi.
4.  **`StatsService`**: Perhitungan statistik dan chart.

## 3. Alur Kerja Aplikasi (Workflow)

1.  **Start Aplikasi**: `streamlit run streamlit_app.py`.
2.  **Beranda**: User disambut dengan ringkasan sistem.
3.  **Import & Exploration**:
    *   User upload file.
    *   `DataService` memvalidasi dan menggabungkan dengan Geodata.
    *   Visualisasi awal (Peta & Grafik) ditampilkan.
4.  **Prediction**:
    *   User memilih model dari Sidebar.
    *   `ModelService` melakukan prediksi.
    *   Hasil ditampilkan dalam Peta Probabilitas (`GeoService`) dan Tabel Metrik (`StatsService` logic).

## 4. Panduan Debugging

### Masalah 1: Peta Kosong / Tidak Muncul
*   **Cek**: `backend/services.py` -> Class `GeoService` -> method `match_regions`.
*   **Solusi**: Pastikan nama daerah di data cocok dengan Shapefile.

### Masalah 2: ImportError / ModuleNotFoundError
*   **Penyebab**: Struktur folder berubah tapi cache pyc lama masih ada.
*   **Solusi**: Hapus folder `__pycache__` dan restart terminal.

### Masalah 3: Gagal Load Model
*   **Solusi**: Pastikan file `.pkl` ada di folder `Model/` dan namanya sesuai dengan yang terdaftar di `ModelService.model_mapping`.

## 5. Tips Pengembangan

*   **Menambah Fitur**: Tambahkan method baru di class yang sesuai dalam `backend/services.py`.
*   **Ubah Config**: Edit `backend/settings.py` untuk mengganti path atau konstanta.

---
*Dibuat oleh AI Assistant untuk memudahkan pengembangan dan pemeliharaan kode.*
