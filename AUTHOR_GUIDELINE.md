# Panduan Pengembang (Author Guideline)

Dokumen ini dibuat untuk membantu pengembang memahami struktur proyek, alur data, dan cara melakukan debugging pada aplikasi **Thesis-GUI**.

## 1. Struktur Proyek

Berikut adalah penjelasan singkat mengenai folder dan file utama dalam proyek ini:

```
Thesis-GUI/
├── backend/                   # Logika bisnis utama (Backend)
│   ├── config/                # Konfigurasi (path, settings)
│   ├── data/                  # Tempat penyimpanan data sementara/cache
│   ├── services/              # Layanan inti (Core Services)
│   │   ├── data_service.py    # Manipulasi data (Load CSV/Excel, Filtering)
│   │   ├── geo_service.py     # Logika Peta (Shapefile, GeoJSON, Choropleth)
│   │   └── stats_service.py   # Perhitungan statistik & KPI
│   └── utils/                 # Fungsi bantuan (Validators, Helpers)
├── Geodata Jawa Tengah/       # File Shapefile (.shp) peta dasar
├── streamlit_app.py           # Entry point aplikasi (Frontend Streamlit)
└── requirements.txt           # Daftar library Python yang dibutuhkan
```

## 2. Komponen Utama

### A. Frontend (`streamlit_app.py`)
File ini mengatur tampilan antarmuka (UI). Tidak boleh ada logika bisnis yang rumit di sini. Frontend hanya memanggil fungsi dari `backend/services`.

### B. Backend Services
1.  **`data_service.py`**:
    *   Bertanggung jawab membaca file CSV/Excel.
    *   Melakukan *cleaning* dan validasi data.
    *   Menggabungkan (merge) data statistik dengan data spasial (Geodata).
2.  **`geo_service.py`**:
    *   Membaca Shapefile `JawaTengah.shp`.
    *   Mengubahnya menjadi format GeoJSON agar ringan ditampilkan di web.
    *   Mencocokkan nama daerah (misal: "Kota Semarang" vs "Semarang Kota") menggunakan fungsi normalisasi.
3.  **`stats_service.py`**:
    *   Menghitung KPI (Key Performance Indicators) seperti Rata-rata, Min, Max.
    *   Menyiapkan data untuk grafik batang/pie.

## 3. Alur Kerja Aplikasi (Workflow)

1.  **Start Aplikasi**: `streamlit run streamlit_app.py` dijalankan. Config dimuat.
2.  **Upload File**: User mengupload CSV di Menu "Import Data".
3.  **Processing**: `data_service.load_file()` dipanggil.
    *   Data divalidasi.
    *   Data otomatis dicocokkan dengan Peta (`_merge_with_geodata`).
4.  **Visualisasi**:
    *   Peta Choropleth memanggil `geo_service.create_choropleth_data()`.
    *   Statistik memanggil `stats_service.calculate_kpi()`.

## 4. Panduan Debugging

Berikut adalah solusi untuk masalah umum yang mungkin terjadi:

### Masalah 1: Peta Kosong / Tidak Muncul
*   **Penyebab**: Nama daerah di Excel tidak cocok dengan nama di Shapefile.
*   **Cek**: Buka `backend/services/geo_service.py` -> lihat fungsi `match_regions`.
*   **Solusi**: Periksa log di terminal. Aplikasi akan mencetak "Unmatched regions". Perbaiki nama di file Excel Anda agar sesuai (misal: "Kab. Batang" mungkin harus ditulis "Batang").

### Masalah 2: Gagal Upload File
*   **Penyebab**: Format kolom salah atau tipe data tidak numerik.
*   **Cek**: `backend/utils/validators.py`.
*   **Solusi**: Pastikan file Excel/CSV memiliki header di baris pertama dan kolom data berisi angka yang valid.

### Masalah 3: Error "ModuleNotFoundError"
*   **Penyebab**: Library belum terinstall atau path salah.
*   **Solusi**: Pastikan menjalankan aplikasi dari folder root (`Thesis-GUI-`) dan environment Python sudah aktif.

### Masalah 4: Perubahan kode tidak nampak
*   **Solusi**: Tekan "Rerun" (biasanya tombol 'r') pada halaman Streamlit atau restart terminal jika mengubah kode di folder `backend/`.

## 5. Tips Pengembangan

*   **Menambah Fitur Baru**: Tambahkan logika di `backend/services/` terlebih dahulu, baru buat UI-nya di `streamlit_app.py`.
*   **Ubah Peta**: Jika ingin mengganti peta (misal ke Jawa Barat), ganti file di folder `Geodata...` dan update path di `backend/config/settings.py`.

---
*Dibuat oleh AI Assistant untuk memudahkan pengembangan dan pemeliharaan kode.*
