# Poverty Depth Index Spatial Analysis

Aplikasi web berbasis Streamlit untuk eksplorasi dan visualisasi data regional Jawa Tengah dengan choropleth map interaktif.

## 📋 Daftar Isi

- [Fitur Utama](#-fitur-utama)
- [Persyaratan Sistem](#-persyaratan-sistem)
- [Instalasi](#-instalasi)
- [Cara Menjalankan Server](#-cara-menjalankan-server)
- [Cara Menggunakan Aplikasi](#-cara-menggunakan-aplikasi)
- [Struktur Folder](#-struktur-folder)
- [Troubleshooting](#-troubleshooting)

## ✨ Fitur Utama

1. **📁 File Upload** - Upload file CSV atau Excel
2. **📋 Data Table Viewer** - Tabel interaktif dengan search dan sort
3. **🔍 Variable Selector** - Filter variabel numerik untuk analisis
4. **📊 Statistik Deskriptif** - Mean, Median, Min, Max, Standard Deviation
5. **🗺️ Choropleth Map** - Peta interaktif Jawa Tengah dengan Folium
6. **📈 Bar Chart** - Top 10 wilayah dengan horizontal bar chart

## 💻 Persyaratan Sistem

- **Python**: 3.8 atau lebih baru
- **OS**: Windows, macOS, atau Linux
- **RAM**: Minimal 4GB
- **Storage**: Minimal 500MB untuk dependencies

## 📦 Instalasi

### 1. Clone atau Download Project

```bash
cd "d:\Kadafi workspace\Entropiata Agency\GUI Skripsi\Thesis-GUI-"
```

### 2. Install Dependencies

Pastikan Anda berada di root folder project, kemudian jalankan:

```bash
pip install -r backend/requirements.txt
```

**Dependencies yang akan diinstall:**
- `streamlit` - Framework web app
- `pandas` - Data manipulation
- `geopandas` - Geodata processing
- `folium` - Interactive maps
- `plotly` - Interactive charts
- `openpyxl` - Excel file support

### 3. Verifikasi Instalasi

Cek apakah semua package terinstall dengan benar:

```bash
python -c "import streamlit; import pandas; import geopandas; import folium; import plotly; print('✓ All packages installed successfully!')"
```

## 🚀 Cara Menjalankan Server

### Metode 1: Command Line (Recommended)

1. **Buka Terminal/Command Prompt**

2. **Navigasi ke folder project:**
   ```bash
   cd "d:\Kadafi workspace\Entropiata Agency\GUI Skripsi\Thesis-GUI-"
   ```

3. **Jalankan Streamlit:**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Tunggu hingga muncul pesan:**
   ```
   You can now view your Streamlit app in your browser.

   Local URL: http://localhost:8501
   Network URL: http://192.168.x.x:8501
   ```

5. **Browser akan otomatis terbuka** ke `http://localhost:8501`
   - Jika tidak otomatis, buka browser dan ketik URL tersebut

### Metode 2: PowerShell (Windows)

```powershell
cd "d:\Kadafi workspace\Entropiata Agency\GUI Skripsi\Thesis-GUI-"
streamlit run streamlit_app.py
```

### Metode 3: VS Code Terminal

1. Buka project di VS Code
2. Buka Terminal (Ctrl + `)
3. Jalankan:
   ```bash
   streamlit run streamlit_app.py
   ```

## 📖 Cara Menggunakan Aplikasi

### 1. Upload Data

1. Klik **"Browse files"** di sidebar kiri
2. Pilih file CSV atau Excel (contoh: `backend/data/sample/JawaTengah.csv`)
3. Tunggu hingga muncul pesan sukses:
   ```
   ✅ File uploaded successfully!
   📊 35 rows × 8 columns
   🗺️ Geodata merge: 35/35 regions (100.0%)
   ```

### 2. Lihat Data Table

Scroll ke bawah untuk melihat tabel interaktif dengan semua data yang diupload.

### 3. Pilih Variabel

Di sidebar, pilih variabel dari dropdown:
- **UMK** - Upah Minimum Kabupaten/Kota
- **TPT** - Tingkat Pengangguran Terbuka
- **P1** - Persentase Penduduk Miskin
- **P2** - Indeks Kedalaman Kemiskinan
- **P3** - Indeks Keparahan Kemiskinan
- **Gini** - Rasio Gini
- **AHH** - Angka Harapan Hidup

### 4. Lihat Statistik

Setelah memilih variabel, akan muncul 5 KPI cards:
- 📊 Mean (rata-rata)
- 📍 Median (nilai tengah)
- ⬇️ Min (nilai minimum)
- ⬆️ Max (nilai maksimum)
- 📏 Std Dev (standar deviasi)

### 5. Eksplorasi Choropleth Map

**Fitur Map:**
- **Hover** pada wilayah untuk melihat nama dan nilai
- **Zoom in/out** dengan scroll mouse atau tombol +/-
- **Pan** dengan drag mouse
- **Color scale** menunjukkan gradient nilai (ungu = rendah, kuning = tinggi)

### 6. Analisis Bar Chart

Bar chart horizontal menampilkan **Top 10 wilayah** berdasarkan variabel yang dipilih:
- Bars berwarna gradient sesuai nilai
- Sorted dari nilai terkecil (atas) ke terbesar (bawah)
- Interactive hover untuk detail

## 📁 Struktur Folder

```
Thesis-GUI-/
├── streamlit_app.py              # Main application file
├── backend/
│   ├── requirements.txt          # Python dependencies
│   ├── services/
│   │   ├── data_service.py       # Data loading & processing
│   │   ├── geo_service.py        # Geodata handling
│   │   └── stats_service.py      # Statistical calculations
│   ├── config/
│   │   └── settings.py           # Configuration
│   ├── utils/
│   │   └── validators.py         # Data validation
│   └── data/
│       ├── sample/
│       │   └── JawaTengah.csv    # Sample dataset
│       └── uploads/              # Uploaded files (auto-created)
├── Geodata Jawa Tengah/
│   ├── JawaTengah.shp            # Shapefile
│   ├── JawaTengah.shx
│   ├── JawaTengah.dbf
│   └── JawaTengah.prj
└── jawa_tengah.geojson           # Cached GeoJSON
```

## 🔧 Troubleshooting

### Server tidak bisa start

**Error:** `ModuleNotFoundError: No module named 'streamlit'`

**Solusi:**
```bash
pip install streamlit
```

### Port 8501 sudah digunakan

**Error:** `OSError: [Errno 98] Address already in use`

**Solusi 1:** Stop server yang sedang running (Ctrl+C)

**Solusi 2:** Gunakan port lain:
```bash
streamlit run streamlit_app.py --server.port 8502
```

### Map tidak muncul

**Penyebab:** Geodata tidak ditemukan

**Solusi:** Pastikan folder `Geodata Jawa Tengah/` ada dan berisi file shapefile lengkap

### Data tidak ter-upload

**Penyebab:** File format tidak didukung atau corrupt

**Solusi:** 
- Pastikan file adalah CSV atau Excel (.xlsx, .xls)
- Cek apakah file memiliki kolom "Kabupaten/Kota"
- Pastikan data numerik tidak mengandung karakter non-numerik

### Browser tidak otomatis terbuka

**Solusi:** Buka manual di browser:
```
http://localhost:8501
```

## 🛑 Cara Menghentikan Server

Tekan **Ctrl + C** di terminal/command prompt tempat server berjalan.

## 📝 Tips & Best Practices

1. **Gunakan data yang clean** - Pastikan tidak ada missing values di kolom region
2. **Format nama wilayah** - Sistem otomatis normalize nama (case-insensitive, trim spaces)
3. **Multiple variables** - Ganti variabel tanpa perlu re-upload file
4. **Refresh data** - Upload file baru akan replace data sebelumnya
5. **Performance** - Untuk dataset besar (>1000 rows), loading mungkin lebih lama

## 📞 Support

Jika mengalami masalah, cek:
1. Python version: `python --version` (harus 3.8+)
2. Streamlit version: `streamlit version`
3. Dependencies: `pip list | grep streamlit`

## 📄 License

© 2026 Sistem Analitik Data - Menu 1: Import & Exploration