import streamlit as st
import pandas as pd
from pathlib import Path
import folium
from branca.colormap import LinearColormap
import streamlit.components.v1 as components
import plotly.express as px
from streamlit_option_menu import option_menu

from backend.services import data_service, stats_service, model_service, geo_service

st.set_page_config(
    page_title="Poverty Depth Index Spatial Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.2rem; color: #666; margin-bottom: 2rem; }
    .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; text-align: center; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    page = option_menu(
        menu_title="Menu Navigasi",
        options=["Homepage", "Import & Exploration", "Prediction"],
        icons=["house", "cloud-upload", "graph-up-arrow"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#f0f2f6"},
            "icon": {"color": "#ff8c00", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "color": "#262730", "--hover-color": "#d0d2d6"},
            "nav-link-selected": {"background-color": "#1f77b4", "color": "#ffffff"},
        },
    )
    st.divider()
    st.caption("© 2026 • Poverty Depth Index Spatial Analysis")


# ── PAGE: HOMEPAGE ────────────────────────────────────────────────────────────

if page == "Homepage":
    st.markdown('<div class="main-header">📊 Poverty Depth Index Spatial Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Sistem Analitik Spasial untuk Provinsi Jawa Tengah</div>', unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.info("### Langkah 1: Import Data")
        st.write("""
        Mulailah dengan mengupload data Indeks Kedalaman Kemiskinan.

        **Fitur:**
        - Upload file CSV/Excel.
        - Eksplorasi Data Tabular.
        - Statistik Deskriptif Otomatis.
        - Visualisasi Awal (Peta & Grafik).
        """)
    with col2:
        st.success("### Langkah 2: Pemodelan")
        st.write("""
        Lakukan prediksi menggunakan model yang tersedia.

        **Model Tersedia:**
        - Global Logistic Regression.
        - Geographically Weighted Logistic Regression (GWLR).
        - GWLR-Semiparametric.
        """)

    st.write("")
    with st.expander("ℹ️ Tentang Aplikasi Ini", expanded=True):
        st.write("""
        Aplikasi ini dikembangkan untuk mempermudah analisis spasial terkait kemiskinan di Jawa Tengah.
        Menggabungkan analisis statistik deskriptif dan pemodelan prediktif spasial untuk memberikan wawasan yang lebih mendalam.

        **Dikembangkan oleh:** Ikmal Thariq Kadafi
        """)


# ── PAGE: IMPORT & EXPLORATION ────────────────────────────────────────────────

elif page == "Import & Exploration":
    st.markdown('<div class="main-header"> Import & Eksplorasi Data</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("📁 Import Data")
        uploaded_file = st.file_uploader(
            "Upload CSV atau Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Format: CSV, Excel (.xlsx, .xls) | Max: 200MB",
        )
        if uploaded_file is not None:
            try:
                temp_path = Path("backend/data/uploads") / uploaded_file.name
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path.write_bytes(uploaded_file.getbuffer())

                df = data_service.load_file(temp_path)
                st.session_state['data'] = df
                st.session_state['file_name'] = uploaded_file.name

                merge_stats = data_service.get_merge_statistics()
                st.success("✅ File uploaded successfully!")
                st.info(f"📊 {len(df)} rows × {len(df.columns)} columns")
                if merge_stats:
                    st.info(f"🗺️ Geodata merge: {merge_stats.get('matched_regions', 0)}/{merge_stats.get('total_geodata_regions', 0)} regions ({merge_stats.get('match_rate', 0)}%)")
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")

        st.divider()
        selected_variable = None
        if 'data' in st.session_state:
            st.header("🔍 Filter Variabel")
            numeric_cols = data_service.get_numeric_columns()
            if numeric_cols:
                selected_variable = st.selectbox("Pilih Variabel:", options=numeric_cols, key='selected_variable')
            else:
                st.warning("No numeric columns found")
        else:
            st.info("Upload a file to start")

    if 'data' in st.session_state:
        df = st.session_state['data']

        st.header("📋 Data Table Viewer")
        st.dataframe(df, use_container_width=True, height=400)

        if selected_variable:
            st.header("📈 Statistik Deskriptif")
            try:
                stats = stats_service.get_statistics(selected_variable)
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Mean", f"{stats['mean']:.2f}")
                c2.metric("Median", f"{stats['median']:.2f}")
                c3.metric("Min", f"{stats['min']:.2f}")
                c4.metric("Max", f"{stats['max']:.2f}")
                c5.metric("Std Dev", f"{stats['std']:.2f}")
            except Exception as e:
                st.error(f"Error calculating statistics: {e}")

            st.header("🗺️ Visualisasi Data")
            col_map, col_chart = st.columns([2, 1])

            with col_map:
                st.subheader("Choropleth Map - Persebaran Regional")
                try:
                    merged_gdf = data_service.get_merged_geodata()
                    if merged_gdf is not None and selected_variable in merged_gdf.columns:
                        gdf_vals = merged_gdf[merged_gdf[selected_variable].notna()].copy()
                        if len(gdf_vals) > 0:
                            bounds = gdf_vals.total_bounds
                            center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
                            m = folium.Map(location=center, zoom_start=8, tiles='OpenStreetMap')
                            colormap = LinearColormap(
                                colors=['#440154', '#2a788e', '#22a884', '#7ad151', '#fde724'],
                                vmin=float(gdf_vals[selected_variable].min()),
                                vmax=float(gdf_vals[selected_variable].max()),
                                caption=selected_variable,
                            )
                            folium.GeoJson(
                                gdf_vals,
                                style_function=lambda f: {
                                    'fillColor': colormap(f['properties'][selected_variable]) if f['properties'].get(selected_variable) else '#cccccc',
                                    'color': 'white', 'weight': 1, 'fillOpacity': 0.7,
                                },
                                tooltip=folium.GeoJsonTooltip(
                                    fields=['NAMOBJ', selected_variable],
                                    aliases=['Wilayah:', f'{selected_variable}:'],
                                    localize=True,
                                ),
                            ).add_to(m)
                            colormap.add_to(m)
                            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
                            components.html(m._repr_html_(), height=500, scrolling=True)
                        else:
                            st.warning(f"No data available for variable '{selected_variable}'")
                    else:
                        st.warning("Geodata not available or variable not found")
                except Exception as e:
                    st.error(f"Error creating map: {e}")

            with col_chart:
                st.subheader("Top 10 Wilayah")
                try:
                    chart_data = stats_service.get_chart_data(selected_variable, top_n=10)
                    chart_df = pd.DataFrame({'Wilayah': chart_data['regions'], selected_variable: chart_data['values']})
                    fig = px.bar(
                        chart_df, y='Wilayah', x=selected_variable, orientation='h',
                        title=f'Top 10 Wilayah - {selected_variable}',
                        labels={'Wilayah': 'Kabupaten/Kota', selected_variable: selected_variable},
                        color=selected_variable, color_continuous_scale='Viridis',
                    )
                    fig.update_layout(height=500, showlegend=False, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating chart: {e}")
    else:
        st.info("👆 Silakan upload file CSV atau Excel di menu sidebar untuk memulai.")


# ── PAGE: PREDICTION ──────────────────────────────────────────────────────────

elif page == "Prediction":
    st.title("Prediksi Model")

    if 'data' not in st.session_state:
        st.warning("⚠️ Silakan import data terlebih dahulu di halaman 'Import & Exploration'.")
    else:
        df = st.session_state['data']

        with st.sidebar:
            st.divider()
            st.header("⚙️ Konfigurasi Model")
            selected_model = st.selectbox("Pilih Model:", model_service.get_available_models())
            st.write("")
            run_button = st.button("🚀 Run Prediction", type="primary", use_container_width=True)

        if run_button:
            with st.spinner(f"Menjalankan prediksi dengan {selected_model}..."):
                try:
                    predictions, probabilities = model_service.predict(df, selected_model)
                    st.session_state.update({'predictions': predictions, 'probabilities': probabilities, 'last_model': selected_model})
                    st.success("✅ Prediksi selesai!")
                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")

        if 'predictions' in st.session_state:
            preds = st.session_state['predictions']
            probs = st.session_state['probabilities']

            st.subheader("📊 Hasil Prediksi")
            st.info(f"Rata-rata Probabilitas: {probs.mean():.4f}")

            col_map, col_metrics = st.columns([2, 1])

            def _make_folium_map(geojson, bounds, colormap, variable):
                center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
                m = folium.Map(location=center, zoom_start=8, tiles='OpenStreetMap')
                folium.GeoJson(
                    geojson,
                    style_function=lambda f: {
                        'fillColor': colormap(f['properties'][variable]) if f['properties'].get(variable) is not None else '#cccccc',
                        'color': 'white', 'weight': 1, 'fillOpacity': 0.7,
                    },
                    tooltip=folium.GeoJsonTooltip(fields=['NAMOBJ', variable], aliases=['Wilayah:', f'{variable}:'], localize=True),
                ).add_to(m)
                colormap.add_to(m)
                m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
                return m

            with col_map:
                region_col = data_service.region_column
                if region_col:
                    df_viz = df.copy()
                    df_viz['Pred_Class_Logit'] = preds
                    df_viz['Pred_Prob_Logit'] = probs

                    try:
                        bounds = geo_service.get_geodata_info()['bounds']

                        st.markdown("### 🗺️ Peta Sebaran Kelas Prediksi")
                        with st.spinner("Membuat Peta Prediksi..."):
                            cdata = geo_service.create_choropleth_data(df_viz, 'Pred_Class_Logit', region_col)
                            cm_class = LinearColormap(colors=['#2c7bb6', '#d7191c'], vmin=0, vmax=1, caption='Pred_Class_Logit (0 vs 1)')
                            m1 = _make_folium_map(cdata['geojson'], bounds, cm_class, 'Pred_Class_Logit')
                            components.html(m1._repr_html_(), height=500)

                        st.write("")
                        st.markdown("### 🗺️ Peta Sebaran Probabilitas")
                        with st.spinner("Membuat Peta Probabilitas..."):
                            cdata_prob = geo_service.create_choropleth_data(df_viz, 'Pred_Prob_Logit', region_col)
                            cm_prob = LinearColormap(colors=['#ffffcc', '#a1dab4', '#41b6c4', '#2c7fb8', '#253494'], vmin=0, vmax=1, caption='Pred_Prob_Logit (Probabilitas)')
                            m2 = _make_folium_map(cdata_prob['geojson'], bounds, cm_prob, 'Pred_Prob_Logit')
                            components.html(m2._repr_html_(), height=500)
                    except Exception as e:
                        st.error(f"Gagal menampilkan peta: {e}")
                else:
                    st.warning("Tidak dapat mendeteksi kolom wilayah untuk peta.")

            with col_metrics:
                st.markdown("### 📏 Evaluasi Model")
                ground_truth_col = next((c for c in ['p1_encoded', 'P1_encoded'] if c in df.columns), None)
                if ground_truth_col:
                    try:
                        metrics = model_service.calculate_metrics(df[ground_truth_col], preds)
                        metrics_df = pd.DataFrame([{'Model': selected_model, **metrics}]).round(4)
                        st.dataframe(metrics_df, hide_index=True, use_container_width=True)
                    except Exception as e:
                        st.error(f"Gagal menghitung metrik: {e}")
                else:
                    st.warning("⚠️ Kolom `p1_encoded` tidak ditemukan. Evaluasi tidak tersedia.")

                st.divider()
                st.markdown("### 📋 Tabel Hasil Prediksi")
                region_col = data_service.region_column or 'Region'
                result_df = pd.DataFrame({'Pred Prob Logit': probs, 'Pred Class Logit': preds})

                detected_col = data_service.region_column
                if not detected_col:
                    candidates = ['kabupaten_kota', 'Kabupaten/Kota', 'region', 'wilayah', 'daerah']
                    detected_col = next((c for c in df.columns if c.lower() in [x.lower() for x in candidates]), None)

                if detected_col and detected_col in df.columns:
                    result_df.insert(0, 'Kabupaten/Kota', df[detected_col])
                elif df.index.dtype in ('object', 'string'):
                    result_df.insert(0, 'Kabupaten/Kota', df.index)
                else:
                    result_df.insert(0, 'Index', df.index)

                st.dataframe(result_df, use_container_width=True, height=500)

            st.divider()

            # Recommendations
            with st.container():
                st.markdown("### 💡 Rekomendasi Kebijakan")
                rec_df = pd.DataFrame({'Region': result_df[result_df.columns[0]], 'Class': result_df['Pred Class Logit']})
                total_regions = len(rec_df)
                class_1_count = (rec_df['Class'] == 1).sum()
                class_0_count = (rec_df['Class'] == 0).sum()
                class_1_pct = class_1_count / total_regions * 100 if total_regions > 0 else 0

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Wilayah", total_regions)
                c2.metric("Kelas 1 (Tinggi)", class_1_count, delta=f"{class_1_pct:.1f}%", delta_color="inverse")
                c3.metric("Kelas 0 (Rendah)", class_0_count, delta=f"{100 - class_1_pct:.1f}%" if total_regions > 0 else "0%", delta_color="normal")
                if class_1_count > class_0_count:
                    c4.metric("Status", "Perlu Perhatian", delta="Mayoritas Tinggi", delta_color="inverse")
                else:
                    c4.metric("Status", "Cukup Baik", delta="Mayoritas Rendah", delta_color="normal")

                st.write("")
                class_1_regions = rec_df[rec_df['Class'] == 1]['Region'].tolist()
                class_0_regions = rec_df[rec_df['Class'] == 0]['Region'].tolist()

                def _show_regions(regions):
                    cols = st.columns(5)
                    for i, r in enumerate(regions):
                        cols[i % 5].markdown(f"• {r}")

                with st.expander(f"🚨 **Daerah Prioritas Tinggi** ({class_1_count})", expanded=True):
                    if class_1_count > 0:
                        st.markdown("**Daftar Daerah:**")
                        _show_regions(class_1_regions)
                        st.markdown("---")
                        st.markdown("""#### 🎯 Rekomendasi Intervensi
**1. Bantuan Sosial Intensif**
- Tingkatkan cakupan bansos dan validasi data penerima.
- Prioritaskan keluarga rentan dengan dependency ratio tinggi.

**2. Akses Layanan Dasar**
- Perbaiki infrastruktur rumah tidak layak huni (RTLH).
- Tingkatkan akses sanitasi dan air bersih.

**3. Ekonomi Lokal**
- Pelatihan kerja berbasis kompetensi lokal.
- Fasilitasi akses modal dan pasar untuk UMKM.""")
                    else:
                        st.success("✅ Tidak ada!")

                with st.expander(f"✅ **Daerah Status Baik** ({class_0_count})", expanded=True):
                    if class_0_count > 0:
                        st.markdown("**Daftar Daerah:**")
                        _show_regions(class_0_regions)
                        st.markdown("---")
                        st.markdown("""#### 🎯 Rekomendasi Pemeliharaan
**1. Pertahankan Program**
- Lanjutkan program efektif dan dokumentasikan praktik baik.

**2. Pencegahan**
- Monitor indikator dini untuk mencegah penurunan status.
- Siapkan jaring pengaman sosial adaptif.

**3. Peningkatan**
- Tingkatkan kualitas layanan publik digital.
- Dorong inovasi dan investasi hijau.""")
                    else:
                        st.warning("⚠️ Perlu perhatian!")

                with st.expander("📋 **Rekomendasi Umum**", expanded=True):
                    st.markdown("""#### 🔄 Monitoring & Evaluasi
**1. Update Data Berkala**: Lakukan pengumpulan data minimal 6 bulan sekali.
**2. Evaluasi Program**: Ukur dampak program terhadap indikator kemiskinan secara rutin.
**3. Kolaborasi**: Sinergi program antara pemerintah provinsi, kabupaten, dan desa.
**4. Analisis Spasial**: Manfaatkan peta kerawanan untuk targeting program yang lebih presisi.""")



st.divider()
st.caption("© 2026 Sistem Analitik Data | Poverty Depth Index Spatial Analysis")
