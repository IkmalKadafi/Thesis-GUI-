"""
Streamlit Data Exploration App
Menu 1: Import Data & Exploration
Menu 2: Prediction
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from backend.services import data_service, stats_service, model_service, geo_service
import folium
from branca.colormap import LinearColormap
import streamlit.components.v1 as components
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Poverty Depth Index Spatial Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
# st.markdown('<div class="main-header">📊 Poverty Depth Index Spatial Analysis</div>', unsafe_allow_html=True) # Moved to Home
# st.markdown('<div class="sub-header">Sistem Analitik Spasial</div>', unsafe_allow_html=True) # Moved to Home

from streamlit_option_menu import option_menu

# Sidebar Menu
with st.sidebar:
    # st.image("https://img.icons8.com/clouds/100/000000/statistics.png", width=80) 
    
    # Custom Menu
    page = option_menu(
        menu_title="Menu Navigasi",  # Required
        options=["Homepage", "Import & Exploration", "Prediction"],  # Required
        icons=["house", "cloud-upload", "graph-up-arrow"],  # Optional
        menu_icon="cast",  # Optional
        default_index=0,  # Optional
        styles={
            "container": {"padding": "0!important", "background-color": "#262730"},
            "icon": {"color": "orange", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#1f77b4"},
        }
    )
    
    st.divider()
    st.caption("© 2026 • Poverty Depth Index Spatial Analysis")

# PAGE: HOMEPAGE (BERANDA)
if page == "Homepage":
    # Hero Section
    st.markdown('<div class="main-header">📊 Poverty Depth Index Spatial Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Sistem Analitik Spasial untuk Provinsi Jawa Tengah</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Grid Layout for Options
    col_home_1, col_home_2 = st.columns(2)
    
    with col_home_1:
        st.info("### 📂 Langkah 1: Import Data")
        st.write("""
        Mulailah dengan mengupload data Indeks Kedalaman Kemiskinan.
        
        **Fitur:**
        - Upload file CSV/Excel.
        - Eksplorasi Data Tabular.
        - Statistik Deskriptif Otomatis.
        - Visualisasi Awal (Peta & Grafik).
        """)
    
    with col_home_2:
        st.success("### 🔮 Langkah 2: Prediksi")
        st.write("""
        Lakukan prediksi menggunakan model Machine Learning yang telah dilatih.
        
        **Model Tersedia:**
        - Global Logistic Regression.
        - Geographically Weighted Logistic Regression (GWLR).
        - MGWLR Semiparametric.
        """)

    st.write("")
    st.write("")
    
    # About Section
    with st.expander("ℹ️ Tentang Aplikasi Ini", expanded=True):
        st.write("""
        Aplikasi ini dikembangkan untuk mempermudah analisis spasial terkait kemiskinan di Jawa Tengah.
        Menggabungkan analisis statistik deskriptif dan pemodelan prediktif spasial untuk memberikan wawasan yang lebih mendalam.
        
        **Dikembangkan oleh:** Ikmal Thariq Kadafi
        """)

# PAGE 1: IMPORT & EXPLORATION
elif page == "Import & Exploration":
    st.markdown('<div class="main-header">📂 Import & Eksplorasi Data</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("📁 Import Data")
        uploaded_file = st.file_uploader(
            "Upload CSV atau Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Format: CSV, Excel (.xlsx, .xls) | Max: 200MB"
        )
        
        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                temp_path = Path("backend/data/uploads") / uploaded_file.name
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load data
                df = data_service.load_file(temp_path)
                st.session_state['data'] = df
                st.session_state['file_name'] = uploaded_file.name
                
                # Get merge statistics
                merge_stats = data_service.get_merge_statistics()
                
                st.success(f"✅ File uploaded successfully!")
                st.info(f"📊 {len(df)} rows × {len(df.columns)} columns")
                if merge_stats:
                    st.info(f"🗺️ Geodata merge: {merge_stats.get('matched_regions', 0)}/{merge_stats.get('total_geodata_regions', 0)} regions ({merge_stats.get('match_rate', 0)}%)")
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
        
        st.divider()
        
        # Variable selector
        selected_variable = None
        if 'data' in st.session_state:
            st.header("🔍 Filter Variabel")
            numeric_cols = data_service.get_numeric_columns()
            
            if numeric_cols:
                selected_variable = st.selectbox(
                    "Pilih Variabel:",
                    options=numeric_cols,
                    key='selected_variable'
                )
            else:
                st.warning("No numeric columns found")
        else:
            st.info("Upload a file to start")

    # Main Content Page 1
    if 'data' in st.session_state:
        df = st.session_state['data']
        
        # Data Table Section
        st.header("📋 Data Table Viewer")
        st.dataframe(
            df,
            use_container_width=True,
            height=400
        )
        
        # Statistics Section
        if selected_variable:
            st.header("📈 Statistik Deskriptif")
            
            try:
                stats = stats_service.get_statistics(selected_variable)
                
                # Display metrics in columns
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("📊 Mean", f"{stats['mean']:.2f}")
                with col2:
                    st.metric("📍 Median", f"{stats['median']:.2f}")
                with col3:
                    st.metric("⬇️ Min", f"{stats['min']:.2f}")
                with col4:
                    st.metric("⬆️ Max", f"{stats['max']:.2f}")
                with col5:
                    st.metric("📏 Std Dev", f"{stats['std']:.2f}")
                    
            except Exception as e:
                st.error(f"Error calculating statistics: {str(e)}")
            
            # Visualizations Section
            st.header("🗺️ Visualisasi Data")
            
            col_map, col_chart = st.columns([2, 1])
            
            # Choropleth Map
            with col_map:
                st.subheader("Choropleth Map - Persebaran Regional")
                
                try:
                    merged_gdf = data_service.get_merged_geodata()
                    
                    if merged_gdf is not None and selected_variable in merged_gdf.columns:
                        # Filter data with values
                        gdf_with_values = merged_gdf[merged_gdf[selected_variable].notna()].copy()
                        
                        if len(gdf_with_values) > 0:
                            # Calculate map center
                            bounds = gdf_with_values.total_bounds
                            center_lat = (bounds[1] + bounds[3]) / 2
                            center_lon = (bounds[0] + bounds[2]) / 2
                            
                            # Create Folium map
                            m = folium.Map(
                                location=[center_lat, center_lon],
                                zoom_start=8,
                                tiles='OpenStreetMap'
                            )
                            
                            # Create color scale
                            min_val = float(gdf_with_values[selected_variable].min())
                            max_val = float(gdf_with_values[selected_variable].max())
                            
                            colormap = LinearColormap(
                                colors=['#440154', '#2a788e', '#22a884', '#7ad151', '#fde724'],
                                vmin=min_val,
                                vmax=max_val,
                                caption=selected_variable
                            )
                            
                            # Add GeoJSON layer
                            folium.GeoJson(
                                gdf_with_values,
                                style_function=lambda feature: {
                                    'fillColor': colormap(feature['properties'][selected_variable]) 
                                        if feature['properties'].get(selected_variable) else '#cccccc',
                                    'color': 'white',
                                    'weight': 1,
                                    'fillOpacity': 0.7
                                },
                                tooltip=folium.GeoJsonTooltip(
                                    fields=['NAMOBJ', selected_variable],
                                    aliases=['Wilayah:', f'{selected_variable}:'],
                                    localize=True
                                )
                            ).add_to(m)
                            
                            # Add color scale legend
                            colormap.add_to(m)
                            
                            # Fit bounds
                            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
                            
                            # Display map using HTML
                            map_html = m._repr_html_()
                            components.html(map_html, height=500, scrolling=True)
                        else:
                            st.warning(f"No data available for variable '{selected_variable}'")
                    else:
                        st.warning("Geodata not available or variable not found")
                        
                except Exception as e:
                    st.error(f"Error creating map: {str(e)}")
            
            # Bar Chart
            with col_chart:
                st.subheader("Top 10 Wilayah")
                
                try:
                    chart_data = stats_service.get_chart_data(selected_variable, top_n=10)
                    
                    # Create DataFrame for chart
                    chart_df = pd.DataFrame({
                        'Wilayah': chart_data['regions'],
                        selected_variable: chart_data['values']
                    })
                    
                    fig = px.bar(
                        chart_df,
                        y='Wilayah',  # Wilayah on Y-axis for horizontal bars
                        x=selected_variable,  # Values on X-axis
                        orientation='h',  # Horizontal orientation
                        title=f'Top 10 Wilayah - {selected_variable}',
                        labels={'Wilayah': 'Kabupaten/Kota', selected_variable: selected_variable},
                        color=selected_variable,
                        color_continuous_scale='Viridis'
                    )
                    
                    # Update layout for better appearance
                    fig.update_layout(
                        height=500,
                        showlegend=False,
                        yaxis={'categoryorder': 'total ascending'}  # Sort by value
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error creating chart: {str(e)}")
    else:
        # Welcome message
        st.info("👆 Silakan upload file CSV atau Excel di menu sidebar untuk memulai.")


# PAGE 2: PREDICTION

elif page == "Prediction":
    st.title("🔮 Prediksi Model")
    
    if 'data' not in st.session_state:
        st.warning("⚠️ Silakan import data terlebih dahulu di halaman 'Import & Exploration'.")
    else:
        df = st.session_state['data']
        
        # --- Sidebar: Model Selection & Execution ---
        with st.sidebar:
            st.divider()
            st.header("⚙️ Konfigurasi Model")
            
            available_models = model_service.get_available_models()
            selected_model = st.selectbox(
                "Pilih Model:", 
                available_models,
            )
            
            st.write("") # Spacer
            run_button = st.button("🚀 Run Prediction", type="primary", use_container_width=True)
        
        # --- Execution ---
        if run_button:
            with st.spinner(f"Menjalankan prediksi dengan {selected_model}..."):
                try:
                    # Run prediction
                    predictions, probabilities = model_service.predict(df, selected_model)
                    
                    # Save results to session state to persist
                    st.session_state['predictions'] = predictions
                    st.session_state['probabilities'] = probabilities
                    st.session_state['last_model'] = selected_model
                    
                    st.success("✅ Prediksi selesai!")
                    
                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan saat prediksi: {str(e)}")
        
        # --- Results Section ---
        if 'predictions' in st.session_state:
            preds = st.session_state['predictions']
            probs = st.session_state['probabilities']
            
            # --- Results: Probability ---
            st.subheader("📊 Hasil Prediksi")
            
            col_prob_info, col_metrics_info = st.columns(2)
            with col_prob_info:
                 st.info(f"Rata-rata Probabilitas: {probs.mean():.4f}")
            
            
            # --- Layout: Map (Left) & Metrics (Right) ---
            col_res_map, col_res_metrics = st.columns([2, 1])
            
            # 1. Map p1_predicted
            with col_res_map:
                st.markdown("### 🗺️ Peta Sebaran Pred_Class_Logit (Kelas Prediksi)")
                
                try:
                    # Add prediction results to dataframe for mapping
                    df_viz = df.copy()
                    df_viz['Pred_Class_Logit'] = preds # Using Class Prediction (0/1) as requested
                    
                    # Merge with geodata
                    # Note: We need to use data_service to merge effectively, 
                    # but we can't easily inject the new column into the service without reloading.
                    # So we use the helper logic here.
                    
                    # from services.geo_service import geo_service
                    # geo_service is already imported at top level from backend.services
                    # Assuming we have a region column. Let's try to detect it from the previous load.
                    region_col = data_service.region_column
                    
                    if region_col:
                        choropleth_data = geo_service.create_choropleth_data(
                            df_viz, 
                            variable='Pred_Class_Logit',
                            region_col=region_col
                        )
                        
                        geojson = choropleth_data['geojson']
                        
                        # Create Map
                        bounds = geo_service.get_geodata_info()['bounds']
                        center_lat = (bounds[1] + bounds[3]) / 2
                        center_lon = (bounds[0] + bounds[2]) / 2
                        
                        m_pred = folium.Map(
                            location=[center_lat, center_lon],
                            zoom_start=8,
                            tiles='OpenStreetMap'
                        )
                        
                        # Color scale for Binary (0 vs 1)
                        # 0 = Low Risk/Negative (e.g., Blue/Green), 1 = High Risk/Positive (e.g., Red/Orange)
                        colormap = LinearColormap(
                            colors=['#2c7bb6', '#d7191c'], # Blue to Red
                            vmin=0,
                            vmax=1,
                            caption='Pred_Class_Logit (0 vs 1)'
                        )
                        
                        folium.GeoJson(
                            geojson,
                            style_function=lambda feature: {
                                'fillColor': colormap(feature['properties']['Pred_Class_Logit']) 
                                    if feature['properties'].get('Pred_Class_Logit') is not None else '#cccccc',
                                'color': 'white',
                                'weight': 1,
                                'fillOpacity': 0.7
                            },
                            tooltip=folium.GeoJsonTooltip(
                                fields=['NAMOBJ', 'Pred_Class_Logit'],
                                aliases=['Wilayah:', 'Kelas Prediksi:'],
                                localize=True
                            )
                        ).add_to(m_pred)
                        
                        colormap.add_to(m_pred)
                        m_pred.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
                        
                        components.html(m_pred._repr_html_(), height=500)
                        
                        # --- Second Map: Pred_Prob_Logit (Probability) ---
                        st.write("")
                        st.write("")
                        st.write("")
                        st.markdown("### 🗺️ Peta Sebaran Pred_Prob_Logit (Probabilitas)")
                        
                        df_viz['Pred_Prob_Logit'] = probs
                        choropleth_data_prob = geo_service.create_choropleth_data(
                            df_viz, 
                            variable='Pred_Prob_Logit',
                            region_col=region_col
                        )
                        geojson_prob = choropleth_data_prob['geojson']
                        
                        m_prob = folium.Map(
                            location=[center_lat, center_lon],
                            zoom_start=8,
                            tiles='OpenStreetMap'
                        )
                        
                        # Color scale for Probability (0 to 1) - Gradient
                        colormap_prob = LinearColormap(
                            colors=['#ffffcc', '#a1dab4', '#41b6c4', '#2c7fb8', '#253494'],
                            vmin=0,
                            vmax=1,
                            caption='Pred_Prob_Logit (Probabilitas)'
                        )
                        
                        folium.GeoJson(
                            geojson_prob,
                            style_function=lambda feature: {
                                'fillColor': colormap_prob(feature['properties']['Pred_Prob_Logit']) 
                                    if feature['properties'].get('Pred_Prob_Logit') is not None else '#cccccc',
                                'color': 'white',
                                'weight': 1,
                                'fillOpacity': 0.7
                            },
                            tooltip=folium.GeoJsonTooltip(
                                fields=['NAMOBJ', 'Pred_Prob_Logit'],
                                aliases=['Wilayah:', 'Probabilitas:'],
                                localize=True
                            )
                        ).add_to(m_prob)
                        
                        colormap_prob.add_to(m_prob)
                        m_prob.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
                        components.html(m_prob._repr_html_(), height=500)
                        
                    else:
                        st.warning("Tidak dapat mendeteksi kolom wilayah untuk peta.")
                        
                except Exception as e:
                    st.error(f"Gagal menampilkan peta: {e}")
            
            # 2. Metrics (Evaluation) & Result Table
            with col_res_metrics:
                st.markdown("### 📏 Evaluasi Model")
                
                # Check for ground truth column (case insensitive)
                ground_truth_col = None
                if 'p1_encoded' in df.columns:
                    ground_truth_col = 'p1_encoded'
                elif 'P1_encoded' in df.columns:
                    ground_truth_col = 'P1_encoded'
                
                if ground_truth_col:
                    try:
                        # Calculate metrics
                        y_true = df[ground_truth_col]
                        y_pred = preds
                        
                        metrics = model_service.calculate_metrics(y_true, y_pred)
                        
                        # Display Metrics as Table
                        # Match user request: Model, Accuracy, Precision, Recall, F1-Score
                        metrics_df = pd.DataFrame([{
                            'Model': selected_model,
                            'Accuracy': metrics['Accuracy'],
                            'Precision': metrics['Precision'],
                            'Recall': metrics['Recall'],
                            'F1-Score': metrics['F1-Score']
                        }]).round(4)
                        
                        st.dataframe(metrics_df, hide_index=True, use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"Gagal menghitung metrik: {e}")
                else:
                    st.warning("⚠️ Kolom `p1_encoded` tidak ditemukan. Evaluasi tidak tersedia.")

                st.divider()
                st.markdown("### 📋 Tabel Hasil Prediksi")
                
                # Create Prediction Result Table
                # Needs: Kabupaten/Kota, Pred Prob Logit, Pred Class Logit
                # Try to find region column
                region_col = data_service.region_column
                if not region_col:
                     # Fallback: use index if it looks like regions, or just use index
                     region_col = 'Region'
                
                result_df = pd.DataFrame({
                    'Pred Prob Logit': probs,
                    'Pred Class Logit': preds
                })
                
                # Add region column if available in original df
                if data_service.region_column and data_service.region_column in df.columns:
                    result_df.insert(0, 'Kabupaten/Kota', df[data_service.region_column])
                else:
                    # If index is meaningful, use it
                    result_df.insert(0, 'Index', df.index)
                
                # Format formatting for display
                # Note: st.dataframe allows formatting but here we just round for simplicity
                
                st.dataframe(
                    result_df, 
                    use_container_width=True,
                    height=500
                )

# Footer
st.divider()
st.caption("© 2026 Sistem Analitik Data | Poverty Depth Index Spatial Analysis")
