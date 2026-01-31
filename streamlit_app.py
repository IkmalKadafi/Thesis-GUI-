"""
Streamlit Data Exploration App
Menu 1: Import Data & Exploration
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from services.data_service import data_service
from services.stats_service import stats_service
import folium
from branca.colormap import LinearColormap
import streamlit.components.v1 as components

# Page config
st.set_page_config(
    page_title="Menu 1: Import Data & Exploration",
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
st.markdown('<div class="main-header">📊 Menu 1: Import Data & Exploration</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Sistem Analitik Berbasis Python</div>', unsafe_allow_html=True)

# Sidebar
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
            selected_variable = None
    else:
        st.info("Upload a file to start")
        selected_variable = None

# Main content
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
                import traceback
                st.code(traceback.format_exc())
        
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
                
                # Use Plotly for horizontal bar chart
                import plotly.express as px
                
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
    st.info("👆 Upload a CSV or Excel file from the sidebar to start exploring your data")
    
    st.markdown("""
    ### Features:
    - 📁 **File Upload**: Support for CSV and Excel files
    - 📋 **Interactive Table**: View and search your data
    - 🔍 **Variable Filtering**: Select numeric variables to analyze
    - 📊 **Statistics**: Mean, median, min, max, standard deviation
    - 🗺️ **Choropleth Map**: Geographic visualization for Jawa Tengah
    - 📈 **Bar Chart**: Top 10 regions comparison
    
    ### Supported Data:
    - Regional data for Jawa Tengah (35 kabupaten/kota)
    - Automatic geodata matching and merging
    - Multiple numeric variables for analysis
    """)

# Footer
st.divider()
st.caption("© 2026 Sistem Analitik Data | Menu 1: Import & Exploration")
