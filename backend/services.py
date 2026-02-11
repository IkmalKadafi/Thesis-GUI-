"""
Unified Service Module
Contains: GeoService, ModelService, DataService, StatsService
"""
import pickle
import json
import pandas as pd
import numpy as np
import geopandas as gpd
import statsmodels.api as sm
import streamlit as st
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

# Unified Imports
from backend.settings import (
    MODEL_DIR, 
    JAWA_TENGAH_SHAPEFILE, 
    JAWA_TENGAH_GEOJSON_CACHE,
    GEOJSON_DIR
)
from backend.utils import (
    normalize_region_name, 
    validate_dataframe, 
    get_region_column, 
    ValidationError
)


# ==========================================
# GEO SERVICE
# ==========================================
# Cached function for loading shapefile
@st.cache_resource
def _load_shapefile_cached(shapefile_path: str) -> gpd.GeoDataFrame:
    """Load shapefile with caching for performance"""
    if not Path(shapefile_path).exists():
        raise FileNotFoundError(
            f"Shapefile not found: {shapefile_path}\n"
            "Please ensure 'Geodata Jawa Tengah' folder exists in project root."
        )
    
    try:
        # Load shapefile using geopandas
        gdf = gpd.read_file(shapefile_path)
        
        # Ensure it's in WGS84 (EPSG:4326) for web mapping
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        
        print(f"[OK] Loaded {len(gdf)} regions from Jawa Tengah shapefile (cached)")
        return gdf
        
    except Exception as e:
        raise RuntimeError(f"Error loading shapefile: {e}")


# Cached function for creating choropleth data
@st.cache_data
def _create_choropleth_data_cached(
    df: pd.DataFrame, 
    variable: str, 
    region_col: str,
    geojson_data: Dict[str, Any],
    region_mapping: Dict[str, int]
) -> Dict[str, Any]:
    """
    Create choropleth map data with caching
    
    Args:
        df: DataFrame with data
        variable: Column to visualize
        region_col: Column containing region names
        geojson_data: GeoJSON dictionary
        region_mapping: Dictionary mapping region names to feature indices
        
    Returns:
        Dict with updated GeoJSON and stats
    """
    # Create matching dataframe
    match_df = pd.DataFrame({
        'region': df[region_col],
        'value': df[variable],
        'region_normalized': df[region_col].apply(normalize_region_name)
    }).dropna(subset=['value'])
    
    # Match regions logic
    matched_values = {}
    unmatched_regions = []
    
    for _, row in match_df.iterrows():
        region_norm = row['region_normalized']
        region_orig = row['region']
        value = row['value']
        
        # Try to find matching GeoJSON feature
        feature_idx = None
        
        # Try exact normalized match first
        if region_norm in region_mapping:
            feature_idx = region_mapping[region_norm]
        # Try original name lowercase
        elif region_orig.lower().strip() in region_mapping:
            feature_idx = region_mapping[region_orig.lower().strip()]
        else:
            # Try fuzzy matching (contains check)
            for geo_region, idx in region_mapping.items():
                if region_norm in geo_region or geo_region in region_norm:
                    feature_idx = idx
                    break
        
        if feature_idx is not None:
            matched_values[feature_idx] = value
        else:
            unmatched_regions.append(region_orig)
            
    # Add values to GeoJSON features
    geojson_with_values = json.loads(json.dumps(geojson_data))  # Deep copy
    for feature_idx, value in matched_values.items():
        if feature_idx < len(geojson_with_values['features']):
            geojson_with_values['features'][feature_idx]['properties'][variable] = value
    
    return {
        "geojson": geojson_with_values,
        "values": matched_values,
        "variable": variable,
        "matched_count": len(matched_values),
        "total_regions": len(geojson_data['features']),
        "unmatched_regions": unmatched_regions,
        "match_rate": round(len(matched_values) / len(geojson_data['features']) * 100, 2)
    }

class GeoService:
    """Service for handling geospatial operations"""
    
    def __init__(self):
        self.geodataframe: Optional[gpd.GeoDataFrame] = None
        self.geojson_data: Dict[str, Any] = {}
        self.region_mapping: Dict[str, str] = {}
    
    def load_shapefile(self) -> gpd.GeoDataFrame:
        """Load Jawa Tengah shapefile (with caching)"""
        if self.geodataframe is None:
            # Use cached loader
            self.geodataframe = _load_shapefile_cached(str(JAWA_TENGAH_SHAPEFILE))
        
        return self.geodataframe
    
    def load_geojson(self) -> Dict[str, Any]:
        """Load or convert shapefile to GeoJSON"""
        if self.geojson_data:
            return self.geojson_data
        
        # Try to load cached GeoJSON first
        if JAWA_TENGAH_GEOJSON_CACHE.exists():
            try:
                with open(JAWA_TENGAH_GEOJSON_CACHE, 'r', encoding='utf-8') as f:
                    self.geojson_data = json.load(f)
                print(f"[OK] Loaded cached GeoJSON from {JAWA_TENGAH_GEOJSON_CACHE.name}")
                
                # Build region mapping
                self._build_region_mapping()
                
                return self.geojson_data
            except Exception as e:
                print(f"Warning: Could not load cached GeoJSON: {e}")
        
        # Load shapefile and convert to GeoJSON
        gdf = self.load_shapefile()
        
        # Convert to GeoJSON format
        self.geojson_data = json.loads(gdf.to_json())
        
        # Cache the GeoJSON for faster subsequent loads
        try:
            GEOJSON_DIR.mkdir(parents=True, exist_ok=True)
            with open(JAWA_TENGAH_GEOJSON_CACHE, 'w', encoding='utf-8') as f:
                json.dump(self.geojson_data, f, ensure_ascii=False, indent=2)
            print(f"[OK] Cached GeoJSON to {JAWA_TENGAH_GEOJSON_CACHE.name}")
        except Exception as e:
            print(f"Warning: Could not cache GeoJSON: {e}")
        
        # Build region mapping
        self._build_region_mapping()
        
        return self.geojson_data
    
    def _build_region_mapping(self):
        """Build mapping of normalized region names to GeoJSON feature indices"""
        if not self.geojson_data or 'features' not in self.geojson_data:
            return
        
        for idx, feature in enumerate(self.geojson_data['features']):
            if 'properties' in feature:
                # Get region name from NAMOBJ field
                region_name = feature['properties'].get('NAMOBJ')
                
                if region_name:
                    # Normalize the region name
                    normalized = normalize_region_name(region_name)
                    # Map to feature index for easy lookup
                    self.region_mapping[normalized] = idx
                    
                    # Also map the original name
                    self.region_mapping[region_name.lower().strip()] = idx
    
    def match_regions(self, df: pd.DataFrame) -> Tuple[Dict[int, float], List[str]]:
        """Match data regions with GeoJSON regions"""
        # Load GeoJSON if not loaded
        if not self.geojson_data:
            self.load_geojson()
        
        matched_values = {}
        unmatched_regions = []
        
        for _, row in df.iterrows():
            region_norm = row['region_normalized']
            region_orig = row['region']
            value = row['value']
            
            # Try to find matching GeoJSON feature
            feature_idx = None
            
            # Try exact normalized match first
            if region_norm in self.region_mapping:
                feature_idx = self.region_mapping[region_norm]
            # Try original name lowercase
            elif region_orig.lower().strip() in self.region_mapping:
                feature_idx = self.region_mapping[region_orig.lower().strip()]
            else:
                # Try fuzzy matching (contains check)
                for geo_region, idx in self.region_mapping.items():
                    if region_norm in geo_region or geo_region in region_norm:
                        feature_idx = idx
                        break
            
            if feature_idx is not None:
                matched_values[feature_idx] = float(value)
            else:
                unmatched_regions.append(region_orig)
        
        return matched_values, unmatched_regions
    
    def create_choropleth_data(
        self, 
        df: pd.DataFrame, 
        variable: str,
        region_col: str = None
    ) -> Dict[str, Any]:
        """Create choropleth map data (uses cached helper)"""
        # Load GeoJSON
        geojson = self.load_geojson()
        
        # Prepare data for matching
        if region_col is None:
            # Try to find region column
            region_candidates = ['kabupaten_kota', 'Kabupaten/Kota', 'region', 'wilayah', 'daerah']
            for col in region_candidates:
                if col in df.columns:
                    region_col = col
                    break
            
            if region_col is None:
                raise ValueError("Could not find region column in DataFrame")
        
        # Call cached helper
        return _create_choropleth_data_cached(
            df, 
            variable, 
            region_col, 
            self.geojson_data,
            self.region_mapping
        )
    
    def get_geodata_info(self) -> Dict[str, Any]:
        """Get information about loaded geodata"""
        gdf = self.load_shapefile()
        
        return {
            "total_regions": len(gdf),
            "region_names": sorted(gdf['NAMOBJ'].tolist()),
            "bounds": gdf.total_bounds.tolist(),  # [minx, miny, maxx, maxy]
            "crs": str(gdf.crs),
            "columns": gdf.columns.tolist()
        }


# ==========================================
# MODEL SERVICE
# ==========================================

# Cached function for loading models
@st.cache_resource
def _load_model_cached(model_path: str) -> Any:
    """Load model from pickle with caching for performance"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            print(f"[OK] Loaded model: {Path(model_path).name} (cached)")
            return model
    except Exception as e:
        raise RuntimeError(f"Error loading model {Path(model_path).name}: {e}")


class ModelService:
    """Service for model loading and prediction"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        # Mapping: Display Name -> Filename
        self.model_mapping = {
            "Model Global Logistic Regression": "model_logistik_global.pkl",
            "Model Geographically Weighted Logistic Regression": "gwlr_model.pkl",
            "Model Geographically Weighted Logistic Regression Semiparametric": "mgwlr_model.pkl"
        }
    
    def load_model(self, model_identifier: str) -> Any:
        """Load a model from disk (with caching)"""
        # Resolve filename if display name is passed
        filename = self.model_mapping.get(model_identifier, model_identifier)
        
        model_path = MODEL_DIR / filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Use cached loader
        return _load_model_cached(str(model_path))
    
    def get_available_models(self) -> List[str]:
        """Get list of available model display names"""
        return list(self.model_mapping.keys())
    
    def _predict_gwlr(self, model_data: Dict, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Predict using GWLR model with local parameters
        
        Args:
            model_data: Dictionary containing GWLR model components
                - params: Local parameters (n x k+1) array
                - scaler: StandardScaler object
                - predictor: List of predictor column names
                - bandwidth: Optimal bandwidth
                - target: Target column name
            df: DataFrame with input data
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        # Extract model components
        # Handle both model structures:
        # 1. Original user's code: {'results': GWR_results_object, ...}
        # 2. Converted model: {'params': numpy_array, ...}
        if 'results' in model_data:
            # Original structure from user's code
            results = model_data['results']
            params = results.params  # Extract params from results object
        elif 'params' in model_data:
            # Converted structure
            params = model_data['params']
        else:
            raise ValueError("Model must have either 'results' or 'params' key")
            
        scaler = model_data['scaler']  # StandardScaler
        predictor_cols = model_data['predictor']  # List of predictor column names
        
        # Preprocessing: Create required columns if they don't exist
        df_processed = df.copy()
        
        # CRITICAL: Check if all predictor columns already exist
        # If yes, SKIP preprocessing to avoid corrupting correct data
        all_cols_exist = all(col in df_processed.columns for col in predictor_cols)
        
        print(f"[GWLR DEBUG] Predictor columns needed: {predictor_cols}")
        print(f"[GWLR DEBUG] All columns exist: {all_cols_exist}")
        print(f"[GWLR DEBUG] Available columns: {list(df_processed.columns)}")
        
        if all_cols_exist:
            print(f"[GWLR DEBUG] All predictor columns already exist, skipping preprocessing")
        else:
            print(f"[GWLR DEBUG] Some predictor columns missing, applying preprocessing...")
            
            # Column mapping and creation (only if needed)
            # DepRatio: Dependency Ratio
            if 'DepRatio' not in df_processed.columns:
                if 'Jumlah Penduduk' in df_processed.columns:
                    df_processed['DepRatio'] = 50.0  # Placeholder value
                else:
                    df_processed['DepRatio'] = 50.0 # Force placeholder
                    print("[GWLR DEBUG] Warning: 'DepRatio' missing, using placeholder 50.0")
            
            # Industri: Industrial data
            if 'Industri' not in df_processed.columns:
                if 'PDRB ' in df_processed.columns:
                    df_processed['Industri'] = df_processed['PDRB '].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
                else:
                    print("[GWLR DEBUG] Error: 'Industri' (PDRB) not found")
            
            # RumahLayak: Decent housing
            if 'RumahLayak' not in df_processed.columns:
                if 'Sanitasi Layak' in df_processed.columns:
                    df_processed['RumahLayak'] = df_processed['Sanitasi Layak'].astype(str).str.replace(',', '.', regex=False).astype(float)
                else:
                    print("[GWLR DEBUG] Error: 'RumahLayak' not found")
            
            # Sanitasi: Sanitation
            if 'Sanitasi' not in df_processed.columns:
                if 'Sanitasi Layak' in df_processed.columns:
                    df_processed['Sanitasi'] = df_processed['Sanitasi Layak'].astype(str).str.replace(',', '.', regex=False).astype(float)
                else:
                    print("[GWLR DEBUG] Error: 'Sanitasi' not found")
            
            # TPT: Tingkat Pengangguran Terbuka
            if 'TPT' not in df_processed.columns:
                if 'Tingkat Pengangguran Terbuka (TPT)' in df_processed.columns:
                    df_processed['TPT'] = df_processed['Tingkat Pengangguran Terbuka (TPT)'].astype(str).str.replace(',', '.').astype(float)
                else:
                    print("[GWLR DEBUG] Error: 'TPT' not found")
            
            # UMK: Upah Minimum Kabupaten/Kota
            if 'UMK' not in df_processed.columns:
                if 'Upah Minimum Kabupaten/Kota (UMK)' in df_processed.columns:
                    df_processed['UMK'] = df_processed['Upah Minimum Kabupaten/Kota (UMK)'].astype(str).str.replace('.', '').str.replace(',', '.').astype(float)
                else:
                    print("[GWLR DEBUG] Error: 'UMK' not found")
        
        # CRITICAL: Prepare for re-indexing
        # df_processed needs to track the original index so we can return predictions in correct order
        df_processed = df_processed.copy()
        df_processed['_original_index'] = df.index
        
        # CRITICAL: Merge with shapefile and sort by ORDER
        # This ensures data order matches training data order
        # User's original code does this merge and sort before training
        print(f"[GWLR DEBUG] Starting shapefile merge...")
        try:
            import geopandas as gpd
            from pathlib import Path
            
            shapefile_path = Path(__file__).parent.parent / "Geodata Jawa Tengah" / "JawaTengah.shp"
            print(f"[GWLR DEBUG] Shapefile path: {shapefile_path}")
            print(f"[GWLR DEBUG] Shapefile exists: {shapefile_path.exists()}")
            
            if shapefile_path.exists():
                # Load shapefile
                gdf = gpd.read_file(shapefile_path)
                gdf['ORDER'] = gdf.index
                print(f"[GWLR DEBUG] Loaded shapefile with {len(gdf)} regions")
                
                # Determine which column to use for merging
                # Try 'Kabupaten/Kota' first, then 'NAMOBJ'
                merge_col = None
                if 'Kabupaten/Kota' in df_processed.columns:
                    merge_col = 'Kabupaten/Kota'
                elif 'NAMOBJ' in df_processed.columns:
                    merge_col = 'NAMOBJ'
                else:
                    # If neither exists, skip merge and hope data is already sorted
                    print("Warning: No merge column found, using data as-is")
                    df_final = df_processed
                
                if merge_col:
                    print(f"[GWLR DEBUG] Merging on column: {merge_col}")
                    # Merge with shapefile
                    df_final = gdf.merge(df_processed, left_on="NAMOBJ", right_on=merge_col, how='inner')
                    print(f"[GWLR DEBUG] After merge: {len(df_final)} rows")
                    
                    # Sort by ORDER (critical for GWLR!)
                    df_final = df_final.sort_values("ORDER").reset_index(drop=True)
                    print(f"[GWLR DEBUG] After sort: {len(df_final)} rows")
                    print(f"[GWLR DEBUG] First 5 kabupaten: {df_final['Kabupaten/Kota'].head().tolist()}")
                    
                    print(f"[GWLR DEBUG] Merged with shapefile and sorted by ORDER")
                else:
                    df_final = df_processed
            else:
                print(f"Warning: Shapefile not found at {shapefile_path}, using data as-is")
                df_final = df_processed
                
        except Exception as e:
            print(f"Warning: Could not merge with shapefile: {e}")
            print("Using data as-is without shapefile sorting")
            df_final = df_processed
        
        # Prepare data (use df_final which is sorted correctly)
        X = df_final[predictor_cols].values
        n = len(X)
        
        # Validate number of observations matches model
        if n != params.shape[0]:
            print(f"[GWLR DEBUG] FATAL ERROR: Observation count mismatch. Data: {n}, Model: {params.shape[0]}")
            raise ValueError(
                f"Number of observations ({n}) does not match model training size ({params.shape[0]}). "
                f"GWLR requires the same observations in the same order as training data."
            )
        
        # Standardize features using the scaler from training (BEST PRACTICE)
        # The scaler was fitted during model training and saved in the pickle
        # This ensures consistency: same mean and std used for both training and prediction
        X_scaled = scaler.transform(X)
        
        # Calculate eta (linear predictor) for each observation using local parameters
        # eta_i = β0_i + X_scaled_i @ β_i[1:]
        eta_gwlr = np.zeros(n)
        
        for i in range(n):
            beta_i = params[i]  # Local parameters for observation i: [β0_i, β1_i, ..., βk_i]
            eta_gwlr[i] = beta_i[0] + X_scaled[i] @ beta_i[1:]
        
        # Convert eta to probabilities using logistic function
        # p = 1 / (1 + exp(-eta))
        prob_values = 1 / (1 + np.exp(-eta_gwlr))
        
        # Classify using threshold 0.5
        predictions = (prob_values >= 0.5).astype(int)
        
        # CRITICAL: Reorder predictions to match original dataframe index
        # Currently predictions correspond to df_final rows (sorted by ORDER)
        # We need to map them back to df.index using _original_index
        if '_original_index' in df_final.columns:
            print("[GWLR DEBUG] Reordering predictions to match original index")
            
            # Create series with original index as the index
            pred_series = pd.Series(predictions, index=df_final['_original_index'])
            prob_series = pd.Series(prob_values, index=df_final['_original_index'])
            
            # Sort by index to match df.index order (assuming df.index is sorted or we reindex)
            pred_series = pred_series.reindex(df.index)
            prob_series = prob_series.reindex(df.index)
            
            return pred_series, prob_series
        else:
            print("[GWLR DEBUG] Warning: Original index lost, returning as-is (may be scrambled)")
            return pd.Series(predictions, index=df.index), pd.Series(prob_values, index=df.index)

    def _predict_mgwlr(self, model_data: Dict, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Predict using MGWLR (Mixed Geographically Weighted Logistic Regression) model
        
        MGWLR combines:
        - Global variables (Z): Fixed effects across all locations
        - Local variables (X): Spatially varying effects
        
        Args:
            model_data: Dictionary containing MGWLR model components
                - X_local: List of local variable names
                - Z_global: List of global variable names
                - Beta_Local: Local parameters (n x k_local) array
                - res_global: Global GLM results with params
                - Bandwidth_optimal: Optimal bandwidth
                - (optional) scaler_X, scaler_Z, coordinates
            df: DataFrame with input data
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        print("[MGWLR DEBUG] Starting MGWLR prediction")
        
        # Extract model components
        X_local_cols = model_data.get('X_local', ['DepRatio', 'RumahLayak', 'Sanitasi'])
        Z_global_cols = model_data.get('Z_global', ['UMK', 'Industri', 'TPT'])
        Beta_Local = model_data['Beta_Local']  # (n, k_local) array
        res_global = model_data['res_global']
        gamma = res_global.params  # Global coefficients [intercept, γ_Z1, γ_Z2, ...]
        
        print(f"[MGWLR DEBUG] X_local (local vars): {X_local_cols}")
        print(f"[MGWLR DEBUG] Z_global (global vars): {Z_global_cols}")
        print(f"[MGWLR DEBUG] Beta_Local shape: {Beta_Local.shape}")
        print(f"[MGWLR DEBUG] Gamma (global params) shape: {gamma.shape}")
        
        # Preprocessing: Create required columns if they don't exist
        df_processed = df.copy()
        
        # Check if all columns already exist
        all_cols = X_local_cols + Z_global_cols
        all_cols_exist = all(col in df_processed.columns for col in all_cols)
        
        print(f"[MGWLR DEBUG] All columns exist: {all_cols_exist}")
        print(f"[MGWLR DEBUG] Available columns: {list(df_processed.columns)}")
        
        if not all_cols_exist:
            print(f"[MGWLR DEBUG] Applying preprocessing...")
            
            # Same preprocessing as GWLR
            if 'DepRatio' not in df_processed.columns:
                df_processed['DepRatio'] = 50.0
                print("[MGWLR DEBUG] Added DepRatio placeholder")
            
            if 'Industri' not in df_processed.columns and 'PDRB ' in df_processed.columns:
                df_processed['Industri'] = df_processed['PDRB '].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
            
            if 'RumahLayak' not in df_processed.columns and 'Sanitasi Layak' in df_processed.columns:
                df_processed['RumahLayak'] = df_processed['Sanitasi Layak'].astype(str).str.replace(',', '.', regex=False).astype(float)
            
            if 'Sanitasi' not in df_processed.columns and 'Sanitasi Layak' in df_processed.columns:
                df_processed['Sanitasi'] = df_processed['Sanitasi Layak'].astype(str).str.replace(',', '.', regex=False).astype(float)
            
            if 'TPT' not in df_processed.columns and 'Tingkat Pengangguran Terbuka (TPT)' in df_processed.columns:
                df_processed['TPT'] = df_processed['Tingkat Pengangguran Terbuka (TPT)'].astype(str).str.replace(',', '.').astype(float)
            
            if 'UMK' not in df_processed.columns and 'Upah Minimum Kabupaten/Kota (UMK)' in df_processed.columns:
                df_processed['UMK'] = df_processed['Upah Minimum Kabupaten/Kota (UMK)'].astype(str).str.replace('.', '').str.replace(',', '.').astype(float)
        
        # CRITICAL: Prepare for re-indexing
        df_processed = df_processed.copy()
        df_processed['_original_index'] = df.index
        
        # CRITICAL: Merge with shapefile and sort by ORDER
        print(f"[MGWLR DEBUG] Starting shapefile merge...")
        try:
            import geopandas as gpd
            from pathlib import Path
            
            shapefile_path = Path(__file__).parent.parent / "Geodata Jawa Tengah" / "JawaTengah.shp"
            print(f"[MGWLR DEBUG] Shapefile path: {shapefile_path}")
            print(f"[MGWLR DEBUG] Shapefile exists: {shapefile_path.exists()}")
            
            if shapefile_path.exists():
                gdf = gpd.read_file(shapefile_path)
                gdf['ORDER'] = gdf.index
                print(f"[MGWLR DEBUG] Loaded shapefile with {len(gdf)} regions")
                
                merge_col = None
                if 'Kabupaten/Kota' in df_processed.columns:
                    merge_col = 'Kabupaten/Kota'
                elif 'NAMOBJ' in df_processed.columns:
                    merge_col = 'NAMOBJ'
                else:
                    print("Warning: No merge column found, using data as-is")
                    df_final = df_processed
                
                if merge_col:
                    print(f"[MGWLR DEBUG] Merging on column: {merge_col}")
                    df_final = gdf.merge(df_processed, left_on="NAMOBJ", right_on=merge_col, how='inner')
                    print(f"[MGWLR DEBUG] After merge: {len(df_final)} rows")
                    
                    df_final = df_final.sort_values("ORDER").reset_index(drop=True)
                    print(f"[MGWLR DEBUG] After sort: {len(df_final)} rows")
                    print(f"[MGWLR DEBUG] First 5 kabupaten: {df_final['Kabupaten/Kota'].head().tolist()}")
                else:
                    df_final = df_processed
            else:
                print(f"Warning: Shapefile not found at {shapefile_path}, using data as-is")
                df_final = df_processed
                
        except Exception as e:
            print(f"Warning: Could not merge with shapefile: {e}")
            print("Using data as-is without shapefile sorting")
            df_final = df_processed
        
        # Extract variables
        X_local = df_final[X_local_cols].values  # (n, k_local)
        Z_global = df_final[Z_global_cols].values  # (n, k_global)
        n = len(df_final)
        
        print(f"[MGWLR DEBUG] Data observations: {n}, Model observations: {Beta_Local.shape[0]}")
        
        # Validate number of observations
        if n != Beta_Local.shape[0]:
            print(f"[MGWLR DEBUG] FATAL ERROR: Observation count mismatch. Data: {n}, Model: {Beta_Local.shape[0]}")
            raise ValueError(
                f"Number of observations ({n}) does not match model training size ({Beta_Local.shape[0]}). "
                f"MGWLR requires the same observations in the same order as training data."
            )
        
        # Check if scalers are available (try both key naming conventions)
        scaler_X = None
        scaler_Z = None
        
        # Try new naming convention first (Scaler_X_local, Scaler_Z_global)
        if 'Scaler_X_local' in model_data and 'Scaler_Z_global' in model_data:
            print("[MGWLR DEBUG] Using scalers from model (Scaler_X_local, Scaler_Z_global)")
            scaler_X = model_data['Scaler_X_local']
            scaler_Z = model_data['Scaler_Z_global']
        # Try old naming convention (scaler_X, scaler_Z)
        elif 'scaler_X' in model_data and 'scaler_Z' in model_data:
            print("[MGWLR DEBUG] Using scalers from model (scaler_X, scaler_Z)")
            scaler_X = model_data['scaler_X']
            scaler_Z = model_data['scaler_Z']
        
        if scaler_X is not None and scaler_Z is not None:
            X_local_scaled = scaler_X.transform(X_local)
            Z_global_scaled = scaler_Z.transform(Z_global)
        else:
            print("[MGWLR DEBUG] WARNING: No scalers found, using raw values")
            print("[MGWLR DEBUG] This may cause incorrect predictions!")
            X_local_scaled = X_local
            Z_global_scaled = Z_global
        
        # Calculate predictions using simpler approach
        # eta_i = gamma[0] + Z_global[i] @ gamma[1:] + X_local[i] @ Beta_Local[i]
        print("[MGWLR DEBUG] Calculating predictions...")
        
        predictions = np.zeros(n)
        probabilities = np.zeros(n)
        
        
        for i in range(n):
            # Global component: intercept + Z_global @ gamma
            eta_global = gamma[0] + Z_global_scaled[i] @ gamma[1:]
            
            # Local component: X_local @ Beta_Local[i]
            eta_local = X_local_scaled[i] @ Beta_Local[i]
            
            # Total linear predictor
            eta_i = eta_global + eta_local
            
            # Probability
            prob_i = 1 / (1 + np.exp(-eta_i))
            probabilities[i] = prob_i
            predictions[i] = 1 if prob_i >= 0.5 else 0
            
            # Debug first 3 observations
            if i < 3:
                print(f"[MGWLR DEBUG] Obs {i} ({df_final.iloc[i]['Kabupaten/Kota'] if 'Kabupaten/Kota' in df_final.columns else i}):")
                print(f"  gamma[0]={gamma[0]:.4f}, eta_global={eta_global:.4f}, eta_local={eta_local:.4f}")
                print(f"  eta_total={eta_i:.4f}, prob={prob_i:.4f}, pred={predictions[i]}")
        
        print("[MGWLR DEBUG] Prediction complete")
        print(f"[MGWLR DEBUG] Predictions summary: {predictions[:5]} (first 5)")
        print(f"[MGWLR DEBUG] Probabilities summary: {probabilities[:5]} (first 5)")
        
        # If P1_encoded available (ground truth), check accuracy
        if 'P1_encoded' in df_final.columns:
            from sklearn.metrics import accuracy_score
            y_true = df_final['P1_encoded'].values
            acc = accuracy_score(y_true, predictions)
            print(f"[MGWLR DEBUG] Calculated accuracy (in sorted order): {acc}")
        
        
        # CRITICAL: Reorder predictions to match original dataframe index
        # Currently predictions correspond to df_final rows (sorted by ORDER for MGWLR)
        # We need to map them back to df.index using _original_index so they align with the frontend df
        if '_original_index' in df_final.columns:
            print("[MGWLR DEBUG] Reordering predictions to match original index")
            
            # Create series with original index as the index
            pred_series = pd.Series(predictions, index=df_final['_original_index'])
            prob_series = pd.Series(probabilities, index=df_final['_original_index'])
            
            # Remove duplicates in index if any (shouldn't happen if 1-to-1)
            if pred_series.index.duplicated().any():
                 print("[MGWLR DEBUG] Warning: Duplicate indices found during reordering")
                 pred_series = pred_series[~pred_series.index.duplicated(keep='first')]
                 prob_series = prob_series[~prob_series.index.duplicated(keep='first')]
            
            # Sort by index to match df.index order
            pred_series = pred_series.reindex(df.index)
            prob_series = prob_series.reindex(df.index)
            
            return pred_series, prob_series
        else:
            print("[MGWLR DEBUG] Warning: Original index lost, returning as-is (may be scrambled)")
            return pd.Series(predictions, index=df_final.index), pd.Series(probabilities, index=df_final.index)

    def predict(self, df: pd.DataFrame, model_name: str) -> Tuple[pd.Series, pd.Series]:
        """Run prediction on data using specified model"""
        model = self.load_model(model_name)
        
        # Prepare data based on model type
        filename = self.model_mapping.get(model_name, model_name)
        
        # Check if model is GWLR or MGWLR (stored as dictionary)
        if filename in ["gwlr_model.pkl", "mgwlr_model.pkl"]:
            if not isinstance(model, dict):
                raise ValueError(f"Expected GWLR/MGWLR model to be a dictionary, got {type(model)}")
            
            # Distinguish between GWLR and MGWLR
            if filename == "mgwlr_model.pkl":
                print("[DEBUG] Using MGWLR prediction method")
                return self._predict_mgwlr(model, df)
            else:
                print("[DEBUG] Using GWLR prediction method")
                return self._predict_gwlr(model, df)

        
        if filename == "model_logistik_global.pkl":
            # Specific logic for Global Logistic Regression
            feature_cols = ['DepRatio', 'UMK', 'Industri', 'TPT', 'RumahLayak', 'Sanitasi']
            
            # Check if columns exist
            missing_cols = [col for col in feature_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns for this model: {', '.join(missing_cols)}")
            
            X = df[feature_cols].copy()
            X = sm.add_constant(X)
        else:
            # Default fallback for other models
            X = df.select_dtypes(include=[np.number]).dropna()
            if 'p1_encoded' in X.columns:
                 X = X.drop(columns=['p1_encoded'])
        
        try:
            # Check for predict_proba method (sklearn style)
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)
                # Take probability of positive class (usually index 1)
                prob_values = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
                preds = model.predict(X)
            
            # Check for statsmodels prediction (returns probabilities by default for Logit)
            elif hasattr(model, 'predict'):
                # For statsmodels Logit, predict() returns probabilities
                raw_prediction = model.predict(X)
                
                # Force conversion to numeric and handle potential Series/ndarray issues
                try:
                    # Ensure it's a pandas Series
                    if not isinstance(raw_prediction, pd.Series):
                        raw_prediction = pd.Series(raw_prediction, index=X.index)
                    
                    # Convert to numeric, coercing errors
                    prob_values = pd.to_numeric(raw_prediction, errors='coerce').fillna(0)
                    
                    # Check if values are effectively probabilities (mostly between 0 and 1)
                    # We assume statsmodels predict() returns probabilities for Logit
                    # If max value > 1.5, it might be linear predictor, but default is probability.
                    
                    # Convert to class (threshold 0.5)
                    preds = (prob_values >= 0.5).astype(int)
                    
                except Exception as conversion_error:
                     print(f"Warning: Could not convert statsmodels output to probability: {conversion_error}")
                     # Fallback
                     preds = raw_prediction
                     prob_values = np.zeros(len(preds))

            else:
                 # Check if it's a statsmodels result or similar wrapper
                if hasattr(model, 'prediksi'): # Custom method name?
                     preds = model.prediksi(X)
                     prob_values = np.zeros(len(preds))
                else:
                    raise AttributeError("Model does not have predictable method")
            
            return pd.Series(preds, index=X.index), pd.Series(prob_values, index=X.index)
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

    def calculate_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """Calculate classification metrics"""
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1-Score": f1_score(y_true, y_pred)
        }


# ==========================================
# DATA SERVICE
# ==========================================
class DataService:
    """Service for data processing operations"""
    
    def __init__(self):
        self.current_data: Optional[pd.DataFrame] = None
        self.region_column: Optional[str] = None
        self.numeric_columns: List[str] = []
        self.merged_geodata: Optional[gpd.GeoDataFrame] = None
        self.merge_stats: Dict[str, Any] = {}
    
    def load_file(self, file_path: Path) -> pd.DataFrame:
        """Load CSV or Excel file"""
        try:
            # Determine file type and load
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                raise ValidationError(f"Unsupported file type: {file_path.suffix}")
            
            # Validate dataframe
            is_valid, message = validate_dataframe(df)
            if not is_valid:
                raise ValidationError(message)
            
            # Store data
            self.current_data = df
            self.region_column = get_region_column(df)
            self.numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
            
            # Auto-merge with geodata
            self._merge_with_geodata()
            
            return df
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Error loading file: {str(e)}")
    
    def get_numeric_columns(self) -> List[str]:
        """Get list of numeric column names"""
        return self.numeric_columns
    
    def get_variable_data(self, variable: str) -> pd.Series:
        """Get data for a specific variable"""
        if self.current_data is None:
            raise ValidationError("No data loaded")
        
        if variable not in self.current_data.columns:
            raise ValidationError(f"Variable '{variable}' not found")
        
        return self.current_data[variable]
    
    def _merge_with_geodata(self):
        """Automatically merge current data with geodata"""
        if self.current_data is None or self.region_column is None:
            return
        
        try:
             # Use the global geo_service instance defined at end of file
             # But here we might not have it yet if classes defined first.
             # Better to define class methods or pass dependency.
             # Or simply use the class logic directly since we are in same file.
             
             # Actually, we can instantiate local GeoService since it's stateless except cache
             # But better to use shared one.
             # We will handle this by deferring to the global instance which we will assume exists
             # OR effectively, we can just instantiate one here if needed, but cache is in instance.
             # Let's trust the singleton at the bottom.
             pass 
             
             # IMPLEMENTATION using local imports or assuming global availability?
             # Python looks up globals at runtime. So if we define geo_service at the bottom, 
             # and call this method AFTER definition, it works.
             global geo_service
             
             gdf = geo_service.load_shapefile()
             
             data_normalized = self.current_data.copy()
             data_normalized['region_normalized'] = data_normalized[self.region_column].apply(normalize_region_name)
             
             gdf['region_normalized'] = gdf['NAMOBJ'].apply(normalize_region_name)
             
             self.merged_geodata = gdf.merge(
                data_normalized[[self.region_column, 'region_normalized'] + self.numeric_columns],
                on='region_normalized',
                how='left'
             )
             
             total_geodata_regions = len(gdf)
             matched_regions = self.merged_geodata[self.numeric_columns[0]].notna().sum() if self.numeric_columns else 0
             unmatched_in_data = set(data_normalized['region_normalized']) - set(gdf['region_normalized'])
             
             self.merge_stats = {
                'total_geodata_regions': total_geodata_regions,
                'matched_regions': int(matched_regions),
                'match_rate': round(matched_regions / total_geodata_regions * 100, 2) if total_geodata_regions > 0 else 0,
                'unmatched_data_regions': list(unmatched_in_data)
             }
             
             print(f"[OK] Auto-merged data with geodata: {self.merge_stats['matched_regions']}/{self.merge_stats['total_geodata_regions']} regions matched ({self.merge_stats['match_rate']}%)")

        except Exception as e:
            print(f"Warning: Could not merge with geodata: {e}")
            self.merged_geodata = None
            self.merge_stats = {}

    def get_merged_geodata(self) -> Optional[gpd.GeoDataFrame]:
        return self.merged_geodata
    
    def get_merge_statistics(self) -> Dict[str, Any]:
        return self.merge_stats


# ==========================================
# STATS SERVICE
# ==========================================
class StatsService:
    """Service for statistical calculations"""
    
    @staticmethod
    def calculate_stats(series: pd.Series) -> Dict[str, float]:
        """Calculate descriptive statistics"""
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "count": 0}
        
        return {
            "mean": float(clean_series.mean()),
            "median": float(clean_series.median()),
            "min": float(clean_series.min()),
            "max": float(clean_series.max()),
            "std": float(clean_series.std()),
            "count": int(len(clean_series))
        }
    
    @staticmethod
    def get_top_n(df: pd.DataFrame, variable_col: str = 'value', n: int = 10) -> pd.DataFrame:
        """Get top N rows by variable value"""
        sorted_df = df.sort_values(by=variable_col, ascending=False)
        return sorted_df.head(n)
    
    def get_statistics(self, variable: str) -> Dict[str, float]:
        """Get statistics for a variable from current data"""
        global data_service
        
        if data_service.current_data is None:
            raise ValidationError("No data loaded")
        
        if variable not in data_service.current_data.columns:
            raise ValidationError(f"Variable '{variable}' not found")
        
        return self.calculate_stats(data_service.current_data[variable])
    
    def get_chart_data(self, variable: str, top_n: int = 10) -> Dict:
        """Get chart data for top N regions"""
        global data_service
        
        if data_service.current_data is None:
            raise ValidationError("No data loaded")
        
        if variable not in data_service.current_data.columns:
            raise ValidationError(f"Variable '{variable}' not found")
        
        region_col = data_service.region_column
        if region_col is None:
            raise ValidationError("No region column found")
        
        chart_df = data_service.current_data[[region_col, variable]].copy()
        chart_df = chart_df.dropna()
        
        top_df = self.get_top_n(chart_df, variable_col=variable, n=top_n)
        
        return {
            'regions': top_df[region_col].tolist(),
            'values': top_df[variable].tolist(),
            'variable': variable
        }


# ==========================================
# GLOBAL INSTANCES
# ==========================================
# Instantiate services to be used across the app
geo_service = GeoService()
model_service = ModelService()
data_service = DataService()
stats_service = StatsService()
