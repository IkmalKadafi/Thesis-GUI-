"""
Unified Service Module
Contains: GeoService, ModelService, DataService, StatsService
"""
import json
import pickle
import pandas as pd
import numpy as np
import geopandas as gpd
import statsmodels.api as sm
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
class GeoService:
    """Service for GeoJSON operations with Jawa Tengah geodata"""
    
    def __init__(self):
        self.geodataframe: Optional[gpd.GeoDataFrame] = None
        self.geojson_data: Dict[str, Any] = {}
        self.region_mapping: Dict[str, str] = {}
    
    def load_shapefile(self) -> gpd.GeoDataFrame:
        """Load Jawa Tengah shapefile"""
        if self.geodataframe is not None:
            return self.geodataframe
        
        if not JAWA_TENGAH_SHAPEFILE.exists():
            raise FileNotFoundError(
                f"Shapefile not found: {JAWA_TENGAH_SHAPEFILE}\n"
                "Please ensure 'Geodata Jawa Tengah' folder exists in project root."
            )
        
        try:
            # Load shapefile using geopandas
            self.geodataframe = gpd.read_file(JAWA_TENGAH_SHAPEFILE)
            
            # Ensure it's in WGS84 (EPSG:4326) for web mapping
            if self.geodataframe.crs is not None and self.geodataframe.crs.to_epsg() != 4326:
                self.geodataframe = self.geodataframe.to_crs(epsg=4326)
            
            print(f"✓ Loaded {len(self.geodataframe)} regions from Jawa Tengah shapefile")
            
            return self.geodataframe
            
        except Exception as e:
            raise RuntimeError(f"Error loading shapefile: {e}")
    
    def load_geojson(self) -> Dict[str, Any]:
        """Load or convert shapefile to GeoJSON"""
        if self.geojson_data:
            return self.geojson_data
        
        # Try to load cached GeoJSON first
        if JAWA_TENGAH_GEOJSON_CACHE.exists():
            try:
                with open(JAWA_TENGAH_GEOJSON_CACHE, 'r', encoding='utf-8') as f:
                    self.geojson_data = json.load(f)
                print(f"✓ Loaded cached GeoJSON from {JAWA_TENGAH_GEOJSON_CACHE.name}")
                
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
            print(f"✓ Cached GeoJSON to {JAWA_TENGAH_GEOJSON_CACHE.name}")
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
        """Create choropleth map data"""
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
        
        # Create matching dataframe
        match_df = pd.DataFrame({
            'region': df[region_col],
            'value': df[variable],
            'region_normalized': df[region_col].apply(normalize_region_name)
        }).dropna(subset=['value'])
        
        # Match regions
        matched_values, unmatched = self.match_regions(match_df)
        
        # Add values to GeoJSON features
        geojson_with_values = json.loads(json.dumps(geojson))  # Deep copy
        for feature_idx, value in matched_values.items():
            if feature_idx < len(geojson_with_values['features']):
                geojson_with_values['features'][feature_idx]['properties'][variable] = value
        
        return {
            "geojson": geojson_with_values,
            "values": matched_values,
            "variable": variable,
            "matched_count": len(matched_values),
            "total_regions": len(geojson['features']),
            "unmatched_regions": unmatched,
            "match_rate": round(len(matched_values) / len(geojson['features']) * 100, 2)
        }
    
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
class ModelService:
    """Service for handling ML models and predictions"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        # Mapping: Display Name -> Filename
        self.model_mapping = {
            "Model Global Logistic Regression": "model_logistik_global.pkl",
            "Model Geographically Weighted Logistic Regression": "gwlr_model.pkl",
            "Model Geographically Weighted Logistic Regression Semiparametric": "mgwlr_model.pkl"
        }
    
    def get_available_models(self) -> List[str]:
        """Get list of available model display names"""
        return list(self.model_mapping.keys())
    
    def load_model(self, model_identifier: str) -> Any:
        """Load a model from disk"""
        # Resolve filename if display name is passed
        filename = self.model_mapping.get(model_identifier, model_identifier)
        
        if filename in self.models:
            return self.models[filename]
            
        model_path = MODEL_DIR / filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                self.models[filename] = model
                print(f"✓ Loaded model: {filename}")
                return model
        except Exception as e:
            raise RuntimeError(f"Error loading model {filename}: {e}")

    def predict(self, df: pd.DataFrame, model_name: str) -> Tuple[pd.Series, pd.Series]:
        """Run prediction on data using specified model"""
        model = self.load_model(model_name)
        
        # Prepare data based on model type
        filename = self.model_mapping.get(model_name, model_name)
        
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
            # Default fallback for other models (GWLR/MGWLR)
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
             
             print(f"✓ Auto-merged data with geodata: {self.merge_stats['matched_regions']}/{self.merge_stats['total_geodata_regions']} regions matched ({self.merge_stats['match_rate']}%)")

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
