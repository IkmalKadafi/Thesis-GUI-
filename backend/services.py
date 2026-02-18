import pickle
import json
import pandas as pd
import numpy as np
import geopandas as gpd
import statsmodels.api as sm
import streamlit as st
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from backend.settings import MODEL_DIR, JAWA_TENGAH_SHAPEFILE, JAWA_TENGAH_GEOJSON_CACHE, GEOJSON_DIR
from backend.utils import normalize_region_name, validate_dataframe, get_region_column, ValidationError


# GeoService helpers 

@st.cache_resource
def _load_shapefile_cached(shapefile_path: str) -> gpd.GeoDataFrame:
    if not Path(shapefile_path).exists():
        raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")
    gdf = gpd.read_file(shapefile_path)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf


@st.cache_data
def _create_choropleth_data_cached(
    df: pd.DataFrame,
    variable: str,
    region_col: str,
    geojson_data: Dict[str, Any],
    region_mapping: Dict[str, int],
) -> Dict[str, Any]:
    match_df = pd.DataFrame({
        'region': df[region_col],
        'value': df[variable],
        'region_normalized': df[region_col].apply(normalize_region_name),
    }).dropna(subset=['value'])

    matched_values, unmatched_regions = {}, []
    for _, row in match_df.iterrows():
        norm, orig, val = row['region_normalized'], row['region'], row['value']
        idx = region_mapping.get(norm) or region_mapping.get(orig.lower().strip())
        if idx is None:
            idx = next((i for k, i in region_mapping.items() if norm in k or k in norm), None)
        if idx is not None:
            matched_values[idx] = val
        else:
            unmatched_regions.append(orig)

    geojson_copy = json.loads(json.dumps(geojson_data))
    for feat_idx, val in matched_values.items():
        if feat_idx < len(geojson_copy['features']):
            geojson_copy['features'][feat_idx]['properties'][variable] = val

    total = len(geojson_data['features'])
    return {
        "geojson": geojson_copy,
        "values": matched_values,
        "variable": variable,
        "matched_count": len(matched_values),
        "total_regions": total,
        "unmatched_regions": unmatched_regions,
        "match_rate": round(len(matched_values) / total * 100, 2),
    }


class GeoService:
    def __init__(self):
        self.geodataframe: Optional[gpd.GeoDataFrame] = None
        self.geojson_data: Dict[str, Any] = {}
        self.region_mapping: Dict[str, int] = {}

    def load_shapefile(self) -> gpd.GeoDataFrame:
        if self.geodataframe is None:
            self.geodataframe = _load_shapefile_cached(str(JAWA_TENGAH_SHAPEFILE))
        return self.geodataframe

    def load_geojson(self) -> Dict[str, Any]:
        if self.geojson_data:
            return self.geojson_data
        if JAWA_TENGAH_GEOJSON_CACHE.exists():
            try:
                with open(JAWA_TENGAH_GEOJSON_CACHE, 'r', encoding='utf-8') as f:
                    self.geojson_data = json.load(f)
                self._build_region_mapping()
                return self.geojson_data
            except Exception:
                pass
        gdf = self.load_shapefile()
        self.geojson_data = json.loads(gdf.to_json())
        try:
            GEOJSON_DIR.mkdir(parents=True, exist_ok=True)
            with open(JAWA_TENGAH_GEOJSON_CACHE, 'w', encoding='utf-8') as f:
                json.dump(self.geojson_data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        self._build_region_mapping()
        return self.geojson_data

    def _build_region_mapping(self):
        for idx, feature in enumerate(self.geojson_data.get('features', [])):
            name = feature.get('properties', {}).get('NAMOBJ')
            if name:
                self.region_mapping[normalize_region_name(name)] = idx
                self.region_mapping[name.lower().strip()] = idx

    def create_choropleth_data(self, df: pd.DataFrame, variable: str, region_col: str = None) -> Dict[str, Any]:
        geojson = self.load_geojson()
        if region_col is None:
            for col in ['kabupaten_kota', 'Kabupaten/Kota', 'region', 'wilayah', 'daerah']:
                if col in df.columns:
                    region_col = col
                    break
            if region_col is None:
                raise ValueError("Could not find region column in DataFrame")
        return _create_choropleth_data_cached(df, variable, region_col, self.geojson_data, self.region_mapping)

    def get_geodata_info(self) -> Dict[str, Any]:
        gdf = self.load_shapefile()
        return {
            "total_regions": len(gdf),
            "region_names": sorted(gdf['NAMOBJ'].tolist()),
            "bounds": gdf.total_bounds.tolist(),
            "crs": str(gdf.crs),
            "columns": gdf.columns.tolist(),
        }


# ModelService helpers 

@st.cache_resource
def _load_model_cached(model_path: str) -> Any:
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading model {Path(model_path).name}: {e}")


def _merge_sort_by_shapefile(df: pd.DataFrame) -> pd.DataFrame:
    """Merge df with shapefile and sort by ORDER to match training data order."""
    shapefile_path = Path(__file__).parent.parent / "Geodata Jawa Tengah" / "JawaTengah.shp"
    if not shapefile_path.exists():
        return df
    try:
        gdf = gpd.read_file(shapefile_path)
        gdf['ORDER'] = gdf.index
        merge_col = next((c for c in ['Kabupaten/Kota', 'NAMOBJ'] if c in df.columns), None)
        if merge_col is None:
            return df
        merged = gdf.merge(df, left_on="NAMOBJ", right_on=merge_col, how='inner')
        return merged.sort_values("ORDER").reset_index(drop=True)
    except Exception:
        return df


def _apply_column_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Create model feature columns from raw data columns if missing."""
    col_map = {
        'DepRatio': lambda d: pd.Series(50.0, index=d.index),
        'Industri': lambda d: d['PDRB '].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float) if 'PDRB ' in d.columns else None,
        'RumahLayak': lambda d: d['Sanitasi Layak'].astype(str).str.replace(',', '.', regex=False).astype(float) if 'Sanitasi Layak' in d.columns else None,
        'Sanitasi': lambda d: d['Sanitasi Layak'].astype(str).str.replace(',', '.', regex=False).astype(float) if 'Sanitasi Layak' in d.columns else None,
        'TPT': lambda d: d['Tingkat Pengangguran Terbuka (TPT)'].astype(str).str.replace(',', '.').astype(float) if 'Tingkat Pengangguran Terbuka (TPT)' in d.columns else None,
        'UMK': lambda d: d['Upah Minimum Kabupaten/Kota (UMK)'].astype(str).str.replace('.', '').str.replace(',', '.').astype(float) if 'Upah Minimum Kabupaten/Kota (UMK)' in d.columns else None,
    }
    for col, fn in col_map.items():
        if col not in df.columns:
            result = fn(df)
            if result is not None:
                df[col] = result
    return df


class ModelService:
    def __init__(self):
        self.model_mapping = {
            "Model Global Logistic Regression": "model_logistik_global.pkl",
            "Model Geographically Weighted Logistic Regression": "gwlr_model.pkl",
            "Model Geographically Weighted Logistic Regression Semiparametric": "mgwlr_model.pkl",
        }

    def load_model(self, model_identifier: str) -> Any:
        filename = self.model_mapping.get(model_identifier, model_identifier)
        model_path = MODEL_DIR / filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        return _load_model_cached(str(model_path))

    def get_available_models(self) -> List[str]:
        return list(self.model_mapping.keys())

    def _prepare_df(self, df: pd.DataFrame, required_cols: List[str]) -> pd.DataFrame:
        """Preprocess, merge-sort with shapefile, and track original index."""
        df_proc = df.copy()
        if not all(c in df_proc.columns for c in required_cols):
            df_proc = _apply_column_preprocessing(df_proc)
        df_proc['_original_index'] = df.index
        return _merge_sort_by_shapefile(df_proc)

    def _reindex_to_original(
        self, predictions: np.ndarray, probabilities: np.ndarray,
        df_final: pd.DataFrame, df_orig: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        if '_original_index' not in df_final.columns:
            return pd.Series(predictions, index=df_final.index), pd.Series(probabilities, index=df_final.index)
        orig_idx = df_final['_original_index']
        pred_s = pd.Series(predictions, index=orig_idx)
        prob_s = pd.Series(probabilities, index=orig_idx)
        if pred_s.index.duplicated().any():
            pred_s = pred_s[~pred_s.index.duplicated(keep='first')]
            prob_s = prob_s[~prob_s.index.duplicated(keep='first')]
        return pred_s.reindex(df_orig.index), prob_s.reindex(df_orig.index)

    def _predict_gwlr(self, model_data: Dict, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        params = model_data['results'].params if 'results' in model_data else model_data['params']
        scaler = model_data['scaler']
        predictor_cols = model_data['predictor']

        df_final = self._prepare_df(df, predictor_cols)
        X = df_final[predictor_cols].values
        n = len(X)

        if n != params.shape[0]:
            raise ValueError(f"Observation count mismatch: data={n}, model={params.shape[0]}")

        X_scaled = scaler.transform(X)
        eta = np.array([params[i][0] + X_scaled[i] @ params[i][1:] for i in range(n)])
        probs = 1 / (1 + np.exp(-eta))
        preds = (probs >= 0.5).astype(int)

        return self._reindex_to_original(preds, probs, df_final, df)

    def _predict_mgwlr(self, model_data: Dict, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        X_local_cols = model_data.get('X_local', ['DepRatio', 'RumahLayak', 'Sanitasi'])
        Z_global_cols = model_data.get('Z_global', ['UMK', 'Industri', 'TPT'])
        Beta_Local = model_data['Beta_Local']
        gamma = model_data['res_global'].params

        df_final = self._prepare_df(df, X_local_cols + Z_global_cols)
        n = len(df_final)

        if n != Beta_Local.shape[0]:
            raise ValueError(f"Observation count mismatch: data={n}, model={Beta_Local.shape[0]}")

        X_local = df_final[X_local_cols].values
        Z_global = df_final[Z_global_cols].values

        scaler_X = model_data.get('Scaler_X_local') or model_data.get('scaler_X')
        scaler_Z = model_data.get('Scaler_Z_global') or model_data.get('scaler_Z')
        if scaler_X and scaler_Z:
            X_local = scaler_X.transform(X_local)
            Z_global = scaler_Z.transform(Z_global)

        eta = np.array([
            gamma[0] + Z_global[i] @ gamma[1:] + X_local[i] @ Beta_Local[i]
            for i in range(n)
        ])
        probs = 1 / (1 + np.exp(-eta))
        preds = (probs >= 0.5).astype(int)

        return self._reindex_to_original(preds, probs, df_final, df)

    def predict(self, df: pd.DataFrame, model_name: str) -> Tuple[pd.Series, pd.Series]:
        model = self.load_model(model_name)
        filename = self.model_mapping.get(model_name, model_name)

        if filename in ("gwlr_model.pkl", "mgwlr_model.pkl"):
            if not isinstance(model, dict):
                raise ValueError(f"Expected dict model, got {type(model)}")
            return self._predict_mgwlr(model, df) if filename == "mgwlr_model.pkl" else self._predict_gwlr(model, df)

        if filename == "model_logistik_global.pkl":
            feature_cols = ['DepRatio', 'UMK', 'Industri', 'TPT', 'RumahLayak', 'Sanitasi']
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing columns: {', '.join(missing)}")
            X = sm.add_constant(df[feature_cols].copy())
        else:
            X = df.select_dtypes(include=[np.number]).dropna()
            if 'p1_encoded' in X.columns:
                X = X.drop(columns=['p1_encoded'])

        try:
            if hasattr(model, 'predict_proba'):
                probs_arr = model.predict_proba(X)
                prob_values = probs_arr[:, 1] if probs_arr.shape[1] > 1 else probs_arr[:, 0]
                preds = model.predict(X)
            elif hasattr(model, 'predict'):
                raw = model.predict(X)
                if not isinstance(raw, pd.Series):
                    raw = pd.Series(raw, index=X.index)
                prob_values = pd.to_numeric(raw, errors='coerce').fillna(0)
                preds = (prob_values >= 0.5).astype(int)
            elif hasattr(model, 'prediksi'):
                preds = model.prediksi(X)
                prob_values = np.zeros(len(preds))
            else:
                raise AttributeError("Model has no predictable method")
            return pd.Series(preds, index=X.index), pd.Series(prob_values, index=X.index)
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

    def calculate_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1-Score": f1_score(y_true, y_pred),
        }


# ── DataService ───────────────────────────────────────────────────────────────

class DataService:
    def __init__(self):
        self.current_data: Optional[pd.DataFrame] = None
        self.region_column: Optional[str] = None
        self.numeric_columns: List[str] = []
        self.merged_geodata: Optional[gpd.GeoDataFrame] = None
        self.merge_stats: Dict[str, Any] = {}

    def load_file(self, file_path: Path) -> pd.DataFrame:
        try:
            suffix = file_path.suffix.lower()
            if suffix == '.csv':
                df = pd.read_csv(file_path)
            elif suffix in ('.xlsx', '.xls'):
                df = pd.read_excel(file_path)
            else:
                raise ValidationError(f"Unsupported file type: {suffix}")

            is_valid, message = validate_dataframe(df)
            if not is_valid:
                raise ValidationError(message)

            self.current_data = df
            self.region_column = get_region_column(df)
            self.numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
            self._merge_with_geodata()
            return df
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Error loading file: {e}")

    def get_numeric_columns(self) -> List[str]:
        return self.numeric_columns

    def get_variable_data(self, variable: str) -> pd.Series:
        if self.current_data is None:
            raise ValidationError("No data loaded")
        if variable not in self.current_data.columns:
            raise ValidationError(f"Variable '{variable}' not found")
        return self.current_data[variable]

    def _merge_with_geodata(self):
        if self.current_data is None or self.region_column is None:
            return
        try:
            global geo_service
            gdf = geo_service.load_shapefile()
            data_norm = self.current_data.copy()
            data_norm['region_normalized'] = data_norm[self.region_column].apply(normalize_region_name)
            gdf['region_normalized'] = gdf['NAMOBJ'].apply(normalize_region_name)

            cols = [self.region_column, 'region_normalized'] + self.numeric_columns
            self.merged_geodata = gdf.merge(data_norm[cols], on='region_normalized', how='left')

            total = len(gdf)
            matched = self.merged_geodata[self.numeric_columns[0]].notna().sum() if self.numeric_columns else 0
            unmatched = set(data_norm['region_normalized']) - set(gdf['region_normalized'])
            self.merge_stats = {
                'total_geodata_regions': total,
                'matched_regions': int(matched),
                'match_rate': round(matched / total * 100, 2) if total > 0 else 0,
                'unmatched_data_regions': list(unmatched),
            }
        except Exception as e:
            print(f"Warning: Could not merge with geodata: {e}")
            self.merged_geodata = None
            self.merge_stats = {}

    def get_merged_geodata(self) -> Optional[gpd.GeoDataFrame]:
        return self.merged_geodata

    def get_merge_statistics(self) -> Dict[str, Any]:
        return self.merge_stats


# ── StatsService ──────────────────────────────────────────────────────────────

class StatsService:
    @staticmethod
    def calculate_stats(series: pd.Series) -> Dict[str, float]:
        s = series.dropna()
        if s.empty:
            return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "count": 0}
        return {
            "mean": float(s.mean()), "median": float(s.median()),
            "min": float(s.min()), "max": float(s.max()),
            "std": float(s.std()), "count": int(len(s)),
        }

    @staticmethod
    def get_top_n(df: pd.DataFrame, variable_col: str = 'value', n: int = 10) -> pd.DataFrame:
        return df.sort_values(by=variable_col, ascending=False).head(n)

    def get_statistics(self, variable: str) -> Dict[str, float]:
        global data_service
        if data_service.current_data is None:
            raise ValidationError("No data loaded")
        if variable not in data_service.current_data.columns:
            raise ValidationError(f"Variable '{variable}' not found")
        return self.calculate_stats(data_service.current_data[variable])

    def get_chart_data(self, variable: str, top_n: int = 10) -> Dict:
        global data_service
        if data_service.current_data is None:
            raise ValidationError("No data loaded")
        if variable not in data_service.current_data.columns:
            raise ValidationError(f"Variable '{variable}' not found")
        region_col = data_service.region_column
        if region_col is None:
            raise ValidationError("No region column found")
        chart_df = data_service.current_data[[region_col, variable]].dropna()
        top_df = self.get_top_n(chart_df, variable_col=variable, n=top_n)
        return {'regions': top_df[region_col].tolist(), 'values': top_df[variable].tolist(), 'variable': variable}


# ── Global instances ──────────────────────────────────────────────────────────

geo_service = GeoService()
model_service = ModelService()
data_service = DataService()
stats_service = StatsService()
