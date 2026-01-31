"""
GeoJSON matching service for choropleth maps with Jawa Tengah shapefile
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import geopandas as gpd
from config.settings import (
    JAWA_TENGAH_SHAPEFILE, 
    JAWA_TENGAH_GEOJSON_CACHE,
    GEOJSON_DIR
)
from utils.helpers import normalize_region_name


class GeoService:
    """Service for GeoJSON operations with Jawa Tengah geodata"""
    
    def __init__(self):
        self.geodataframe: Optional[gpd.GeoDataFrame] = None
        self.geojson_data: Dict[str, Any] = {}
        self.region_mapping: Dict[str, str] = {}
    
    def load_shapefile(self) -> gpd.GeoDataFrame:
        """
        Load Jawa Tengah shapefile
        
        Returns:
            GeoDataFrame with Jawa Tengah regions
        """
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
        """
        Load or convert shapefile to GeoJSON
        
        Returns:
            GeoJSON data as dictionary
        """
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
        """
        Match data regions with GeoJSON regions
        
        Args:
            df: DataFrame with 'region', 'region_normalized' and 'value' columns
            
        Returns:
            Tuple of (matched_values_dict, unmatched_regions_list)
            matched_values_dict maps feature index to value
        """
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
    
    def merge_with_geodata(
        self, 
        df: pd.DataFrame, 
        region_col: str, 
        value_col: str
    ) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
        """
        Merge DataFrame with geodata
        
        Args:
            df: DataFrame with data to merge
            region_col: Name of region column in df
            value_col: Name of value column in df
            
        Returns:
            Tuple of (merged GeoDataFrame, statistics dict)
        """
        # Load shapefile
        gdf = self.load_shapefile()
        
        # Prepare data for merging
        merge_df = df[[region_col, value_col]].copy()
        merge_df['region_normalized'] = merge_df[region_col].apply(normalize_region_name)
        
        # Normalize geodata region names for matching
        gdf['region_normalized'] = gdf['NAMOBJ'].apply(normalize_region_name)
        
        # Merge on normalized region names
        merged_gdf = gdf.merge(
            merge_df,
            on='region_normalized',
            how='left'
        )
        
        # Calculate statistics
        total_geodata_regions = len(gdf)
        matched_regions = merged_gdf[value_col].notna().sum()
        unmatched_in_data = set(merge_df['region_normalized']) - set(gdf['region_normalized'])
        
        stats = {
            'total_geodata_regions': total_geodata_regions,
            'matched_regions': int(matched_regions),
            'match_rate': round(matched_regions / total_geodata_regions * 100, 2),
            'unmatched_data_regions': list(unmatched_in_data)
        }
        
        return merged_gdf, stats
    
    def create_choropleth_data(
        self, 
        df: pd.DataFrame, 
        variable: str,
        region_col: str = None
    ) -> Dict[str, Any]:
        """
        Create choropleth map data
        
        Args:
            df: DataFrame with region and variable data
            variable: Variable name to visualize
            region_col: Optional region column name (auto-detected if None)
            
        Returns:
            Dictionary with GeoJSON and values
        """
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
        """
        Get information about loaded geodata
        
        Returns:
            Dictionary with geodata information
        """
        gdf = self.load_shapefile()
        
        return {
            "total_regions": len(gdf),
            "region_names": sorted(gdf['NAMOBJ'].tolist()),
            "bounds": gdf.total_bounds.tolist(),  # [minx, miny, maxx, maxy]
            "crs": str(gdf.crs),
            "columns": gdf.columns.tolist()
        }


# Global instance
geo_service = GeoService()
