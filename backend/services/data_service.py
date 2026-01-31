"""
Data processing service
"""
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import geopandas as gpd
from utils.validators import validate_dataframe, get_region_column, ValidationError
from utils.helpers import normalize_region_name


class DataService:
    """Service for data processing operations"""
    
    def __init__(self):
        self.current_data: Optional[pd.DataFrame] = None
        self.region_column: Optional[str] = None
        self.numeric_columns: List[str] = []
        self.merged_geodata: Optional[gpd.GeoDataFrame] = None
        self.merge_stats: Dict[str, Any] = {}
    
    def load_file(self, file_path: Path) -> pd.DataFrame:
        """
        Load CSV or Excel file
        
        Args:
            file_path: Path to the file
            
        Returns:
            DataFrame with loaded data
            
        Raises:
            ValidationError: If file cannot be loaded or is invalid
        """
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
    
    def get_data(
        self, 
        page: int = 1, 
        page_size: int = 50,
        search: Optional[str] = None,
        sort_by: Optional[str] = None,
        sort_order: str = 'asc'
    ) -> Dict[str, Any]:
        """
        Get paginated data
        
        Args:
            page: Page number (1-indexed)
            page_size: Number of rows per page
            search: Search term to filter data
            sort_by: Column name to sort by
            sort_order: Sort order ('asc' or 'desc')
            
        Returns:
            Dictionary with paginated data
        """
        if self.current_data is None:
            return {
                "data": [],
                "total": 0,
                "page": page,
                "page_size": page_size,
                "total_pages": 0
            }
        
        df = self.current_data.copy()
        
        # Apply search filter
        if search:
            mask = df.astype(str).apply(
                lambda x: x.str.contains(search, case=False, na=False)
            ).any(axis=1)
            df = df[mask]
        
        # Apply sorting
        if sort_by and sort_by in df.columns:
            ascending = sort_order.lower() == 'asc'
            df = df.sort_values(by=sort_by, ascending=ascending)
        
        # Calculate pagination
        total = len(df)
        total_pages = (total + page_size - 1) // page_size
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        # Get page data
        page_data = df.iloc[start_idx:end_idx]
        
        # Convert to list of dictionaries
        data_list = page_data.to_dict('records')
        
        return {
            "data": data_list,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages
        }
    
    def get_variable_data(self, variable: str) -> pd.Series:
        """
        Get data for a specific variable
        
        Args:
            variable: Variable name
            
        Returns:
            Series with variable data
            
        Raises:
            ValidationError: If variable not found
        """
        if self.current_data is None:
            raise ValidationError("No data loaded")
        
        if variable not in self.current_data.columns:
            raise ValidationError(f"Variable '{variable}' not found")
        
        return self.current_data[variable]
    
    def get_region_variable_data(self, variable: str) -> pd.DataFrame:
        """
        Get region and variable data
        
        Args:
            variable: Variable name
            
        Returns:
            DataFrame with region and variable columns
        """
        if self.current_data is None:
            raise ValidationError("No data loaded")
        
        if variable not in self.numeric_columns:
            raise ValidationError(f"Variable '{variable}' is not numeric")
        
        # Create dataframe with region and variable
        df = pd.DataFrame({
            'region': self.current_data[self.region_column],
            'value': self.current_data[variable]
        })
        
        # Normalize region names
        df['region_normalized'] = df['region'].apply(normalize_region_name)
        
        # Remove missing values
        df = df.dropna(subset=['value'])
        
        return df
    
    def _merge_with_geodata(self):
        """Automatically merge current data with geodata"""
        if self.current_data is None or self.region_column is None:
            return
        
        try:
            from services.geo_service import geo_service
            
            # Load geodataframe
            gdf = geo_service.load_shapefile()
            
            # Normalize region names in both dataframes
            from services.geo_service import normalize_region_name
            
            # Create normalized region column in data
            data_normalized = self.current_data.copy()
            data_normalized['region_normalized'] = data_normalized[self.region_column].apply(normalize_region_name)
            
            # Create normalized region column in geodata
            gdf['region_normalized'] = gdf['NAMOBJ'].apply(normalize_region_name)
            
            # Merge geodata with ALL numeric columns from data
            self.merged_geodata = gdf.merge(
                data_normalized[[self.region_column, 'region_normalized'] + self.numeric_columns],
                on='region_normalized',
                how='left'
            )
            
            # Calculate merge statistics
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
            
            if self.merge_stats['unmatched_data_regions']:
                print(f"  Unmatched regions: {', '.join(list(self.merge_stats['unmatched_data_regions'])[:5])}{'...' if len(self.merge_stats['unmatched_data_regions']) > 5 else ''}")
        
        except Exception as e:
            print(f"Warning: Could not merge with geodata: {e}")
            import traceback
            traceback.print_exc()
            self.merged_geodata = None
            self.merge_stats = {}
    
    def get_merged_geodata(self) -> Optional[gpd.GeoDataFrame]:
        """
        Get merged geodataframe with all variables
        
        Returns:
            GeoDataFrame with merged data or None if no data loaded
        """
        return self.merged_geodata
    
    def get_merge_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about geodata merge
        
        Returns:
            Dictionary with merge statistics
        """
        return self.merge_stats


# Global instance
data_service = DataService()
