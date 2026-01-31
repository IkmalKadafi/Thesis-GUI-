"""
Statistics calculation service
"""
from typing import Dict
import pandas as pd
import numpy as np
from utils.validators import ValidationError


class StatsService:
    """Service for statistical calculations"""
    
    @staticmethod
    def calculate_stats(series: pd.Series) -> Dict[str, float]:
        """
        Calculate descriptive statistics
        
        Args:
            series: Pandas series with numeric data
            
        Returns:
            Dictionary with statistics
        """
        # Remove NaN values
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return {
                "mean": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std": 0.0,
                "count": 0
            }
        
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
        """
        Get top N rows by variable value
        
        Args:
            df: DataFrame with data
            variable_col: Column name for the variable
            n: Number of top rows to return
            
        Returns:
            DataFrame with top N rows sorted descending
        """
        # Sort by value descending
        sorted_df = df.sort_values(by=variable_col, ascending=False)
        
        # Get top N
        top_n = sorted_df.head(n)
        
        return top_n
    
    @staticmethod
    def calculate_distribution(series: pd.Series) -> Dict[str, float]:
        """
        Calculate distribution statistics (quartiles)
        
        Args:
            series: Pandas series with numeric data
            
        Returns:
            Dictionary with quartile values
        """
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return {
                "q25": 0.0,
                "q50": 0.0,
                "q75": 0.0
            }
        
        return {
            "q25": float(clean_series.quantile(0.25)),
            "q50": float(clean_series.quantile(0.50)),
            "q75": float(clean_series.quantile(0.75))
        }
    
    def get_statistics(self, variable: str) -> Dict[str, float]:
        """
        Get statistics for a variable from current data
        
        Args:
            variable: Variable name
            
        Returns:
            Dictionary with statistics
        """
        from services.data_service import data_service
        
        if data_service.current_data is None:
            raise ValidationError("No data loaded")
        
        if variable not in data_service.current_data.columns:
            raise ValidationError(f"Variable '{variable}' not found")
        
        return self.calculate_stats(data_service.current_data[variable])
    
    def get_chart_data(self, variable: str, top_n: int = 10) -> Dict:
        """
        Get chart data for top N regions
        
        Args:
            variable: Variable name
            top_n: Number of top regions
            
        Returns:
            Dictionary with regions and values
        """
        from services.data_service import data_service
        
        if data_service.current_data is None:
            raise ValidationError("No data loaded")
        
        if variable not in data_service.current_data.columns:
            raise ValidationError(f"Variable '{variable}' not found")
        
        # Get region column
        region_col = data_service.region_column
        if region_col is None:
            raise ValidationError("No region column found")
        
        # Create dataframe with region and value
        chart_df = data_service.current_data[[region_col, variable]].copy()
        chart_df = chart_df.dropna()
        
        # Get top N
        top_df = self.get_top_n(chart_df, variable_col=variable, n=top_n)
        
        return {
            'regions': top_df[region_col].tolist(),
            'values': top_df[variable].tolist(),
            'variable': variable
        }


# Global instance
stats_service = StatsService()
