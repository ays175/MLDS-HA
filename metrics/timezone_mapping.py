"""
Market-Aware Timezone Mapping System
Provides country-to-timezone mapping for accurate temporal analysis
"""

import pytz
from typing import Dict, Optional
from datetime import datetime
import pandas as pd

class MarketTimezoneMapper:
    """Maps countries/markets to their respective timezones for temporal analysis"""
    
    def __init__(self):
        self.country_timezone_map = {
            # European Markets
            "France": "Europe/Paris",
            "Germany": "Europe/Berlin", 
            "Italy": "Europe/Rome",
            "Spain": "Europe/Madrid",
            "Switzerland": "Europe/Zurich",
            "Czech Republic": "Europe/Prague",
            "Portugal": "Europe/Lisbon",
            "Poland": "Europe/Warsaw",
            "Turkey": "Europe/Istanbul",
            "Greece": "Europe/Athens",
            "United Kingdom": "Europe/London",
            "Bulgaria": "Europe/Sofia",
            "Romania": "Europe/Bucharest",
            "Serbia": "Europe/Belgrade",
            "Scandinavia": "Europe/Stockholm",  # Default for Nordic region
            
            # Other Markets
            "Middle East": "Asia/Dubai",  # Default for MENA region
            "Canada": "America/Toronto",  # Default for Canada
            "Singapore": "Asia/Singapore",
            
            # Global/Default
            "Global": "UTC",  # Keep global content in UTC
        }
        
        # Reverse mapping for quick lookups
        self.timezone_country_map = {tz: country for country, tz in self.country_timezone_map.items()}
    
    def get_timezone(self, country: str) -> str:
        """Get timezone string for a country"""
        return self.country_timezone_map.get(country, "UTC")
    
    def get_pytz_timezone(self, country: str) -> pytz.BaseTzInfo:
        """Get pytz timezone object for a country"""
        tz_string = self.get_timezone(country)
        return pytz.timezone(tz_string)
    
    def convert_utc_to_local(self, utc_datetime: pd.Timestamp, country: str) -> pd.Timestamp:
        """Convert UTC datetime to local market time"""
        if pd.isna(utc_datetime):
            return utc_datetime
            
        # Ensure UTC timezone
        if utc_datetime.tz is None:
            utc_datetime = utc_datetime.tz_localize('UTC')
        elif utc_datetime.tz != pytz.UTC:
            utc_datetime = utc_datetime.tz_convert('UTC')
        
        # Convert to local timezone
        local_tz = self.get_pytz_timezone(country)
        local_datetime = utc_datetime.tz_convert(local_tz)
        
        return local_datetime
    
    def convert_series_to_local(self, utc_series: pd.Series, country_series: pd.Series) -> pd.Series:
        """Convert a series of UTC datetimes to local times based on country series"""
        # Initialize result series with same index
        result = pd.Series(index=utc_series.index, dtype=object)
        
        for country in country_series.unique():
            if pd.isna(country):
                continue
                
            mask = country_series == country
            local_tz = self.get_pytz_timezone(country)
            
            # Convert UTC to local for this country's posts
            utc_subset = utc_series[mask]
            if not utc_subset.empty:
                # Convert each timestamp individually to avoid dtype issues
                local_timestamps = []
                for utc_ts in utc_subset:
                    if pd.isna(utc_ts):
                        local_timestamps.append(utc_ts)
                    else:
                        # Ensure it's a pandas Timestamp
                        if not isinstance(utc_ts, pd.Timestamp):
                            utc_ts = pd.Timestamp(utc_ts)
                        
                        # Ensure UTC timezone
                        if utc_ts.tz is None:
                            utc_ts = utc_ts.tz_localize('UTC')
                        elif utc_ts.tz != pytz.UTC:
                            utc_ts = utc_ts.tz_convert('UTC')
                        
                        # Convert to local timezone
                        local_ts = utc_ts.tz_convert(local_tz)
                        local_timestamps.append(local_ts)
                
                # Assign back to result
                result.loc[mask] = local_timestamps
        
        return result
    
    def get_market_summary(self) -> Dict[str, Dict]:
        """Get summary of all markets and their timezones"""
        summary = {}
        for country, tz_string in self.country_timezone_map.items():
            tz = pytz.timezone(tz_string)
            now_utc = datetime.now(pytz.UTC)
            now_local = now_utc.astimezone(tz)
            
            summary[country] = {
                "timezone": tz_string,
                "current_time": now_local.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "utc_offset": now_local.strftime("%z"),
                "dst_active": bool(now_local.dst())
            }
        
        return summary
    
    def validate_country(self, country: str) -> bool:
        """Check if country is supported"""
        return country in self.country_timezone_map
    
    def get_supported_countries(self) -> list:
        """Get list of all supported countries"""
        return list(self.country_timezone_map.keys())

# Global instance for easy import
timezone_mapper = MarketTimezoneMapper()

def get_timezone_mapper() -> MarketTimezoneMapper:
    """Get the global timezone mapper instance"""
    return timezone_mapper

if __name__ == "__main__":
    # Test the timezone mapper
    mapper = MarketTimezoneMapper()
    
    print("🌍 MARKET TIMEZONE MAPPING SYSTEM")
    print("=" * 50)
    
    # Show all markets
    summary = mapper.get_market_summary()
    for country, info in summary.items():
        print(f"{country:15} | {info['timezone']:20} | {info['current_time']}")
    
    # Test conversion
    print(f"\n🔄 CONVERSION TEST:")
    utc_time = pd.Timestamp("2024-12-19T13:55:26.000Z")
    print(f"UTC Time: {utc_time}")
    
    for country in ["France", "Germany", "Singapore"]:
        local_time = mapper.convert_utc_to_local(utc_time, country)
        print(f"{country}: {local_time}")
