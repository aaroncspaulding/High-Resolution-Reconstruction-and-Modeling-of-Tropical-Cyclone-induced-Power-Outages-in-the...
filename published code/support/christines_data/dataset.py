from __future__ import annotations
import geopandas as gpd
from .constants import COUNTY_SHP

def load_counties(county_file=COUNTY_SHP) -> gpd.GeoDataFrame:
    counties = gpd.read_file(county_file)
    counties['GEOID'] = counties['GEOID'].astype(str).str.zfill(5)
    return counties
