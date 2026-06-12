import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Initialize FastAPI app
app = FastAPI(title="SSH Visualizer API")

# Automatically open the default web browser upon startup
@app.on_event("startup")
async def startup_event():
    import threading
    import time
    import webbrowser
    def open_browser():
        time.sleep(1.2)  # Give Uvicorn a moment to bind to the port
        try:
            webbrowser.open("http://127.0.0.1:8000")
        except Exception as e:
            print(f"Failed to automatically open browser: {e}")
    threading.Thread(target=open_browser, daemon=True).start()

# Custom JSON Sanitizer to handle NaNs and Inf values for Plotly/JS compatibility
def sanitize_array(arr):
    if arr is None:
        return None
    # Replace NaN and Inf with None (JSON null)
    arr_float = np.ascontiguousarray(arr, dtype=np.float64)
    has_nan_or_inf = np.isnan(arr_float) | np.isinf(arr_float)
    
    # We convert to object array to allow None values, or simply handle it line by line
    if np.any(has_nan_or_inf):
        # Convert to list and replace float('nan') / float('inf') with None
        lst = arr_float.tolist()
        def clean_nested_list(item):
            if isinstance(item, list):
                return [clean_nested_list(x) for x in item]
            if item is None or np.isnan(item) or np.isinf(item):
                return None
            return item
        return clean_nested_list(lst)
    return arr_float.tolist()

class SSHDataManager:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.files = [
            os.path.join(base_dir, "Data", "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D_2018.nc"),
            os.path.join(base_dir, "Data", "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D_2019.nc")
        ]
        self.topo_file = r"E:\Work2\Topography\etopo2022_111.5_128.5_13.5_30.5_geotiff.tiff"
        self.stations_file = os.path.join(base_dir, "pin_point.csv")
        self.geojson_file = os.path.join(base_dir, "Data", "coastline.geojson")
        
        self.full_ds = None
        self.ds = None
        self.dates = []
        self.stations = []
        self.coastlines_cache = []
        
        self.load_data()
        self.load_coastlines()
        self.mooring_file = os.path.join(base_dir, "Data", "mooring_time_depth.npz")
        self.mooring_data = None
        self.load_mooring()

    def load_mooring(self):
        if os.path.exists(self.mooring_file):
            print(f"Loading mooring data from {self.mooring_file}")
            try:
                self.mooring_data = np.load(self.mooring_file)
                print("Successfully loaded mooring data.")
            except Exception as e:
                print(f"Error loading mooring npz file: {e}")
        else:
            print(f"Mooring npz file not found at {self.mooring_file}")

    def load_data(self):
        print("Loading netCDF datasets...")
        datasets = []
        for f in self.files:
            if os.path.exists(f):
                print(f"Opening {f}")
                ds_single = xr.open_dataset(f)
                datasets.append(ds_single)
            else:
                print(f"File not found: {f}")
        
        if not datasets:
            raise FileNotFoundError("No NetCDF files were found at the specified paths!")
        
        # Combine datasets along time dimension
        print("Combining datasets...")
        self.full_ds = xr.concat(datasets, dim='time')
        
        # Ensure all coordinates are strictly ascending for Plotly/Leaflet compliance
        print("Sorting coordinates to ascending...")
        self.full_ds = self.full_ds.sortby('latitude').sortby('longitude')
        
        # Default crop to: Lon 118-126E, Lat 19-27N
        self.set_bounds(118.0, 126.0, 19.0, 27.0)
        
        # Load stations from CSV
        if os.path.exists(self.stations_file):
            try:
                df = pd.read_csv(self.stations_file, header=None, names=['name', 'lon', 'lat'])
                self.stations = df[['name', 'lon', 'lat']].values.tolist()
                print(f"Loaded {len(self.stations)} stations from {self.stations_file}")
            except Exception as e:
                print(f"Error loading stations: {e}")

    def load_coastlines(self):
        import json
        if os.path.exists(self.geojson_file):
            print(f"Loading pre-computed coastlines from cache file: {self.geojson_file}")
            try:
                with open(self.geojson_file, 'r', encoding='utf-8') as f:
                    geojson = json.load(f)
                
                coastlines = []
                for feature in geojson.get("features", []):
                    coords = feature["geometry"]["coordinates"]
                    for poly in coords:
                        lon_list = [pt[0] for pt in poly]
                        lat_list = [pt[1] for pt in poly]
                        coastlines.append({"lon": lon_list, "lat": lat_list})
                self.coastlines_cache = coastlines
                print(f"Successfully loaded {len(self.coastlines_cache)} coastline polygons from cache.")
                return
            except Exception as e:
                print(f"Error reading coastline.geojson: {e}. Re-generating...")

        # If cache doesn't exist, pre-compute from GeoTIFF once
        if os.path.exists(self.topo_file):
            print("Pre-computing high-resolution coastlines from GeoTIFF once...")
            try:
                from PIL import Image
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                
                with Image.open(self.topo_file) as img:
                    z = np.array(img)
                    z_asc = np.flipud(z) # Flip to match ascending coordinates
                    
                    lon_asc = np.linspace(111.5, 128.5, 4080)
                    lat_asc = np.linspace(13.5, 30.5, 4080)
                
                fig, ax = plt.subplots()
                # Extract filled contour at sea-level (z=0)
                cs = ax.contourf(lon_asc[::2], lat_asc[::2], z_asc[::2, ::2], levels=[0, 10000])
                plt.close(fig)
                
                features = []
                coastlines = []
                
                def add_poly(pts):
                    pts_list = pts.tolist()
                    features.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [pts_list]
                        },
                        "properties": {}
                    })
                    coastlines.append({
                        "lon": pts[:, 0].tolist(),
                        "lat": pts[:, 1].tolist()
                    })
                
                for path in cs.get_paths():
                    vertices = path.vertices
                    codes = path.codes
                    
                    if codes is None:
                        if len(vertices) > 3:
                            add_poly(vertices[::2])
                    else:
                        current_pts = []
                        for pt, code in zip(vertices, codes):
                            if code == 1: # MOVETO (start of a new separate segment)
                                if len(current_pts) > 3:
                                    add_poly(np.array(current_pts)[::2])
                                current_pts = [pt.tolist()]
                            else:
                                current_pts.append(pt.tolist())
                        if len(current_pts) > 3:
                            add_poly(np.array(current_pts)[::2])
                
                # Write to GeoJSON cache file
                geojson_data = {
                    "type": "FeatureCollection",
                    "features": features
                }
                with open(self.geojson_file, 'w', encoding='utf-8') as f:
                    json.dump(geojson_data, f)
                
                self.coastlines_cache = coastlines
                print(f"Successfully generated and cached {len(self.coastlines_cache)} coastline polygons to {self.geojson_file}")
            except Exception as e:
                print(f"Error generating coastlines: {e}")

    def set_bounds(self, lon_min, lon_max, lat_min, lat_max):
        # Since coordinates are guaranteed to be ascending, simple slices are robust!
        self.ds = self.full_ds.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max))
        self.dates = self.ds.time.values

    def get_data_for_index(self, index, var_name):
        date_val = self.dates[index]
        data_slice = self.ds.isel(time=index)
        
        lon = data_slice.longitude.values
        lat = data_slice.latitude.values
        z = data_slice[var_name].values
        
        if var_name == 'sla':
            u = data_slice['ugosa'].values
            v = data_slice['vgosa'].values
        else:
            u = data_slice['ugos'].values
            v = data_slice['vgos'].values
        
        return lon, lat, z, u, v, date_val

    def get_average_data(self, start_date, end_date, var_name):
        ds_period = self.ds.sel(time=slice(start_date, end_date))
        if len(ds_period.time) == 0:
            raise ValueError("No data available in the selected date range!")
            
        lon = ds_period.longitude.values
        lat = ds_period.latitude.values
        z = ds_period[var_name].mean(dim='time').values
        
        if var_name == 'sla':
            u = ds_period['ugosa'].mean(dim='time').values
            v = ds_period['vgosa'].mean(dim='time').values
        else:
            u = ds_period['ugos'].mean(dim='time').values
            v = ds_period['vgos'].mean(dim='time').values
            
        date_label = f"Mean Field ({pd.to_datetime(ds_period.time.values[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(ds_period.time.values[-1]).strftime('%Y-%m-%d')})"
        return lon, lat, z, u, v, date_label

# Global data manager instance
try:
    data_manager = SSHDataManager()
except Exception as e:
    print(f"Data loading failed: {e}")
    data_manager = None

@app.get("/", response_class=HTMLResponse)
def get_home():
    # Read index.html from local folder
    index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>index.html not found! Please create it in the workspace.</h1>", status_code=404)

@app.get("/api/init")
def get_init_metadata():
    if not data_manager:
        raise HTTPException(status_code=500, detail="Data Manager not initialized. Verify dataset paths.")
    
    # Convert dates to string ISO format
    dates_str = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in data_manager.dates]
    
    # Global limits for SLA and ADT for locking scale
    sla_vals = data_manager.ds['sla'].values
    adt_vals = data_manager.ds['adt'].values
    
    sla_limits = [float(np.nanmin(sla_vals)), float(np.nanmax(sla_vals))]
    adt_limits = [float(np.nanmin(adt_vals)), float(np.nanmax(adt_vals))]
    
    return {
        "dates": dates_str,
        "stations": data_manager.stations,
        "bounds": {
            "lon_min": float(data_manager.ds.longitude.min().values),
            "lon_max": float(data_manager.ds.longitude.max().values),
            "lat_min": float(data_manager.ds.latitude.min().values),
            "lat_max": float(data_manager.ds.latitude.max().values)
        },
        "limits": {
            "sla": sla_limits,
            "adt": adt_limits
        }
    }

@app.get("/api/slice")
def get_slice(
    index: int = Query(0, description="Time index"),
    var_name: str = Query("sla", description="Variable: sla or adt"),
    lon_min: float = Query(None),
    lon_max: float = Query(None),
    lat_min: float = Query(None),
    lat_max: float = Query(None),
    avg_start: str = Query(None, description="Start date for temporal average YYYY-MM-DD"),
    avg_end: str = Query(None, description="End date for temporal average YYYY-MM-DD")
):
    if not data_manager:
        raise HTTPException(status_code=500, detail="Data Manager not initialized.")

    # Dynamically update bounds if requested coordinates differ
    if None not in (lon_min, lon_max, lat_min, lat_max):
        data_manager.set_bounds(lon_min, lon_max, lat_min, lat_max)

    try:
        if avg_start and avg_end:
            lon, lat, z, u, v, date_label = data_manager.get_average_data(avg_start, avg_end, var_name)
            date_str = date_label
        else:
            if index < 0 or index >= len(data_manager.dates):
                raise HTTPException(status_code=400, detail=f"Index out of range. Max index is {len(data_manager.dates)-1}")
            lon, lat, z, u, v, date_val = data_manager.get_data_for_index(index, var_name)
            date_str = pd.to_datetime(date_val).strftime("%Y-%m-%d")
        
        # Calculate stats
        vmin = float(np.nanmin(z)) if not np.all(np.isnan(z)) else 0.0
        vmax = float(np.nanmax(z)) if not np.all(np.isnan(z)) else 0.0
        vmean = float(np.nanmean(z)) if not np.all(np.isnan(z)) else 0.0
        vstd = float(np.nanstd(z)) if not np.all(np.isnan(z)) else 0.0
        
        # Fetch pre-computed coastlines instantly from memory cache!
        coastlines = data_manager.coastlines_cache

        return JSONResponse(content={
            "date": date_str,
            "lon": lon.tolist(),
            "lat": lat.tolist(),
            "z": sanitize_array(z),
            "u": sanitize_array(u),
            "v": sanitize_array(v),
            "stats": {
                "min": vmin,
                "max": vmax,
                "mean": vmean,
                "std": vstd
            },
            "coastlines": coastlines
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/mooring")
def get_mooring_data():
    if not data_manager or data_manager.mooring_data is None:
        raise HTTPException(status_code=500, detail="Mooring data not initialized.")
    
    res = {}
    for key in ['ckm1', 'ckm6', 'ckm3']:
        t_key = f"t_{key}"
        dep_key = f"deps_{key}"
        v_key = f"v_{key}"
        
        if t_key in data_manager.mooring_data and dep_key in data_manager.mooring_data and v_key in data_manager.mooring_data:
            t_data = data_manager.mooring_data[t_key]
            t_list = [pd.to_datetime(t).strftime("%Y-%m-%d") for t in t_data]
            
            dep_data = data_manager.mooring_data[dep_key]
            dep_list = (-dep_data).tolist()
            
            v_data = data_manager.mooring_data[v_key]
            
            res[key] = {
                "t": t_list,
                "deps": dep_list,
                "v": sanitize_array(v_data)
            }
            
    return JSONResponse(content=res)

if __name__ == "__main__":
    import uvicorn
    # Start web server on localhost port 8000
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
