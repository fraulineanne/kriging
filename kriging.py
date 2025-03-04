import base64
from flask import Flask
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
from pykrige.ok import OrdinaryKriging
from io import BytesIO

app = Flask(__name__)

def sensor_data(schema, table_name, clean=True):
    url = f"https://sync.upcare.ph/api/sensorinsights/{schema}/{table_name}/latest?time_bucket=1h"
    headers = {"accept": "application/json"}
    response = requests.get(url, headers=headers)

    df = pd.DataFrame({
        'pm25': [x['pm25'] for x in response.json()],
        'latitude': [x['gps_lat'] for x in response.json()] ,
        'longitude': [x['gps_long'] for x in response.json()]
    })

    if clean:
        return df.dropna().drop(df[(df['latitude'] == 14.649929) & (df['longitude'] == 121.068478)].index ) # remove data from lab

    return df

def generate_kriging_map():
    # get latest sensor data
    pasig = sensor_data('pasig_v2', 'data') 
    up = sensor_data('renetzero', 'data')

    df = pd.concat([pasig, up], axis=0)

    latitude = df['latitude'].values
    longitude = df['longitude'].values
    pm25 = df['pm25'].values
    
    # Define the grid for interpolation
    gridx = np.linspace(min(longitude) - 0.01, max(longitude) + 0.01, 100)
    gridy = np.linspace(min(latitude) - 0.01, max(latitude) + 0.01, 100)    

    min_lon, max_lon = min(gridx), max(gridx)
    min_lat, max_lat = min(gridy), max(gridy)

    # Create the Kriging Model
    OK = OrdinaryKriging(longitude, 
                        latitude, 
                        pm25, 
                        variogram_model='linear', 
                        coordinates_type='geographic', 
                        verbose=False, 
                        enable_plotting=False)
    z, _ = OK.execute('grid', gridx, gridy) 

    # Plot Kriging output
    # aspect_ratio = (max_lon - min_lon) / (max_lat - min_lat)
    # fig, ax = plt.subplots(figsize=(8 * aspect_ratio, 8))
    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.imshow(z, 
                extent=[min_lon, max_lon, min_lat, max_lat],
                origin='lower', 
                cmap='viridis_r', 
                alpha=1)

    ax.axis('off')

    # Save to in-memory file
    img_io = BytesIO()
    plt.savefig(img_io, format="png", bbox_inches='tight', pad_inches=0, transparent=True)
    img_io.seek(0)
    
    with open("debug_image.png", "wb") as f:
        f.write(img_io.getbuffer())  

    img_base64 = base64.b64encode(img_io.read()).decode("utf-8")
    img_data_uri = f"data:image/png;base64,{img_base64}"

    return img_data_uri, min_lat, max_lat, min_lon, max_lon, df



@app.route("/")
def serve_map():
    img_io, min_lat, max_lat, min_lon, max_lon, df = generate_kriging_map()

    # Compute map center
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    # Generate Folium Map dynamically centered
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    # Overlay Kriging PNG on Folium map
    folium.raster_layers.ImageOverlay(
        image=img_io,
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        opacity=.7,
        interactive=True,
        cross_origin=False,
        zindex=1,
    ).add_to(m)

    latitude = df['latitude'].values
    longitude = df['longitude'].values

    # Add points for the measuring stations
    for lat, lon in zip(latitude, longitude):
        folium.CircleMarker(
            location=[lat, lon],
            radius=2,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=1,
            tooltip=df[(df['latitude'] == lat) & (df['longitude'] == lon)].pm25.values[0]
        ).add_to(m)


    return m._repr_html_()

if __name__ == "__main__":
    app.run(debug=True)
