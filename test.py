import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pydeck as pdk

# Directly load the data from LAPD's online data source
DATA_URL = 'https://data.lacity.org/api/views/2nrs-mtv8/rows.csv?accessType=DOWNLOAD'

def load_data():
    # Load data from the URL
    data = pd.read_csv(DATA_URL)
    
    # Data cleaning and preprocessing
    data = data.rename(columns={'LAT': 'lat', 'LON': 'lon'})
    data['lat'] = pd.to_numeric(data['lat'], errors='coerce')
    data['lon'] = pd.to_numeric(data['lon'], errors='coerce')
    data['DATE OCC'] = pd.to_datetime(data['DATE OCC'], errors='coerce')

    return data.dropna(subset=['lat', 'lon'])

def main():
    st.title("Safety LA")
    st.subheader("LA Crime Data Exploration")

    # Load the crime data
    df = load_data()

    # Date range for filtering
    min_occurence = pd.to_datetime(df['DATE OCC']).min().date()
    max_occurence = pd.to_datetime(df['DATE OCC']).max().date()

    st.write("Select Time Range for the map")
    start_time, end_time = st.slider("Timeline", min_value=min_occurence, max_value=max_occurence, value=[min_occurence, max_occurence])
    filtered_data = df[(df['DATE OCC'] >= pd.to_datetime(start_time)) & (df['DATE OCC'] <= pd.to_datetime(end_time))]

    st.write(f"Filtering between {start_time} & {end_time}")
    st.write(f"Data Points: {len(filtered_data)}")

    # Map visualization
    st.map(filtered_data)

    # Detailed area selection
    st.subheader("More Detailed Data")
    option = st.selectbox(
        'Which area you want to know more about?',
        df['AREA NAME'].unique()
    )
    st.write(f'You selected: {option}')
    
    area_filtered = df[df['AREA NAME'] == option]
    st.write('Detailed Crimes Info')
    st.write(area_filtered[['TIME OCC', 'Crm Cd Desc', 'Vict Age', 'Vict Sex', 'LOCATION']])

if __name__ == '__main__':
    main()
