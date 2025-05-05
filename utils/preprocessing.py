# -*- coding: utf-8 -*-
"""
Created on Sun May  4 21:11:01 2025

@author: Hood
"""
import pandas as pd
from timezonefinder import TimezoneFinder
from zoneinfo import ZoneInfo
from geopy.distance import geodesic
import xarray as xr
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def preprocess(raw_input):
    """
    Preprocesses the raw input data by extracting relevant features related to the flight, 
    engines, airport coordinates, local time of departure, temperature, humidity, and radiations.
    
    Args:
        raw_input (pd.DataFrame): Raw input data containing flight-related information.
    
    Returns:
        pd.DataFrame: Preprocessed data with engineered features for model prediction.
    """
   
    print('Preprocessing')
    
    df = raw_input[['Aircraft', 'Engine', 'Take-off Time (UTC)',
                    'Origin Airport', 'Destination Airport',
                    'Distance Flown (km)']].copy()
    df = engines_feature_extraction(df)
    df = get_airport_coordinates(df)
    df = local_time_departure(df)
    df = get_flight_coordinates(df)
    df = get_temperature_humidity(df)
    df = get_LW_SH_radiations(df)
    

    df = df.drop(columns = ['Mid_Lat', 'Mid_Lon',
       'Departure_Threshold_Lat', 'Departure_Threshold_Lon',
       'Departure_Cruise_Lat', 'Departure_Cruise_Lon', 'Arrival_Cruise_Lat',
       'Arrival_Cruise_Lon', 'Arrival_Threshold_Lat', 'Arrival_Threshold_Lon',
       'Cruise_Altitude_m', 'Cruise_Speed_kmh',
       'Time_Departure_Threshold', 'Time_Departure_Cruise', 'Time_midway',
       'Time_Arrival_Cruise', 'Time_Arrival_Threshold'])
    
    
    return df

def engines_feature_extraction(df):
    """
    Extracts features related to the aircraft engine including engine code, 
    particle emission index, and fuel flow rate. Computes the particle emission rate 
    based on the engine and fuel flow.
    
    Args:
        df (pd.DataFrame): DataFrame with flight data that includes engine types.
    
    Returns:
        pd.DataFrame: DataFrame with additional engine-related features.
    """
    
    #engines ranked based on contrail probability observed in training data (lowest to highest)
    engine_to_int = {'CFM56-5B4/P': 0,
                     'CFM56-5B6/P': 1,
                     'CFM56-5B6/3': 2,
                     'CFM56-5B4/3': 3,
                     'V2527-A5': 4,
                     'CFM56-5B3/3': 5,
                     'PW1127G-JM': 6,
                     'V2533-A5': 7,
                     'PW1133G-JM': 8}
    df['Engine_Code'] = df['Engine'].map(engine_to_int)
    
    # Particle emission index (particles/kg fuel)
    engine_nvpm = {
        'PW1127G-JM': 2e15,
        'PW1133G-JM': 3e15,
        'CFM56-5B3/3': 3e15,
        'CFM56-5B4/3': 3e15,
        'CFM56-5B6/3': 2.5e15,
        'CFM56-5B4/P': 3e15,
        'CFM56-5B6/P': 2.5e15,
        'V2527-A5': 2.5e15,
        'V2533-A5': 3e15
    }
    
    # Cruise fuel flow rates (kg/s)
    fuel_flow_kg_s = {
        'PW1127G-JM': 0.6,
        'PW1133G-JM': 0.65,
        'CFM56-5B3/3': 0.5,
        'CFM56-5B4/3': 0.5,
        'CFM56-5B6/3': 0.5,
        'CFM56-5B4/P': 0.5,
        'CFM56-5B6/P': 0.5,
        'V2527-A5': 0.39,
        'V2533-A5': 0.39
    }
    
    # Map both dictionaries
    df['nvPM_number'] = df['Engine'].map(engine_nvpm)
    df['Fuel_Flow_kg_s'] = df['Engine'].map(fuel_flow_kg_s)
    
    # Compute particles per second
    df['nvPM_particles_per_s'] = df['nvPM_number'] * df['Fuel_Flow_kg_s']
    df = df.drop(columns=['nvPM_number','Fuel_Flow_kg_s','Engine'])
    
    return df
    
    
def local_time_departure(df):
    """
    Converts UTC time of departure to local time based on the airport's location 
    and adds the hour of departure to the dataset.
    
    Args:
        df (pd.DataFrame): DataFrame containing flight data including UTC departure time.
    
    Returns:
        pd.DataFrame: DataFrame with added local departure time and departure hour.
    """
    tf = TimezoneFinder()
    
    def utc_to_local(row):
        tz_str = tf.timezone_at(lat=row['Origin_Lat'], lng=row['Origin_Lon'])
        if tz_str:
            utc_time = pd.to_datetime(row['Take-off Time (UTC)'])
            utc_time = utc_time.tz_localize('UTC')
            local_time = utc_time.astimezone(ZoneInfo(tz_str)).replace(tzinfo=None) 
            return local_time
        return pd.NaT
    
    df['Take-off Time Local'] = df.apply(utc_to_local, axis=1)
    df['Take-off Hour'] = df['Take-off Time Local'].dt.hour
    
    df = df.drop(columns=['Take-off Time Local'])
    return df


def get_airport_coordinates(df):
    """
    Adds the geographical coordinates (latitude, longitude, and altitude) 
    of the origin and destination airports to the dataset.
    
    Args:
        df (pd.DataFrame): DataFrame containing flight data including airport codes.
    
    Returns:
        pd.DataFrame: DataFrame with added geographical coordinates for airports.
    """
    
    airports = pd.read_csv('data/airports.csv')
    df = df.merge(airports[['ident', 'latitude_deg', 'longitude_deg', 'elevation_ft']], 
              left_on='Origin Airport', right_on='ident', 
              how='left', suffixes=('_Origin', '_Temp'))
    
    df = df.rename(columns={'latitude_deg': 'Origin_Lat', 
                            'longitude_deg': 'Origin_Lon', 
                            'elevation_ft': 'Origin_Airport_Altitude_m'})
    
    df = df.merge(airports[['ident', 'latitude_deg', 'longitude_deg', 'elevation_ft']], 
                  left_on='Destination Airport', right_on='ident', 
                  how='left', suffixes=('_Origin', '_Dest'))
    
    df = df.rename(columns={'latitude_deg': 'Dest_Lat', 
                            'longitude_deg': 'Dest_Lon', 
                            'elevation_ft': 'Destination_Airport_Altitude_m'})
    
    df['Mid_Lat'] = (df['Origin_Lat'] + df['Dest_Lat']) / 2
    df['Mid_Lon'] = (df['Origin_Lon'] + df['Dest_Lon']) / 2
    
    
    df = df.drop(columns=['Origin Airport', 'Destination Airport', 'ident_Origin', 'ident_Dest'])
    
    def compute_distance(row):
        origin = (row['Origin_Lat'], row['Origin_Lon'])
        dest = (row['Dest_Lat'], row['Dest_Lon'])
        return geodesic(origin, dest).kilometers
    
    mask = df['Distance Flown (km)'].isna()
    df.loc[mask, 'Distance Flown (km)'] = df[mask].apply(compute_distance, axis=1)

    
    return df

def get_flight_coordinates(df):
    """
    Calculates the flight coordinates at key stages: departure threshold, cruise, 
    and arrival threshold for both the origin and destination airports.
    
    Args:
        df (pd.DataFrame): DataFrame containing flight data with airport and aircraft information.
    
    Returns:
        pd.DataFrame: DataFrame with calculated flight coordinates for key stages of the flight.
    """
    
    aircraft_data = {
    "A320": {"cruise_speed_kmh": 800, "cruise_altitude_m": 12000, "climb_rate_mpm": 500},
    "A20N": {"cruise_speed_kmh": 820, "cruise_altitude_m": 12000, "climb_rate_mpm": 530},
    "A321": {"cruise_speed_kmh": 830, "cruise_altitude_m": 12000, "climb_rate_mpm": 500},
    "A319": {"cruise_speed_kmh": 780, "cruise_altitude_m": 12000, "climb_rate_mpm": 470},
    "A21N": {"cruise_speed_kmh": 850, "cruise_altitude_m": 13000, "climb_rate_mpm": 530}}
    threshold_altitude_m = 10000  # Fixed threshold altitude for contrail creation
    
    def calculate_coordinates(row):
        aircraft_type = row['Aircraft'] 
        aircraft = aircraft_data[aircraft_type]
        cruise_altitude = aircraft["cruise_altitude_m"]
        cruise_speed_kmh = aircraft["cruise_speed_kmh"]
        climb_rate_mpm = aircraft["climb_rate_mpm"]
        time_to_threshold = (threshold_altitude_m - row['Origin_Airport_Altitude_m']) / climb_rate_mpm
        time_to_cruise = (cruise_altitude - row['Origin_Airport_Altitude_m']) / climb_rate_mpm
        
        cruise_speed_mps = aircraft["cruise_speed_kmh"] * 1000 / 3600
        
        distance_to_threshold = cruise_speed_mps * time_to_threshold
        distance_to_cruise = cruise_speed_mps * time_to_cruise
        
        departure_threshold_lat = row['Origin_Lat'] + (distance_to_threshold / 111000) #one lat degree is 111km
        departure_threshold_lon = row['Origin_Lon'] + (distance_to_threshold / 111000)
        
        departure_cruise_lat = row['Origin_Lat'] + (distance_to_cruise / 111000)
        departure_cruise_lon = row['Origin_Lon'] + (distance_to_cruise / 111000)
        
        arrival_cruise_lat = row['Dest_Lat'] + (distance_to_cruise / 111000)
        arrival_cruise_lon = row['Dest_Lon'] + (distance_to_cruise / 111000)
        
        arrival_threshold_lat = row['Dest_Lat'] + (distance_to_threshold / 111000)
        arrival_threshold_lon = row['Dest_Lon'] + (distance_to_threshold / 111000)
        
        return pd.Series([departure_threshold_lat, departure_threshold_lon, 
                          departure_cruise_lat, departure_cruise_lon, 
                          arrival_cruise_lat, arrival_cruise_lon, 
                          arrival_threshold_lat, arrival_threshold_lon,
                          cruise_altitude, cruise_speed_kmh],
                         index=['Departure_Threshold_Lat', 'Departure_Threshold_Lon', 
                                'Departure_Cruise_Lat', 'Departure_Cruise_Lon', 
                                'Arrival_Cruise_Lat', 'Arrival_Cruise_Lon', 
                                'Arrival_Threshold_Lat', 'Arrival_Threshold_Lon',
                                'Cruise_Altitude_m', 'Cruise_Speed_kmh'])
    
    df[['Departure_Threshold_Lat', 'Departure_Threshold_Lon', 
        'Departure_Cruise_Lat', 'Departure_Cruise_Lon', 
        'Arrival_Cruise_Lat', 'Arrival_Cruise_Lon', 
        'Arrival_Threshold_Lat', 'Arrival_Threshold_Lon',
        'Cruise_Altitude_m', 'Cruise_Speed_kmh',
        ]] = df.apply(calculate_coordinates, axis=1)
    
    df['Flight_Duration'] = df['Distance Flown (km)'] / df['Cruise_Speed_kmh'] 
    
    df['Take-off Time (UTC)'] = pd.to_datetime(df['Take-off Time (UTC)'])
    df['Take-off Time (UTC)'] = df['Take-off Time (UTC)'].dt.round('h') 
    
    df['Time_Departure_Threshold'] = df['Take-off Time (UTC)'] + pd.to_timedelta(0.25, unit = 'h')
    df['Time_Departure_Threshold'] = df['Time_Departure_Threshold'].dt.round('h')
    
    df['Time_Departure_Cruise'] = df['Take-off Time (UTC)'] +  pd.to_timedelta(0.5, unit = 'h')
    df['Time_Departure_Cruise'] = df['Time_Departure_Cruise'].dt.round('h')
    
    df['Time_midway'] = df['Take-off Time (UTC)'] + pd.to_timedelta(df['Flight_Duration']/2, unit = 'h')
    df['Time_midway'] = df['Time_midway'].dt.round('h')
    
    df['Time_Arrival_Cruise'] = df['Take-off Time (UTC)'] + pd.to_timedelta(df['Flight_Duration'] - 0.5, unit = 'h')
    df['Time_Arrival_Cruise'] = df['Time_Arrival_Cruise'].dt.round('h')
    
    df['Time_Arrival_Threshold'] = df['Take-off Time (UTC)'] + pd.to_timedelta(df['Flight_Duration'] - 0.25, unit = 'h')
    df['Time_Arrival_Threshold'] = df['Time_Arrival_Threshold'].dt.round('h')
    
    #Now we can drop the locations of airports aswell as aircraft type.
    df = df.drop(columns = ['Origin_Lat','Origin_Lon','Dest_Lat','Dest_Lon','Origin_Airport_Altitude_m','Destination_Airport_Altitude_m','Aircraft','Take-off Time (UTC)'])
    
    return df


def get_temperature_humidity(df):
    """
    Extracts temperature, humidity, and wind components (u, v) at different flight segments (departure/arrival thresholds and cruise)
    using pressure-level weather data. Computes wind speeds and applies PCA to reduce wind-related features to 2 principal components.

    Args:
        df (pd.DataFrame): DataFrame containing time and coordinates for flight segments.

    Returns:
        pd.DataFrame: Updated DataFrame with temperature, humidity, and 2 PCA wind features.
    """
    
    ds = xr.open_dataset("data/jan2025_T_H_W.grib", engine="cfgrib", decode_timedelta=False)
    def get_variables(departure_time, lat, lon, pressure_level, variables, column_names):
        selected_data = ds.sel(
            time=departure_time,
            latitude=lat,
            longitude=lon,
            isobaricInhPa=pressure_level,
            method='nearest'
        )
        result = {}
        for var, col_name in zip(variables, column_names):
            variable_value = float(selected_data[var].values)
            result[col_name] = variable_value
        return result


    df[['T_Threshold_1', 'H_Threshold_1','U_Threshold_1', 'V_Threshold_1']] = df.apply(
        lambda row: pd.Series(get_variables( 
            row['Time_Departure_Threshold'], 
            row['Departure_Threshold_Lat'],
            row['Departure_Threshold_Lon'], 
            pressure_level=250.0,
            variables=['t','r','u','v'],  
            column_names=['T_Threshold_1', 'H_Threshold_1','U_Threshold_1', 'V_Threshold_1']  
        )), axis=1)
    df[['T_Cruise_1', 'H_Cruise_1','U_Cruise_1', 'V_Cruise_1']] = df.apply(
        lambda row: pd.Series(get_variables(
            row['Time_Departure_Threshold'], 
            row['Departure_Cruise_Lat'], 
            row['Departure_Cruise_Lon'], 
            pressure_level=200.0,
            variables=['t','r','u','v'],  
            column_names=['T_Cruise_1', 'H_Cruise_1','U_Cruise_1', 'V_Cruise_1'] 
        )), axis=1)
    df[['T_Cruise_2', 'H_Cruise_2','U_Cruise_2', 'V_Cruise_2']] = df.apply(
        lambda row: pd.Series(get_variables( 
            row['Time_Arrival_Cruise'], 
            row['Arrival_Cruise_Lat'], 
            row['Arrival_Cruise_Lon'], 
            pressure_level=200.0,
            variables=['t','r','u','v'],  
            column_names=['T_Cruise_2', 'H_Cruise_2','U_Cruise_2', 'V_Cruise_2']
        )), axis=1)
    df[['T_Threshold_2', 'H_Threshold_2','U_Threshold_2', 'V_Threshold_2']] = df.apply(
        lambda row: pd.Series(get_variables(
            row['Time_Arrival_Threshold'], 
            row['Arrival_Threshold_Lat'], 
            row['Arrival_Threshold_Lon'], 
            pressure_level=250.0,
            variables=['t','r','u','v'],  
            column_names=['T_Threshold_2', 'H_Threshold_2','U_Threshold_2', 'V_Threshold_2']
        )), axis=1)
    
    df['WindSpeed_Threshold_1'] = np.sqrt(df['U_Threshold_1']**2 + df['V_Threshold_1']**2)
    df['WindSpeed_Cruise_1'] = np.sqrt(df['U_Cruise_1']**2 + df['V_Cruise_1']**2)
    df['WindSpeed_Cruise_2'] = np.sqrt(df['U_Cruise_2']**2 + df['V_Cruise_2']**2)
    df['WindSpeed_Threshold_2'] = np.sqrt(df['U_Threshold_2']**2 + df['V_Threshold_2']**2)
    
    scaler = StandardScaler()
    X = df[['WindSpeed_Threshold_1','WindSpeed_Threshold_2', 'WindSpeed_Cruise_1', 'WindSpeed_Cruise_2']]
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    df[['W_PCA_1','W_PCA_2']] = pca.fit_transform(X_scaled)
    df.drop(columns=[
        'U_Threshold_1', 'V_Threshold_1',
        'U_Cruise_1', 'V_Cruise_1',
        'U_Cruise_2', 'V_Cruise_2',
        'U_Threshold_2', 'V_Threshold_2',
        'WindSpeed_Threshold_1','WindSpeed_Threshold_2', 
        'WindSpeed_Cruise_1', 'WindSpeed_Cruise_2'
    ], inplace=True)
    return df
        
    
def get_LW_SH_radiations(df):
    """
    Extracts longwave (strd) and shortwave (ssrd) radiations at various flight segments using reanalysis data. 
    Handles missing values by searching nearby grid points. Applies PCA to reduce radiation features to a single component.
    
    Args:
        df (pd.DataFrame): DataFrame with time and coordinates for flight segments.
    
    Returns:
        pd.DataFrame: Updated DataFrame with a single PCA radiation feature.
    """
    ds = xr.open_dataset("data/jan2025_LW_SW.grib", engine="cfgrib", decode_timedelta=False)
    def get_variables(departure_time, lat, lon, variables, column_names):
        try:
            base_time = pd.to_datetime(departure_time)
            step_index = base_time.hour
    
            selected_data = ds.sel(
                time=base_time,
                step=step_index,
                latitude=lat,
                longitude=lon,
                method='nearest'
            )
            if not np.isnan(selected_data['ssrd']) and not np.isnan(selected_data['strd']):
                result = {
                    col_name: float(selected_data[var].values)
                    for var, col_name in zip(variables, column_names)
                }
                return result
    
            radius = 0.1
            #these is a lot of points with nan values so i make sure we pull a point with actual values
            while True:
                radius = round(radius, 6)
                candidates = [
                    (lat + radius, lon),
                    (lat - radius, lon),
                    (lat, lon + radius),
                    (lat, lon - radius)
                ]
    
                for new_lat, new_lon in candidates:
                    selected_data = ds.sel(
                        time=base_time,
                        step=step_index,
                        latitude=new_lat,
                        longitude=new_lon,
                        method='nearest'
                    )
    
                    if not np.isnan(selected_data['ssrd']) and not np.isnan(selected_data['strd']):
                        result = {
                            col_name: float(selected_data[var].values)
                            for var, col_name in zip(variables, column_names)
                        }
                        return result
    
                radius = round(radius + 0.1, 6)
    
        except Exception as e:
            print(f"ERROR at {departure_time}, lat={lat}, lon={lon}: {e}")
            return {col: None for col in column_names}
    
    
    df[['LW_Threshold_1', 'SW_Threshold_1']] = df.apply(
        lambda row: pd.Series(get_variables( 
            row['Time_Departure_Threshold'], 
            row['Departure_Threshold_Lat'],
            row['Departure_Threshold_Lon'],
            variables=['strd', 'ssrd'],  
            column_names=['LW_Threshold_1', 'SW_Threshold_1']  
        )), axis=1)
    df[['LW_Cruise_1', 'SW_Cruise_1']] = df.apply(
        lambda row: pd.Series(get_variables(
            row['Time_Departure_Threshold'], 
            row['Departure_Cruise_Lat'], 
            row['Departure_Cruise_Lon'],
            variables=['strd', 'ssrd'],  
            column_names=['LW_Cruise_1', 'SW_Cruise_1'] 
        )), axis=1)
    df[['LW_Cruise_2', 'SW_Cruise_2']] = df.apply(
        lambda row: pd.Series(get_variables( 
            row['Time_Arrival_Cruise'], 
            row['Arrival_Cruise_Lat'], 
            row['Arrival_Cruise_Lon'],
            variables=['strd', 'ssrd'], 
            column_names=['LW_Cruise_2', 'SW_Cruise_2'] 
        )), axis=1)
    df[['LW_Threshold_2', 'SW_Threshold_2']] = df.apply(
        lambda row: pd.Series(get_variables(
            row['Time_Arrival_Threshold'], 
            row['Arrival_Threshold_Lat'], 
            row['Arrival_Threshold_Lon'],
            variables=['strd', 'ssrd'], 
            column_names=['LW_Threshold_2', 'SW_Threshold_2'] 
        )), axis=1)
    columns=['LW_Threshold_1', 'SW_Threshold_1','LW_Cruise_1', 'SW_Cruise_1','LW_Cruise_2', 'SW_Cruise_2','LW_Threshold_2', 'SW_Threshold_2']
    scaler = StandardScaler()
    X = df[columns]
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=1)  
    df['LW_SW_PC'] = pca.fit_transform(X_scaled)
    df = df.drop(columns = columns)
    
    
    return df
