import os
import numpy as np
from glob import glob
import pandas as pd
from tqdm import tqdm
import warnings

from pycontrails.core.fuel import JetA
from pycontrails.core.flight import Flight
from pycontrails.models.cocip import Cocip
from pycontrails.models.emissions import Emissions
from pycontrails.models.humidity_scaling import ExponentialBoostLatitudeCorrectionHumidityScaling
from pycontrails.models.ps_model.ps_model import PSFlight
from pycontrails.physics.geo import advect_latitude, advect_longitude, haversine
from RMS.Formats.FFfile import filenameToDatetime

from RMS.Astrometry.ApplyAstrometry import GeoHt2xy

# Ellipsoid parameters: semi major axis in metres, reciprocal flattening.
GRS80 = 6378137, 298.257222100882711
WGS84 = 6378137, 298.257223563

class FlightData:
    def __init__(self, file_paths, output_meta_file, output_wypts_file, output_contrails_file, center_coordinate, platepar_files, 
                 max_camera_depth=50000, simulate_contrails=False, era5_met=None, era5_rad=None, cam_frame_time=20,
                 flight_id_filter=[], bound_range=None, cache_flight_data=True, calculate_image_heading=True,
                 extract_good_candidates_only=True):

        if cache_flight_data:
            if not os.path.exists(output_meta_file) or not os.path.exists(output_wypts_file):
                os.makedirs(os.path.dirname(output_meta_file), exist_ok=True)
                self._preprocess_flight_data(file_paths, output_meta_file, output_wypts_file, center_coordinate)
            
            # Load the processed flight data
            meta_data = pd.read_parquet(output_meta_file)
            wypts_data = pd.read_parquet(output_wypts_file)
    
            # Simulate contrails
            if simulate_contrails:
                if not os.path.exists(output_contrails_file):
                    assert era5_met is not None and era5_rad is not None, "Need to provide ERA5 data path to simulate contrails"
                    self._simulate_contrails(output_contrails_file, meta_data, wypts_data, era5_met, era5_rad, cam_frame_time=cam_frame_time)

                contrails = pd.read_parquet(output_contrails_file)
            else:
                contrails = None
        else:
            meta_data, wypts_data = self._preprocess_flight_data(file_paths, output_meta_file, output_wypts_file, center_coordinate, cache_flight_data=cache_flight_data)
            
            if simulate_contrails:
                assert era5_met is not None and era5_rad is not None, "Need to provide ERA5 data path to simulate contrails"
                contrails = self._simulate_contrails(output_contrails_file, meta_data, wypts_data, era5_met, era5_rad, cam_frame_time=cam_frame_time, cache_flight_data=cache_flight_data)

        # Add camera data to flight and contrail data
        wypts_data = add_camera_information(wypts_data, platepar_files, center_coordinate=center_coordinate, calculate_image_heading=calculate_image_heading)

        if simulate_contrails:
            contrails = add_camera_information(contrails, platepar_files, center_coordinate=center_coordinate, calculate_image_heading=False)

        # Interpolate flight data to match camera frame time (TODO: can be optimized)
        wypts_data = self._interpolate_flight_data(wypts_data, cam_frame_time=cam_frame_time)

        # Filter using bounding box
        if center_coordinate is not None and bound_range is not None:
            longitude_range = (center_coordinate[1] - bound_range, center_coordinate[1] + bound_range)
            latitude_range = (center_coordinate[0] - bound_range, center_coordinate[0] + bound_range)
            wypts_data = wypts_data[(wypts_data['longitude'] >= longitude_range[0]) & (wypts_data['longitude'] <= longitude_range[1])]
            wypts_data = wypts_data[(wypts_data['latitude'] >= latitude_range[0]) & (wypts_data['latitude'] <= latitude_range[1])]

            if simulate_contrails:
                contrails = contrails[(contrails['longitude'] >= longitude_range[0]) & (contrails['longitude'] <= longitude_range[1])]
                contrails = contrails[(contrails['latitude'] >= latitude_range[0]) & (contrails['latitude'] <= latitude_range[1])]

        # Additionally filter so that only flights which are in frame at some point in their trajectory
        filter_string = f'in_frame'
        wypts_within_depth = wypts_data.query(filter_string)
        valid_flight_ids = set(wypts_within_depth['flight_id'].unique())
        wypts_data = wypts_data[wypts_data['flight_id'].isin(valid_flight_ids)]
        if simulate_contrails:
            contrails = contrails[contrails['flight_id'].isin(valid_flight_ids)]
        
        # Additionally filter so that only flights which are near cameras at some point in their trajectory
        filter_string = f'haversine_distance < {max_camera_depth}'
        wypts_within_depth = wypts_data.query(filter_string)
        valid_flight_ids = set(wypts_within_depth['flight_id'].unique())
        wypts_data = wypts_data[wypts_data['flight_id'].isin(valid_flight_ids)]
        if simulate_contrails:
            contrails = contrails[contrails['flight_id'].isin(valid_flight_ids)]

        # Extract based on number of in frame waypoints
        # For now, just remove flights with extremely high number of waypoints
        # print(f"Number of flights: {len(wypts_data['flight_id'].unique())}")
        if extract_good_candidates_only:
            max_waypoints = 50
            number_of_in_frame_waypoints_per_flight_id = wypts_data.groupby('flight_id').apply(lambda x: x['in_frame'].sum())
            # print(f"Min number of waypoints in frame {np.min(number_of_in_frame_waypoints_per_flight_id)}")
            # print(f"Max number of waypoints in frame {np.max(number_of_in_frame_waypoints_per_flight_id)}")
            
            # The sum will be in per-second
            # print(number_of_in_frame_waypoints_per_flight_id)
            valid_flight_ids = number_of_in_frame_waypoints_per_flight_id[(number_of_in_frame_waypoints_per_flight_id <= max_waypoints * 10)].index
            wypts_data = wypts_data[wypts_data['flight_id'].isin(valid_flight_ids)]
            if simulate_contrails:
                contrails = contrails[contrails['flight_id'].isin(valid_flight_ids)]

        # print(f"Number of flights after filtering for num waypoints in frame: {len(wypts_data['flight_id'].unique())}")

        # Print min haversine distance for each flight_id
        # distances = []
        # for flight_id in valid_flight_ids:
            # distances.append((flight_id, np.min(wypts_data[wypts_data['flight_id'] == flight_id]['haversine_distance'])))

        # print(np.min([x[1] for x in distances]))
        # print(np.max([x[1] for x in distances]))

        # print(distances)

        nums = []
        for flight_id in valid_flight_ids:
            nums.append(np.min(wypts_data[wypts_data['flight_id'] == flight_id]['haversine_distance']))

        # Check if flight_id_filter is provided (in which case, only keep those flight ids)
        if len(flight_id_filter) > 0:
            wypts_data = wypts_data[wypts_data['flight_id'].isin(flight_id_filter)]
            if simulate_contrails:
                contrails = contrails[contrails['flight_id'].isin(flight_id_filter)]

        # Group data into dict of timestamps (TODO: can be optimized)
        self.hashed_flight_data = {}
        self.hashed_contrail_data = {}

        # Times in unix
        if len(wypts_data) != 0:
            min_time = wypts_data['time'].min().timestamp()
            max_time = wypts_data['time'].max().timestamp()

            for curr_time in range(int(min_time), int(max_time) + cam_frame_time, cam_frame_time):
                curr_pd = wypts_data[wypts_data['time'] == pd.to_datetime(curr_time, unit='s')]
                self.hashed_flight_data[curr_time] = curr_pd

            if simulate_contrails:
                min_time = contrails['time'].min().timestamp()
                max_time = contrails['time'].max().timestamp()

                for curr_time in range(int(min_time), int(max_time) + cam_frame_time, cam_frame_time):
                    curr_contrail_pd = contrails[contrails['time'] == pd.to_datetime(curr_time, unit='s')]
                    self.hashed_contrail_data[curr_time] = curr_contrail_pd

    def _preprocess_flight_data(self, files, output_meta_file_name, output_wypts_file_name, middle_lat_lon, cache_flight_data=True):
        spatial_bbox_buffer = 5  # Add +-`spatial_bbox_buffer` to longitude and latitude
        
        extract_cols = [
            'ingestion_time', 'icao_address', 'timestamp', 'latitude', 'longitude',
            'altitude_baro', 'callsign', 'tail_number', 'source', 'collection_type',
            'flight_number', 'aircraft_type_icao', 'aircraft_type_name',
            'airline_name', 'departure_airport_icao', 'arrival_airport_icao', 'heading'
        ]

        df = pd.read_parquet(files, columns=extract_cols)
    
        # Filter waypoints
        is_in_lon = df["longitude"].between(
            (middle_lat_lon[1] - spatial_bbox_buffer), (middle_lat_lon[1] + spatial_bbox_buffer), inclusive="both"
        )
        is_in_lat = df["latitude"].between(
            (middle_lat_lon[0] - spatial_bbox_buffer), (middle_lat_lon[0] + spatial_bbox_buffer), inclusive="both"
        )
        
        # Remove waypoints below 20,000 feet, not interested in LTO data points
        is_above_alt = (df["altitude_baro"] > 20000)
        
        # Discard erroneous waypoints where ingestion time is significantly different than the timestamp
        is_time = ((pd.to_datetime(df["ingestion_time"]) - pd.to_datetime(df["timestamp"])).dt.seconds) < 300
        df = df[is_in_lon & is_in_lat & is_above_alt & is_time].copy()
        
        # Create unique flight ID
        df["flight_id"] = df["icao_address"] + '_' + df["callsign"]
        
        # Resample waypoints to every 30 s - Reduce RAM intensity
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.index = df["timestamp"]
        df = df.groupby("flight_id").resample("30s").first()
        is_not_nan = df["longitude"].notna()
        df = df[is_not_nan].copy()
        df.reset_index(inplace=True, drop=True)
        df.rename(columns={"altitude_baro": "altitude_ft", "timestamp": "time"}, inplace=True)

        warnings.filterwarnings('ignore')

        MODEL_FUEL = JetA()

        cols_wypts = [
            'longitude', 'latitude', 'altitude_ft', 'time', 'source', 'collection_type', 'heading'
        ]

        grouped = list(df.groupby("flight_id"))

        flights = list()

        for flt_id, flt_wypts in tqdm(grouped):
            attrs = {
                "flight_id": flt_id,
                "icao_address": flt_wypts["icao_address"].iloc[0],
                "callsign": flt_wypts["callsign"].iloc[0],
                "tail_number": flt_wypts["tail_number"].iloc[0],
                "flight_number": flt_wypts["flight_number"].iloc[0],
                "aircraft_type": flt_wypts["aircraft_type_icao"].iloc[0],
                "aircraft_type_name": flt_wypts["aircraft_type_name"].iloc[0],
                "airline_name": flt_wypts["airline_name"].iloc[0],
                "origin_airport": flt_wypts["departure_airport_icao"].iloc[0],
                "destination_airport": flt_wypts["arrival_airport_icao"].iloc[0],
            }
            
            fl = Flight(data=flt_wypts[cols_wypts].copy(), attrs=attrs, flight_id=flt_id, fuel=MODEL_FUEL)
            fl.resample_and_fill(freq="30s")

            fl['segment_length'] = fl.segment_length()

            if len(fl.dataframe) > 0:
                flights.append(fl)

        output_wypts = list()
        output_metadata = list()

        for flt in tqdm(flights):
            flt_meta = extract_flight_metadata(flt)
            flt_wypts = flt.dataframe
            flt_wypts["flight_id"] = flt.attrs["flight_id"]
            output_metadata.append(flt_meta)
            output_wypts.append(flt_wypts)

        output_metadata = pd.concat(output_metadata, axis=1).T
        output_wypts = pd.concat(output_wypts)

        # Save files
        if cache_flight_data:
            output_metadata.to_parquet(output_meta_file_name)
            output_wypts.to_parquet(output_wypts_file_name)
            
        return output_metadata, output_wypts

    def _interpolate_flight_data(self, wypts_data, cam_frame_time=20):
        new_wypts_data = []
        flight_ids = wypts_data['flight_id'].unique()

        for flight_id in flight_ids:
            flight = wypts_data[wypts_data['flight_id'] == flight_id]
            new_flight = flight.copy()

            min_time = flight['time'].min()
            max_time = flight['time'].max()

            new_flight = new_flight.set_index('time').resample(f'{cam_frame_time}s').asfreq().reset_index()

            # Combine with old times
            new_flight = pd.concat([new_flight, flight]).sort_values('time').reset_index(drop=True).drop_duplicates()

            # Interpolate values
            new_flight = new_flight.interpolate(method='linear', limit_direction='both')

            # Fill NaN values with nearest valid
            new_flight = new_flight.fillna(method='ffill')
            new_flight = new_flight.fillna(method='bfill')

            new_flight = new_flight.set_index('time').resample(f'{cam_frame_time}S').asfreq().reset_index()

            # Only keep values values min and max time
            new_flight = new_flight[(new_flight['time'] >= min_time) & (new_flight['time'] <= max_time)]

            new_wypts_data.append(new_flight)

        new_wypts_data = pd.concat(new_wypts_data)
        return new_wypts_data

    def _simulate_contrails(self, output_contrails_file, metadata, flt_wypts, era5_met, era5_rad, cam_frame_time=20, cache_flight_data=True):
        # Initialise models
        ps = PSFlight(copy_source=False)
        emissions = Emissions(copy_source=False)

        use_cols = ['longitude', 'latitude', 'altitude_ft', 'time']

        flights = []

        for i in tqdm(range(len(metadata))):
            # Exclude aircraft types not covered by PS model
            if ps.check_aircraft_type_availability(metadata["aircraft_type_icao"].iloc[i], raise_error=False) is False:
                continue
            
            attrs = {
                "flight_id": metadata["flight_id"].iloc[i],
                "aircraft_type": metadata["aircraft_type_icao"].iloc[i], 
                "tail_number": metadata["tail_number"].iloc[i], 
            }
            
            is_flight = flt_wypts["flight_id"] == metadata["flight_id"].iloc[i]

            if np.sum(is_flight) < 2:
                continue

            flight = Flight(
                data=flt_wypts[is_flight][use_cols].copy(), 
                attrs=attrs,
                fuel=JetA()
                
            )

            # Get meteorology at each waypoint
            flight["air_temperature"] = flight.intersect_met(era5_met["air_temperature"])
            flight["u_wind"] = flight.intersect_met(era5_met["eastward_wind"])
            flight["v_wind"] = flight.intersect_met(era5_met["northward_wind"])
            flight["specific_humidity"] = flight.intersect_met(era5_met["specific_humidity"])

            # Calculate speeds
            flight["ground_speed"] = flight.segment_groundspeed()
            flight["true_airspeed"] = flight.segment_true_airspeed(flight["u_wind"], flight["v_wind"])

            # Simulate aircraft performance
            flight = ps.eval(flight)

            # Simulate emissions
            flight = emissions.eval(source=flight)
            flights.append(flight)

        # Initialise CoCiP
        params = {
            "humidity_scaling": ExponentialBoostLatitudeCorrectionHumidityScaling(copy_source=False),
            "dt_integration": np.timedelta64(cam_frame_time, "s"),
            "max_age": np.timedelta64(12, "h"),
            "radiative_heating_effects": True,
            "unterstrasser_ice_survival_fraction": True,
            "contrail_contrail_overlapping": False, 
        }

        cocip = Cocip(met=era5_met, rad=era5_rad, params=params)
        cocip.eval(flights)

        # retrieve the contrail output
        contrails = cocip.contrail

        if cache_flight_data:
            contrails.to_parquet(output_contrails_file)

        return contrails
    
    def fetch_flight_data(self, unix_time):
        if unix_time in self.hashed_flight_data:
            return self.hashed_flight_data[unix_time]
        else:
            return []

    def fetch_contrail_data(self, unix_time):
        if unix_time in self.hashed_contrail_data:
            return self.hashed_contrail_data[unix_time]
        else:
            return []
        
def extract_flight_metadata(fl: Flight) -> pd.Series:
    meta = {
        "flight_id": fl.attrs["flight_id"],
        "callsign": fl.attrs["callsign"],
        "icao_address": fl.attrs["icao_address"],
        "flight_number": fl.attrs["flight_number"],
        "tail_number": fl.attrs["tail_number"],
        "aircraft_type_icao": fl.attrs["aircraft_type"],
        "first_waypoint_time": fl.dataframe["time"].min(),
        "last_waypoint_time": fl.dataframe["time"].max(),
        "total_distance_km": np.nansum(fl["segment_length"]) / 1000,
        "longitude_min": fl.dataframe["longitude"].min(),
        "longitude_max": fl.dataframe["longitude"].max(),
        "latitude_min": fl.dataframe["latitude"].min(),
        "latitude_max": fl.dataframe["latitude"].max(),
        "altitude_min": fl.dataframe["altitude_ft"].min(),
        "altitude_max": fl.dataframe["altitude_ft"].max(),
        "altitude_first_waypoint": fl["altitude_ft"][0],
        "altitude_last_waypoint": fl["altitude_ft"][-1],
        "n_waypoints": len(fl.dataframe),
    }

    return pd.Series(meta)

def gmn_project_to_cameras(platepar_dict, sorted_platepar_keys, lons, lats, altitudes, center_coordinate=None):
    lons = np.array(lons)
    lats = np.array(lats)
    altitudes = np.array(altitudes)

    x_coords = []
    y_coords = []
    valids = []
    haversine_distances = []

    for i in range(len(lons)):
        xy = GeoHt2xy(platepar_dict[sorted_platepar_keys[i]], lats[i], lons[i], altitudes[i])
        x_coords.append(xy[0])
        y_coords.append(xy[1])
        valids.append((xy[0] > 0) & (xy[0] < 1280) & (xy[1] > 0) & (xy[1] < 720))   # TODO: Hardcoded image size

        if center_coordinate is not None:
            haversine_distances.append(haversine(lons[i], lats[i], center_coordinate[1], center_coordinate[0]))

    return np.array(x_coords), np.array(y_coords), np.array(valids), np.array(haversine_distances)
    

def add_camera_information(pd_data, platepar_dict, center_coordinate=None, calculate_image_heading=False):
    output_pd_data = pd_data.copy()
    
    if 'altitude_ft' in output_pd_data.columns:
        # Flight data is stored in ft
        altitude_meters = output_pd_data['altitude_ft'] * 0.3048
    elif 'altitude' in output_pd_data.columns:
        # Contrails data is only stored in m
        altitude_meters = output_pd_data['altitude']
    else:
        raise ValueError('Unexpected data format. Data should either only have `altitude_ft` or `altitude`')
    
    # Get a list of platepar names depending on the timestamp
    sorted_platepar_keys = []
    platepar_times = [(platepar_name, filenameToDatetime(platepar_name).timestamp()) for platepar_name in platepar_dict.keys()]
    platepar_times.sort(key=lambda x: x[1])

    for time in output_pd_data['time']:
        # Get the nearest platepar_name to the current time
        platepar_name = min(platepar_times, key=lambda x: abs(x[1] - time.timestamp()))[0]
        sorted_platepar_keys.append(platepar_name)

    x_coords, y_coords, valids, haversine_distances = gmn_project_to_cameras(platepar_dict, sorted_platepar_keys, output_pd_data['longitude'],
                                                                          output_pd_data['latitude'], altitude_meters,
                                                                          center_coordinate=center_coordinate)

    output_pd_data['x_coord'] = x_coords
    output_pd_data['y_coord'] = y_coords
    output_pd_data['in_frame'] = valids

    if center_coordinate is not None:
        output_pd_data['haversine_distance'] = haversine_distances
    
    if calculate_image_heading:
        # Rough calculation to get direction of airplane but in the image domain (only for flight waypoints)
        # print(pd_data['heading'][:10])
        next_latitude = output_pd_data['latitude'] + 1e-2 * np.cos(pd_data['heading'] * np.pi / 180)
        next_longitude = output_pd_data['longitude'] + 1e-2 * np.sin(pd_data['heading'] * np.pi / 180)

        next_x_coords, next_y_coords, _, _ = gmn_project_to_cameras(platepar_dict, sorted_platepar_keys, next_longitude, next_latitude, altitude_meters)

        output_pd_data["image_heading"] = 180 * np.arctan2(-(next_x_coords - x_coords), -(next_y_coords - y_coords)) / np.pi

    return output_pd_data
    
def advect_waypoints(last_frame_flight_info, platepar_dict, era5_met, curr_time):
    # Convert data to Flight object
    df = pd.DataFrame()
    df['longitude'] = last_frame_flight_info['advected_longitudes']
    df['latitude'] = last_frame_flight_info['advected_latitudes']
    df['altitude'] = last_frame_flight_info['heights_for_advection']
    df['time'] = pd.to_datetime(curr_time, unit='s')
    fl = Flight(df)

    fl['era5_east_wind'] = fl.intersect_met(era5_met['eastward_wind'], method='linear')
    fl['era5_north_wind'] = fl.intersect_met(era5_met['northward_wind'], method='linear')
    delta_time = np.timedelta64(int(round(curr_time - last_frame_flight_info['time'])), 's')

    advected_longitudes = advect_longitude(df['longitude'], df['latitude'], fl['era5_east_wind'], delta_time)
    advected_latitudes = advect_latitude(df['latitude'], fl['era5_north_wind'], delta_time)

    # Project to camera
    advected_df = pd.DataFrame()
    advected_df['longitude'] = advected_longitudes
    advected_df['latitude'] = advected_latitudes
    advected_df['altitude'] = df['altitude']
    advected_df['time'] = df['time']
    
    advected_df = add_camera_information(advected_df, platepar_dict)

    return advected_latitudes.tolist(), advected_longitudes.tolist(), advected_df['x_coord'].tolist(), advected_df['y_coord'].tolist()
