import json
from glob import glob
import os

from RMS.Formats.Platepar import Platepar
import RMS.ConfigReader as cr
from RMS.Astrometry.ApplyAstrometry import rotationWrtHorizon

from pycontrails.datalib.ecmwf import ERA5
from pycontrails.models.cocip import Cocip

def load_platepars(folder_path):
    platepar_file = glob(os.path.join(folder_path, '*platepar*_flux_recalibrated*'))[0]
    config = os.path.join(folder_path, '.config')

    config = cr.loadConfigFromDirectory(config, folder_path)
    center_coordinate = None

    with open(platepar_file) as f:
        recalibrated_platepars_dict = json.load(f)

        # Convert the dictionary of recalibrated platepars to a dictionary of Platepar objects
        recalibrated_platepars = {}
        file_names = []
        for ff_name in recalibrated_platepars_dict:
            file_names.append(ff_name)

            if center_coordinate is None:
                center_coordinate = (recalibrated_platepars_dict[ff_name]['lat'], recalibrated_platepars_dict[ff_name]['lon'])

            pp = Platepar()
            pp.loadFromDict(recalibrated_platepars_dict[ff_name], use_flat=config.use_flat)

            new_rotation_from_horiz = rotationWrtHorizon(pp)
            pp.rotation_from_horiz = new_rotation_from_horiz

            recalibrated_platepars[ff_name] = pp

    return recalibrated_platepars, center_coordinate

def get_flight_files(times, flight_data_base_path='data/flight_data/'):
    all_flight_data_files = []
    for (year, month, day, hour) in times:
        date = f'{year}{month:02d}{day:02d}'
        time_prefix = f'{hour:02d}'

        flight_data_file_paths = glob(os.path.join(flight_data_base_path, f'{year}_{month:02d}/{date}-{time_prefix}*.pq'))
        all_flight_data_files.extend(flight_data_file_paths)
    
    return sorted(all_flight_data_files)

def get_era5_data(year, month, day, hour, end_year, end_month, end_day, end_hour, file_path_met):
    # Load ERA5 data
    era5_levels = [
    150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600,
    650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000,
    ]

    era5_met = ERA5(
        time=(f'{year}-{month:02d}-{day:02d} {hour:02d}:00:00', f'{end_year}-{end_month:02d}-{end_day:02d} {end_hour}:00:00'),
        # time=("2023-08-09 18:00:00", "2023-08-10 12:00:00"),
        # time=("2023-08-20 09:00:00", "2023-08-20 23:00:00"),
        variables=Cocip.met_variables,
        pressure_levels=era5_levels,
        paths=file_path_met,
        # cachestore=None
    )

    met = era5_met.open_metdataset()

    return met