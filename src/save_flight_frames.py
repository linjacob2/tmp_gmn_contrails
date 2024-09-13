# Function for extracting candidate contrails (in the form of a image sequence) from a video file.
# This is done by checking for flight trajectories intersecting with the cameras.

# TODO: Timestamp of the videos could still be off by roughly 10 seconds at most

import shutil
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import json
import warnings
warnings.filterwarnings("ignore")

from utils.flight_utils import FlightData, advect_waypoints
from collections import defaultdict
from RMS.Formats.FFfile import filenameToDatetime
from utils.data_loading_utils import load_platepars

import imageio
from glob import glob
from datetime import datetime, timezone
from PIL import Image

def save_flight_frames(output_dir, dataset, video_dir, flight_data_paths, output_meta_file, output_wypts_file,
                       era5_met=None, flight_id_filter=[], cache_flight_data=True, num_frames_after_flight=100, verbose=True):
    assert dataset in ['gmn']

    platepar_files, center_coordinate = load_platepars(video_dir)
    video_file = glob(os.path.join(video_dir, '*timelapse*.mp4'))[0]

    flight_data = FlightData(flight_data_paths, output_meta_file, output_wypts_file, None,
                             center_coordinate, platepar_files,
                             era5_met=era5_met, era5_rad=None, simulate_contrails=False,
                             flight_id_filter=flight_id_filter, bound_range=1.5,
                             cache_flight_data=cache_flight_data, cam_frame_time=1,
                             calculate_image_heading=False)

    os.makedirs(output_dir, exist_ok=True)

    if len(flight_data.hashed_flight_data) == 0:
        print("No flight data found, exiting")

        # If finished with the video, mark it by saving a 'done.txt' file
        with open(os.path.join(output_dir, 'done.txt'), 'w') as f:
            date_today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f'Done, {date_today}')

        return
    
    curr_time = filenameToDatetime(video_file).timestamp() + 5  # Just try to offset the timestamp error a bit, most accurate would be to read off the timestamp text in the video

    vid = imageio.get_reader(video_file,  'ffmpeg')

    curr_in_frame_flights = dict()
    exited_flights = dict()

    for frame_idx, image in enumerate(vid):
        # Fetch flight data for the current time
        fetched_flights = flight_data.fetch_flight_data(int(round(curr_time)))

        if len(fetched_flights) > 0:
            in_frame_flights = fetched_flights[fetched_flights[f'in_frame']]

            this_frame_ids = set()
            if len(in_frame_flights) > 0:

                for _, flight in in_frame_flights.iterrows():
                    # Make sure that flight did not disappear and reappear
                    if flight['flight_id'] in exited_flights:
                        print(f'Found flight which reappeared: {flight["flight_id"]}')
                        continue                        

                    this_frame_ids.add(flight['flight_id'])

                    flight_id = flight['flight_id']

                    # Create folders
                    metadata_dir = os.path.join(output_dir, flight_id, 'metadata')
                    inframe_image_dir = os.path.join(output_dir, flight_id, 'inframe_images')
                    afterframe_image_dir = os.path.join(output_dir, flight_id, 'afterframe_images')
                    os.makedirs(inframe_image_dir, exist_ok=True)
                    os.makedirs(afterframe_image_dir, exist_ok=True)
                    os.makedirs(metadata_dir, exist_ok=True)

                    inframe_image_file_path = os.path.join(inframe_image_dir, f'{str(frame_idx).zfill(5)}.jpg')

                    if flight['flight_id'] not in curr_in_frame_flights:
                        curr_in_frame_flights[flight_id] = dict({'frames_with_flight': [],
                                                                 'frames_after_flight': [],
                                                                 })

                    # Do advection here
                    if len(curr_in_frame_flights[flight_id]['frames_with_flight']) > 0:
                        (advected_latitudes, advected_longitudes,
                        advected_x_coords, advected_y_coords) = advect_waypoints(curr_in_frame_flights[flight_id]['frames_with_flight'][-1],
                                                                                 platepar_files, era5_met, curr_time)
                        
                        heights_for_advection = curr_in_frame_flights[flight_id]['frames_with_flight'][-1]['heights_for_advection'].copy()
                    else:
                        advected_latitudes = []
                        advected_longitudes = []
                        advected_x_coords = []
                        advected_y_coords = []
                        heights_for_advection = []

                    # Save current waypoint for future advection
                    advected_latitudes.append(flight['latitude'])
                    advected_longitudes.append(flight['longitude'])
                    advected_x_coords.append(flight['x_coord'])
                    advected_y_coords.append(flight['y_coord'])
                    heights_for_advection.append(flight['altitude_ft'] * 0.3048)

                    new_entry = dict({'latitude': flight['latitude'], 'longitude': flight['longitude'],
                                      'x_coord': flight['x_coord'], 'y_coord': flight['y_coord'],
                                      'advected_latitudes': advected_latitudes, 'advected_longitudes': advected_longitudes,
                                      'advected_x_coords': advected_x_coords, 'advected_y_coords': advected_y_coords,
                                      'heights_for_advection': heights_for_advection,
                                      'file_path': inframe_image_file_path, 'time': curr_time})
                    
                    curr_in_frame_flights[flight_id]['frames_with_flight'].append(new_entry)

                    # Check if image is already saved
                    if not os.path.exists(inframe_image_file_path):
                        Image.fromarray(image[:, :, 0], mode="L").save(inframe_image_file_path)

            # Check if any flights have left the frame
            for flight_id in list(curr_in_frame_flights.keys()):
                if flight_id not in this_frame_ids:
                    exited_flights[flight_id] = curr_in_frame_flights[flight_id]
                    exited_flights[flight_id]['frame_exited'] = frame_idx
                    del curr_in_frame_flights[flight_id]


        for flight_id in list(exited_flights.keys()):
            metadata_dir = os.path.join(output_dir, flight_id, 'metadata')
            inframe_image_dir = os.path.join(output_dir, flight_id, 'inframe_images')
            afterframe_image_dir = os.path.join(output_dir, flight_id, 'afterframe_images')
            afterframe_image_file_path = os.path.join(afterframe_image_dir, f'{str(frame_idx).zfill(5)}.jpg')

            if frame_idx - exited_flights[flight_id]['frame_exited'] > num_frames_after_flight:

                # Write JSON file
                with open(os.path.join(metadata_dir, f'{flight_id}.json'), 'w') as f:
                    json.dump(exited_flights[flight_id], f)

                del exited_flights[flight_id]

                continue

            # Advect flights
            if len(exited_flights[flight_id]['frames_after_flight']) > 0:
                last_entry = exited_flights[flight_id]['frames_after_flight'][-1]
            else:
                last_entry = exited_flights[flight_id]['frames_with_flight'][-1]

            # Do advection here
            (advected_latitudes, advected_longitudes,
            advected_x_coords, advected_y_coords) = advect_waypoints(last_entry, platepar_files, era5_met, curr_time)
                
            heights_for_advection = last_entry['heights_for_advection'].copy()

            new_entry = dict({'latitude': last_entry['latitude'], 'longitude': last_entry['longitude'],
                                'x_coord': last_entry['x_coord'], 'y_coord': last_entry['y_coord'],
                                'advected_latitudes': advected_latitudes, 'advected_longitudes': advected_longitudes,
                                'advected_x_coords': advected_x_coords, 'advected_y_coords': advected_y_coords,
                                'heights_for_advection': heights_for_advection,
                                'file_path': afterframe_image_file_path, 'time': curr_time})
                
            exited_flights[flight_id]['frames_after_flight'].append(new_entry)

            # Check if image is already saved
            if not os.path.exists(afterframe_image_file_path):
                Image.fromarray(image[:, :, 0], mode="L").save(afterframe_image_file_path)

        curr_time += 10.248   # Add average time between frames

    # If finished with the video, mark it by saving a 'done.txt' file
    with open(os.path.join(output_dir, 'done.txt'), 'w') as f:
        date_today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f'Done, {date_today}')

    # Also copy over the platepar and config files
    platepar_file = glob(os.path.join(video_dir, '*platepar*_flux_recalibrated*'))[0]
    config = os.path.join(video_dir, '.config')

    output_platepar_file = os.path.join(output_dir, os.path.basename(platepar_file))
    output_config = os.path.join(output_dir, os.path.basename(config))

    shutil.copy(platepar_file, output_platepar_file)
    shutil.copy(config, output_config)
