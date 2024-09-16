import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import shutil

import warnings
from glob import glob
import json
import torchvision.transforms as transforms
from sam2.build_sam import build_sam2_video_predictor
from sam2.utils.misc import get_connected_components
from utils.data_loading_utils import load_platepars

from RMS.Formats.FFfile import filenameToDatetime

from RMS.Astrometry.ApplyAstrometry import XyHt2Geo

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def _preprocess_images(metadata, output_folder):
    query = []

    for frame in metadata['frames_with_flight']:
        query.append((frame['x_coord'], frame['y_coord']))

    # x_coords = first_frame['advected_x_coords']
    # y_coords = first_frame['advected_y_coords']
    # query = [[x_coords[x], y_coords[x]] for x in range(len(x_coords))]

    # Find difference to nearest 90 degrees
    angle = np.arctan2(query[-1][1] - query[0][1], query[-1][0] - query[0][0]) * 180 / np.pi
    nearest_90 = round(angle / 90) * 90
    diff_angle = -(nearest_90 - angle)
    query = np.array(query, dtype=np.float32)

    # Rotate all images and save to a temporary folder
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
        
    os.makedirs(output_folder, exist_ok=True)
    for frame in metadata['frames_after_flight']:
        img = Image.open(frame['file_path']).convert('RGB')
        orig_w, orig_h = img.size
        img = img.rotate(diff_angle, expand=True)
        expanded_w, expanded_h = img.size
        img.save(os.path.join(output_folder, os.path.basename(frame['file_path'])))

    # Calculate the expansion in the rotated images
    image_centre = (orig_w // 2, orig_h // 2)
    rot_matrix = np.array([[np.cos(np.radians(diff_angle)), -np.sin(np.radians(diff_angle))], [np.sin(np.radians(diff_angle)), np.cos(np.radians(diff_angle))]])
    
    corners = np.array([[0, 0], [orig_w, 0], [orig_w, orig_h], [0, orig_h]])
    corners = corners - image_centre
    corners = np.dot(corners, rot_matrix)
    corners = corners + image_centre
    min_x = np.min(corners[:, 0])
    min_y = np.min(corners[:, 1])
    max_x = np.max(corners[:, 0])
    max_y = np.max(corners[:, 1])
    min_offset = np.array([min_x, min_y])
    max_offset = np.array([max_x, max_y])

    return query, diff_angle, min_offset, max_offset, orig_w, orig_h, expanded_w, expanded_h

def segment_contrails(flight_dir, sam2_checkpoint="./sam2_checkpoints/sam2_hiera_large.pt", model_cfg="sam2_hiera_l.yaml",
                      num_query_frames=1, box_margin=10, binary_threshold=0.0, debug=False):
    # flight_dir: directory name with flight_id as the name e.g. "3C6514_DLH413"

    # use bfloat16
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # flight_id = os.path.basename(flight_dir)
    meta_data = glob(os.path.join(flight_dir, 'metadata/*.json'))[0]
    with open(meta_data, 'r') as f:
        data = json.load(f)

    tmp_sam2_folder = 'tmp_sam2_folder_output'
    query, diff_angle, min_offset, max_offset, orig_w, orig_h, expanded_w, expanded_h = _preprocess_images(data, tmp_sam2_folder)

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    inference_state = predictor.init_state(video_path=tmp_sam2_folder)

    # Add box labels for the first frame
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

    # rotate query points around diff_angle
    image_centre = (orig_w // 2, orig_h // 2)
    query = query - image_centre
    rot_matrix = np.array([[np.cos(np.radians(diff_angle)), -np.sin(np.radians(diff_angle))], [np.sin(np.radians(diff_angle)), np.cos(np.radians(diff_angle))]])
    query = np.dot(query, rot_matrix)
    query = query + image_centre
    query = query - min_offset

    margin = box_margin
    min_x = np.min(query[:, 0]) - margin
    max_x = np.max(query[:, 0]) + margin
    min_y = np.min(query[:, 1]) - margin
    max_y = np.max(query[:, 1]) + margin
    
    # Clip all values to fit image frame
    min_x = max(0, min_x)
    max_x = min(expanded_w, max_x)
    min_y = max(0, min_y)
    max_y = min(expanded_h, max_y)

    box = np.array([min_x, min_y, max_x, max_y], dtype=np.float32)

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=ann_obj_id,
        # points=query,
        box=box,
        # labels=labels,
    )

    first_frame_mask_features = extract_mask_features((out_mask_logits[0, 0, :, :] > binary_threshold).cpu().numpy(), bbox=[int(min_x), int(min_y), int(max_x), int(max_y)])

    # Visualise outputs if debug=True
    if debug:
        debug_folder = os.path.join(flight_dir, 'debug')
        os.makedirs(debug_folder, exist_ok=True)

        frame_names = [
            p for p in os.listdir(tmp_sam2_folder)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
        ]
        frame_names = sorted(frame_names)

        labels = torch.ones(query.shape[0], dtype=torch.int64)

        plt.figure(figsize=(12, 8))
        plt.title(f"frame {0}")
        plt.imshow(Image.open(os.path.join(tmp_sam2_folder, frame_names[0])))
        show_points(query, labels, plt.gca())
        show_box(box, plt.gca())
        show_mask((out_mask_logits[0] > binary_threshold).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

        plt.savefig(os.path.join(debug_folder, f'{str(0).zfill(5)}.png'))

        plt.close('all')

    # Propagate the predictions across the video
    # run propagation throughout the video and collect the results in a dict
    output_dir = os.path.join(flight_dir, 'sam2_output')
    os.makedirs(output_dir, exist_ok=True)
    masks = []

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        out_mask_logits = transforms.functional.affine(out_mask_logits, angle=diff_angle, translate=(min_offset[0], min_offset[1]), scale=1.0, shear=0.0)
        out_mask_logits = out_mask_logits[:, :, :orig_h, :orig_w]

        # Save as binary image for efficient storage
        out_mask_logits = (out_mask_logits > binary_threshold).squeeze().cpu().numpy()
        masks.append(out_mask_logits)
        mask = Image.fromarray(out_mask_logits)
        mask.save(os.path.join(output_dir, f'{str(out_frame_idx).zfill(5)}.png'), bits=1, optimize=True)

    # Get temporal mask features
    temporal_mask_features = extract_temporal_mask_features(masks)
    combined_features = {**first_frame_mask_features, **temporal_mask_features}

    # Save mask features as a json file
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump({'mask_features': combined_features}, f)

    # Remove temporary folder
    shutil.rmtree(tmp_sam2_folder)

def extract_mask_features(mask, bbox=None, device='cuda'):
    features = dict()
    
    # Only extract features from the largest connected component in the mask
    # This is important, as speckles can often result in false positives when extracting features
    # such as length/width ratio of a contrail
    try:
        labels, areas = get_connected_components(torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device))
        max_area_idx = torch.argmax(areas)
        max_area_label = labels.flatten()[max_area_idx]
        mask = (labels == max_area_label).cpu().numpy().squeeze()
    except:
        warnings.warn('Skipping SAM2 post-processing, and also single connected component filtering for the contrail feature extraction from the SAM2 mask. This is likely due to SAM2 CUDA failing to build.',
                      category=UserWarning,
                      stacklevel=2,)

    # Convert bbox of form ([min_x, min_y, max_x, max_y]) to binary mask
    if bbox is not None:
        bbox_mask = np.zeros_like(mask)
        bbox_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1

        # Calculate intersection over union of mask and bbox_mask
        intersection = np.logical_and(mask, bbox_mask)
        union = np.logical_or(mask, bbox_mask)
        iou = np.sum(intersection) / np.sum(union)
        features['iou'] = iou

        # Check how much percentage of the mask is inside the bbox mask
        if np.sum(mask) != 0:
            features['mask_in_bbox'] = np.sum(intersection) / np.sum(mask)
        else:
            features['mask_in_bbox'] = 0

    # In case detection without flight data is desired, simpler metrics could be used, e.g. total area of the mask
    features['total_area'] = int(np.sum(mask))

    # Get length / width of the mask. TODO: this could be more accurate in case the contrail is not aligned with the bbox.
    # Currently it should be aligned to some extent since the mask is already rotated to match the flight waypoints.
    y_coords, x_coords = np.where(mask)
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    length = max_x - min_x
    width = max_y - min_y
    
    # It is not guaranteed that the flight waypoints are horizontal instead of vertical. So assign "length" to the longer side.
    if length < width:
        length, width = width, length

    features['length'] = int(length)
    features['width'] = int(width)

    return features

def extract_temporal_mask_features(masks, device='cuda'):
    features = dict()
    masks = torch.from_numpy(np.array(masks)).float().to(device)

    # Get change in mask for each time step
    diff_masks = masks[1:, :, :] - masks[:-1, :, :]
    per_frame_temporal_variability = torch.mean(torch.abs(diff_masks), dim=(1, 2))
    features['temporal_variability'] = torch.mean(per_frame_temporal_variability ** 2).item()

    return features

def _lift_single_mask_to_geo(platepar, mask, height):
    # Get x, y coordinates of all ones in the mask
    y_coords, x_coords = np.where(mask)

    # Project to longitude latitude (code from Luc is not vectorised, so it needs to be looped)
    lats = []
    lons = []
    for i in range(len(x_coords)):
        lat, lon = XyHt2Geo(platepar, x_coords[i], y_coords[i], height)
        lats.append(lat)
        lons.append(lon)
    
    return np.array(lats), np.array(lons)

def masks_to_geo(base_dir, contrail_flights):
    # contrail_flights is a list of flight_ids where contrails have been detected

    # Load platepar files
    platepar_files, center_coordinate = load_platepars(base_dir)

    # Go through all flight directories in base_dir
    flight_dirs = [f.path for f in os.scandir(base_dir) if f.is_dir()]

    for flight_dir in flight_dirs:
        flight_id = os.path.basename(flight_dir)

        if flight_id not in contrail_flights:
            continue

        output_dir = os.path.join(flight_dir, 'mask_geo')
        os.makedirs(output_dir, exist_ok=True)

        meta_data_file = glob(os.path.join(flight_dir, 'metadata/*.json'))

        if len(meta_data_file) == 0:
            print(f"No metadata found for {flight_dir} (most likely no frames saved)")
            continue

        with open(meta_data_file[0], 'r') as f:
            meta_data = json.load(f)

        # Get height of the contrail
        first_mask_frame = meta_data['frames_with_flight'][0]
        contrail_height = first_mask_frame['heights_for_advection'][0]
        time = first_mask_frame['time']

        # TODO: Get platepar closest in time
        platepar_times = [(platepar_name, filenameToDatetime(platepar_name).timestamp()) for platepar_name in platepar_files.keys()]
        platepar_times.sort(key=lambda x: x[1])
        platepar_name = min(platepar_times, key=lambda x: abs(x[1] - time))[0]
        platepar = platepar_files[platepar_name]

        # Get masks
        mask_dir = os.path.join(flight_dir, 'sam2_output')
        mask_files = glob(os.path.join(mask_dir, '*.png'))

        output_dict = {}
        output_dict['masks'] = {}

        # Project image corners to geo by making a fake mask
        fake_mask = np.zeros_like(np.array(Image.open(mask_files[0])))
        fake_mask[0, 0] = 1
        fake_mask[0, -1] = 1
        fake_mask[-1, 0] = 1
        fake_mask[-1, -1] = 1
        lats, lons = _lift_single_mask_to_geo(platepar, fake_mask, contrail_height)
        output_dict['image_corners'] = {'lats': lats.tolist(), 'lons': lons.tolist()}

        for mask_file in mask_files:
            mask = Image.open(mask_file)
            mask = np.array(mask)

            # Get world coordinates
            lats, lons = _lift_single_mask_to_geo(platepar, mask, contrail_height)
            output_dict['masks'][mask_file] = {'lats': lats.tolist(), 'lons': lons.tolist()}

        # Save to json
        with open(os.path.join(output_dir, 'mask_geo.json'), 'w') as f:
            json.dump(output_dict, f)

