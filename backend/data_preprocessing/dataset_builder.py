import os
import re
import json
import csv
import pandas as pd
import numpy as np
import cv2


JSON_DIR = "backend/data_preprocessing/json_data"
CSV_DIR = "backend/data_preprocessing/csv_data"


def get_sorted_json_files_train(base_folder): # todo make similar function for testing without splitting in left, right, middle subfolders
    """
    Collects and sorts JSON file paths from 'left', 'right', and 'middle' subfolders
    based on frame numbers.

    Args:
    - base_folder (str): Path to the main folder containing 'left', 'right', and 'middle'.

    Returns:
    - List of JSON file paths sorted by frame number.
    """
    json_files = []

    pattern = re.compile(r'frame_(\d+)_keypoints\.json')

    for subfolder in ['left', 'right', 'middle']:
        subfolder_path = os.path.join(base_folder, subfolder)
        if os.path.exists(subfolder_path):
            for file in os.listdir(subfolder_path):
                match = pattern.match(file)
                if match:
                    frame_number = int(match.group(1))
                    file_path = os.path.join(subfolder_path, file)
                    if not is_empty_json(file_path):
                        json_files.append((frame_number, file_path))

    json_files.sort(key=lambda x: x[0])

    return [file_path for _, file_path in json_files]


def get_sorted_json_files_test(base_folder):

    json_files = []
    pattern = re.compile(r'frame_(\d+)_keypoints\.json')
    for file in os.listdir(base_folder):
        match = pattern.match(file)
        if match:
            frame_number = int(match.group(1))
            file_path = os.path.join(base_folder, file)
            if not is_empty_json(file_path):
                json_files.append((frame_number, file_path))

    json_files.sort(key=lambda x: x[0])

    return [file_path for _, file_path in json_files]

def is_empty_json(file_path):
    """
    Checks if a JSON file is empty.

    Args:
    - file_path (str): Path to the JSON file.

    Returns:
    - True if the JSON is empty, False otherwise.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return "people" in data and isinstance(data["people"],
                                                   list) and len(
                data["people"]) == 0
    except (json.JSONDecodeError, FileNotFoundError):
        return True


def get_csv_openpose_train(sorted_json_files, output_csv):
    # todo add description
    body_parts = [
        "Nose",
        "Neck",
        "RShoulder",
        "RElbow",
        "RWrist",
        "LShoulder",
        "LElbow",
        "LWrist",
        "MidHip",
        "RHip",
        "RKnee",
        "RAnkle",
        "LHip",
        "LKnee",
        "LAnkle",
        "REye",
        "LEye",
        "REar",
        "LEar",
        "LBigToe",
        "LSmallToe",
        "LHeel",
        "RBigToe",
        "RSmallToe",
        "RHeel"
    ]

    all_data = []
    valid_json_files = []
    modified_headers = []
    for header in body_parts:
        modified_headers.append(header + "_x")
        modified_headers.append(header + "_y")
    modified_headers.append("phase")
    modified_headers.append("img_name")
    for json_file in sorted_json_files:
        with open(json_file) as f:
            labels = json.load(f)
            people = labels["people"]
            if len(people) != 0:
                pose_keypoints = labels["people"][0]["pose_keypoints_2d"]
                coordinates = []
                to_delete = False
                for i in range(0, len(pose_keypoints), 3):
                    # if not (pose_keypoints[i] == 0 or pose_keypoints[
                    #     i + 1] == 0): #todo check this condition
                    coordinates.append(pose_keypoints[i])
                    coordinates.append(pose_keypoints[i + 1])
                    # else:
                    #     to_delete = True
                    #     break

                if not to_delete:
                    turn_phase = os.path.basename(os.path.dirname(json_file))
                    coordinates.append(turn_phase)
                    base_name = os.path.basename(json_file)
                    cropped_name = base_name.replace('_keypoints.json',
                                                     '')
                    valid_json_files.append(cropped_name)

                    coordinates.append(cropped_name)
                    all_data.append(coordinates)

    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(modified_headers)
        for coordinates in all_data:
            csvwriter.writerow(coordinates)

    return valid_json_files


def get_csv_openpose_test(sorted_json_files, output_csv):
    # todo add description
    body_parts = [
        "Nose",
        "Neck",
        "RShoulder",
        "RElbow",
        "RWrist",
        "LShoulder",
        "LElbow",
        "LWrist",
        "MidHip",
        "RHip",
        "RKnee",
        "RAnkle",
        "LHip",
        "LKnee",
        "LAnkle",
        "REye",
        "LEye",
        "REar",
        "LEar",
        "LBigToe",
        "LSmallToe",
        "LHeel",
        "RBigToe",
        "RSmallToe",
        "RHeel"
    ]

    all_data = []
    valid_json_files = []
    modified_headers = []
    for header in body_parts:
        modified_headers.append(header + "_x")
        modified_headers.append(header + "_y")
    modified_headers.append("img_name")
    for json_file in sorted_json_files:
        with open(json_file) as f:
            labels = json.load(f)
            people = labels["people"]
            if len(people) != 0:
                pose_keypoints = labels["people"][0]["pose_keypoints_2d"]
                coordinates = []
                to_delete = False
                for i in range(0, len(pose_keypoints), 3):
                    # if not (pose_keypoints[i] == 0 or pose_keypoints[
                    #     i + 1] == 0):
                    coordinates.append(pose_keypoints[i])
                    coordinates.append(pose_keypoints[i + 1])
                    # else:
                    #     to_delete = True
                    #     break

                if not to_delete:

                    base_name = os.path.basename(json_file)
                    cropped_name = base_name.replace('_keypoints.json',
                                                     '')
                    valid_json_files.append(cropped_name)
                    coordinates.append(cropped_name)
                    all_data.append(coordinates)

    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(modified_headers)
        for coordinates in all_data:
            csvwriter.writerow(coordinates)

    return valid_json_files

def count_turn_transitions(csv_path):
    # todo add description
    with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)
        if "phase" not in header[-1].lower():
            raise ValueError("The last column must contain 'phase' values.")

        phases = [row[-1].strip() for row in reader]

    left_after_right = 0
    right_after_left = 0
    correct_right_middle_right = 0
    correct_left_middle_left = 0

    prev_phase = None
    prev_prev_phase = None

    for phase in phases:
        if prev_phase == "right" and phase == "left":
            left_after_right += 1
        if prev_phase == "left" and phase == "right":
            right_after_left += 1
        if prev_prev_phase == "right" and prev_phase == "middle" and phase == "right":
            correct_right_middle_right += 1
        if prev_prev_phase == "left" and prev_phase == "middle" and phase == "left":
            correct_left_middle_left += 1

        prev_prev_phase = prev_phase
        prev_phase = phase

    return {
        "left_after_right": left_after_right,
        "right_after_left": right_after_left,
        "right_middle_right": correct_right_middle_right,
        "left_middle_left": correct_left_middle_left
    }


def sort_images_by_runs_json_based(json_files, input_dir, output_dir):
    """
    Iterates through JSON files, finds the corresponding image by frame number,
    and sorts images into ski run folders based on user input.

    Args:
        json_root (str): Root directory containing sorted JSON files (left, middle, right).
        input_dir (str): Directory containing images named as frame_XXXX.jpg.
        output_dir (str): Directory where sorted runs will be stored.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_list = []
    for json_path in json_files:
        filename = os.path.basename(json_path)
        frame_number = filename.split("_")[1]

        image_filename = f"frame_{frame_number}.jpg"
        image_path = os.path.join(input_dir, image_filename)

        if os.path.exists(image_path):
            image_list.append((image_path, json_path))

    display_idx = 0
    run_counter = 1
    run_dir = os.path.join(output_dir, f"run{run_counter}")
    os.makedirs(run_dir, exist_ok=True)
    while display_idx < len(image_list):
        image_path, json_path = image_list[display_idx]
        frame_number = os.path.basename(image_path).split("_")[1].split(".")[0]

        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read image: {image_path}")
            display_idx += 1
            continue

        cv2.imshow(f"Frame {frame_number} - Run {run_counter}", img)
        print(f"Processing: {json_path} -> {image_path} (Run {run_counter})")

        key = cv2.waitKey(0)

        if key in [27, ord('x')]:
            break
        elif key in [3, ord('n')]:
            display_idx += 1
        elif key in [2, ord('b')]:
            display_idx = max(0, display_idx - 1)
        elif key == ord('r'):
            run_counter += 1
            run_dir = os.path.join(output_dir, f"run{run_counter}")
            os.makedirs(run_dir, exist_ok=True)
            print(f"New run started: {run_dir}")

        cv2.imwrite(os.path.join(run_dir, os.path.basename(image_path)), img)
        cv2.destroyAllWindows()
    cv2.destroyAllWindows()


def get_first_image_indices(output_dir):
    """
    Get the indices (positions) of the first image from each run folder inside the output directory,
    in sorted order, considering cumulative indices across all runs.

    Args:
        output_dir (str): Path to the directory containing run folders.

    Returns:
        list: A list of indices representing the position of the first image in each run folder,
              sorted by run number, with cumulative indexing.
    """
    first_image_indices = []
    cumulative_index = 0
    run_folders = sorted(
        [f for f in os.listdir(output_dir) if
         f.startswith("run") and os.path.isdir(os.path.join(output_dir, f))],
        key=lambda x: int(re.search(r'\d+', x).group())
    )

    for run_folder in run_folders:
        run_path = os.path.join(output_dir, run_folder)
        images = sorted(
            [f for f in os.listdir(run_path) if
             f.lower().endswith(('.jpg', '.png'))]
        )
        if images:
            first_image_indices.append(cumulative_index)
            cumulative_index += len(images)

    return first_image_indices


def scale_neck(df, output_name):
    # todo add description
    df = df.copy()
    neck_x = df["Neck_x"]
    neck_y = df["Neck_y"]
    x_columns = [col for col in df.columns if col.endswith("_x")]
    y_columns = [col for col in df.columns if col.endswith("_y")]

    df[x_columns] -= neck_x.values.reshape(-1, 1)
    df[y_columns] -= neck_y.values.reshape(-1, 1)

    df.to_csv(output_name, index=False)


def scale_b_b(df, output_name):
    # todo add description
    x_columns = [col for col in df.columns if col.endswith("_x")]
    y_columns = [col for col in df.columns if col.endswith("_y")]
    min_x = df[x_columns].min(axis=1)
    max_x = df[x_columns].max(axis=1)
    min_y = df[y_columns].min(axis=1)
    max_y = df[y_columns].max(axis=1)
    width = max_x - min_x
    height = max_y - min_y

    print(width, height)
    df[x_columns] = (df[x_columns] - min_x.values.reshape(-1,
                                                          1)) / width.values.reshape(
        -1, 1)
    df[y_columns] = (df[y_columns] - min_y.values.reshape(-1,
                                                          1)) / height.values.reshape(
        -1, 1)

    df.to_csv(output_name, index=False)


def add_neck_knee_dist(df, output_folder, train=False):
    # todo add description
    def euclidean_distance(x1, y1, x2, y2):
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    df['Nose_left_knee_dist'] = euclidean_distance(df['Nose_x'],
                                                   df['Nose_y'],
                                                   df['LKnee_x'],
                                                   df['LKnee_y'])

    df['Nose_right_knee_dist'] = euclidean_distance(df['Nose_x'],
                                                    df['Nose_y'],
                                                    df['RKnee_x'],
                                                    df['RKnee_y'])

    columns = list(df.columns)

    columns.remove('Nose_right_knee_dist')
    columns.remove('Nose_left_knee_dist')
    if train:
        columns.insert(-2, 'Nose_right_knee_dist')
        columns.insert(-3, 'Nose_left_knee_dist')
    else:
        columns.insert(-1, 'Nose_right_knee_dist')
        columns.insert(-2, 'Nose_left_knee_dist')
    df = df[columns]
    df.to_csv(output_folder, index=False)


def build_dataset(coordinates_folder_name, train=False):
    coordinates_path = os.path.join(JSON_DIR, coordinates_folder_name)
    csv_path = os.path.join(CSV_DIR, f"{coordinates_folder_name}.csv")

    # 1. sort json files in numeric order to reconstruct the initial frames sequence
    if train:
        sorted_json_files = get_sorted_json_files_train(coordinates_path)
    else:
        sorted_json_files = get_sorted_json_files_test(coordinates_path)

    # 2. write the coordinates and labels to csv
    if train:
        get_csv_openpose_train(sorted_json_files, csv_path)
    else:
        get_csv_openpose_test(sorted_json_files, csv_path)

    output_csv_scaled_b_b_path = os.path.join(CSV_DIR, f"{coordinates_folder_name}_scaled_b_b.csv")
    output_csv_scaled_b_b_neck_path = os.path.join(CSV_DIR,f"{coordinates_folder_name}_scaled_b_b_neck.csv")
    output_csv_scaled_b_b_neck_knee_dist_path = os.path.join(CSV_DIR,f"{coordinates_folder_name}_scaled_b_b_neck_knee_dist.csv")

    # 3. scale coordinates using bounding box
    df = pd.read_csv(csv_path)
    scale_b_b(df, output_csv_scaled_b_b_path)

    # 4. move the coordinates centre to the neck coordinate
    df = pd.read_csv(output_csv_scaled_b_b_path)
    scale_neck(df, output_csv_scaled_b_b_neck_path)

    # 5. add a new feature - the distance between neck and knees
    df = pd.read_csv(output_csv_scaled_b_b_neck_path)
    add_neck_knee_dist(df, output_csv_scaled_b_b_neck_knee_dist_path, train)
    return output_csv_scaled_b_b_neck_knee_dist_path


    # DRAFT
    # STEPS

    # len pose_keypoints_2d 75
    # sorted_json_files = get_sorted_json_files(base_folder)
    # print("openpose detected ", len(sorted_json_files))
    #
    # input_dir = "png_data/gs_training_folder_fps10"
    # output_dir = "png_data/gs_training_folder_fps10_by_runs_after_preprocessing"
    #
    # file_names = get_csv_openpose(sorted_json_files, output_csv)
    # # sort_images_by_runs_json_based(file_names, input_dir, output_dir)
