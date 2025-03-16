from flask import Blueprint, request
import os
import subprocess
import glob
import shutil
import threading

api_bp = Blueprint("api", __name__)

ALLOWED_EXTENSIONS = ["mp4", "mov", "mkv"] # todo check allowed extensions for ffmpeg
VIDEO_DIR = "backend/data_preprocessing/videos"
FRAMES_DIR = "backend/data_preprocessing/frames"
JSON_DIR = "backend/data_preprocessing/json_data"
BATCH_SIZE = 10

@api_bp.get('/')
def home():
    return 'Flask is running!'

# todo move to service all the logic
@api_bp.post('/extract-frames')
def extract_frames():

    full_video_name = request.get_json()['video_name']
    fps = request.get_json()['fps']

    video_name = full_video_name.rsplit('.', 1)[0]
    video_extension = full_video_name.rsplit('.', 1)[1].lower()
    print("video_name", video_name)
    print("video_extension", video_extension)

    video_path = os.path.join(VIDEO_DIR, full_video_name)
    frames_name = f'{video_name}_fps{fps}'
    frames_path = os.path.join(FRAMES_DIR, frames_name)
    print('FRAMES path', frames_path)

    os.makedirs(frames_path, exist_ok=True)

    output_pattern = os.path.join(frames_path, "frame_%04d.jpg")
    command = ["ffmpeg", "-i", video_path, "-vf", f"fps={fps}", output_pattern]

    result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        return {"errors": [result.stderr]}, 422 # todo check status codes everywhere
    else:
        return {"frames_name": frames_name}, 201 # todo check camel case here and in frontend

@api_bp.post('/video-upload')
def upload_video():
    video = request.files['file']
    video_extension = video.filename.rsplit('.', 1)[1].lower()

    if not video_extension in ALLOWED_EXTENSIONS:
        return {"errors": ["Video format must be .mp4, .mov or .mkv"]}, 422

    if video.filename != "":
        os.makedirs(VIDEO_DIR, exist_ok=True)
        video_path = os.path.join(VIDEO_DIR, video.filename)
        try:
            video.save(video_path)
            return {"filename": video.filename}, 201
        except Exception as err:
            return {"errors": [str(err)]}, 422
    return {"errors": ["missing filename"]}, 422 # todo chack if jsonify is needed


@api_bp.post('/extract-keypoints')
def start_keypoints_extraction():

    frames_name = request.get_json()['frames_name']
    print("frames name received ", frames_name)
    frames_path = os.path.join(FRAMES_DIR, frames_name)
    output_json_path = os.path.join(JSON_DIR, frames_name)

    thread = threading.Thread(target=extract_keypoints, args=(frames_path, output_json_path))
    thread.start()
    return {'json_folder_name': frames_name}, 200


@api_bp.post('/extract-keypoints-info')
def get_keypoints_extraction_info():
    frames_name = request.get_json()['frames_name']
    frames_path = os.path.join(FRAMES_DIR, frames_name)
    output_json_path = os.path.join(JSON_DIR, frames_name)
    total_cnt = sum(os.path.isfile(os.path.join(frames_path, file)) and not file.startswith('.') for file in os.listdir(frames_path))
    processed_cnt = sum(len(files) for _, _, files in os.walk(output_json_path))
    return {"processed_cnt": processed_cnt, "total_cnt": total_cnt, "json_folder_name": frames_name}, 200



def extract_keypoints(frames_path, output_json_path):
    sh_script = "./backend/routes/openpose_runner.sh"

    if os.path.exists(output_json_path):
        shutil.rmtree(output_json_path)
    os.makedirs(output_json_path, exist_ok=True)
    image_files = sorted(glob.glob(os.path.join(frames_path, "*.jpg")) +
                         glob.glob(os.path.join(frames_path, "*.png")))

    if not image_files:
        return {'error': 'No images found in the directory'}, 400

    for i in range(0, len(image_files), BATCH_SIZE):
        batch_images = image_files[i:i + BATCH_SIZE]
        print(f"Processing batch {i//BATCH_SIZE + 1}: {len(batch_images)} images")

        batch_folder = os.path.join(frames_path, f"batch_{i//BATCH_SIZE}")
        os.makedirs(batch_folder, exist_ok=True)

        for img in batch_images:
            shutil.copy(img, batch_folder)

        batch_json_folder = os.path.join(output_json_path, f"batch_{i//BATCH_SIZE}")
        os.makedirs(batch_json_folder, exist_ok=True)

        args = [sh_script, batch_folder, batch_json_folder]
        try:
            subprocess.run(args, check=True)
        except subprocess.CalledProcessError as e:
            return {'errors': [f'Error during script execution: {str(e)}']}, 500
        shutil.rmtree(batch_folder)

    all_json_files = glob.glob(os.path.join(output_json_path, "batch_*", "*.json"))
    for json_file in all_json_files:
        shutil.move(json_file, output_json_path)

    for batch_folder in glob.glob(os.path.join(output_json_path, "batch_*")):
        shutil.rmtree(batch_folder)

    print(f"Extraction completed for: {frames_path}")
