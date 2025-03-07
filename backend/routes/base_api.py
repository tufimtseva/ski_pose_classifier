from flask import Blueprint, request
import os
import subprocess


api_bp = Blueprint("api", __name__)

ALLOWED_EXTENSIONS = ["mp4", "mov", "mkv"] # todo check allowed extensions fro ffmpeg
VIDEO_DIR = "backend/data_preprocessing/videos"
FRAMES_DIR = "backend/data_preprocessing/frames"

@api_bp.get('/')
def home():
    return 'Flask is running!'

@api_bp.post('/extract-frames')
def extract_frames():

    video_name = request.get_json()['video_name']
    fps = request.get_json()['fps']
    video_path = os.path.join(VIDEO_DIR, video_name)
    frames_path = os.path.join(FRAMES_DIR, f'{video_name}_fps{fps}')

    # video_path, frames_path, fps = "gs_training.mp4", "gs_training_folder_fps10", 10
    os.makedirs(frames_path, exist_ok=True)

    output_pattern = os.path.join(frames_path, "frame_%04d.jpg")
    command = ["ffmpeg", "-i", video_path, "-vf", f"fps={fps}", output_pattern]

    result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        return {"errors": [result.stderr]}, 422 # todo check status codes everywhere
    else:
        return {"frames_path": frames_path}, 201 # todo check camel case here and in frontend

@api_bp.post('/video-upload')
def upload_video():
    video = request.files['file']

    if not video.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
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
