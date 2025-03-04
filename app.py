from flask import Flask
import os
import subprocess

app = Flask(__name__)

@app.route('/')
def home():
    return 'Flask is running! Go to /extract_frames to process video.'

@app.route('/extract_frames')
def extract_frames():
    video_path, frames_path, fps = "gs_training.mp4", "gs_training_folder_fps10", 10
    if not os.path.exists(video_path):
        return f"Error: Video file '{video_path}' not found!"

    os.makedirs(frames_path, exist_ok=True)

    output_pattern = os.path.join(frames_path, "frame_%04d.jpg")
    command = ["ffmpeg", "-i", video_path, "-vf", f"fps={fps}", output_pattern]

    result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        return f"FFmpeg Error: {result.stderr}"
    else:
        return f"Frames saved in {frames_path}"

if __name__ == '__main__':
    app.run()
