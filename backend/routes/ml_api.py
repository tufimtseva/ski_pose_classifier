from flask import Blueprint, request, jsonify, g, send_from_directory
from backend.services.classifier_service import ClassifierService
import os
import base64

ml_api_bp = Blueprint("ml_api", __name__)


IMAGE_DIR = "backend/data_preprocessing/png_data/gs_training_folder_fps10"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def init_classifier_service(coordinates_path):
    if not hasattr(g,
                   'classifier_service'):
        g.classifier_service = ClassifierService(coordinates_path)
    return g.classifier_service


@ml_api_bp.post("/train")
def train():  # ="json_data/gs_training_fps10_sorted_json"

    data = request.get_json()
    coordinates_path = data.get("coordinates_path")

    if not coordinates_path:
        return jsonify({"error": "coordinates_path is required"}), 400

    # service = ClassifierService(coordinates_path) # todo check the right place to instansiate
    service = init_classifier_service(coordinates_path)
    service.train_classifier()
    response = jsonify({"message": "Training completed successfully"})
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response, 200  # todo change output


@ml_api_bp.get("/classify")
def classify():
    # data = request.get_json()
    # coordinates_path = data.get("coordinates_path")
    #
    # if not coordinates_path:
    #     return jsonify({"error": "coordinates_path is required"}), 400

    # service = ClassifierService(coordinates_path) # todo check the right place to instansiate
    coordinates_path = "backend/data_preprocessing/json_data/gs_training_fps10_sorted_json"
    service = init_classifier_service(coordinates_path)
    classified_img_names = service.classify_images(coordinates_path)
    # response = jsonify(
    #     {
    #         'left': classified_img_names["left"],
    #         'middle': classified_img_names["middle"],
    #         'right': classified_img_names["right"]
    #     })
    classified_images = {
        "left": [encode_image(os.path.join(IMAGE_DIR, f"{name}.jpg")) for name in classified_img_names["left"]],
        "middle": [encode_image(os.path.join(IMAGE_DIR, f"{name}.jpg")) for name in classified_img_names["middle"]],
        "right": [encode_image(os.path.join(IMAGE_DIR, f"{name}.jpg")) for name in classified_img_names["right"]],
    }

    response = jsonify(classified_images)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response, 200
