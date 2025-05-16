from flask import Blueprint, request, jsonify, g, send_from_directory
from backend.services.classifier_service import ClassifierServiceCoordinates, \
    ClassifierServiceImages
import os
import base64

ml_api_bp = Blueprint("ml_api", __name__)

IMAGE_DIR = "backend/data_preprocessing/frames"
TRAIN_COORDINATES_FOLDER = "gs_training_fps10_sorted_json"


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def init_classifier_service_coordinates(coordinates_folder):
    if not hasattr(g, 'classifier_service'):
        service = ClassifierServiceCoordinates(coordinates_folder)
        g.classifier_service = service
        g.train_json_coordinates_folder = service.train_json_coordinates_folder
    return g.classifier_service

def init_classifier_service_images():
    if not hasattr(g, 'classifier_service'):
        service = ClassifierServiceImages()
        g.classifier_service = service
    return g.classifier_service


def get_train_folder():
    if not hasattr(g, 'train_json_coordinates_folder'):
        g.train_json_coordinates_folder = TRAIN_COORDINATES_FOLDER
    return g.train_json_coordinates_folder


@ml_api_bp.post("/train")
def train():
    data = request.get_json()
    json_folder_name = data.get("json_folder_name")

    if not json_folder_name:
        return jsonify({"error": "json_folder_name is required"}), 400

    service = init_classifier_service_coordinates(json_folder_name)
    service.train_classifier()
    response = jsonify({"message": "Training completed successfully"})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response, 200


@ml_api_bp.post("/classify-coordinates")
def classify_coordinates():
    data = request.get_json()
    json_folder_name = data.get("json_folder_name")
    if not json_folder_name:
        return jsonify({"error": "json_folder_name is required"}), 400

    train_json_coordinates_folder = get_train_folder()
    service = init_classifier_service_coordinates(train_json_coordinates_folder)
    # classified_img_names = service.classify_images(json_folder_name)
    groupped_classified_img_names, turn_probs = service.classify_images(
        json_folder_name)
    image_folder = os.path.join(IMAGE_DIR, f"{json_folder_name}_cropped")

    classified_images = {
        "left": [
            [encode_image(os.path.join(image_folder, f"{name}.jpg")) for name in
             group]
            for group in groupped_classified_img_names["left"]],
        "middle": [
            [encode_image(os.path.join(image_folder, f"{name}.jpg")) for name in
             group]
            for group in groupped_classified_img_names["middle"]],
        "right": [
            [encode_image(os.path.join(image_folder, f"{name}.jpg")) for name in
             group]
            for group in groupped_classified_img_names["right"]],
    }

    turn_probs = [arr.tolist() for arr in turn_probs]
    response = jsonify(
        {
            "images": classified_images,
            "probabilities": turn_probs
        })

    # response = jsonify(classified_images)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response, 200


@ml_api_bp.post("/classify-images")
def classify_images():
    data = request.get_json()
    image_folder_name = data.get("image_folder_name")
    if not image_folder_name:
        return jsonify({"error": "image_folder_name is required"}), 400

    service = init_classifier_service_images()
    image_folder = os.path.join(IMAGE_DIR, image_folder_name)
    groupped_classified_img_names, turn_probs = service.classify(image_folder)

    classified_images = {
        "left": [
            [encode_image(os.path.join(image_folder, f"{name}.jpg")) for name in
             group]
            for group in groupped_classified_img_names["left"]],
        "middle": [
            [encode_image(os.path.join(image_folder, f"{name}.jpg")) for name in
             group]
            for group in groupped_classified_img_names["middle"]],
        "right": [
            [encode_image(os.path.join(image_folder, f"{name}.jpg")) for name in
             group]
            for group in groupped_classified_img_names["right"]],
    }

    turn_probs = [arr.tolist() for arr in turn_probs]
    response = jsonify(
        {
            "images": classified_images,
            "probabilities": turn_probs
        })

    # response = jsonify(classified_images)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response, 200