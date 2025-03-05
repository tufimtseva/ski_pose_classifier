from flask import Blueprint, request, jsonify, g
from services.classifier_service import ClassifierService


ml_api_bp = Blueprint("ml_api", __name__)


def init_classifier_service(coordinates_path):
    if not hasattr(g, 'classifier_service'):  # If the service isn't initialized yet
        g.classifier_service = ClassifierService(coordinates_path)
    return g.classifier_service

@ml_api_bp.post("/train")
def train(): #="json_data/gs_training_fps10_sorted_json"

    data = request.get_json()
    coordinates_path = data.get("coordinates_path")

    if not coordinates_path:
        return jsonify({"error": "coordinates_path is required"}), 400

    # service = ClassifierService(coordinates_path) # todo check the right place to instansiate
    service = init_classifier_service(coordinates_path)
    service.train_classifier()
    return jsonify({"message": "Training completed successfully"}), 200 # todo change output


@ml_api_bp.post("/classify")
def classify():
    data = request.get_json()
    coordinates_path = data.get("coordinates_path")

    if not coordinates_path:
        return jsonify({"error": "coordinates_path is required"}), 400

    # service = ClassifierService(coordinates_path) # todo check the right place to instansiate
    service = init_classifier_service(coordinates_path)
    service.classify_images(coordinates_path)
    return jsonify({"message": "Classifying completed successfully"}), 200 # todo change output
