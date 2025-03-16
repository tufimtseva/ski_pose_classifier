from backend.data_preprocessing import dataset_builder
from backend.ml_model.model_training import SkiPoseClassifier


class ClassifierService:
    def __init__(self, train_coordinates_folder):
        # todo maybe use get_instance or flask ...
        self.train_json_coordinates_folder = train_coordinates_folder
        train_dataset_path = dataset_builder.build_dataset(self.train_json_coordinates_folder, train=True)
        self.ski_pose_classifier = SkiPoseClassifier(train_dataset_path)  # todo check if this is the right place to instantiate

    def train_classifier(self):
        self.ski_pose_classifier.train()

    def classify_images(self, coordinates_folder_name):
        dataset_path = dataset_builder.build_dataset(coordinates_folder_name)
        return self.ski_pose_classifier.classify(dataset_path)
