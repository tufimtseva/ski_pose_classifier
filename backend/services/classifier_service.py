from backend.data_preprocessing import dataset_builder
from backend.ml_model.classifier_pipeline import SkiPoseClassifierCoordinates, SkiPoseClassifierImages


class ClassifierServiceCoordinates:
    def __init__(self, train_coordinates_folder):
        # todo maybe use get_instance or flask ...
        self.train_json_coordinates_folder = train_coordinates_folder
        train_dataset_path = dataset_builder.build_dataset_coordinates(self.train_json_coordinates_folder, train=True)
        self.ski_pose_classifier = SkiPoseClassifierCoordinates(train_dataset_path)

    def train_classifier(self):
        self.ski_pose_classifier.train()

    def classify_images(self, coordinates_folder_name):
        dataset_path = dataset_builder.build_dataset_coordinates(coordinates_folder_name)
        return self.ski_pose_classifier.classify(dataset_path)



class ClassifierServiceImages:
    def __init__(self):
        self.ski_pose_classifier = SkiPoseClassifierImages()

    def classify(self, image_folder_name):
        data_loader, img_names = dataset_builder.build_dataset_images(image_folder_name, shuffle=False)
        return self.ski_pose_classifier.classify(data_loader, img_names)
