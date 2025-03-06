from backend.data_preprocessing import dataset_builder
from backend.ml_model.model_training import SkiPoseClassifier



class ClassifierService: # todo rename to ml service?
    def __init__(self, training_coordinates_path):
        self.training_coordinates_path = training_coordinates_path
        train_dataset_path = dataset_builder.build_dataset(self.training_coordinates_path)
        self.ski_pose_classifier = SkiPoseClassifier(train_dataset_path) # todo check if this is the right place to instantiate
        # self.lstm_path = "saved_lstm.pth"
        # self.mlp_path = "saved_mlp.pkl"

    def train_classifier(self): # todo pass source_data_path as argument?
        # dataset_path = dataset_builder.build_dataset(self.training_coordinates_path)
        # classifier = SkiPoseClassifier(dataset_path)
        self.ski_pose_classifier.train()
    def classify_images(self, test_coordinates_path):
        dataset_path = dataset_builder.build_dataset(test_coordinates_path)
        # classifier = SkiPoseClassifier(dataset_path)
        return self.ski_pose_classifier.classify(dataset_path)



