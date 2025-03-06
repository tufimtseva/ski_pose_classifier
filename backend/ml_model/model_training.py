import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import torch
import torch.nn as nn
import joblib

import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
# todo check if needed keras

from backend.ml_model.lstm_classifier import LSTMClassifier
# from ml_model.mlp_classifier import MLPClassifier

class SkiPoseClassifier:
    def __init__(self, training_coordinates_path):
        self.training_coordinates_path = training_coordinates_path
        self.lstm_path = "backend/saved_lstm.pth"
        self.mlp_path = "backend/saved_mlp.pkl"
        # self.lstm_path = lstm_path
        # self.mlp_path = mlp_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = 3
        self.hidden_dim = 128
        self.num_layers = 1
        self.output_dim = 3
        self.batch_size = 32



    def __load_data(self, source_path):
        df = pd.read_csv(source_path)

        X = df.iloc[:, :-2]
        y = df.iloc[:, -2]

        img_names = df.iloc[:, -1]

        from sklearn.preprocessing import LabelEncoder
        self.le = LabelEncoder()
        self.le.fit(["left", "middle", "right"])
        y = self.le.transform(y)

        # indices = np.arange(len(df)) # todo check if needed

        return X, y, img_names
    def __preprocess_training_data(self, shuffle):
        X, y, img_names = self.__load_data(self.training_coordinates_path)

        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, test_size=0.15, random_state=1, shuffle=shuffle)
        # X_train, X_val, y_train, y_val = train_test_split(
        #     X_train, y_train, test_size=0.176, random_state=1,
        #     shuffle=shuffle)
        # return X_train, y_train, X_val, y_val, X_test, y_test, img_names

        # Split X, y, and img_names into train and test sets
        X_train, X_test, y_train, y_test, img_names_train, img_names_test = train_test_split(
            X, y, img_names, test_size=0.15, random_state=1, shuffle=shuffle)

        # Split X_train, y_train, and img_names_train into train and validation sets
        X_train, X_val, y_train, y_val, img_names_train, img_names_val = train_test_split(
            X_train, y_train, img_names_train, test_size=0.176, random_state=1,
            shuffle=shuffle)

        return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val), np.array(X_test), np.array(y_test), np.array(img_names_train), np.array(img_names_val), np.array(img_names_test)

    def __calculate_metrics(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1_score_res = f1_score(y_test, y_pred, average='weighted')

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 score:", f1_score_res)

    def __train_mlp(self, X_train, y_train, X_val, y_val):
        # todo add params search
        self.mlp = MLPClassifier()
        self.mlp.fit(X_train, y_train)
        y_pred = self.mlp.predict(X_val)
        self.__calculate_metrics(y_val, y_pred)

        # cm = confusion_matrix(y_val, y_pred)
        # ConfusionMatrixDisplay(confusion_matrix=cm).plot()

    # def __train_mlp(self, X_train, y_train, X_val, y_val):
    #
    #     X_train = np.array(X_train)
    #     X_val = np.array(X_val)
    #     categorical_y_train = keras.utils.to_categorical(y_train)
    #     categorical_y_val = keras.utils.to_categorical(y_val)
    #
    #     X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
    #     y_train_tensor = torch.tensor(categorical_y_train,
    #                                   dtype=torch.float32).to(self.device)
    #     X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
    #     y_val_tensor = torch.tensor(categorical_y_val, dtype=torch.float32).to(
    #         self.device)
    #
    #     batch_size = 32
    #     train_loader = data.DataLoader(
    #         data.TensorDataset(X_train_tensor, y_train_tensor), shuffle=True,
    #         batch_size=batch_size, drop_last=True)
    #     val_loader = data.DataLoader(
    #         data.TensorDataset(X_val_tensor, y_val_tensor), shuffle=True,
    #         batch_size=batch_size, drop_last=True)
    #
    #     self.mlp = MLPClassifier().to(self.device)
    #     criterion = nn.CrossEntropyLoss()
    #     optimizer = optim.Adam(self.mlp.parameters(), lr=0.001)
    #
    #     epochs = 70
    #     for epoch in range(epochs):
    #         self.mlp.train()
    #         total_train_loss = 0
    #
    #         for X_batch, y_batch in train_loader:
    #             optimizer.zero_grad()
    #             preds = self.mlp(X_batch)
    #             loss = criterion(preds, y_batch)
    #             loss.backward()
    #             optimizer.step()
    #             total_train_loss += loss.item()
    #
    #         avg_train_loss = total_train_loss / len(train_loader)
    #
    #         self.mlp.eval()
    #         total_val_loss = 0
    #         correct = 0
    #         total = 0
    #
    #         with torch.no_grad():
    #             for X_batch, y_batch in val_loader:
    #                 preds = self.mlp(X_batch)
    #                 loss = criterion(preds, y_batch)
    #                 total_val_loss += loss.item()
    #
    #                 predicted_classes = preds.argmax(dim=1)
    #                 true_classes = y_batch.argmax(
    #                     dim=1)
    #                 correct += (predicted_classes == true_classes).sum().item()
    #                 total += y_batch.size(0)
    #
    #         avg_val_loss = total_val_loss / len(val_loader)
    #         val_accuracy = correct / total
    #
    #         print(
    #             f"Epoch {epoch + 1}/{epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_accuracy:.4f}")

    def __classify_mlp(self, X):
        if self.mlp is None:
            raise ValueError("MLP model not loaded")
        return self.mlp.predict_log_proba(X)

        # self.mlp.eval()
        # X = np.array(X)
        # X_train_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        #
        # with torch.no_grad():
        #     logits = self.mlp(X_train_tensor)
        #
        #
        # probs = F.softmax(logits, dim=1).cpu().numpy()
        # log_probs = np.log(probs + 1e-10)
        #
        # print("Probabilities Shape:", probs.shape)
        # print("Log Probabilities Shape:", log_probs.shape)
        # return log_probs
    def __create_lstm_dataset(self, logits, labels, img_names, lookback):
        # run_starts = [21, 86, 175, 243, 342, 421, 509, 589, 640, 688, 739, 811,
        #               893, 947, 1034, 1100, 1144] # todo use these indices with real indice of logits
        X, y = [], []
        target_img_names = []
        for i in range(len(logits) - lookback):
            inputs = logits[i: i + lookback]
            target = labels[i + lookback]
            # if any(start >= i and start <= i + lookback for start in
            #        run_starts):
            #     continue
            target_img_names.append(img_names[i + lookback])
            X.append(inputs)
            y.append(target)
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        return torch.tensor(X), torch.tensor(y), target_img_names

    def __train_lstm(self, X_train, y_train, X_val, y_val, target_img_names_val):

        self.lstm = LSTMClassifier(self.input_dim,  self.hidden_dim, self.num_layers, self.output_dim,
                               self.batch_size, self.device)
        self.lstm.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.lstm.parameters(), lr=0.0001)

        img_names_dataset = StringDataset(target_img_names_val)
        # todo pass target_img_names_train also in the train loop?

        train_loader = DataLoader(data.TensorDataset(X_train, y_train),
                                  shuffle=False, batch_size=self.batch_size,
                                  drop_last=True)
        val_loader = DataLoader(data.TensorDataset(X_val, y_val), shuffle=False,
                                batch_size=self.batch_size,
                                drop_last=True)
        img_names_loader = DataLoader(img_names_dataset, batch_size=self.batch_size,
                                      shuffle=False)

        # todo make separate function for epoch_train
        epochs = 100
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(epochs):
            self.lstm.train()
            total_train_loss = 0
            train_correct, train_total = 0, 0
            val_logits = []

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device).float(), labels.to(
                    self.device).long()
                optimizer.zero_grad()
                preds = self.lstm(inputs)
                loss = criterion(preds, labels)
                loss.backward()
                total_train_loss += loss.item()
                pred_classes = preds.argmax(dim=1)
                train_correct += (pred_classes == labels).sum().item()
                train_total += labels.numel()

                optimizer.step()

            train_loss = total_train_loss / train_total
            train_accuracy = train_correct / train_total
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            self.lstm.eval()
            total_val_loss = 0
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for val_data, imgs in zip(val_loader, img_names_loader):
                    inputs, val_labels = val_data[0], val_data[1]
                    inputs, val_labels = inputs.to(
                        self.device).float(), val_labels.to(
                        self.device).long()
                    val_preds = self.lstm(inputs)

                    val_loss = criterion(val_preds, val_labels)
                    total_val_loss += val_loss.item()
                    val_logits.append(val_preds)
                    pred_classes = val_preds.argmax(dim=1)
                    val_correct += (pred_classes == val_labels).sum().item()
                    val_total += val_labels.numel()

            val_loss = total_val_loss / val_total
            val_accuracy = val_correct / val_total
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            print(f"Epoch {epoch + 1}/{epochs}")
            print(
                f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%")
            print(
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy * 100:.2f}%")
            print("=" * 40)


    def __load_mlp_model(self):
        # if self.mlp is None: # todo check for this
        self.mlp = joblib.load(self.mlp_path)
        print(f"MLP model loaded successfully from {self.mlp_path}")

    def __load_lstm_model(self):
        # if self.lstm is None:
            # self.lstm = joblib.load(self.lstm_path)
        self.lstm = LSTMClassifier(input_dim=3, hidden_dim=128,
                                   num_layers=1, output_dim=3,
                                   batch_size=self.batch_size, device=self.device)
        self.lstm.load_state_dict(
            torch.load(self.lstm_path, map_location=self.device))
        self.lstm.to(self.device)
        self.lstm.eval()
        print(f"LSTM model loaded successfully from {self.lstm_path}")


    def train(self):
        X_train, y_train, X_val, y_val, X_test, y_test, img_names_train, img_names_val, img_names_test = self.__preprocess_training_data(shuffle=True)
        self.__train_mlp(X_train, y_train, X_val, y_val)

        joblib.dump(self.mlp, self.mlp_path)
        print(f"MLP model saved to {self.mlp_path}")
        # torch.save(self.mlp.st), self.lstm_path)

        X_train, y_train, X_val, y_val, X_test, y_test, img_names_train, img_names_val, img_names_test = self.__preprocess_training_data(shuffle=False)
        # print("img_names_train:\n")
        # for i in img_names_train:
        #     print(i)
        # print("-------------------------")
        # print("img_names_val:\n")
        # for i in img_names_val:
        #     print(i)
        # print("-------------------------")
        # print("img_names_test:\n")
        # for i in img_names_test:
        #     print(i)

        logits_train = self.__classify_mlp(X_train)

        features, targets, target_img_names = self.__create_lstm_dataset(
            logits_train, y_train, img_names_train, lookback=10)

        print("LSTM Input shape:", features.shape)
        print("LSTM Target shape:", targets.shape)

        train_size = int(len(features) * 0.2)
        X_val, X_train = features[:train_size], features[train_size:]  # todo create create_lstm_dataset separately for validation
        y_val, y_train = targets[:train_size], targets[train_size:]
        target_img_names_val, target_img_names_train = target_img_names[:train_size], target_img_names[train_size:]

        self.__train_lstm(X_train, y_train, X_val, y_val, target_img_names_val)
        torch.save(self.lstm.state_dict(), self.lstm_path)
        print(f"LSTM model saved to {self.lstm_path}")


    def __classify_lstm(self, X):
        if self.lstm is None:
            raise ValueError("LSTM model not loaded")

        # X = np.array(X)
        X_lstm_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        # img_names_dataset = StringDataset(img_names)
        data_loader = DataLoader(data.TensorDataset(X_lstm_tensor),
                                  shuffle=False, batch_size=self.batch_size,
                                  drop_last=True)
        # img_names_loader = DataLoader(img_names_dataset, batch_size=self.batch_size,
        #                               shuffle=False) # todo check if needed
        all_probs = []
        self.lstm.eval()
        with torch.no_grad():
            for inputs in data_loader:
                inputs = inputs[0].to(self.device) # inputs[0]?
                logits = self.lstm(inputs)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())

        probs = np.concatenate(all_probs, axis=0)
        predicted_classes = np.argmax(probs, axis=1)
        turn_phases = self.le.inverse_transform(predicted_classes)
        return list(turn_phases)

    def __sort_by_turn_phases(self, img_names, turn_phases):
        classified_img_names = {'left': [], 'middle': [], 'right': []}

        for img, turn in zip(img_names, turn_phases):
            if turn in classified_img_names:
                classified_img_names[turn].append(img)

        return classified_img_names


    def classify(self, coordinates_path): # todo source_data_path or coordinates_path?

        self.__load_mlp_model()
        self.__load_lstm_model()
        # X, y, img_names = self.__load_data(coordinates_path)
        X_train, y_train, X_val, y_val, X_test, y_test, img_names_train, img_names_val, img_names_test = self.__preprocess_training_data(
            shuffle=False)  # todo for test purposes, use self.__load_data(coordinates_path) later later for separate csv with frames from a single video


        # X, y = X_train[:21], y_train[:21]
        X, y = X_test, y_test

        logits = self.__classify_mlp(X)

        features, targets, img_names = self.__create_lstm_dataset(
            logits, y, img_names_test, lookback=10)

        print("LSTM Input shape:", features.shape)
        print("LSTM Target shape:", targets.shape)

        turn_phases = self.__classify_lstm(features)


        return self.__sort_by_turn_phases(img_names, turn_phases)
        # for i, p in zip(img_names, turn_names):
        #     print(f"img name, predicted turn phase: {i}, {p}")






class StringDataset(Dataset): # todo move to separate class
    def __init__(self, string_list):
        self.string_list = string_list

    def __len__(self):
        return len(self.string_list)

    def __getitem__(self, idx):
        return self.string_list[idx]





