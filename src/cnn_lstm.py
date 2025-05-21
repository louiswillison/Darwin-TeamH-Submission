import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import classify


def run_model(csv_path, roc_filename, cm_filename):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("> Using GPU for training")

    def run_training_cnn(train_loader, test_loader, lr, epochs, dropout, n, t):
        # Initialise the model
        model = CNN_LSTM(num_classes, n, dropout)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_cnn(model, train_loader, optimizer, criterion, epochs)
        metrics = evaluate_cnn(model, test_loader)
        return metrics

    def reshape_for_cnn(X, n):
        num_samples, num_features = X.shape
        desired_size = (n, n)
        required_size = desired_size[0] * desired_size[1]

        if num_features < required_size:
            # Pad with zeros
            padded = np.zeros((num_samples, required_size))
            padded[:, :num_features] = X
            X_reshaped = padded.reshape(num_samples, 1, *desired_size)
        elif num_features > required_size:
            # Truncate
            X_reshaped = X[:, :required_size].reshape(num_samples, 1, *desired_size)
        else:
            # Perfect fit
            X_reshaped = X.reshape(num_samples, 1, *desired_size)

        return torch.tensor(X_reshaped, dtype=torch.float32)

    # for compare and gemaps
    def prepare_data_from_1D(path, epochs, lr, dropout):
        df = pd.read_csv(path)
        labels = df.iloc[:, 0].values
        features = df.iloc[:, 1:].values

        # dimensions for reshaping
        vector_length = features.shape[1]
        sqrt = math.ceil(math.sqrt(vector_length))

        # scale and smote
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        sm = SMOTE(random_state=42)
        X, y = sm.fit_resample(features, labels)

        # reshape to 2D
        X = reshape_for_cnn(X, sqrt)

        y = torch.tensor(y, dtype=torch.long)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        return run_training_cnn(
            train_loader, test_loader, lr, epochs, dropout, sqrt, sqrt
        )

    class CNN_LSTM(nn.Module):
        def __init__(self, num_classes, n_features, dropout):
            super(CNN_LSTM, self).__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 16, 3, 1, 2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 32, 3, 1, 2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout(dropout),
            )

            # Dummy input to calculate LSTM input size
            with torch.no_grad():
                dummy_input = torch.zeros(1, 1, n_features, 1)
                cnn_out = self.conv_layers(dummy_input)
                _, c, h, w = cnn_out.shape
                self.lstm_input_size = c * h

            self.lstm = nn.LSTM(
                input_size=self.lstm_input_size,
                hidden_size=64,
                num_layers=3,
                batch_first=True,
                bidirectional=True,
                dropout=dropout,
            )

            self.fc = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes),
            )

        def forward(self, x):

            x = self.conv_layers(x)
            x = x.permute(0, 3, 1, 2)
            x = x.flatten(2)
            lstm_out, _ = self.lstm(x)
            out = lstm_out[:, -1, :]

            return self.fc(out)

    # Training function for cnn
    def train_cnn(model, train_loader, optimizer, criterion, epochs):
        for epoch in tqdm(range(epochs), leave=False):
            model.train()
            running_loss = 0.0
            train_correct = 0
            total_train = 0

            for inputs, labels in tqdm(train_loader, leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                # forward pass
                outputs = model(inputs).squeeze()
                labels = labels.long()
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                # backward pass and optimisation
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                train_correct += (preds == labels).sum().item()
                total_train += labels.size(0)

            train_accuracy = train_correct / total_train

    # Testing function for cnn
    def evaluate_cnn(model, data_loader):
        model.eval()
        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        # False negatives and false positives
        false_negs = np.sum((all_preds == 0) & (all_labels == 1))
        false_pos = np.sum((all_preds == 1) & (all_labels == 0))
        true_pos = np.sum((all_preds == 1) & (all_labels == 1))
        true_negs = np.sum((all_preds == 0) & (all_labels == 0))

        # Avoid divide by zero
        sensitivity = true_pos / (true_pos + false_negs + 1e-10)
        specificity = true_negs / (true_negs + false_pos + 1e-10)

        # Compute metrics
        f1 = f1_score(all_labels, all_preds, average="weighted")
        roc_auc = roc_auc_score(all_labels, all_probs)

        classify.save_roc_curve_2(
            all_labels,
            all_probs,
            roc_filename,
        )

        classify.save_confusion_matrix(
            all_labels,
            all_preds,
            cm_filename,
        )

        return (
            roc_auc,
            f1,
            sensitivity,
            specificity,
        )

    num_classes = 2
    epochs = 50
    lr = 0.001
    dropout = 0.4

    return prepare_data_from_1D(csv_path, epochs, lr, dropout)
