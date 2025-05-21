import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import classify


def run_model(csv_path, roc_filename, cm_filename):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("> Using GPU for training")

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
    def prepare_data_from_1D(path):
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
        return train_loader, test_loader, sqrt

    class CNN(nn.Module):
        def __init__(self, num_classes, n, t, dropout):
            super(CNN, self).__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 16, 3, 1, 2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 32, 3, 1, 2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, 3, 1, 2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.MaxPool2d(2, 2),
                nn.Dropout(dropout),
            )

            # Use dummy input to calculate flattened shape
            with torch.no_grad():
                dummy_input = torch.zeros(1, 1, n, t)
                dummy_output = self.conv_layers(dummy_input)
                flattened_size = dummy_output.view(1, -1).size(1)

            self.fc_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flattened_size, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 2),
            )

        def forward(self, x):
            x = self.conv_layers(x)
            x = self.fc_layers(x)
            return x

    # Training function for cnn
    def train_cnn(model, train_loader, optimizer, criterion, epochs):
        for epoch in tqdm(range(epochs), leave=False):

            model.to(device)
            model.train()
            running_loss = 0.0
            train_correct = 0
            total_train = 0

            for inputs, labels in tqdm(train_loader, leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                # forward pass
                outputs = model(inputs)
                labels = labels.long()
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                # backward pass and optimisation
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                train_correct += (preds == labels).sum().item()
                total_train += labels.size(0)

    # Testing function for cnn
    def evaluate_cnn(model, data_loader):
        model.eval()

        model.to(device)
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
        false_negs = 0
        for i in range(len(all_preds) - 1):
            if all_preds[i] == 0 and all_preds[i] != all_labels[i]:
                false_negs += 1
        false_pos = 0
        for i in range(len(all_preds) - 1):
            if all_preds[i] == 1 and all_preds[i] != all_labels[i]:
                false_pos += 1
        true_pos = np.count_nonzero(all_labels == 1)
        true_negs = np.count_nonzero(all_labels == 0)

        sensitivity = true_pos / (true_pos + false_negs)
        specificity = true_negs / (true_negs + false_pos)

        # Compute F1 score
        f1 = f1_score(all_labels, all_preds, average="weighted")
        roc_auc = roc_auc_score(all_labels, all_probs)

        return roc_auc, f1, sensitivity, specificity, all_labels, all_probs, all_preds

    # runs training and testing for cnn
    def run_training_cnn(lr, epochs, dropout, n, t):
        # Initialise the model
        model = CNN(num_classes, n, t, dropout)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_cnn(model, train_loader, optimizer, criterion, epochs)
        metrics = evaluate_cnn(model, test_loader)
        return metrics

    num_classes = 2

    train_loader, test_loader, n = prepare_data_from_1D(csv_path)
    t = n

    auc, f1, sensitivity, specificity, y_test, y_score, y_pred = run_training_cnn(
        0.001, 25, 0.4, n, t
    )

    classify.save_roc_curve_2(
        y_test,
        y_score,
        roc_filename,
    )
    classify.save_confusion_matrix(
        y_test,
        y_pred,
        cm_filename,
    )
    return (
        auc,
        f1,
        sensitivity,
        specificity,
    )
