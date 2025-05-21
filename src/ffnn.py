from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

    def prepare_data_from_csv(csv_path, batch_size=32, test_size=0.2):
        # max did most of this

        df = pd.read_csv(csv_path, on_bad_lines="skip", delimiter=",", skiprows=1)
        # Split by label
        negatives = df[df.iloc[:, 0] == 0]
        positives = df[df.iloc[:, 0] == 1]

        # Drop % of negatives randomly for equal class distribution
        negatives_downsampled = negatives.sample(
            frac=len(positives) / len(negatives), random_state=42
        )  # Keep 60%
        balanced_df = pd.concat([negatives_downsampled, positives])

        # Shuffle the final dataset
        balanced_df = balanced_df.sample(frac=1.0, random_state=42).reset_index(
            drop=True
        )

        # display positive and negative counts
        negatives = balanced_df[balanced_df.iloc[:, 0] == 0]
        positives = balanced_df[balanced_df.iloc[:, 0] == 1]

        # Drop rows with any NaN values
        balanced_df.dropna(inplace=True)

        labels = balanced_df.iloc[:, 0].values.astype(np.int64)
        features = balanced_df.iloc[:, 1:].values.astype(np.float32)

        num_samples = features.shape[0]

        # scale and flatten
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        features = features.reshape(num_samples, -1)

        X = torch.tensor(features)
        y = torch.tensor(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        # trying pca
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)

        train_loader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False
        )

        return train_loader, test_loader

    class FFNN(nn.Module):
        def __init__(self, input_size, num_classes, dropout):
            super(FFNN, self).__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 2),
            )

        def forward(self, x):
            return self.net(x)

    def train_ffnn(train_loader, test_loader, epochs, lr, dropout):
        input_size = next(iter(train_loader))[0].shape[1]
        model = FFNN(input_size, 2, dropout)
        model.to(device)
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

        # Training loop
        for epoch in tqdm(range(epochs), leave=False):
            running_loss = 0.0
            model.train()
            for inputs, labels in tqdm(train_loader, leave=False):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Testing
        return test_ffnn(model, test_loader)

    def test_ffnn(model, loader):

        model.eval()
        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in loader:
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

    train_loader, test_loader = prepare_data_from_csv(csv_path)

    return train_ffnn(train_loader, test_loader, 80, 0.001, 0.4)
