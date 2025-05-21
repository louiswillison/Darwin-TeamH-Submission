import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import classify


def run_model(csv_path, roc_filename, cm_filename):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("> Using GPU for training")

    class ResNet50(nn.Module):
        def __init__(self, num_classes):
            super(ResNet50, self).__init__()

            self.resnet50 = models.resnet50(pretrained=True)

            # Modify the first convolutional layer to accept 1-channel MFCC input
            self.resnet50.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_classes)

        def forward(self, x):
            return self.resnet50(x)

    # Training function for ResNet50
    def train_RN50(model, train_loader, criterion, optimizer):
        model.to(device)
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.to(torch.long))

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    # Testing function for ResNet50
    def test_RN50(model, test_loader, criterion):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                probs = F.softmax(outputs, dim=1)
                predicted = torch.argmax(probs, dim=1)

                all_probs.extend(probs[:, 1].cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        f1 = f1_score(all_labels, all_preds, average="macro")
        roc_auc = roc_auc_score(all_labels, all_probs)

        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        return roc_auc, sensitivity, specificity, f1, all_labels, all_probs, all_preds

    # Runs training and testing for ResNet50
    def run_training_RN50(
        train_loader, test_loader, num_classes=2, num_epochs=5, learning_rate=0.0001
    ):
        model = ResNet50(num_classes).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in tqdm(range(num_epochs), leave=False):
            train_loss, train_acc = train_RN50(
                model, train_loader, criterion, optimizer
            )

        roc_auc, sens, spec, f1, all_labels, all_probs, all_preds = test_RN50(
            model, test_loader, criterion
        )

        classify.save_roc_curve_2(
            all_labels,
            all_probs,
            roc_filename,
        )
        classify.save_confusion_matrix(all_labels, all_preds, cm_filename)
        return (
            roc_auc,
            f1,
            sens,
            spec,
        )

    def reshape_for_resnet(X, desired_size=(224, 224)):
        num_samples, num_features = X.shape
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

    start_time = time.time()

    df = pd.read_csv(csv_path, on_bad_lines="skip")
    labels = df.iloc[:, 0].values
    features = df.iloc[:, 1:].values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    sm = SMOTE(random_state=42)
    X, y = sm.fit_resample(features, labels)
    X = reshape_for_resnet(X)
    y = torch.tensor(y, dtype=torch.long)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return run_training_RN50(
        train_loader, test_loader, num_epochs=15, learning_rate=0.001
    )
