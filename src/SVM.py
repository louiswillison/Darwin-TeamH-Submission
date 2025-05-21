import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import classify


def prepare_data(csv_path):
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    neg = df[df.iloc[:, 0] == 0]
    pos = df[df.iloc[:, 0] == 1]
    neg = neg.sample(len(pos), random_state=42)
    df = pd.concat([neg, pos]).sample(frac=1, random_state=42).reset_index(drop=True)

    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, y_train, X_test, y_test


def run_model(csv_path, roc_filename, cm_filename):
    x_train, y_train, x_test, y_test = prepare_data(csv_path)
    model = SVC(kernel="rbf", C=1.0, gamma="scale")
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_score = model.decision_function(x_test)  # Probabilities for ROC AUC

    roc_auc = roc_auc_score(y_test, y_score)
    f1 = f1_score(y_test, y_pred, average="weighted")
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

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
        roc_auc,
        f1,
        sensitivity,
        specificity,
    )
