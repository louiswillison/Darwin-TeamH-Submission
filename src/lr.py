import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

import classify


def prepare_data(csv_path):
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    neg = df[df.iloc[:, 0] == 0]
    pos = df[df.iloc[:, 0] == 1]
    neg = neg.sample(len(pos), random_state=42)
    df = pd.concat([neg, pos]).sample(frac=1, random_state=42).reset_index(drop=True)

    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y


def run_model(csv_path, roc_filename, cm_filename):
    x, y = prepare_data(csv_path)
    model = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        C=0.0001,
        max_iter=1000,
    )
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    probs = cross_val_predict(model, x, y, cv=cv, method="predict_proba")[:, 1]
    preds = (probs >= 0.5).astype(int)

    roc_auc = roc_auc_score(y, probs)
    f1 = f1_score(y, preds, average="weighted")
    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    classify.save_roc_curve_2(y, probs, roc_filename)

    classify.save_confusion_matrix(
        y,
        preds,
        cm_filename,
    )

    return (
        roc_auc,
        f1,
        sensitivity,
        specificity,
    )
