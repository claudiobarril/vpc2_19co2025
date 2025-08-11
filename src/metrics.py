import numpy as np
from sklearn.metrics import recall_score, f1_score, accuracy_score
from sympy.printing.pytorch import torch
from sklearn.metrics import precision_recall_fscore_support


@torch.no_grad()
def eval_metrics(model, valid_loader, device, label_map=None):
    model.eval()
    y_true, y_pred = [], []

    for x, y in valid_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        y_true.append(y.cpu().numpy())
        y_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # Accuracy global
    acc_global = accuracy_score(y_true, y_pred)

    # Identificar el ID de la clase 'Sano'
    sano_id = None
    if label_map is not None:
        for k, v in label_map.items():
            if str(v).lower() == "sano":
                sano_id = int(k)
                break

    if sano_id is None:
        print("Available labels in label_map:", label_map)
        raise ValueError("No se pudo encontrar la clase 'Sano' en label_map")

    print(f"ID de la clase 'Sano': {sano_id}")

    # Métricas binarias: Sano vs Enfermo
    y_true_sano = (y_true == sano_id).astype(int)
    y_pred_sano = (y_pred == sano_id).astype(int)
    recall_sano = recall_score(y_true_sano, y_pred_sano, zero_division=0)
    f1_sano = f1_score(y_true_sano, y_pred_sano, zero_division=0)

    # Métricas para enfermedades (todo lo que NO es 'Sano')
    y_true_enf = (y_true != sano_id).astype(int)
    y_pred_enf = (y_pred != sano_id).astype(int)
    recall_enf = recall_score(y_true_enf, y_pred_enf, zero_division=0)
    f1_enf = f1_score(y_true_enf, y_pred_enf, zero_division=0)

    return acc_global, recall_sano, f1_sano, recall_enf, f1_enf


@torch.no_grad()
def metrics_sano_enfermo(model, loader, device, label_map):
    sano_id = next(k for k, v in label_map.items() if str(v).lower() == "sano")
    model.eval()
    y_true, y_pred = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        pred = logits.argmax(1).cpu().numpy()
        y_true.append(yb.numpy())
        y_pred.append(pred)
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # 3) binarizar (Sano=1 / Enfermo=0) para métricas por "Sanas"
    yt_sano = (y_true == sano_id).astype(int)
    yp_sano = (y_pred == sano_id).astype(int)
    P_sano, R_sano, F1_sano, _ = precision_recall_fscore_support(
        yt_sano, yp_sano, average="binary", zero_division=0
    )

    # 4) binarizar (Enfermo=1 / Sano=0) para métricas por "Enfermas"
    yt_enf = (y_true != sano_id).astype(int)
    yp_enf = (y_pred != sano_id).astype(int)
    P_enf, R_enf, F1_enf, _ = precision_recall_fscore_support(
        yt_enf, yp_enf, average="binary", zero_division=0
    )

    # 5) imprimir tabla
    print("\nClass Weights: equilibrando el aprendizaje\n")
    print(f"{'':12s} {'Precision':>10s} {'Recall':>10s} {'F1-score':>10s}")
    print("-" * 46)
    print(f"{'Sanas':12s} {P_sano:10.4f} {R_sano:10.4f} {F1_sano:10.4f}")
    print(f"{'Enfermas':12s} {P_enf:10.4f} {R_enf:10.4f} {F1_enf:10.4f}")

    return {
        "Sanas": {"precision": P_sano, "recall": R_sano, "f1": F1_sano},
        "Enfermas": {"precision": P_enf, "recall": R_enf, "f1": F1_enf},
    }