from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
import torchmetrics
import torch.nn as nn


def train(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        data: dict,
        epochs: int,
        scheduler=None,
        tb_writer=None,
        log_interval: int = 10,
        early_stop_patience: int = 5,
        early_stop_metric: str = "valid_loss",
        grad_clip_norm=None,
        use_amp: bool = True,
        best_ckpt_path: str = "mejor_modelo.pth",
):
    from tqdm.auto import tqdm
    import torch

    # Importar AMP de forma compatible
    try:
        # PyTorch >= 2.0
        from torch.amp import autocast, GradScaler

        scaler = GradScaler(enabled=use_amp)

        def autocast_context():
            return autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")

    except ImportError:
        # PyTorch < 2.0
        from torch.cuda.amp import autocast, GradScaler

        autocast_context = lambda: autocast()
        scaler = GradScaler(enabled=use_amp)

    train_loader = data["train"]
    valid_loader = data["valid"]
    device = next(model.parameters()).device
    device_type = (
        "cuda" if torch.cuda.is_available() and device.type == "cuda" else "cpu"
    )

    # Crear métricas
    num_classes = len(set(train_loader.dataset.df["Label_id"]))
    train_metrics = create_metrics(num_classes, device)
    valid_metrics = create_metrics(num_classes, device)

    # Early stopping
    best_score = float("inf") if "loss" in early_stop_metric else 0.0
    patience_counter = 0

    # History expandido
    history = {
        "train_loss": [],
        "valid_loss": [],
        "train_accuracy": [],
        "valid_accuracy": [],
        "train_f1_macro": [],
        "valid_f1_macro": [],
        "train_f1_weighted": [],
        "valid_f1_weighted": [],
        "learning_rate": [],
    }

    for epoch in range(1, epochs + 1):
        # =================
        # TRAINING
        # =================
        model.train()
        reset_metrics(train_metrics)
        running_loss = 0.0

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch}/{epochs} [Train]",
        )

        for step, (x, y) in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with autocast_context():
                    logits = model(x)
                    loss = criterion(logits, y)
            else:
                logits = model(x)
                loss = criterion(logits, y)

            # Backward y step
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            running_loss += loss.item()
            update_metrics(train_metrics, logits, y)

            if step % log_interval == 0:
                avg_loss = running_loss / (step + 1)
                current_acc = train_metrics["accuracy"].compute().item()
                pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{current_acc:.4f}")

        # Compute train metrics
        epoch_train_loss = running_loss / len(train_loader)
        train_scores = compute_metrics(train_metrics)

        # =================
        # VALIDATION
        # =================
        model.eval()
        reset_metrics(valid_metrics)
        val_running_loss = 0.0

        with torch.inference_mode():
            for x, y in valid_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                if use_amp:
                    with autocast_context():
                        logits = model(x)
                        loss = criterion(logits, y)
                else:
                    logits = model(x)
                    loss = criterion(logits, y)

                val_running_loss += loss.item()
                update_metrics(valid_metrics, logits, y)

        epoch_val_loss = val_running_loss / len(valid_loader)
        valid_scores = compute_metrics(valid_metrics)

        # Guardar en history
        history["train_loss"].append(epoch_train_loss)
        history["valid_loss"].append(epoch_val_loss)
        history["train_accuracy"].append(train_scores["accuracy"])
        history["valid_accuracy"].append(valid_scores["accuracy"])
        history["train_f1_macro"].append(train_scores["f1_macro"])
        history["valid_f1_macro"].append(valid_scores["f1_macro"])
        history["train_f1_weighted"].append(train_scores["f1_weighted"])
        history["valid_f1_weighted"].append(valid_scores["f1_weighted"])

        # Learning rate actual
        current_lr = optimizer.param_groups[0]["lr"]
        history["learning_rate"].append(current_lr)

        # Print resultados
        print(f"Epoch {epoch} | LR: {current_lr:.2e}")
        print(
            f"  Train - Loss: {epoch_train_loss:.4f}, Acc: {train_scores['accuracy']:.4f}, "
            f"F1: {train_scores['f1_macro']:.4f}"
        )
        print(
            f"  Valid - Loss: {epoch_val_loss:.4f}, Acc: {valid_scores['accuracy']:.4f}, "
            f"F1: {valid_scores['f1_macro']:.4f}"
        )

        # TensorBoard logging
        if tb_writer:
            # Métricas básicas
            tb_writer["train"].add_scalar("loss", epoch_train_loss, epoch)
            tb_writer["valid"].add_scalar("loss", epoch_val_loss, epoch)
            tb_writer["train"].add_scalar("accuracy", train_scores["accuracy"], epoch)
            tb_writer["valid"].add_scalar("accuracy", valid_scores["accuracy"], epoch)

            # Métricas F1
            tb_writer["train"].add_scalar("f1_macro", train_scores["f1_macro"], epoch)
            tb_writer["valid"].add_scalar("f1_macro", valid_scores["f1_macro"], epoch)
            tb_writer["train"].add_scalar(
                "f1_weighted", train_scores["f1_weighted"], epoch
            )
            tb_writer["valid"].add_scalar(
                "f1_weighted", valid_scores["f1_weighted"], epoch
            )

            # Learning rate
            tb_writer["train"].add_scalar("learning_rate", current_lr, epoch)

            # Flush
            tb_writer["train"].flush()
            tb_writer["valid"].flush()

        # Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                if early_stop_metric == "valid_loss":
                    scheduler.step(epoch_val_loss)
                elif early_stop_metric == "valid_f1_macro":
                    scheduler.step(valid_scores["f1_macro"])
            else:

                scheduler.step()
        # Early stopping
        if early_stop_metric == "valid_loss":
            current_score = epoch_val_loss
            is_better = current_score < best_score
        elif early_stop_metric == "valid_f1_macro":
            current_score = valid_scores["f1_macro"]
            is_better = current_score > best_score
        else:
            raise ValueError(f"Métrica no soportada: {early_stop_metric}")

        if is_better:
            best_score = current_score
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt_path)
            print(
                f"✅ Mejor modelo guardado en '{best_ckpt_path}' ({early_stop_metric}={current_score:.4f})"
            )
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"⏹️ Early stopping en epoch {epoch} (paciencia agotada)")
                break

    return history


def create_metrics(num_classes, device):
    metrics = {
        "accuracy": torchmetrics.classification.MulticlassAccuracy(
            num_classes=num_classes, average="macro"
        ).to(device),
        "f1_macro": torchmetrics.classification.MulticlassF1Score(
            num_classes=num_classes, average="macro"
        ).to(device),
        "f1_weighted": torchmetrics.classification.MulticlassF1Score(
            num_classes=num_classes, average="weighted"
        ).to(device),
        "precision_macro": torchmetrics.classification.MulticlassPrecision(
            num_classes=num_classes, average="macro"
        ).to(device),
        "recall_macro": torchmetrics.classification.MulticlassRecall(
            num_classes=num_classes, average="macro"
        ).to(device),
        # Para análisis per-class (opcional)
        "f1_per_class": torchmetrics.classification.MulticlassF1Score(
            num_classes=num_classes, average="none"
        ).to(device),
    }
    return metrics


def reset_metrics(metrics):
    """Reset todas las métricas."""
    for metric in metrics.values():
        metric.reset()


def update_metrics(metrics, logits, targets):
    """Actualizar todas las métricas."""
    for metric in metrics.values():
        metric.update(logits, targets)


def compute_metrics(metrics):
    """Computar todas las métricas y retornar dict."""
    results = {}
    for name, metric in metrics.items():
        if name == "f1_per_class":
            per_class_scores = metric.compute()
            results[name] = per_class_scores.mean().item()

        else:
            results[name] = metric.compute().item()
    return results