"""PyTorch 학습 백엔드."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Coroutine

import numpy as np


class TorchTrainer:
    """PyTorch 기반 모델 빌드 및 학습."""

    async def train(
        self,
        layers: list[dict],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
        optimizer_config: dict,
        loss_config: dict,
        callback_configs: Any,
        epochs: int,
        batch_size: int,
        ws_callback: Callable | None,
        pre_built_model: Any = None,
    ) -> dict:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── 입력 shape 보정 ─────────────────────────────
        first_layer_type = layers[0]["type"] if layers else ""
        needs_channel = first_layer_type in ("Conv2D", "Conv2DTranspose", "MaxPooling2D")
        needs_seq_channel = first_layer_type == "Conv1D"

        if needs_channel and X_train.ndim == 3:
            # (batch, H, W) → (batch, 1, H, W) — PyTorch는 channels-first
            X_train = X_train[:, np.newaxis, ...]
            if X_val is not None and X_val.ndim == 3:
                X_val = X_val[:, np.newaxis, ...]
        elif needs_channel and X_train.ndim == 4 and X_train.shape[-1] in (1, 3, 4):
            # (batch, H, W, C) → (batch, C, H, W) — NHWC → NCHW 변환
            X_train = np.transpose(X_train, (0, 3, 1, 2))
            if X_val is not None and X_val.ndim == 4:
                X_val = np.transpose(X_val, (0, 3, 1, 2))
        elif needs_seq_channel and X_train.ndim == 2:
            # (batch, seq_len) → (batch, 1, seq_len) — channels-first
            X_train = X_train[:, np.newaxis, :]
            if X_val is not None and X_val.ndim == 2:
                X_val = X_val[:, np.newaxis, :]

        # ── 손실 함수 타입 결정 ────────────────────────
        loss_type = loss_config.get("type", "sparse_categorical_crossentropy")
        is_regression = loss_type in ("mse", "mae", "mean_squared_error", "mean_absolute_error")
        is_bce = loss_type == "binary_crossentropy"
        needs_float_target = is_regression or is_bce

        # ── 라벨 shape 보정 ─────────────────────────
        # 분류용 CrossEntropyLoss는 (N,) 형태의 1D 라벨이 필요
        # 회귀(MSE/MAE)나 BCE는 다차원 타겟 유지
        if not needs_float_target:
            if y_train is not None and hasattr(y_train, 'ndim') and y_train.ndim > 1:
                y_train = y_train.ravel()
            if y_val is not None and hasattr(y_val, 'ndim') and y_val.ndim > 1:
                y_val = y_val.ravel()

        # 분류 라벨이 0-based가 아닌 경우 보정 (예: wine quality 3~8 → 0~5)
        if not needs_float_target and y_train is not None:
            y_min = int(np.min(y_train))
            if y_min != 0:
                y_train = y_train - y_min
                if y_val is not None and len(y_val) > 0:
                    y_val = y_val - y_min

        # ── Embedding 레이어 확인 (정수 입력 필요) ────
        has_embedding = any(l.get("type") == "Embedding" for l in layers)

        # ── 데이터 준비 ──────────────────────────────
        if has_embedding:
            X_t = torch.tensor(np.asarray(X_train), dtype=torch.long)
        else:
            X_t = torch.tensor(np.asarray(X_train, dtype=np.float32), dtype=torch.float32)

        # y_train이 None인 경우 더미 라벨 생성
        if y_train is None:
            if needs_float_target:
                y_train = np.zeros(len(X_train), dtype=np.float32)
            else:
                y_train = np.zeros(len(X_train), dtype=np.int64)

        if needs_float_target:
            y_t = torch.tensor(np.asarray(y_train, dtype=np.float32), dtype=torch.float32)
        else:
            y_t = torch.tensor(np.asarray(y_train), dtype=torch.long)
        train_ds = TensorDataset(X_t, y_t)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            X_v = torch.tensor(np.asarray(X_val), dtype=torch.long if has_embedding else torch.float32)
            if needs_float_target:
                y_v = torch.tensor(np.asarray(y_val, dtype=np.float32), dtype=torch.float32)
            else:
                y_v = torch.tensor(np.asarray(y_val), dtype=torch.long)
            val_ds = TensorDataset(X_v, y_v)
            val_loader = DataLoader(val_ds, batch_size=batch_size)

        # ── 출력 클래스 수 자동 결정 ──────────────────
        n_classes = None
        if not needs_float_target and y_train is not None:
            n_classes = int(np.max(y_train)) + 1
            # 마지막 Dense 레이어의 units를 클래스 수에 맞게 조정
            for i in range(len(layers) - 1, -1, -1):
                if layers[i].get("type") == "Dense":
                    act = layers[i].get("activation", "relu")
                    if act in ("softmax", "linear") or i == len(layers) - 1:
                        if layers[i].get("units", 0) != n_classes:
                            layers[i] = {**layers[i], "units": n_classes}
                    break

        # ── 모델 빌드 ────────────────────────────────
        if pre_built_model is not None:
            model = pre_built_model
            model = model.to(device)
        else:
            input_shape = X_train.shape[1:]
            model = self._build_model(nn, layers, input_shape, is_regression)
            model = model.to(device)

        if ws_callback:
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            await ws_callback({
                "type": "DEVICE_INFO",
                "device": str(device),
                "gpu_name": gpu_name,
            })

        # ── 옵티마이저 ───────────────────────────────
        opt_type = optimizer_config.get("type", "adam")
        lr = float(optimizer_config.get("learning_rate", 0.001))

        if opt_type == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif opt_type == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=float(optimizer_config.get("momentum", 0.9)),
            )
        elif opt_type == "rmsprop":
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
        elif opt_type == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=float(optimizer_config.get("weight_decay", 0.01)),
            )
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # ── 손실 함수 ────────────────────────────────
        if loss_type in ("sparse_categorical_crossentropy", "categorical_crossentropy"):
            criterion = nn.CrossEntropyLoss()
        elif loss_type == "binary_crossentropy":
            criterion = nn.BCEWithLogitsLoss()
        elif loss_type in ("mse", "mean_squared_error"):
            criterion = nn.MSELoss()
        elif loss_type in ("mae", "mean_absolute_error"):
            criterion = nn.L1Loss()
        else:
            criterion = nn.CrossEntropyLoss()

        # ── LR 스케줄러 ──────────────────────────────
        scheduler = None
        if callback_configs:
            cb_list = callback_configs if isinstance(callback_configs, list) else [callback_configs] if isinstance(callback_configs, dict) else []
            for cb_cfg in cb_list:
                cb_type = cb_cfg.get("type", "")
                if cb_type == "ReduceLROnPlateau":
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        factor=float(cb_cfg.get("factor", 0.1)),
                        patience=int(cb_cfg.get("patience", 10)),
                    )
                elif cb_type == "StepLR":
                    scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer,
                        step_size=int(cb_cfg.get("step_size", 10)),
                        gamma=float(cb_cfg.get("gamma", 0.1)),
                    )
                elif cb_type == "CosineAnnealing":
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=epochs,
                    )

        # ── 학습 ─────────────────────────────────────
        history: dict[str, list[float]] = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        start_time = time.time()
        best_val_loss = float("inf")
        patience_counter = 0
        patience = 5

        # Early Stopping 설정
        if callback_configs:
            cb_list = callback_configs if isinstance(callback_configs, list) else [callback_configs] if isinstance(callback_configs, dict) else []
            for cb_cfg in cb_list:
                if cb_cfg.get("type") == "EarlyStopping":
                    patience = int(cb_cfg.get("patience", 5))

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                output = model(batch_x)

                if is_regression or is_bce:
                    # Scalar output (batch, 1) → (batch,)
                    if output.dim() == 2 and output.size(-1) == 1:
                        output = output.squeeze(-1)
                    # Flatten both for element-wise comparison if shapes don't match
                    if output.shape != batch_y.shape:
                        output = output.reshape(output.size(0), -1)
                        batch_y = batch_y.reshape(batch_y.size(0), -1)
                        # If target has fewer features, expand to match output
                        if batch_y.size(-1) < output.size(-1):
                            batch_y = batch_y.expand_as(output)
                    loss = criterion(output, batch_y)
                else:
                    loss = criterion(output, batch_y)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch_x.size(0)
                total += batch_y.size(0)

                if is_bce:
                    predicted = (output > 0).long()
                    if predicted.shape != batch_y.shape:
                        predicted = predicted.reshape(batch_y.shape[0], -1)
                    correct += (predicted == batch_y.long()).sum().item()
                elif not is_regression:
                    _, predicted = torch.max(output, 1)
                    correct += (predicted == batch_y).sum().item()

            epoch_loss = running_loss / total
            epoch_acc = correct / total if not is_regression else 0.0
            history["loss"].append(epoch_loss)
            history["accuracy"].append(epoch_acc)

            # 검증
            val_loss = 0.0
            val_acc = 0.0
            if val_loader:
                model.eval()
                val_running = 0.0
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        output = model(batch_x)

                        if is_regression or is_bce:
                            if output.dim() == 2 and output.size(-1) == 1:
                                output = output.squeeze(-1)
                            if output.shape != batch_y.shape:
                                output = output.reshape(output.size(0), -1)
                                batch_y = batch_y.reshape(batch_y.size(0), -1)
                                if batch_y.size(-1) < output.size(-1):
                                    batch_y = batch_y.expand_as(output)
                            loss = criterion(output, batch_y)
                        else:
                            loss = criterion(output, batch_y)

                        val_running += loss.item() * batch_x.size(0)
                        val_total += batch_y.size(0)

                        if is_bce:
                            predicted = (output > 0).long()
                            if predicted.shape != batch_y.shape:
                                predicted = predicted.reshape(batch_y.shape[0], -1)
                            val_correct += (predicted == batch_y.long()).sum().item()
                        elif not is_regression:
                            _, predicted = torch.max(output, 1)
                            val_correct += (predicted == batch_y).sum().item()

                val_loss = val_running / val_total
                val_acc = val_correct / val_total if not is_regression else 0.0

            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)

            # LR 스케줄러 업데이트
            current_lr = optimizer.param_groups[0]["lr"]
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss if val_loader else epoch_loss)
                else:
                    scheduler.step()
                current_lr = optimizer.param_groups[0]["lr"]

            if ws_callback:
                await ws_callback(
                    {
                        "type": "EPOCH_UPDATE",
                        "epoch": epoch + 1,
                        "total_epochs": epochs,
                        "loss": epoch_loss,
                        "val_loss": val_loss,
                        "accuracy": epoch_acc,
                        "val_accuracy": val_acc,
                        "lr": current_lr,
                        "elapsed_sec": round(time.time() - start_time, 2),
                    }
                )

            # Early stopping
            if val_loader and val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            elif val_loader:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # ── 저장 ─────────────────────────────────────
        model_path = "outputs/model.pth"
        Path("outputs").mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)

        if ws_callback:
            await ws_callback(
                {
                    "type": "TRAINING_COMPLETE",
                    "model_path": model_path,
                    "metrics": {
                        "loss": history["loss"][-1],
                        "accuracy": history["accuracy"][-1],
                        "val_loss": history["val_loss"][-1] if history["val_loss"] else 0,
                        "val_accuracy": history["val_accuracy"][-1] if history["val_accuracy"] else 0,
                    },
                }
            )

        return {"model": model, "history": history, "model_path": model_path}

    # ── 모델 빌드 ────────────────────────────────────
    def _build_model(self, nn, layers: list[dict], input_shape: tuple, is_regression: bool):
        """레이어 config 리스트를 기반으로 nn.Sequential 모델을 빌드."""
        model_layers = []
        in_features = int(np.prod(input_shape))
        in_channels = input_shape[0] if len(input_shape) >= 3 else 1
        spatial = list(input_shape[1:]) if len(input_shape) >= 3 else list(input_shape)
        needs_flatten = len(input_shape) > 1
        mode = "spatial" if len(input_shape) >= 2 and any(
            cfg["type"] in ("Conv2D", "Conv2DTranspose", "MaxPooling2D") for cfg in layers
        ) else "flat"

        for layer_cfg in layers:
            ltype = layer_cfg["type"]

            if ltype == "Conv2D":
                out_ch = layer_cfg.get("filters", 32)
                ks = layer_cfg.get("kernel_size", 3)
                pad = ks // 2 if layer_cfg.get("padding", "same") == "same" else 0
                stride = layer_cfg.get("strides", layer_cfg.get("stride", 1))
                model_layers.append(nn.Conv2d(in_channels, out_ch, ks, stride=stride, padding=pad))
                act = layer_cfg.get("activation", "relu")
                if act == "relu":
                    model_layers.append(nn.ReLU())
                elif act == "sigmoid":
                    model_layers.append(nn.Sigmoid())
                elif act == "tanh":
                    model_layers.append(nn.Tanh())
                in_channels = out_ch
                # Update spatial dims for valid padding
                if spatial and pad == 0:
                    spatial = [(s - ks) // stride + 1 for s in spatial]
                elif spatial and stride > 1:
                    spatial = [(s + 2*pad - ks) // stride + 1 for s in spatial]
                needs_flatten = True

            elif ltype == "Conv1D":
                out_ch = layer_cfg.get("filters", 64)
                ks = layer_cfg.get("kernel_size", 3)
                pad = ks // 2 if layer_cfg.get("padding", "same") == "same" else 0
                # Embedding 후 Conv1D: (batch, seq, embed) → (batch, embed, seq) 변환 필요
                if mode == "flat" and in_features > 0:
                    model_layers.append(_PermuteWrapper((0, 2, 1)))
                    in_channels = in_features
                model_layers.append(nn.Conv1d(in_channels, out_ch, ks, padding=pad))
                act = layer_cfg.get("activation", "relu")
                if act == "relu":
                    model_layers.append(nn.ReLU())
                in_channels = out_ch
                needs_flatten = True

            elif ltype == "Conv2DTranspose":
                out_ch = layer_cfg.get("filters", 32)
                ks = layer_cfg.get("kernel_size", 3)
                stride = layer_cfg.get("strides", layer_cfg.get("stride", 1))
                pad = ks // 2
                out_pad = stride - 1 if stride > 1 else 0
                model_layers.append(nn.ConvTranspose2d(in_channels, out_ch, ks, stride=stride, padding=pad, output_padding=out_pad))
                act = layer_cfg.get("activation", "relu")
                if act == "relu":
                    model_layers.append(nn.ReLU())
                elif act == "sigmoid":
                    model_layers.append(nn.Sigmoid())
                in_channels = out_ch
                if spatial and stride > 1:
                    spatial = [s * stride for s in spatial]

            elif ltype == "MaxPooling2D":
                ps = layer_cfg.get("pool_size", 2)
                model_layers.append(nn.MaxPool2d(ps))
                # spatial 크기 업데이트
                if spatial:
                    spatial = [s // ps for s in spatial]
                needs_flatten = True

            elif ltype == "GlobalAveragePooling2D":
                if needs_flatten:
                    model_layers.append(nn.AdaptiveAvgPool2d(1))
                    model_layers.append(nn.Flatten())
                    in_features = in_channels
                    needs_flatten = False
                # else: already flat (e.g. after PretrainedModel), skip

            elif ltype == "GlobalAveragePooling1D":
                if needs_flatten:
                    model_layers.append(nn.AdaptiveAvgPool1d(1))
                    model_layers.append(nn.Flatten())
                    in_features = in_channels
                    needs_flatten = False

            elif ltype == "Flatten":
                model_layers.append(nn.Flatten())
                if mode == "spatial" and spatial:
                    in_features = in_channels * int(np.prod(spatial))
                needs_flatten = False

            elif ltype == "Dense":
                if needs_flatten:
                    model_layers.append(nn.Flatten())
                    if mode == "spatial" and spatial:
                        in_features = in_channels * int(np.prod(spatial))
                    needs_flatten = False

                units = layer_cfg.get("units", 128)
                model_layers.append(nn.Linear(in_features, units))

                activation = layer_cfg.get("activation", "relu")
                if activation == "relu":
                    model_layers.append(nn.ReLU())
                elif activation == "sigmoid":
                    model_layers.append(nn.Sigmoid())
                elif activation == "tanh":
                    model_layers.append(nn.Tanh())
                elif activation == "softmax":
                    pass  # CrossEntropyLoss handles softmax
                elif activation == "swish":
                    model_layers.append(nn.SiLU())

                in_features = units

            elif ltype == "Dropout":
                model_layers.append(nn.Dropout(p=layer_cfg.get("rate", 0.5)))

            elif ltype in ("BatchNormalization", "BatchNorm"):
                if needs_flatten:
                    # Conv 이후 → BatchNorm2d
                    model_layers.append(nn.BatchNorm2d(in_channels))
                else:
                    model_layers.append(nn.BatchNorm1d(in_features))

            elif ltype == "Embedding":
                # Embedding → 커스텀 래퍼 필요
                vocab_size = layer_cfg.get("input_dim", layer_cfg.get("vocab_size", 10000))
                embed_dim = layer_cfg.get("output_dim", layer_cfg.get("embedding_dim", 128))
                model_layers.append(_EmbeddingWrapper(nn, vocab_size, embed_dim))
                in_features = embed_dim
                needs_flatten = False
                mode = "flat"

            elif ltype == "LSTM":
                hidden = layer_cfg.get("units", 64)
                return_seq = layer_cfg.get("return_sequences", False)
                bidir = False
                # Transition from Conv1D: permute (batch, C, L) → (batch, L, C)
                if needs_flatten and mode != "spatial":
                    model_layers.append(_PermuteWrapper((0, 2, 1)))
                    in_features = in_channels
                    needs_flatten = False
                model_layers.append(_RNNWrapper(nn, "LSTM", in_features, hidden, return_seq, bidir))
                in_features = hidden * (2 if bidir else 1)
                needs_flatten = False
                mode = "flat"

            elif ltype == "Bidirectional":
                inner = layer_cfg.get("layer", {})
                hidden = inner.get("units", 64)
                return_seq = inner.get("return_sequences", False)
                rnn_type = inner.get("type", "LSTM")
                if needs_flatten and mode != "spatial":
                    model_layers.append(_PermuteWrapper((0, 2, 1)))
                    in_features = in_channels
                    needs_flatten = False
                model_layers.append(_RNNWrapper(nn, rnn_type, in_features, hidden, return_seq, True))
                in_features = hidden * 2
                needs_flatten = False
                mode = "flat"

            elif ltype == "GRU":
                hidden = layer_cfg.get("units", 64)
                return_seq = layer_cfg.get("return_sequences", False)
                if needs_flatten and mode != "spatial":
                    model_layers.append(_PermuteWrapper((0, 2, 1)))
                    in_features = in_channels
                    needs_flatten = False
                model_layers.append(_RNNWrapper(nn, "GRU", in_features, hidden, return_seq, False))
                in_features = hidden
                needs_flatten = False
                mode = "flat"

            elif ltype == "Reshape":
                target_shape = layer_cfg.get("target_shape", (-1,))
                if isinstance(target_shape, str):
                    target_shape = tuple(int(s.strip()) for s in target_shape.split(","))
                else:
                    target_shape = tuple(int(s) for s in target_shape)
                model_layers.append(_ReshapeWrapper(target_shape))
                # 마지막 차원을 in_features로 설정 (RNN 입력용)
                in_features = target_shape[-1] if target_shape[-1] > 0 else int(np.prod([s for s in target_shape if s > 0]))
                needs_flatten = False

            elif ltype == "TransformerBlock":
                # Transformer → 간소화된 FFN (시퀀스 마지막 타임스텝 사용)
                ff_dim = layer_cfg.get("ff_dim", 128)
                model_layers.append(_RNNWrapper(nn, "GRU", in_features, ff_dim, False, False))
                in_features = ff_dim
                needs_flatten = False
                mode = "flat"

            elif ltype == "PretrainedModel":
                # PretrainedModel → 간단한 Conv 블록으로 대체
                import torch.nn as nn_mod
                out_ch = 64
                model_layers.append(nn.Conv2d(in_channels, 32, 3, padding=1))
                model_layers.append(nn.ReLU())
                model_layers.append(nn.Conv2d(32, out_ch, 3, padding=1))
                model_layers.append(nn.ReLU())
                model_layers.append(nn.AdaptiveAvgPool2d(1))
                model_layers.append(nn.Flatten())
                in_features = out_ch
                in_channels = out_ch
                needs_flatten = False
                mode = "flat"

        return _FlexSequential(model_layers)


class _EmbeddingWrapper:
    """nn.Embedding을 Sequential에서 사용할 수 있게 래핑."""
    def __init__(self, nn, vocab_size, embed_dim):
        import torch.nn as nn_mod
        self._embed = nn_mod.Embedding(vocab_size, embed_dim)
    def __call__(self, x):
        return self._embed(x.long())
    def parameters(self):
        return self._embed.parameters()
    def train(self, mode=True):
        self._embed.train(mode)
    def eval(self):
        self._embed.eval()
    def to(self, device):
        self._embed = self._embed.to(device)
        return self


class _RNNWrapper:
    """LSTM/GRU를 Sequential에서 사용할 수 있게 래핑."""
    def __init__(self, nn, rnn_type, input_size, hidden_size, return_sequences, bidirectional):
        import torch.nn as nn_mod
        rnn_cls = nn_mod.LSTM if rnn_type == "LSTM" else nn_mod.GRU
        self._rnn = rnn_cls(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)
        self._return_seq = return_sequences
    def __call__(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (batch, features) → (batch, 1, features)
        output, _ = self._rnn(x)
        if self._return_seq:
            return output
        return output[:, -1, :]  # 마지막 타임스텝
    def parameters(self):
        return self._rnn.parameters()
    def train(self, mode=True):
        self._rnn.train(mode)
    def eval(self):
        self._rnn.eval()
    def to(self, device):
        self._rnn = self._rnn.to(device)
        return self


class _PermuteWrapper:
    """텐서 차원 순서를 변환."""
    def __init__(self, dims):
        self._dims = dims
    def __call__(self, x):
        return x.permute(*self._dims)
    def parameters(self):
        return iter([])
    def train(self, mode=True):
        pass
    def eval(self):
        pass
    def to(self, device):
        return self


class _ReshapeWrapper:
    """Reshape을 Sequential에서 사용할 수 있게 래핑."""
    def __init__(self, target_shape):
        self._shape = target_shape
    def __call__(self, x):
        return x.contiguous().view(x.size(0), *self._shape)
    def parameters(self):
        return iter([])
    def train(self, mode=True):
        pass
    def eval(self):
        pass
    def to(self, device):
        return self


class _FlexSequential:
    """커스텀 래퍼도 포함 가능한 유연한 Sequential."""
    def __init__(self, layers):
        import torch.nn as nn_mod
        self._layers = layers
        # nn.Module인 것만 모아서 parameters 제공
        self._modules = nn_mod.ModuleList([l for l in layers if isinstance(l, nn_mod.Module)])
        self._custom = [l for l in layers if not isinstance(l, nn_mod.Module)]

    def __call__(self, x):
        for layer in self._layers:
            x = layer(x)
        return x

    def parameters(self):
        import itertools
        all_params = list(self._modules.parameters())
        for c in self._custom:
            if hasattr(c, 'parameters'):
                all_params.extend(c.parameters())
        return iter(all_params)

    def train(self, mode=True):
        self._modules.train(mode)
        for c in self._custom:
            if hasattr(c, 'train'):
                c.train(mode)
        return self

    def eval(self):
        self._modules.eval()
        for c in self._custom:
            if hasattr(c, 'eval'):
                c.eval()
        return self

    def to(self, device):
        self._modules = self._modules.to(device)
        for i, c in enumerate(self._custom):
            if hasattr(c, 'to'):
                self._custom[i] = c.to(device)
        return self

    def state_dict(self):
        return self._modules.state_dict()
