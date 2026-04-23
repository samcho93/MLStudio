"""TensorFlow/Keras 학습 백엔드."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Callable, Coroutine

import numpy as np


class TFTrainer:
    """TensorFlow/Keras 기반 모델 빌드 및 학습."""

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
        import tensorflow as tf

        # ── 입력 shape 보정 ─────────────────────────────
        # Conv2D/Conv1D가 첫 레이어인 경우 채널 차원 자동 추가
        first_layer_type = layers[0]["type"] if layers else ""
        needs_channel = first_layer_type in ("Conv2D", "Conv2DTranspose", "MaxPooling2D")
        needs_seq_channel = first_layer_type == "Conv1D"

        if needs_channel and X_train.ndim == 3:
            # (batch, H, W) → (batch, H, W, 1) 흑백 이미지
            X_train = X_train[..., np.newaxis]
            if X_val is not None and X_val.ndim == 3:
                X_val = X_val[..., np.newaxis]
        elif needs_seq_channel and X_train.ndim == 2:
            # (batch, seq_len) → (batch, seq_len, 1)
            X_train = X_train[..., np.newaxis]
            if X_val is not None and X_val.ndim == 2:
                X_val = X_val[..., np.newaxis]

        # ── 모델 빌드 ────────────────────────────────
        input_shape = X_train.shape[1:]

        # 브랜치(Add/Concat)가 있으면 Functional API, 아니면 Sequential
        has_branch = any(
            cfg.get("type") in ("Add", "Concatenate") for cfg in layers
        )

        if has_branch:
            model = self._build_functional(tf, layers, input_shape)
        else:
            model = self._build_sequential(tf, layers, input_shape)

        # ── 옵티마이저 ───────────────────────────────
        opt_type = optimizer_config.get("type", "adam")
        lr = optimizer_config.get("learning_rate", 0.001)

        if opt_type == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        elif opt_type == "sgd":
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=lr,
                momentum=optimizer_config.get("momentum", 0.9),
            )
        elif opt_type == "rmsprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
        elif opt_type == "adamw":
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=lr,
                weight_decay=optimizer_config.get("weight_decay", 0.01),
            )
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # ── 손실 함수 ────────────────────────────────
        loss = loss_config.get("type", "sparse_categorical_crossentropy")

        # ── 메트릭 ───────────────────────────────────
        metrics = ["accuracy"]

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # ── 콜백 ─────────────────────────────────────
        callbacks = []

        # WebSocket 전송 콜백
        if ws_callback:

            class WSCallback(tf.keras.callbacks.Callback):
                def __init__(self, cb):
                    super().__init__()
                    self._cb = cb
                    self._start = time.time()

                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    data = {
                        "type": "EPOCH_UPDATE",
                        "epoch": epoch + 1,
                        "total_epochs": self.params.get("epochs", epochs),
                        "loss": float(logs.get("loss", 0)),
                        "val_loss": float(logs.get("val_loss", 0)),
                        "accuracy": float(logs.get("accuracy", 0)),
                        "val_accuracy": float(logs.get("val_accuracy", 0)),
                        "lr": float(
                            tf.keras.backend.get_value(
                                self.model.optimizer.learning_rate
                            )
                        ),
                        "elapsed_sec": round(time.time() - self._start, 2),
                    }
                    asyncio.get_event_loop().create_task(self._cb(data))

            callbacks.append(WSCallback(ws_callback))

        # 사용자 정의 콜백
        if callback_configs:
            if isinstance(callback_configs, dict):
                callback_configs = [callback_configs]
            elif not isinstance(callback_configs, list):
                callback_configs = []

            for cb_cfg in callback_configs:
                cb_type = cb_cfg.get("type", "")
                if cb_type == "EarlyStopping":
                    callbacks.append(
                        tf.keras.callbacks.EarlyStopping(
                            patience=cb_cfg.get("patience", 5),
                            monitor=cb_cfg.get("monitor", "val_loss"),
                            min_delta=cb_cfg.get("min_delta", 0.001),
                            restore_best_weights=True,
                        )
                    )
                elif cb_type == "ModelCheckpoint":
                    filepath = cb_cfg.get("filepath", "outputs/best_model.keras")
                    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
                    callbacks.append(
                        tf.keras.callbacks.ModelCheckpoint(
                            filepath=filepath,
                            monitor=cb_cfg.get("monitor", "val_loss"),
                            save_best_only=cb_cfg.get("save_best_only", True),
                        )
                    )
                elif cb_type == "ReduceLROnPlateau":
                    callbacks.append(
                        tf.keras.callbacks.ReduceLROnPlateau(
                            factor=cb_cfg.get("factor", 0.1),
                            patience=cb_cfg.get("patience", 10),
                            monitor=cb_cfg.get("monitor", "val_loss"),
                        )
                    )

        # ── 학습 실행 ────────────────────────────────
        validation_data = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            validation_data = (X_val, y_val)

        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=0,
        )

        # ── 모델 저장 ────────────────────────────────
        model_path = "outputs/model.keras"
        Path("outputs").mkdir(exist_ok=True)
        model.save(model_path)

        # ── 결과 반환 ────────────────────────────────
        hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}

        if ws_callback:
            await ws_callback(
                {
                    "type": "TRAINING_COMPLETE",
                    "model_path": model_path,
                    "metrics": {
                        k: float(vals[-1]) for k, vals in history.history.items()
                    },
                }
            )

        return {"model": model, "history": hist, "model_path": model_path}

    # ── Sequential 모델 빌드 ─────────────────────────
    def _build_sequential(self, tf, layers: list[dict], input_shape: tuple):
        model = tf.keras.Sequential()
        first_layer = True

        for layer_cfg in layers:
            ltype = layer_cfg["type"]
            kwargs: dict[str, Any] = {}

            if first_layer:
                kwargs["input_shape"] = input_shape
                first_layer = False

            layer = self._create_layer(tf, layer_cfg, kwargs)
            if layer is not None:
                model.add(layer)

        return model

    # ── Functional 모델 빌드 (Add/Concat 지원) ───────
    def _build_functional(self, tf, layers: list[dict], input_shape: tuple):
        inp = tf.keras.Input(shape=input_shape)
        x = inp

        for layer_cfg in layers:
            ltype = layer_cfg["type"]

            if ltype == "Add" and "branches" in layer_cfg:
                branches_out = []
                for branch_layers in layer_cfg["branches"]:
                    bx = inp
                    for bl in branch_layers:
                        layer = self._create_layer(tf, bl, {})
                        if layer is not None:
                            bx = layer(bx)
                    branches_out.append(bx)
                x = tf.keras.layers.Add()(branches_out)

            elif ltype == "Concatenate" and "branches" in layer_cfg:
                branches_out = []
                for branch_layers in layer_cfg["branches"]:
                    bx = inp
                    for bl in branch_layers:
                        layer = self._create_layer(tf, bl, {})
                        if layer is not None:
                            bx = layer(bx)
                    branches_out.append(bx)
                axis = layer_cfg.get("axis", -1)
                x = tf.keras.layers.Concatenate(axis=axis)(branches_out)

            else:
                layer = self._create_layer(tf, layer_cfg, {})
                if layer is not None:
                    x = layer(x)

        return tf.keras.Model(inputs=inp, outputs=x)

    # ── 단일 레이어 생성 ─────────────────────────────
    def _create_layer(self, tf, layer_cfg: dict, kwargs: dict):
        ltype = layer_cfg["type"]

        if ltype == "Dense":
            return tf.keras.layers.Dense(
                layer_cfg.get("units", 128),
                activation=layer_cfg.get("activation", "relu"),
                **kwargs,
            )

        elif ltype == "Conv2D":
            return tf.keras.layers.Conv2D(
                layer_cfg.get("filters", 32),
                layer_cfg.get("kernel_size", 3),
                activation=layer_cfg.get("activation", "relu"),
                padding=layer_cfg.get("padding", "same"),
                **kwargs,
            )

        elif ltype == "Conv1D":
            return tf.keras.layers.Conv1D(
                layer_cfg.get("filters", 64),
                layer_cfg.get("kernel_size", 3),
                activation=layer_cfg.get("activation", "relu"),
                padding=layer_cfg.get("padding", "same"),
                **kwargs,
            )

        elif ltype == "Conv2DTranspose":
            return tf.keras.layers.Conv2DTranspose(
                layer_cfg.get("filters", 32),
                layer_cfg.get("kernel_size", 3),
                activation=layer_cfg.get("activation", "relu"),
                padding="same",
                **kwargs,
            )

        elif ltype == "MaxPooling2D":
            return tf.keras.layers.MaxPooling2D(
                pool_size=layer_cfg.get("pool_size", 2), **kwargs
            )

        elif ltype == "Flatten":
            return tf.keras.layers.Flatten(**kwargs)

        elif ltype == "GlobalAveragePooling2D":
            return tf.keras.layers.GlobalAveragePooling2D(**kwargs)

        elif ltype == "GlobalAveragePooling1D":
            return tf.keras.layers.GlobalAveragePooling1D(**kwargs)

        elif ltype == "BatchNormalization":
            return tf.keras.layers.BatchNormalization(**kwargs)

        elif ltype == "Dropout":
            return tf.keras.layers.Dropout(
                rate=layer_cfg.get("rate", 0.5), **kwargs
            )

        elif ltype == "Embedding":
            return tf.keras.layers.Embedding(
                input_dim=layer_cfg.get("input_dim", 10000),
                output_dim=layer_cfg.get("output_dim", 128),
                input_length=layer_cfg.get("max_length"),
                **kwargs,
            )

        elif ltype == "LSTM":
            return tf.keras.layers.LSTM(
                layer_cfg.get("units", 64),
                return_sequences=layer_cfg.get("return_sequences", False),
                **kwargs,
            )

        elif ltype == "GRU":
            return tf.keras.layers.GRU(
                layer_cfg.get("units", 64),
                return_sequences=layer_cfg.get("return_sequences", False),
                **kwargs,
            )

        elif ltype == "Bidirectional":
            inner = layer_cfg.get("layer", {})
            inner_layer = self._create_layer(tf, inner, {})
            if inner_layer is not None:
                return tf.keras.layers.Bidirectional(inner_layer, **kwargs)
            return None

        elif ltype == "Reshape":
            target_shape = layer_cfg.get("target_shape", (-1,))
            if isinstance(target_shape, str):
                target_shape = tuple(int(s.strip()) for s in target_shape.split(","))
            return tf.keras.layers.Reshape(target_shape, **kwargs)

        elif ltype == "MultiHeadAttention":
            # MHA는 Functional API에서만 적절히 동작
            num_heads = layer_cfg.get("num_heads", 8)
            key_dim = layer_cfg.get("key_dim", 64)
            return tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=key_dim, **kwargs
            )

        elif ltype == "TransformerBlock":
            # 커스텀 Transformer 블록 (서브클래싱)
            return _TransformerBlock(
                num_heads=layer_cfg.get("num_heads", 4),
                ff_dim=layer_cfg.get("ff_dim", 128),
                dropout=layer_cfg.get("dropout", 0.1),
            )

        elif ltype == "PretrainedModel":
            model_name = layer_cfg.get("model_name", "MobileNetV2")
            trainable = layer_cfg.get("trainable", False)
            include_top = layer_cfg.get("include_top", False)
            return _get_pretrained(tf, model_name, trainable, include_top, kwargs)

        else:
            # 알 수 없는 레이어 → 무시 (경고 로그)
            print(f"[TFTrainer] Unknown layer type: {ltype}, skipping")
            return None


# ── Transformer Block (Keras 서브클래싱) ─────────────
class _TransformerBlock:
    """Transformer 블록을 감싸는 래퍼.
    Sequential에서는 직접 사용 불가 → Functional API 권장."""

    def __init__(self, num_heads, ff_dim, dropout):
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self._built = False

    def __call__(self, x):
        import tensorflow as tf

        embed_dim = x.shape[-1]

        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=embed_dim
        )(x, x)
        attn = tf.keras.layers.Dropout(self.dropout)(attn)
        x1 = tf.keras.layers.LayerNormalization()(x + attn)

        ffn = tf.keras.layers.Dense(self.ff_dim, activation="relu")(x1)
        ffn = tf.keras.layers.Dense(embed_dim)(ffn)
        ffn = tf.keras.layers.Dropout(self.dropout)(ffn)
        return tf.keras.layers.LayerNormalization()(x1 + ffn)


def _get_pretrained(tf, model_name, trainable, include_top, kwargs):
    """사전학습 모델 로드."""
    models_map = {
        "MobileNetV2": tf.keras.applications.MobileNetV2,
        "ResNet50": tf.keras.applications.ResNet50,
        "EfficientNetB0": tf.keras.applications.EfficientNetB0,
        "VGG16": tf.keras.applications.VGG16,
        "InceptionV3": tf.keras.applications.InceptionV3,
    }

    model_cls = models_map.get(model_name)
    if model_cls is None:
        raise ValueError(f"Unknown pretrained model: {model_name}")

    base = model_cls(weights="imagenet", include_top=include_top)
    base.trainable = trainable
    return base
