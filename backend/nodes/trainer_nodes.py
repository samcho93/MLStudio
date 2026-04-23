"""학습 관련 노드 (Phase 1) — Trainer, Optimizer, Loss, EarlyStopping 등."""

from __future__ import annotations

import time
from typing import Any, Callable, Coroutine

import numpy as np

from .base import BaseNode


# ── Optimizer ────────────────────────────────────────
class OptimizerNode(BaseNode):
    category = "training"
    description = "옵티마이저를 선택합니다."

    def param_schema(self):
        return [
            {
                "name": "type",
                "type": "select",
                "label": "옵티마이저",
                "options": ["adam", "sgd", "rmsprop", "adamw"],
                "default": "adam",
            },
            {"name": "learning_rate", "type": "number", "label": "학습률", "default": 0.001},
            {"name": "momentum", "type": "number", "label": "모멘텀", "default": 0.9},
            {"name": "weight_decay", "type": "number", "label": "Weight Decay", "default": 0.0},
        ]

    def input_ports(self):
        return []

    def output_ports(self):
        return [{"name": "optimizer_config", "type": "config"}]

    async def run(self, params, inputs, ws_callback):
        return {
            "optimizer_config": {
                "type": params.get("type", "adam"),
                "learning_rate": float(params.get("learning_rate", 0.001)),
                "momentum": float(params.get("momentum", 0.9)),
                "weight_decay": float(params.get("weight_decay", 0.0)),
            }
        }


# ── Loss Function ────────────────────────────────────
class LossFunctionNode(BaseNode):
    category = "training"
    description = "손실 함수를 선택합니다."

    def param_schema(self):
        return [
            {
                "name": "type",
                "type": "select",
                "label": "손실 함수",
                "options": [
                    "sparse_categorical_crossentropy",
                    "categorical_crossentropy",
                    "binary_crossentropy",
                    "mse",
                    "mae",
                ],
                "default": "sparse_categorical_crossentropy",
            },
        ]

    def input_ports(self):
        return []

    def output_ports(self):
        return [{"name": "loss_config", "type": "config"}]

    async def run(self, params, inputs, ws_callback):
        return {"loss_config": {"type": params.get("type", "sparse_categorical_crossentropy")}}


# ── Early Stopping ───────────────────────────────────
class EarlyStoppingNode(BaseNode):
    category = "training"
    description = "조기 종료 설정."

    def param_schema(self):
        return [
            {"name": "patience", "type": "number", "label": "인내 횟수", "default": 5},
            {"name": "monitor", "type": "string", "label": "모니터 메트릭", "default": "val_loss"},
            {"name": "min_delta", "type": "number", "label": "최소 변화량", "default": 0.001},
        ]

    def input_ports(self):
        return []

    def output_ports(self):
        return [{"name": "callback_config", "type": "config"}]

    async def run(self, params, inputs, ws_callback):
        return {
            "callback_config": {
                "type": "EarlyStopping",
                "patience": int(params.get("patience", 5)),
                "monitor": params.get("monitor", "val_loss"),
                "min_delta": float(params.get("min_delta", 0.001)),
            }
        }


# ── LR Scheduler ────────────────────────────────────
class LRSchedulerNode(BaseNode):
    category = "training"
    description = "학습률 스케줄러를 설정합니다."

    def param_schema(self):
        return [
            {"name": "type", "type": "select", "label": "스케줄러",
             "options": ["StepLR", "CosineAnnealing", "ReduceOnPlateau", "ExponentialLR"],
             "default": "ReduceOnPlateau"},
            {"name": "factor", "type": "number", "label": "감소 비율", "default": 0.1},
            {"name": "patience", "type": "number", "label": "인내 횟수", "default": 10},
            {"name": "step_size", "type": "number", "label": "스텝 크기", "default": 10},
        ]

    def input_ports(self):
        return []

    def output_ports(self):
        return [{"name": "scheduler_config", "type": "config"}]

    async def run(self, params, inputs, ws_callback):
        return {
            "scheduler_config": {
                "type": params.get("type", "ReduceOnPlateau"),
                "factor": float(params.get("factor", 0.1)),
                "patience": int(params.get("patience", 10)),
                "step_size": int(params.get("step_size", 10)),
            }
        }


# ── Model Checkpoint ─────────────────────────────────
class ModelCheckpointNode(BaseNode):
    category = "training"
    description = "최적 모델을 자동 저장합니다."

    def param_schema(self):
        return [
            {"name": "filepath", "type": "string", "label": "저장 경로", "default": "outputs/best_model.keras"},
            {"name": "monitor", "type": "string", "label": "모니터 메트릭", "default": "val_loss"},
            {"name": "save_best_only", "type": "boolean", "label": "최적만 저장", "default": True},
        ]

    def input_ports(self):
        return []

    def output_ports(self):
        return [{"name": "callback_config", "type": "config"}]

    async def run(self, params, inputs, ws_callback):
        return {
            "callback_config": {
                "type": "ModelCheckpoint",
                "filepath": params.get("filepath", "outputs/best_model.keras"),
                "monitor": params.get("monitor", "val_loss"),
                "save_best_only": params.get("save_best_only", True),
            }
        }


# ── Trainer ──────────────────────────────────────────
class TrainerNode(BaseNode):
    category = "training"
    description = "모델을 조립하고 학습을 실행합니다."

    def param_schema(self):
        return [
            {"name": "epochs", "type": "number", "label": "에포크", "default": 10},
            {"name": "batch_size", "type": "number", "label": "배치 크기", "default": 32},
            {
                "name": "framework",
                "type": "select",
                "label": "프레임워크",
                "options": ["tensorflow", "pytorch"],
                "default": "tensorflow",
            },
        ]

    def input_ports(self):
        return [
            {"name": "layers", "type": "layer_config"},
            {"name": "train_features", "type": "dataset"},
            {"name": "train_labels", "type": "dataset"},
            {"name": "val_features", "type": "dataset"},
            {"name": "val_labels", "type": "dataset"},
            {"name": "optimizer_config", "type": "config"},
            {"name": "loss_config", "type": "config"},
            {"name": "callback_config", "type": "config"},
            {"name": "scheduler_config", "type": "config"},
        ]

    def output_ports(self):
        return [
            {"name": "model", "type": "model"},
            {"name": "history", "type": "history"},
            {"name": "model_path", "type": "string"},
        ]

    async def run(self, params, inputs, ws_callback):
        framework = params.get("framework", "pytorch")
        epochs = int(params.get("epochs", 10))
        batch_size = int(params.get("batch_size", 32))

        layers = inputs.get("layers", [])
        if not isinstance(layers, list):
            layers = [layers] if layers else []

        # layers 리스트에서 데이터(numpy array)와 레이어 설정(dict)을 분리
        # NLP 예제에서 데이터가 layer chain에 섞여 들어오는 패턴 처리
        clean_layers = []
        embedded_data = None
        embedded_labels = None
        pre_built_model = None
        for item in layers:
            if isinstance(item, dict) and "type" in item:
                clean_layers.append(item)
            elif isinstance(item, np.ndarray):
                # 데이터가 layer chain에 포함됨
                if embedded_data is None:
                    embedded_data = item
                else:
                    embedded_labels = item
            elif isinstance(item, dict) and ("features" in item or "train_features" in item):
                # dict 형태의 데이터 출력
                embedded_data = item.get("features") or item.get("train_features")
                embedded_labels = item.get("labels") or item.get("train_labels")
            elif hasattr(item, 'parameters') and callable(getattr(item, 'parameters', None)):
                # 이전 Trainer에서 받은 모델 객체 (Progressive Transfer Learning, Knowledge Distillation)
                pre_built_model = item
        layers = clean_layers

        X_train = inputs.get("train_features")
        y_train = inputs.get("train_labels")
        X_val = inputs.get("val_features")
        y_val = inputs.get("val_labels")

        # 인라인 옵티마이저/손실 파라미터 지원 (NLP 예제 호환)
        opt_cfg = inputs.get("optimizer_config")
        if opt_cfg is None:
            opt_type = params.get("optimizer", "adam")
            lr = float(params.get("learning_rate", 0.001))
            opt_cfg = {"type": opt_type, "learning_rate": lr}

        loss_cfg = inputs.get("loss_config")
        if loss_cfg is None:
            loss_type = params.get("loss", "sparse_categorical_crossentropy")
            loss_cfg = {"type": loss_type}

        # Unknown loss types → 기본 cross_entropy로 매핑
        known_losses = {"sparse_categorical_crossentropy", "categorical_crossentropy",
                       "binary_crossentropy", "mse", "mae", "mean_squared_error", "mean_absolute_error"}
        if loss_cfg.get("type") not in known_losses:
            loss_cfg = {"type": loss_cfg.get("student_loss", "sparse_categorical_crossentropy")}

        # _source_labels 전파 (NLP 패턴: 데이터 노드 → 레이어 체인 → Trainer)
        source_labels = inputs.get("_source_labels")

        # 데이터가 없으면 layer chain에서 추출된 데이터 사용
        if X_train is None and embedded_data is not None:
            X_train = embedded_data
            y_train = embedded_labels

        # 데이터가 없으면 raw layers input에서 데이터 추출 시도
        if X_train is None:
            raw_layers = inputs.get("layers")
            if isinstance(raw_layers, dict):
                if "features" in raw_layers:
                    X_train = raw_layers["features"]
                    y_train = raw_layers.get("labels")
                elif "train_features" in raw_layers:
                    X_train = raw_layers["train_features"]
                    y_train = raw_layers.get("train_labels")
                    X_val = raw_layers.get("test_features") or raw_layers.get("val_features")
                    y_val = raw_layers.get("test_labels") or raw_layers.get("val_labels")

        # _source_labels로 라벨 보완 (NLP 패턴)
        if y_train is None and source_labels is not None:
            y_train = source_labels

        # validation_split 지원
        val_split = float(params.get("validation_split", 0))
        if val_split > 0 and X_val is None and X_train is not None:
            from sklearn.model_selection import train_test_split as _split
            if y_train is not None:
                X_train, X_val, y_train, y_val = _split(
                    X_train, y_train, test_size=val_split, random_state=42
                )
            else:
                X_train, X_val = _split(X_train, test_size=val_split, random_state=42)

        # Autoencoder 패턴: labels가 없고 loss가 MSE/MAE면 features를 labels로 사용
        loss_type = loss_cfg.get("type", "")
        is_autoencoder = (y_train is None and loss_type in ("mse", "mae", "mean_squared_error", "mean_absolute_error"))
        if is_autoencoder and X_train is not None:
            y_train = X_train.copy() if hasattr(X_train, 'copy') else X_train
            if X_val is not None and y_val is None:
                y_val = X_val.copy() if hasattr(X_val, 'copy') else X_val
            # 마지막 Dense 레이어의 유닛 수를 입력 차원에 맞게 자동 조정
            input_dim = int(np.prod(X_train.shape[1:])) if X_train.ndim >= 2 else len(X_train)
            for i in range(len(layers) - 1, -1, -1):
                if layers[i].get("type") == "Dense":
                    layers[i] = {**layers[i], "units": input_dim}
                    break
        cb_cfgs = inputs.get("callback_config")

        # scheduler_config → callback_config에 합침
        sched_cfg = inputs.get("scheduler_config")
        if sched_cfg:
            sched_as_cb = {
                "type": "ReduceLROnPlateau",
                "factor": sched_cfg.get("factor", 0.1),
                "patience": sched_cfg.get("patience", 10),
            }
            if cb_cfgs is None:
                cb_cfgs = sched_as_cb
            elif isinstance(cb_cfgs, dict):
                cb_cfgs = [cb_cfgs, sched_as_cb]
            elif isinstance(cb_cfgs, list):
                cb_cfgs = cb_cfgs + [sched_as_cb]

        if framework == "tensorflow":
            from frameworks.tf_backend import TFTrainer

            trainer = TFTrainer()
        else:
            from frameworks.torch_backend import TorchTrainer

            trainer = TorchTrainer()

        result = await trainer.train(
            layers=layers,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            optimizer_config=opt_cfg,
            loss_config=loss_cfg,
            callback_configs=cb_cfgs,
            epochs=epochs,
            batch_size=batch_size,
            ws_callback=ws_callback,
            pre_built_model=pre_built_model,
        )

        return result


# ── Model Save ───────────────────────────────────────
class ModelSaveNode(BaseNode):
    category = "output"
    description = "학습된 모델을 파일로 저장합니다."

    def param_schema(self):
        return [
            {"name": "filepath", "type": "string", "label": "저장 경로", "default": "outputs/model.keras"},
        ]

    def input_ports(self):
        return [{"name": "model", "type": "model"}]

    def output_ports(self):
        return [{"name": "model_path", "type": "string"}]

    async def run(self, params, inputs, ws_callback):
        from pathlib import Path

        model = inputs.get("model")
        filepath = params.get("filepath", "outputs/model.keras")
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        if model is not None:
            if hasattr(model, 'save'):
                model.save(filepath)
            elif hasattr(model, 'state_dict'):
                import torch
                pth_path = filepath.replace('.keras', '.pth').replace('.h5', '.pth')
                torch.save(model.state_dict(), pth_path)
                filepath = pth_path

        return {"model_path": filepath}
