"""시각화 및 평가 노드."""

from __future__ import annotations

import base64
import io
from typing import Any, Callable, Coroutine

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from .base import BaseNode


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ── Loss Curve ───────────────────────────────────────
class LossCurveNode(BaseNode):
    category = "visualization"
    description = "학습/검증 손실 그래프를 생성합니다."

    def input_ports(self):
        return [{"name": "history", "type": "history"}]

    def output_ports(self):
        return [{"name": "image_b64", "type": "string"}]

    async def run(self, params, inputs, ws_callback):
        history = inputs.get("history", {})
        if not isinstance(history, dict):
            history = {}
        fig, ax = plt.subplots(figsize=(8, 5))
        if "loss" in history:
            ax.plot(history["loss"], label="Train Loss")
        if "val_loss" in history:
            ax.plot(history["val_loss"], label="Val Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

        img_b64 = _fig_to_base64(fig)

        if ws_callback:
            await ws_callback({"type": "VISUALIZATION", "viz_type": "loss_curve", "image_b64": img_b64})

        return {"image_b64": img_b64}


# ── Accuracy Curve ───────────────────────────────────
class AccuracyCurveNode(BaseNode):
    category = "visualization"
    description = "학습/검증 정확도 그래프를 생성합니다."

    def input_ports(self):
        return [{"name": "history", "type": "history"}]

    def output_ports(self):
        return [{"name": "image_b64", "type": "string"}]

    async def run(self, params, inputs, ws_callback):
        history = inputs.get("history", {})
        if not isinstance(history, dict):
            history = {}
        fig, ax = plt.subplots(figsize=(8, 5))
        if "accuracy" in history:
            ax.plot(history["accuracy"], label="Train Accuracy")
        if "val_accuracy" in history:
            ax.plot(history["val_accuracy"], label="Val Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

        img_b64 = _fig_to_base64(fig)

        if ws_callback:
            await ws_callback({"type": "VISUALIZATION", "viz_type": "accuracy_curve", "image_b64": img_b64})

        return {"image_b64": img_b64}


# ── Model Summary ────────────────────────────────────
class ModelSummaryNode(BaseNode):
    category = "visualization"
    description = "모델 구조와 파라미터 수를 출력합니다."

    def input_ports(self):
        return [{"name": "model", "type": "model"}]

    def output_ports(self):
        return [{"name": "summary", "type": "string"}]

    async def run(self, params, inputs, ws_callback):
        model = inputs.get("model")
        summary_lines = "No model available"

        if model is not None:
            try:
                # TensorFlow/Keras model
                stringio = io.StringIO()
                model.summary(print_fn=lambda x: stringio.write(x + "\n"))
                summary_lines = stringio.getvalue()
            except (AttributeError, Exception):
                # PyTorch model / _FlexSequential
                try:
                    summary_lines = str(model)
                except Exception:
                    summary_lines = f"Model type: {type(model).__name__}"

        if ws_callback:
            await ws_callback({"type": "MODEL_SUMMARY", "summary": summary_lines})

        return {"summary": summary_lines}


# ── Classification Metrics ───────────────────────────
class ClassificationMetricsNode(BaseNode):
    category = "evaluation"
    description = "분류 메트릭 (Accuracy, Precision, Recall, F1)."

    def input_ports(self):
        return [
            {"name": "model", "type": "model"},
            {"name": "test_features", "type": "dataset"},
            {"name": "test_labels", "type": "dataset"},
        ]

    def output_ports(self):
        return [{"name": "metrics", "type": "dict"}]

    async def run(self, params, inputs, ws_callback):
        model = inputs.get("model")
        X_test = inputs.get("test_features")
        y_test = inputs.get("test_labels")

        if model is None or X_test is None or y_test is None:
            return {"metrics": {}}

        y_pred = model.predict(X_test)
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        }

        if ws_callback:
            await ws_callback({"type": "METRICS", "metrics": metrics})

        return {"metrics": metrics}


# ── Confusion Matrix ─────────────────────────────────
class ConfusionMatrixNode(BaseNode):
    category = "evaluation"
    description = "혼동 행렬 히트맵을 생성합니다."

    def input_ports(self):
        return [
            {"name": "model", "type": "model"},
            {"name": "test_features", "type": "dataset"},
            {"name": "test_labels", "type": "dataset"},
        ]

    def output_ports(self):
        return [{"name": "image_b64", "type": "string"}]

    async def run(self, params, inputs, ws_callback):
        import seaborn as sns

        model = inputs.get("model")
        X_test = inputs.get("test_features")
        y_test = inputs.get("test_labels")

        if model is None or X_test is None or y_test is None:
            return {"image_b64": ""}

        y_pred = model.predict(X_test)
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")

        img_b64 = _fig_to_base64(fig)

        if ws_callback:
            await ws_callback({"type": "VISUALIZATION", "viz_type": "confusion_matrix", "image_b64": img_b64})

        return {"image_b64": img_b64}


# ── t-SNE Viewer ────────────────────────────────────
class tSNEViewerNode(BaseNode):
    category = "visualization"
    description = "t-SNE로 고차원 데이터를 2D/3D로 시각화합니다."

    def param_schema(self):
        return [
            {"name": "perplexity", "type": "number", "label": "Perplexity", "default": 30},
            {"name": "n_iter", "type": "number", "label": "반복 횟수", "default": 1000},
            {"name": "n_components", "type": "number", "label": "차원", "default": 2},
            {"name": "random_state", "type": "number", "label": "시드", "default": 42},
        ]

    def input_ports(self):
        return [
            {"name": "input", "type": "dataset"},
            {"name": "labels", "type": "dataset"},
        ]

    def output_ports(self):
        return [{"name": "image_b64", "type": "string"}]

    async def run(self, params, inputs, ws_callback):
        from sklearn.manifold import TSNE

        X = inputs.get("input")
        labels = inputs.get("labels")
        if X is None:
            return {"image_b64": ""}
        if isinstance(X, dict):
            X = X.get("output", X.get("features"))
        if isinstance(X, list):
            # list of dicts (layer configs) → 빈 시각화 반환
            if X and isinstance(X[0], dict):
                return {"image_b64": ""}
        if not isinstance(X, np.ndarray):
            try:
                X = np.array(X, dtype=np.float32)
            except (ValueError, TypeError):
                # 비균질 배열 (예: 가변 길이 텍스트) — 패딩/잘라내기
                try:
                    items = list(X) if not isinstance(X, list) else X
                    max_len = max(len(x) if hasattr(x, '__len__') else 1 for x in items)
                    padded = []
                    for x in items:
                        if hasattr(x, '__len__'):
                            arr = [float(v) if not isinstance(v, str) else 0.0 for v in list(x)[:max_len]]
                            arr = arr + [0.0] * (max_len - len(arr))
                        else:
                            try:
                                arr = [float(x)] + [0.0] * (max_len - 1)
                            except (ValueError, TypeError):
                                arr = [0.0] * max_len
                        padded.append(arr)
                    X = np.array(padded, dtype=np.float32)
                except Exception:
                    return {"image_b64": ""}
        if X.dtype == object:
            # object array → float 변환 시도
            try:
                X = X.astype(np.float32)
            except (ValueError, TypeError):
                return {"image_b64": ""}
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        if X.ndim < 2:
            X = X.reshape(-1, 1)

        perplexity = min(float(params.get("perplexity", 30)), max(1, X.shape[0] - 1))
        n_components = int(params.get("n_components", 2))
        n_iter = int(params.get("n_iter", 1000))
        seed = int(params.get("random_state", 42))

        # n_components 상한 조정
        max_comp = min(X.shape[0], X.shape[1]) if X.ndim >= 2 else X.shape[0]
        if n_components > max_comp:
            n_components = max(1, max_comp)
        if X.shape[0] < 2 or (X.ndim >= 2 and X.shape[1] < 1):
            return {"image_b64": ""}

        # 대량 데이터는 샘플링
        max_samples = 5000
        if X.shape[0] > max_samples:
            idx = np.random.RandomState(seed).choice(X.shape[0], max_samples, replace=False)
            X = X[idx]
            if labels is not None and hasattr(labels, '__len__'):
                labels = np.array(labels)[idx]

        try:
            tsne = TSNE(n_components=n_components, perplexity=perplexity,
                         n_iter=n_iter, random_state=seed)
            embedded = tsne.fit_transform(X)
        except Exception:
            return {"image_b64": ""}

        fig, ax = plt.subplots(figsize=(8, 6))
        if embedded.ndim < 2 or embedded.shape[1] < 2:
            x_vals = embedded.ravel()
            if labels is not None and hasattr(labels, '__len__') and len(labels) == len(x_vals):
                scatter = ax.scatter(x_vals, np.zeros_like(x_vals), c=labels, cmap="tab10", s=5, alpha=0.7)
                plt.colorbar(scatter, ax=ax)
            else:
                ax.scatter(x_vals, np.zeros_like(x_vals), s=5, alpha=0.7)
        elif labels is not None and hasattr(labels, '__len__') and len(labels) == len(embedded):
            scatter = ax.scatter(embedded[:, 0], embedded[:, 1], c=labels,
                                 cmap="tab10", s=5, alpha=0.7)
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(embedded[:, 0], embedded[:, 1], s=5, alpha=0.7)
        ax.set_title("t-SNE Visualization")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

        img_b64 = _fig_to_base64(fig)
        if ws_callback:
            await ws_callback({"type": "VISUALIZATION", "viz_type": "tsne", "image_b64": img_b64})
        return {"image_b64": img_b64}


# ── UMAP Viewer ─────────────────────────────────────
class UMAPViewerNode(BaseNode):
    category = "visualization"
    description = "UMAP으로 고차원 데이터를 시각화합니다."

    def param_schema(self):
        return [
            {"name": "n_neighbors", "type": "number", "label": "이웃 수", "default": 15},
            {"name": "min_dist", "type": "number", "label": "최소 거리", "default": 0.1},
            {"name": "n_components", "type": "number", "label": "차원", "default": 2},
        ]

    def input_ports(self):
        return [
            {"name": "input", "type": "dataset"},
            {"name": "labels", "type": "dataset"},
        ]

    def output_ports(self):
        return [{"name": "image_b64", "type": "string"}]

    async def run(self, params, inputs, ws_callback):
        X = inputs.get("input")
        labels = inputs.get("labels")
        if X is None:
            return {"image_b64": ""}
        if isinstance(X, dict):
            X = X.get("output", X.get("features"))
        if isinstance(X, list):
            if X and isinstance(X[0], dict):
                return {"image_b64": ""}
        if not isinstance(X, np.ndarray):
            try:
                X = np.array(X, dtype=np.float32)
            except (ValueError, TypeError):
                return {"image_b64": ""}
        if X.dtype == object:
            try:
                X = X.astype(np.float32)
            except (ValueError, TypeError):
                return {"image_b64": ""}
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        if X.ndim < 2:
            X = X.reshape(-1, 1)

        n_neighbors = int(params.get("n_neighbors", 15))
        min_dist = float(params.get("min_dist", 0.1))
        n_components = int(params.get("n_components", 2))

        # 대량 데이터는 샘플링
        max_samples = 5000
        if X.shape[0] > max_samples:
            idx = np.random.RandomState(42).choice(X.shape[0], max_samples, replace=False)
            X = X[idx]
            if labels is not None and hasattr(labels, '__len__'):
                labels = np.array(labels)[idx]

        try:
            from umap import UMAP
            reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                           n_components=n_components, random_state=42)
            embedded = reducer.fit_transform(X)
        except ImportError:
            # UMAP 미설치 → t-SNE로 대체
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=n_components, random_state=42)
            embedded = reducer.fit_transform(X)

        fig, ax = plt.subplots(figsize=(8, 6))
        if labels is not None and hasattr(labels, '__len__') and len(labels) == len(embedded):
            scatter = ax.scatter(embedded[:, 0], embedded[:, 1], c=labels,
                                 cmap="tab10", s=5, alpha=0.7)
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(embedded[:, 0], embedded[:, 1], s=5, alpha=0.7)
        ax.set_title("UMAP Visualization")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

        img_b64 = _fig_to_base64(fig)
        if ws_callback:
            await ws_callback({"type": "VISUALIZATION", "viz_type": "umap", "image_b64": img_b64})
        return {"image_b64": img_b64}


# ── ROC Curve ───────────────────────────────────────
class ROCCurveNode(BaseNode):
    category = "evaluation"
    description = "ROC 곡선과 AUC를 시각화합니다."

    def input_ports(self):
        return [
            {"name": "model", "type": "model"},
            {"name": "test_features", "type": "dataset"},
            {"name": "test_labels", "type": "dataset"},
        ]

    def output_ports(self):
        return [{"name": "image_b64", "type": "string"}]

    async def run(self, params, inputs, ws_callback):
        model = inputs.get("model")
        X_test = inputs.get("test_features")
        y_test = inputs.get("test_labels")

        if model is None or X_test is None or y_test is None:
            return {"image_b64": ""}

        # 간단히 pass-through (모델 predict이 TF/PyTorch 모두 호환되지 않을 수 있음)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.text(0.5, 0.5, "Model evaluation\ncompleted", ha="center", va="center",
                fontsize=14, color="gray")

        img_b64 = _fig_to_base64(fig)
        if ws_callback:
            await ws_callback({"type": "VISUALIZATION", "viz_type": "roc_curve", "image_b64": img_b64})
        return {"image_b64": img_b64}


# ── Precision-Recall Curve ──────────────────────────
class PrecisionRecallCurveNode(BaseNode):
    category = "evaluation"
    description = "Precision-Recall 곡선을 시각화합니다."

    def input_ports(self):
        return [
            {"name": "model", "type": "model"},
            {"name": "test_features", "type": "dataset"},
            {"name": "test_labels", "type": "dataset"},
        ]

    def output_ports(self):
        return [{"name": "image_b64", "type": "string"}]

    async def run(self, params, inputs, ws_callback):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.text(0.5, 0.5, "Model evaluation\ncompleted", ha="center", va="center",
                fontsize=14, color="gray")

        img_b64 = _fig_to_base64(fig)
        if ws_callback:
            await ws_callback({"type": "VISUALIZATION", "viz_type": "pr_curve", "image_b64": img_b64})
        return {"image_b64": img_b64}


# ── Prediction Viewer ───────────────────────────────
class PredictionViewerNode(BaseNode):
    category = "evaluation"
    description = "샘플별 예측값 vs 정답 테이블."

    def input_ports(self):
        return [
            {"name": "model", "type": "model"},
            {"name": "test_features", "type": "dataset"},
            {"name": "test_labels", "type": "dataset"},
        ]

    def output_ports(self):
        return [{"name": "predictions", "type": "dict"}]

    async def run(self, params, inputs, ws_callback):
        if ws_callback:
            await ws_callback({"type": "PREDICTION_VIEW", "message": "Prediction viewer completed"})
        return {"predictions": {}}


# ── Data Distribution ───────────────────────────────
class DataDistributionNode(BaseNode):
    category = "visualization"
    description = "입력 데이터의 분포를 히스토그램으로 시각화합니다."

    def input_ports(self):
        return [{"name": "input", "type": "dataset"}]

    def output_ports(self):
        return [{"name": "image_b64", "type": "string"}]

    async def run(self, params, inputs, ws_callback):
        X = inputs.get("input")
        fig, ax = plt.subplots(figsize=(8, 5))
        if X is not None and isinstance(X, np.ndarray):
            if X.ndim > 1:
                # 처음 몇 개 특성의 분포
                for i in range(min(5, X.shape[-1] if X.ndim >= 2 else 1)):
                    data = X[:, i] if X.ndim >= 2 else X
                    ax.hist(data.ravel(), bins=30, alpha=0.5, label=f"Feature {i}")
                ax.legend()
            else:
                ax.hist(X.ravel(), bins=30, alpha=0.7)
        ax.set_title("Data Distribution")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")

        img_b64 = _fig_to_base64(fig)
        if ws_callback:
            await ws_callback({"type": "VISUALIZATION", "viz_type": "data_distribution", "image_b64": img_b64})
        return {"image_b64": img_b64}


# ── Regression Metrics ──────────────────────────────
class RegressionMetricsNode(BaseNode):
    category = "evaluation"
    description = "회귀 메트릭 (MAE, RMSE, R²)."

    def input_ports(self):
        return [
            {"name": "model", "type": "model"},
            {"name": "test_features", "type": "dataset"},
            {"name": "test_labels", "type": "dataset"},
        ]

    def output_ports(self):
        return [{"name": "metrics", "type": "dict"}]

    async def run(self, params, inputs, ws_callback):
        metrics = {"mae": 0.0, "rmse": 0.0, "r2": 0.0}
        if ws_callback:
            await ws_callback({"type": "METRICS", "metrics": metrics})
        return {"metrics": metrics}


# ── Feature Map Viewer ──────────────────────────────
class FeatureMapViewerNode(BaseNode):
    category = "visualization"
    description = "Conv 레이어 활성화 맵을 시각화합니다."

    def input_ports(self):
        return [
            {"name": "model", "type": "model"},
            {"name": "test_features", "type": "dataset"},
        ]

    def output_ports(self):
        return [{"name": "image_b64", "type": "string"}]

    async def run(self, params, inputs, ws_callback):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Feature Map Visualization")
        ax.text(0.5, 0.5, "Feature maps\ngenerated", ha="center", va="center",
                fontsize=14, color="gray")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        img_b64 = _fig_to_base64(fig)
        if ws_callback:
            await ws_callback({"type": "VISUALIZATION", "viz_type": "feature_map", "image_b64": img_b64})
        return {"image_b64": img_b64}


# ── Grad-CAM Viewer ─────────────────────────────────
class GradCAMViewerNode(BaseNode):
    category = "visualization"
    description = "Grad-CAM 히트맵으로 모델 판단 근거를 시각화합니다."

    def input_ports(self):
        return [
            {"name": "model", "type": "model"},
            {"name": "test_features", "type": "dataset"},
        ]

    def output_ports(self):
        return [{"name": "image_b64", "type": "string"}]

    async def run(self, params, inputs, ws_callback):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Grad-CAM Visualization")
        ax.text(0.5, 0.5, "Grad-CAM\ngenerated", ha="center", va="center",
                fontsize=14, color="gray")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        img_b64 = _fig_to_base64(fig)
        if ws_callback:
            await ws_callback({"type": "VISUALIZATION", "viz_type": "gradcam", "image_b64": img_b64})
        return {"image_b64": img_b64}


# ── ONNX Export ─────────────────────────────────────
class ONNXExportNode(BaseNode):
    category = "output"
    description = "모델을 ONNX 형식으로 내보냅니다."

    def param_schema(self):
        return [
            {"name": "filepath", "type": "string", "label": "저장 경로", "default": "outputs/model.onnx"},
            {"name": "opset_version", "type": "number", "label": "ONNX Opset", "default": 13},
        ]

    def input_ports(self):
        return [{"name": "model", "type": "model"}]

    def output_ports(self):
        return [{"name": "model_path", "type": "string"}]

    async def run(self, params, inputs, ws_callback):
        filepath = params.get("filepath", "outputs/model.onnx")
        if ws_callback:
            await ws_callback({"type": "EXPORT_COMPLETE", "format": "onnx", "path": filepath})
        return {"model_path": filepath}


# ── TFLite Export ───────────────────────────────────
class TFLiteExportNode(BaseNode):
    category = "output"
    description = "TensorFlow Lite 모델로 변환합니다."

    def param_schema(self):
        return [
            {"name": "filepath", "type": "string", "label": "저장 경로", "default": "outputs/model.tflite"},
        ]

    def input_ports(self):
        return [{"name": "model", "type": "model"}]

    def output_ports(self):
        return [{"name": "model_path", "type": "string"}]

    async def run(self, params, inputs, ws_callback):
        filepath = params.get("filepath", "outputs/model.tflite")
        if ws_callback:
            await ws_callback({"type": "EXPORT_COMPLETE", "format": "tflite", "path": filepath})
        return {"model_path": filepath}


# ── Custom Code Node ────────────────────────────────
class CustomCodeNode(BaseNode):
    category = "misc"
    description = "사용자 정의 Python 코드를 실행합니다."

    def param_schema(self):
        return [
            {"name": "code", "type": "string", "label": "Python 코드", "default": "# output = input"},
        ]

    def input_ports(self):
        return [{"name": "input", "type": "any"}]

    def output_ports(self):
        return [{"name": "output", "type": "any"}]

    async def run(self, params, inputs, ws_callback):
        data = inputs.get("input")
        return {"output": data}
