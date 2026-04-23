"""Model I/O nodes -- ModelLoader and ModelTester."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Coroutine

import numpy as np

from .base import BaseNode

SAVED_MODELS_DIR = Path("outputs/saved_models")


class ModelLoaderNode(BaseNode):
    """Load a previously saved model from disk."""

    category = "output"
    description = "Save/Load from disk. Loads a previously saved model."

    def param_schema(self):
        return [
            {
                "name": "model_name",
                "type": "select",
                "label": "Model Name",
                "options": self._list_saved_models(),
                "default": "",
            },
            {
                "name": "framework",
                "type": "select",
                "label": "Framework",
                "options": ["tensorflow", "pytorch"],
                "default": "tensorflow",
            },
        ]

    def input_ports(self):
        return []

    def output_ports(self):
        return [
            {"name": "model", "type": "model"},
            {"name": "model_info", "type": "any"},
        ]

    async def run(self, params, inputs, ws_callback):
        model_name = params.get("model_name", "")
        if not model_name:
            raise ValueError("ModelLoader: model_name is required.")

        model_dir = SAVED_MODELS_DIR / model_name
        meta_path = model_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"ModelLoader: model '{model_name}' not found at {model_dir}"
            )

        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        framework = metadata.get("framework", params.get("framework", "tensorflow"))

        model = None
        if framework == "tensorflow":
            model = self._load_tf_model(model_dir, metadata)
        elif framework == "pytorch":
            model = self._load_torch_model(model_dir, metadata)

        if model is None:
            raise RuntimeError(f"ModelLoader: failed to load model '{model_name}'")

        # Load scaler if available
        scaler = None
        scaler_path = model_dir / "scaler.pkl"
        if scaler_path.exists():
            import pickle
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            metadata["scaler"] = scaler

        if ws_callback:
            await ws_callback({
                "type": "MODEL_LOADED",
                "name": model_name,
                "framework": framework,
                "input_shape": metadata.get("input_shape"),
                "class_names": metadata.get("class_names"),
                "is_regression": metadata.get("is_regression", False),
            })

        return {"model": model, "model_info": metadata}

    def _load_tf_model(self, model_dir: Path, metadata: dict):
        """Load a TensorFlow/Keras model."""
        import tensorflow as tf

        model_path = model_dir / "model.keras"
        if not model_path.exists():
            # Try .h5 fallback
            model_path = model_dir / "model.h5"
        if not model_path.exists():
            raise FileNotFoundError(f"TF model file not found in {model_dir}")

        return tf.keras.models.load_model(str(model_path))

    def _load_torch_model(self, model_dir: Path, metadata: dict):
        """Load a PyTorch model by rebuilding architecture from layer config."""
        import torch
        import torch.nn as nn

        model_path = model_dir / "model.pth"
        class_json = model_dir / "model_class.json"

        if not model_path.exists():
            raise FileNotFoundError(f"PyTorch model file not found in {model_dir}")
        if not class_json.exists():
            raise FileNotFoundError(
                f"PyTorch model_class.json not found in {model_dir}. "
                "Cannot rebuild model architecture."
            )

        with open(class_json, "r", encoding="utf-8") as f:
            class_info = json.load(f)

        layers = class_info.get("layers", [])
        input_shape = class_info.get("input_shape", metadata.get("input_shape", []))
        is_regression = metadata.get("is_regression", False)

        if not layers or not input_shape:
            raise ValueError(
                "ModelLoader: model_class.json missing 'layers' or 'input_shape'."
            )

        # Rebuild model using TorchTrainer._build_model
        from frameworks.torch_backend import TorchTrainer

        trainer = TorchTrainer()
        model = trainer._build_model(nn, layers, tuple(input_shape), is_regression)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(str(model_path), map_location=device, weights_only=True)
        model._modules.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        return model

    @staticmethod
    def _list_saved_models() -> list[str]:
        """Return a list of saved model names."""
        if not SAVED_MODELS_DIR.exists():
            return []
        names = []
        for d in sorted(SAVED_MODELS_DIR.iterdir()):
            if d.is_dir() and (d / "metadata.json").exists():
                names.append(d.name)
        return names


class ModelTesterNode(BaseNode):
    """Run predictions on test data and compute metrics."""

    category = "evaluation"
    description = "Test a model on data and compute metrics."

    def param_schema(self):
        return [
            {
                "name": "batch_size",
                "type": "number",
                "label": "Batch Size",
                "default": 32,
            },
            {
                "name": "show_samples",
                "type": "number",
                "label": "Show Samples",
                "default": 10,
            },
        ]

    def input_ports(self):
        return [
            {"name": "model", "type": "model"},
            {"name": "test_features", "type": "dataset"},
            {"name": "test_labels", "type": "dataset"},
        ]

    def output_ports(self):
        return [
            {"name": "predictions", "type": "dataset"},
            {"name": "metrics", "type": "any"},
        ]

    async def run(self, params, inputs, ws_callback):
        model = inputs.get("model")
        if model is None:
            raise ValueError("ModelTester: no model provided.")

        X_test = inputs.get("test_features")
        if X_test is None:
            raise ValueError("ModelTester: no test_features provided.")

        y_test = inputs.get("test_labels")
        batch_size = int(params.get("batch_size", 32))
        show_samples = int(params.get("show_samples", 10))

        # Determine framework
        is_pytorch = hasattr(model, "parameters") and callable(
            getattr(model, "parameters", None)
        )

        if not isinstance(X_test, np.ndarray):
            X_test = np.array(X_test, dtype=np.float32)

        if is_pytorch:
            predictions = self._predict_pytorch(model, X_test, batch_size)
        else:
            predictions = self._predict_tf(model, X_test, batch_size)

        # Compute metrics if labels are provided
        metrics = {}
        if y_test is not None:
            if not isinstance(y_test, np.ndarray):
                y_test = np.array(y_test)
            metrics = self._compute_metrics(predictions, y_test)

        # Build sample table
        samples = self._build_samples(predictions, y_test, show_samples)

        if ws_callback:
            # Send serializable results
            result_msg = {
                "type": "TEST_RESULTS",
                "metrics": metrics,
                "samples": samples,
                "num_predictions": len(predictions),
            }
            # Include prediction summary (not the full array)
            if predictions.ndim == 1:
                result_msg["prediction_summary"] = {
                    "mean": float(np.mean(predictions)),
                    "std": float(np.std(predictions)),
                    "min": float(np.min(predictions)),
                    "max": float(np.max(predictions)),
                }
            await ws_callback(result_msg)

        return {"predictions": predictions, "metrics": metrics}

    def _predict_pytorch(self, model, X: np.ndarray, batch_size: int) -> np.ndarray:
        """Run predictions with a PyTorch model."""
        import torch

        model.eval()
        device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device("cpu")

        # Check if model uses Embedding (needs long input)
        has_embedding = False
        if hasattr(model, '_layers'):
            from frameworks.torch_backend import _EmbeddingWrapper
            has_embedding = any(isinstance(l, _EmbeddingWrapper) for l in model._layers)

        all_preds = []
        for i in range(0, len(X), batch_size):
            batch = X[i : i + batch_size]
            if has_embedding:
                tensor = torch.tensor(batch, dtype=torch.long).to(device)
            else:
                tensor = torch.tensor(batch, dtype=torch.float32).to(device)

            with torch.no_grad():
                output = model(tensor)
            all_preds.append(output.cpu().numpy())

        return np.concatenate(all_preds, axis=0)

    def _predict_tf(self, model, X: np.ndarray, batch_size: int) -> np.ndarray:
        """Run predictions with a TensorFlow model."""
        return model.predict(X, batch_size=batch_size, verbose=0)

    def _compute_metrics(self, predictions: np.ndarray, y_true: np.ndarray) -> dict:
        """Compute classification or regression metrics."""
        metrics = {}

        # Flatten labels if needed
        if y_true.ndim > 1 and y_true.shape[-1] == 1:
            y_true = y_true.ravel()

        # Determine if regression or classification
        is_regression = np.issubdtype(y_true.dtype, np.floating) and len(np.unique(y_true)) > 20

        if is_regression:
            # Regression: squeeze predictions
            y_pred = predictions.ravel()
            if len(y_pred) != len(y_true):
                y_pred = y_pred[: len(y_true)]

            from sklearn.metrics import (
                mean_absolute_error,
                mean_squared_error,
                r2_score,
            )

            mse = float(mean_squared_error(y_true, y_pred))
            metrics["mse"] = mse
            metrics["rmse"] = float(np.sqrt(mse))
            metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
            metrics["r2"] = float(r2_score(y_true, y_pred))
            metrics["type"] = "regression"

        else:
            # Classification
            if predictions.ndim == 2 and predictions.shape[1] > 1:
                y_pred = np.argmax(predictions, axis=1)
            elif predictions.ndim == 2 and predictions.shape[1] == 1:
                y_pred = (predictions.ravel() > 0.5).astype(int)
            else:
                y_pred = (predictions.ravel() > 0.5).astype(int)

            y_true_int = y_true.astype(int).ravel()
            y_pred_int = y_pred.astype(int).ravel()

            from sklearn.metrics import (
                accuracy_score,
                classification_report,
                confusion_matrix,
            )

            metrics["accuracy"] = float(accuracy_score(y_true_int, y_pred_int))

            try:
                report = classification_report(
                    y_true_int, y_pred_int, output_dict=True, zero_division=0
                )
                # Convert numpy values to float for JSON serialization
                clean_report = {}
                for k, v in report.items():
                    if isinstance(v, dict):
                        clean_report[k] = {kk: float(vv) for kk, vv in v.items()}
                    else:
                        clean_report[k] = float(v)
                metrics["classification_report"] = clean_report
            except Exception:
                pass

            try:
                cm = confusion_matrix(y_true_int, y_pred_int)
                metrics["confusion_matrix"] = cm.tolist()
            except Exception:
                pass

            metrics["type"] = "classification"

        return metrics

    def _build_samples(
        self, predictions: np.ndarray, y_true: np.ndarray | None, n: int
    ) -> list[dict]:
        """Build a list of sample predictions for display."""
        samples = []
        n = min(n, len(predictions))

        for i in range(n):
            sample: dict[str, Any] = {"index": i}

            if predictions.ndim == 2 and predictions.shape[1] > 1:
                sample["predicted_class"] = int(np.argmax(predictions[i]))
                sample["confidence"] = float(np.max(predictions[i]))
                sample["probabilities"] = [float(p) for p in predictions[i]]
            elif predictions.ndim == 2 and predictions.shape[1] == 1:
                val = float(predictions[i, 0])
                sample["predicted_value"] = val
                sample["predicted_class"] = int(val > 0.5)
            else:
                sample["predicted_value"] = float(predictions[i])

            if y_true is not None and i < len(y_true):
                if y_true.ndim > 1:
                    sample["true_label"] = int(y_true[i].ravel()[0])
                else:
                    sample["true_label"] = (
                        int(y_true[i])
                        if np.issubdtype(y_true.dtype, np.integer)
                        else float(y_true[i])
                    )

            samples.append(sample)

        return samples
