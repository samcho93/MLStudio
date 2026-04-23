"""데이터 관련 노드 (Phase 1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Coroutine

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)

from .base import BaseNode

# 백엔드 루트 디렉토리 (datasets/ 등 상대경로 기준)
_BACKEND_DIR = Path(__file__).resolve().parent.parent


def _resolve_path(raw_path: str) -> Path:
    """상대경로를 백엔드 루트 기준으로 해석. 절대경로는 그대로."""
    p = Path(raw_path)
    if p.is_absolute():
        return p
    resolved = _BACKEND_DIR / p
    if resolved.exists():
        return resolved
    # datasets/ 접두사 없이 들어온 경우
    alt = _BACKEND_DIR / "datasets" / p
    if alt.exists():
        return alt
    return resolved  # 원래 경로 반환 (에러는 호출부에서 처리)


# ── CSV Loader ───────────────────────────────────────
class CSVLoaderNode(BaseNode):
    category = "data"
    description = "CSV 파일을 로드합니다."

    def param_schema(self):
        return [
            {"name": "file_path", "type": "string", "label": "파일 경로"},
            {"name": "target_column", "type": "string", "label": "타겟 컬럼", "default": ""},
            {"name": "separator", "type": "string", "label": "구분자", "default": ","},
        ]

    def input_ports(self):
        return []

    def output_ports(self):
        return [
            {"name": "features", "type": "dataset"},
            {"name": "labels", "type": "dataset"},
            {"name": "dataframe", "type": "dataframe"},
        ]

    async def run(self, params, inputs, ws_callback):
        raw_path = params.get("file_path", "")
        sep = params.get("separator", ",")
        target_col = params.get("target_column", "")
        # NLP 텍스트 토크나이징 파라미터
        text_col = params.get("text_column", "")
        tokenize = params.get("tokenize", False)
        max_vocab = int(params.get("max_vocab", 10000))
        max_length = int(params.get("max_length", 256))
        label_col = params.get("label_column", "")

        file_path = _resolve_path(raw_path)
        if not file_path.exists():
            raise FileNotFoundError(
                f"CSV 파일을 찾을 수 없습니다: {raw_path}\n"
                f"시도한 경로: {file_path}"
            )

        df = pd.read_csv(file_path, sep=sep)

        # ── NLP 텍스트 토크나이징 모드 ──────────────────
        if tokenize and text_col:
            # text_col이 리스트인 경우 (다중 텍스트 컬럼: 문장 유사도 등)
            multi_text = isinstance(text_col, list)
            text_cols = text_col if multi_text else [text_col]

            # 컬럼명 대소문자 무시 매칭
            resolved_cols = []
            for tc in text_cols:
                if tc in df.columns:
                    resolved_cols.append(tc)
                else:
                    found = False
                    for col in df.columns:
                        if col.lower() == tc.lower():
                            resolved_cols.append(col)
                            found = True
                            break
                    if not found:
                        resolved_cols.append(tc)
            text_cols = resolved_cols

            # label_column 결정
            lbl_col = label_col or target_col
            if lbl_col and lbl_col not in df.columns:
                for col in df.columns:
                    if col.lower() == lbl_col.lower():
                        lbl_col = col
                        break
            if not lbl_col:
                # 마지막 컬럼을 라벨로
                lbl_col = df.columns[-1] if len(df.columns) >= 2 else ""

            # 텍스트 추출 (다중 컬럼이면 합쳐서 하나의 텍스트로)
            if multi_text:
                # 다중 컬럼: 각 행의 텍스트를 합침
                texts = []
                for _, row in df.iterrows():
                    combined = " ".join(str(row.get(c, "")) for c in text_cols if c in df.columns)
                    texts.append(combined)
            else:
                tc = text_cols[0]
                texts = df[tc].astype(str).tolist() if tc in df.columns else df.iloc[:, 0].astype(str).tolist()

            # 간단한 단어 단위 토크나이징
            from collections import Counter
            word_counts = Counter()
            for text in texts:
                word_counts.update(text.lower().split())
            vocab = {w: i + 2 for i, (w, _) in enumerate(word_counts.most_common(max_vocab - 2))}
            vocab["<PAD>"] = 0
            vocab["<UNK>"] = 1

            # 텍스트 → 정수 시퀀스 변환 + 패딩
            sequences = []
            for text in texts:
                seq = [vocab.get(w, 1) for w in text.lower().split()][:max_length]
                # 패딩
                seq = seq + [0] * (max_length - len(seq))
                sequences.append(seq)
            X = np.array(sequences, dtype=np.int64)

            # 라벨 처리
            y = None
            if lbl_col and lbl_col in df.columns:
                y = df[lbl_col].values
                if y.dtype == object:
                    y = LabelEncoder().fit_transform(y)

            return {"features": X, "labels": y, "dataframe": df}

        # ── 일반 CSV 모드 ──────────────────────────────
        # target_column이 지정되었으나 컬럼에 없으면 유사 이름 검색
        if target_col and target_col not in df.columns:
            # 대소문자 무시 매칭
            for col in df.columns:
                if col.lower() == target_col.lower():
                    target_col = col
                    break

        if target_col and target_col in df.columns:
            y = df[target_col].values
            X_df = df.drop(columns=[target_col])
            # 숫자형 컬럼만 float으로 변환, 나머지는 LabelEncoder로 처리
            for col in X_df.select_dtypes(include=["object", "category"]).columns:
                X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))
            X = X_df.values.astype(np.float32)
            # 타겟이 문자열이면 인코딩
            if y.dtype == object:
                y = LabelEncoder().fit_transform(y)
            return {"features": X, "labels": y, "dataframe": df}

        # target_column 미지정 → 마지막 컬럼을 타겟으로 자동 선택
        if len(df.columns) >= 2:
            last_col = df.columns[-1]
            y = df[last_col].values
            X_df = df.iloc[:, :-1]
            for col in X_df.select_dtypes(include=["object", "category"]).columns:
                X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))
            X = X_df.values.astype(np.float32)
            if y.dtype == object:
                y = LabelEncoder().fit_transform(y)
            return {"features": X, "labels": y, "dataframe": df}

        return {
            "features": df.select_dtypes(include=[np.number]).values.astype(np.float32),
            "labels": None,
            "dataframe": df,
        }


# ── Numpy Input ──────────────────────────────────────
class NumpyInputNode(BaseNode):
    category = "data"
    description = "내장 데이터셋 또는 Numpy 배열을 로드합니다."

    def param_schema(self):
        return [
            {
                "name": "dataset",
                "type": "select",
                "label": "데이터셋",
                "options": ["mnist", "fashion_mnist", "cifar10", "boston", "iris", "custom"],
                "default": "mnist",
            },
        ]

    def input_ports(self):
        return []

    def output_ports(self):
        return [
            {"name": "train_features", "type": "tensor"},
            {"name": "train_labels", "type": "tensor"},
            {"name": "test_features", "type": "tensor"},
            {"name": "test_labels", "type": "tensor"},
        ]

    async def run(self, params, inputs, ws_callback):
        dataset_name = params.get("dataset", "mnist")

        # sklearn 내장 데이터셋
        if dataset_name in ("iris", "boston"):
            from sklearn.datasets import load_iris
            if dataset_name == "iris":
                d = load_iris()
            else:
                # boston은 제거됨 → california housing 대체
                from sklearn.datasets import fetch_california_housing
                d = fetch_california_housing()
            X, y = d.data.astype(np.float32), d.target
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
            return {
                "train_features": X_tr,
                "train_labels": y_tr,
                "test_features": X_te,
                "test_labels": y_te,
            }

        # TensorFlow/Keras 내장 데이터셋
        if dataset_name in ("mnist", "fashion_mnist", "cifar10"):
            try:
                import tensorflow as tf
                loader = getattr(tf.keras.datasets, dataset_name)
                (x_train, y_train), (x_test, y_test) = loader.load_data()
            except ImportError:
                raise ImportError(
                    f"TensorFlow가 설치되지 않았습니다. "
                    f"`pip install tensorflow`로 설치하세요."
                )
            x_train = x_train.astype(np.float32) / 255.0
            x_test = x_test.astype(np.float32) / 255.0
            # 라벨이 (N,1) 형태면 (N,)으로 flatten
            if y_train.ndim > 1:
                y_train = y_train.ravel()
            if y_test.ndim > 1:
                y_test = y_test.ravel()
            return {
                "train_features": x_train,
                "train_labels": y_train,
                "test_features": x_test,
                "test_labels": y_test,
            }

        # custom 데이터셋 — 더미 데이터 반환 (실제 사용 시 사용자가 데이터 제공)
        if dataset_name == "custom":
            X_dummy = np.random.randn(100, 28, 28).astype(np.float32)
            y_dummy = np.random.randint(0, 10, 100)
            return {
                "train_features": X_dummy[:80],
                "train_labels": y_dummy[:80],
                "test_features": X_dummy[80:],
                "test_labels": y_dummy[80:],
            }

        raise ValueError(f"Unknown dataset: {dataset_name}")


# ── Image Folder ─────────────────────────────────────
class ImageFolderNode(BaseNode):
    category = "data"
    description = "폴더의 이미지를 로드합니다 (서브폴더명 = 클래스)."

    def param_schema(self):
        return [
            {"name": "folder_path", "type": "string", "label": "폴더 경로"},
            {"name": "image_size", "type": "number", "label": "이미지 크기", "default": 224},
            {"name": "batch_size", "type": "number", "label": "배치 크기", "default": 32},
        ]

    def input_ports(self):
        return []

    def output_ports(self):
        return [
            {"name": "features", "type": "dataset"},
            {"name": "labels", "type": "dataset"},
            {"name": "class_names", "type": "list"},
        ]

    async def run(self, params, inputs, ws_callback):
        from PIL import Image

        raw_path = params.get("folder_path", "")
        folder = _resolve_path(raw_path)
        if not folder.exists():
            raise FileNotFoundError(
                f"이미지 폴더를 찾을 수 없습니다: {raw_path}\n"
                f"시도한 경로: {folder}"
            )

        # image_size: 숫자 또는 [w, h] 배열 모두 지원
        img_size_raw = params.get("image_size", 224)
        if isinstance(img_size_raw, (list, tuple)):
            img_size = (int(img_size_raw[0]), int(img_size_raw[1]))
        else:
            img_size = (int(img_size_raw), int(img_size_raw))

        class_dirs = sorted([d for d in folder.iterdir() if d.is_dir()])
        if not class_dirs:
            raise ValueError(f"이미지 폴더에 클래스 서브폴더가 없습니다: {folder}")

        class_names = [d.name for d in class_dirs]
        images, labels = [], []

        for idx, class_dir in enumerate(class_dirs):
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
                    try:
                        img = Image.open(img_path).convert("RGB").resize(img_size)
                        images.append(np.array(img, dtype=np.float32) / 255.0)
                        labels.append(idx)
                    except Exception:
                        continue  # 손상된 이미지 건너뜀

        if not images:
            raise ValueError(f"이미지를 찾을 수 없습니다: {folder}")

        return {
            "features": np.array(images),
            "labels": np.array(labels),
            "class_names": class_names,
        }


# ── Train/Val Split ──────────────────────────────────
class TrainValSplitNode(BaseNode):
    category = "data"
    description = "데이터를 학습/검증/테스트 세트로 분할합니다."

    def param_schema(self):
        return [
            {"name": "val_ratio", "type": "number", "label": "검증 비율", "default": 0.2},
            {"name": "test_ratio", "type": "number", "label": "테스트 비율", "default": 0.0},
            {"name": "random_seed", "type": "number", "label": "시드", "default": 42},
        ]

    def input_ports(self):
        return [
            {"name": "features", "type": "dataset"},
            {"name": "labels", "type": "dataset"},
        ]

    def output_ports(self):
        return [
            {"name": "train_features", "type": "dataset"},
            {"name": "train_labels", "type": "dataset"},
            {"name": "val_features", "type": "dataset"},
            {"name": "val_labels", "type": "dataset"},
            {"name": "test_features", "type": "dataset"},
            {"name": "test_labels", "type": "dataset"},
        ]

    async def run(self, params, inputs, ws_callback):
        X = inputs.get("features")
        y = inputs.get("labels")
        # dict인 경우 features 키 추출
        if isinstance(X, dict):
            y_from_dict = X.get("labels", y)
            X = X.get("features")
            if y is None:
                y = y_from_dict
        if X is None:
            raise ValueError("TrainValSplit: features 입력이 없습니다.")

        # 샘플 수 일치 검증
        if y is not None and hasattr(X, '__len__') and hasattr(y, '__len__'):
            if len(X) != len(y):
                raise ValueError(
                    f"TrainValSplit: features({len(X)}개)와 labels({len(y)}개)의 "
                    f"샘플 수가 일치하지 않습니다. "
                    f"features shape={getattr(X, 'shape', '?')}, "
                    f"labels shape={getattr(y, 'shape', '?')}"
                )

        val_r = float(params.get("val_ratio", 0.2))
        test_r = float(params.get("test_ratio", 0.0))
        seed = int(params.get("random_seed", 42))

        # labels가 None인 경우 (오토인코더 등) — features만으로 분할
        if y is None:
            if test_r > 0:
                X_temp, X_test = train_test_split(X, test_size=test_r, random_state=seed)
            else:
                X_temp, X_test = X, np.array([])
            if val_r > 0:
                adjusted_val = val_r / (1 - test_r) if test_r < 1 else val_r
                X_train, X_val = train_test_split(X_temp, test_size=adjusted_val, random_state=seed)
            else:
                X_train, X_val = X_temp, np.array([])
            return {
                "train_features": X_train,
                "train_labels": None,
                "val_features": X_val,
                "val_labels": None,
                "test_features": X_test,
                "test_labels": None,
            }

        if test_r > 0:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_r, random_state=seed
            )
        else:
            X_temp, X_test = X, np.array([])
            y_temp, y_test = y, np.array([])

        if val_r > 0:
            adjusted_val = val_r / (1 - test_r) if test_r < 1 else val_r
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=adjusted_val, random_state=seed
            )
        else:
            X_train, X_val = X_temp, np.array([])
            y_train = y_temp
            y_val = np.array([])

        return {
            "train_features": X_train,
            "train_labels": y_train,
            "val_features": X_val,
            "val_labels": y_val,
            "test_features": X_test,
            "test_labels": y_test,
        }


# ── Data Inspector ───────────────────────────────────
class DataInspectorNode(BaseNode):
    category = "data"
    description = "데이터의 shape, dtype, 샘플을 확인합니다."

    def input_ports(self):
        return [{"name": "input", "type": "any"}]

    def output_ports(self):
        return [{"name": "output", "type": "any"}]

    async def run(self, params, inputs, ws_callback):
        data = inputs.get("input")
        if data is None:
            data = inputs.get("features")
        # dict인 경우 features 추출
        if isinstance(data, dict):
            data = data.get("features", data.get("output", data))
        info = {}

        if isinstance(data, np.ndarray):
            info = {
                "shape": list(data.shape),
                "dtype": str(data.dtype),
                "min": float(np.nanmin(data)) if data.size > 0 else None,
                "max": float(np.nanmax(data)) if data.size > 0 else None,
                "mean": float(np.nanmean(data)) if data.size > 0 else None,
                "sample": data[:3].tolist() if len(data) >= 3 else data.tolist(),
            }
        elif isinstance(data, pd.DataFrame):
            info = {
                "shape": list(data.shape),
                "columns": list(data.columns),
                "dtypes": {c: str(t) for c, t in data.dtypes.items()},
                "head": data.head(3).to_dict(orient="records"),
            }

        if ws_callback:
            await ws_callback({"type": "DATA_INSPECT", "info": info})

        return {"output": data}


# ── StandardScaler ───────────────────────────────────
class StandardScalerNode(BaseNode):
    category = "data"
    description = "Z-score 정규화 (평균 0, 분산 1)."

    def input_ports(self):
        return [{"name": "input", "type": "dataset"}]

    def output_ports(self):
        return [{"name": "output", "type": "dataset"}]

    async def run(self, params, inputs, ws_callback):
        X = inputs.get("input")
        if X is None:
            # 다른 포트에서 데이터 찾기
            for key in ("features", "output", "data"):
                X = inputs.get(key)
                if X is not None:
                    break
        if X is None:
            raise ValueError("StandardScaler: input이 None입니다.")
        # dict가 들어온 경우 features 키 추출
        if isinstance(X, dict):
            X = X.get("features", X.get("output"))
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float32)
        original_shape = X.shape
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X).astype(np.float32)
        if len(original_shape) > 2:
            X_scaled = X_scaled.reshape(original_shape)
        return {"output": X_scaled}


# ── MinMaxScaler ─────────────────────────────────────
class MinMaxScalerNode(BaseNode):
    category = "data"
    description = "Min-Max 정규화 (0~1 범위)."

    def input_ports(self):
        return [{"name": "input", "type": "dataset"}]

    def output_ports(self):
        return [{"name": "output", "type": "dataset"}]

    async def run(self, params, inputs, ws_callback):
        X = inputs.get("input")
        if X is None:
            for key in ("features", "output", "data"):
                X = inputs.get(key)
                if X is not None:
                    break
        if X is None:
            raise ValueError("MinMaxScaler: input이 None입니다.")
        # dict가 들어온 경우 features 키 추출
        if isinstance(X, dict):
            X = X.get("features", X.get("output"))
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float32)
        original_shape = X.shape
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X).astype(np.float32)
        if len(original_shape) > 2:
            X_scaled = X_scaled.reshape(original_shape)
        return {"output": X_scaled}


# ── Label Encoder ────────────────────────────────────
class LabelEncoderNode(BaseNode):
    category = "data"
    description = "문자열 레이블을 정수로 인코딩합니다."

    def input_ports(self):
        return [{"name": "input", "type": "dataset"}]

    def output_ports(self):
        return [{"name": "output", "type": "dataset"}]

    async def run(self, params, inputs, ws_callback):
        data = inputs.get("input")
        if data is None:
            for key in ("features", "output", "labels"):
                data = inputs.get(key)
                if data is not None:
                    break
        if data is None:
            raise ValueError("LabelEncoder: input이 None입니다.")
        # dict가 들어온 경우 features 키 추출
        if isinstance(data, dict):
            data = data.get("features", data.get("output"))
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        # 스칼라인 경우 1D로 변환
        if data.ndim == 0:
            data = data.reshape(1)

        # 2D 배열인 경우 각 컬럼을 개별 인코딩 (shape 유지)
        if hasattr(data, 'ndim') and data.ndim == 2:
            result = np.zeros_like(data, dtype=np.float32)
            for col in range(data.shape[1]):
                encoder = LabelEncoder()
                col_data = data[:, col]
                if col_data.dtype == object:
                    result[:, col] = encoder.fit_transform(col_data.astype(str))
                else:
                    result[:, col] = col_data.astype(np.float32)
            return {"output": result}

        # 1D 배열 (레이블 인코딩)
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(
            data.ravel() if hasattr(data, 'ravel') else data
        )
        return {"output": y_encoded}


# ── One-Hot Encoder ──────────────────────────────────
class OneHotEncoderNode(BaseNode):
    category = "data"
    description = "범주형 값을 One-Hot 벡터로 변환합니다."

    def input_ports(self):
        return [{"name": "input", "type": "dataset"}]

    def output_ports(self):
        return [{"name": "output", "type": "dataset"}]

    async def run(self, params, inputs, ws_callback):
        y = inputs.get("input")
        if y is None:
            raise ValueError("OneHotEncoder: input이 None입니다.")
        if isinstance(y, dict):
            y = y.get("features", y.get("output"))
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if y.ndim == 0:
            y = y.reshape(1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        encoder = OneHotEncoder(sparse_output=False)
        y_encoded = encoder.fit_transform(y).astype(np.float32)
        return {"output": y_encoded}


# ── PCA ──────────────────────────────────────────────
class PCANode(BaseNode):
    category = "data"
    description = "주성분 분석으로 차원을 축소합니다."

    def param_schema(self):
        return [
            {"name": "n_components", "type": "number", "label": "주성분 수", "default": 2},
        ]

    def input_ports(self):
        return [{"name": "input", "type": "dataset"}]

    def output_ports(self):
        return [{"name": "output", "type": "dataset"}]

    async def run(self, params, inputs, ws_callback):
        from sklearn.decomposition import PCA
        X = inputs["input"]
        if X is None:
            raise ValueError("PCA: input이 None입니다.")
        n = int(params.get("n_components", 2))
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        # n_components를 특성 수 이하로 제한
        n = min(n, X.shape[1], X.shape[0])
        pca = PCA(n_components=n)
        X_reduced = pca.fit_transform(X).astype(np.float32)
        return {"output": X_reduced}


# ── Augmentation ─────────────────────────────────────
class AugmentationNode(BaseNode):
    category = "data"
    description = "이미지 데이터 증강 (회전, 뒤집기 등)."

    def param_schema(self):
        return [
            {"name": "rotation", "type": "number", "label": "회전 각도", "default": 20},
            {"name": "horizontal_flip", "type": "boolean", "label": "수평 뒤집기", "default": True},
            {"name": "zoom_range", "type": "number", "label": "줌 범위", "default": 0.1},
        ]

    def input_ports(self):
        return [{"name": "input", "type": "dataset"}]

    def output_ports(self):
        return [{"name": "output", "type": "dataset"}]

    async def run(self, params, inputs, ws_callback):
        X = inputs["input"]
        # 증강 설정을 레이어 config로 전달 (실제 증강은 Trainer에서 수행)
        augmentation_config = {
            "rotation_range": float(params.get("rotation", 20)),
            "horizontal_flip": params.get("horizontal_flip", True),
            "zoom_range": float(params.get("zoom_range", 0.1)),
        }
        return {"output": X, "augmentation_config": augmentation_config}
