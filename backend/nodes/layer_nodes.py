"""레이어 노드 — 모델 구성용 레이어 정의."""

from __future__ import annotations

from typing import Any

from .base import BaseNode


class LayerNodeBase(BaseNode):
    """레이어 노드는 실행 시 레이어 설정(dict)을 반환한다.
    실제 레이어 생성은 Trainer 노드에서 모델 조립 시 수행."""

    category = "layer"

    def input_ports(self):
        return [{"name": "input", "type": "layer_config"}]

    def output_ports(self):
        return [{"name": "output", "type": "layer_config"}]

    @staticmethod
    def _prev(inputs):
        prev = inputs.get("input", [])
        if not isinstance(prev, list):
            if prev is None:
                return []
            return [prev]
        return prev

    async def run(self, params, inputs, ws_callback):
        """서브클래스에서 _build_output()을 구현하면 labels 자동 전파."""
        result = await self._build_output(params, inputs, ws_callback)
        # _source_labels 자동 전파
        labels = inputs.get("_source_labels")
        if labels is not None:
            result["_source_labels"] = labels
        return result

    async def _build_output(self, params, inputs, ws_callback):
        """서브클래스에서 오버라이드할 실제 출력 생성."""
        return {"output": self._prev(inputs)}


# ── Dense ────────────────────────────────────────────
class DenseLayerNode(LayerNodeBase):
    description = "완전연결(Dense) 레이어."

    def param_schema(self):
        return [
            {"name": "units", "type": "number", "label": "유닛 수", "default": 128},
            {
                "name": "activation",
                "type": "select",
                "label": "활성화 함수",
                "options": ["relu", "sigmoid", "tanh", "softmax", "linear", "swish"],
                "default": "relu",
            },
        ]

    async def _build_output(self, params, inputs, ws_callback):
        return {"output": self._prev(inputs) + [{
            "type": "Dense",
            "units": int(params.get("units", 128)),
            "activation": params.get("activation", "relu"),
        }]}


# ── Conv2D ───────────────────────────────────────────
class Conv2DLayerNode(LayerNodeBase):
    description = "2D 합성곱 레이어."

    def param_schema(self):
        return [
            {"name": "filters", "type": "number", "label": "필터 수", "default": 32},
            {"name": "kernel_size", "type": "number", "label": "커널 크기", "default": 3},
            {"name": "activation", "type": "select", "label": "활성화 함수",
             "options": ["relu", "sigmoid", "tanh", "linear", "swish"], "default": "relu"},
            {"name": "padding", "type": "select", "label": "패딩",
             "options": ["same", "valid"], "default": "same"},
        ]

    async def _build_output(self, params, inputs, ws_callback):
        return {"output": self._prev(inputs) + [{
            "type": "Conv2D",
            "filters": int(params.get("filters", 32)),
            "kernel_size": int(params.get("kernel_size", 3)),
            "activation": params.get("activation", "relu"),
            "padding": params.get("padding", "same"),
        }]}


# ── Conv1D ───────────────────────────────────────────
class Conv1DLayerNode(LayerNodeBase):
    description = "1D 합성곱 레이어 (시계열/NLP)."

    def param_schema(self):
        return [
            {"name": "filters", "type": "number", "label": "필터 수", "default": 64},
            {"name": "kernel_size", "type": "number", "label": "커널 크기", "default": 3},
            {"name": "activation", "type": "select", "label": "활성화 함수",
             "options": ["relu", "sigmoid", "tanh", "linear"], "default": "relu"},
            {"name": "padding", "type": "select", "label": "패딩",
             "options": ["same", "valid"], "default": "same"},
        ]

    async def _build_output(self, params, inputs, ws_callback):
        return {"output": self._prev(inputs) + [{
            "type": "Conv1D",
            "filters": int(params.get("filters", 64)),
            "kernel_size": int(params.get("kernel_size", 3)),
            "activation": params.get("activation", "relu"),
            "padding": params.get("padding", "same"),
        }]}


# ── Conv2DTranspose ──────────────────────────────────
class Conv2DTransposeNode(LayerNodeBase):
    description = "2D 전치 합성곱 (디코더/업샘플링)."

    def param_schema(self):
        return [
            {"name": "filters", "type": "number", "label": "필터 수", "default": 32},
            {"name": "kernel_size", "type": "number", "label": "커널 크기", "default": 3},
            {"name": "activation", "type": "select", "label": "활성화 함수",
             "options": ["relu", "sigmoid", "tanh", "linear"], "default": "relu"},
        ]

    async def _build_output(self, params, inputs, ws_callback):
        return {"output": self._prev(inputs) + [{
            "type": "Conv2DTranspose",
            "filters": int(params.get("filters", 32)),
            "kernel_size": int(params.get("kernel_size", 3)),
            "activation": params.get("activation", "relu"),
        }]}


# ── MaxPooling2D ─────────────────────────────────────
class MaxPooling2DNode(LayerNodeBase):
    description = "2D 최대 풀링 레이어."

    def param_schema(self):
        return [
            {"name": "pool_size", "type": "number", "label": "풀 크기", "default": 2},
        ]

    async def _build_output(self, params, inputs, ws_callback):
        return {"output": self._prev(inputs) + [{
            "type": "MaxPooling2D",
            "pool_size": int(params.get("pool_size", 2)),
        }]}


# ── Flatten ──────────────────────────────────────────
class FlattenNode(LayerNodeBase):
    description = "다차원 텐서를 1D로 변환."

    def param_schema(self):
        return []

    async def _build_output(self, params, inputs, ws_callback):
        return {"output": self._prev(inputs) + [{"type": "Flatten"}]}


# ── GlobalAveragePooling2D ───────────────────────────
class GlobalAveragePooling2DNode(LayerNodeBase):
    description = "공간 평균 풀링 (CNN 헤드)."

    def param_schema(self):
        return []

    async def _build_output(self, params, inputs, ws_callback):
        return {"output": self._prev(inputs) + [{"type": "GlobalAveragePooling2D"}]}


# ── GlobalAveragePooling1D ───────────────────────────
class GlobalAveragePooling1DNode(LayerNodeBase):
    description = "1D 글로벌 평균 풀링."

    def param_schema(self):
        return []

    async def _build_output(self, params, inputs, ws_callback):
        return {"output": self._prev(inputs) + [{"type": "GlobalAveragePooling1D"}]}


# ── BatchNormalization ───────────────────────────────
class BatchNormNode(LayerNodeBase):
    description = "배치 정규화 레이어."

    def param_schema(self):
        return []

    async def _build_output(self, params, inputs, ws_callback):
        return {"output": self._prev(inputs) + [{"type": "BatchNormalization"}]}


# ── Dropout ──────────────────────────────────────────
class DropoutNode(LayerNodeBase):
    description = "드롭아웃 레이어."

    def param_schema(self):
        return [
            {"name": "rate", "type": "number", "label": "드롭아웃 비율", "default": 0.5},
        ]

    async def _build_output(self, params, inputs, ws_callback):
        return {"output": self._prev(inputs) + [{
            "type": "Dropout",
            "rate": float(params.get("rate", 0.5)),
        }]}


# ── Embedding ────────────────────────────────────────
class EmbeddingNode(LayerNodeBase):
    description = "정수 인덱스를 밀집 벡터로 변환 (NLP)."

    def param_schema(self):
        return [
            {"name": "input_dim", "type": "number", "label": "어휘 크기", "default": 10000},
            {"name": "output_dim", "type": "number", "label": "임베딩 차원", "default": 128},
            {"name": "max_length", "type": "number", "label": "최대 시퀀스 길이", "default": 100},
        ]

    async def _build_output(self, params, inputs, ws_callback):
        return {"output": self._prev(inputs) + [{
            "type": "Embedding",
            "input_dim": int(params.get("input_dim", 10000)),
            "output_dim": int(params.get("output_dim", 128)),
            "max_length": int(params.get("max_length", 100)),
        }]}


# ── LSTM ─────────────────────────────────────────────
class LSTMNode(LayerNodeBase):
    description = "LSTM 순환 레이어."

    def param_schema(self):
        return [
            {"name": "units", "type": "number", "label": "유닛 수", "default": 64},
            {"name": "return_sequences", "type": "boolean", "label": "시퀀스 반환", "default": False},
            {"name": "bidirectional", "type": "boolean", "label": "양방향", "default": False},
        ]

    async def _build_output(self, params, inputs, ws_callback):
        cfg = {
            "type": "LSTM",
            "units": int(params.get("units", 64)),
            "return_sequences": params.get("return_sequences", False),
        }
        if params.get("bidirectional", False):
            cfg = {"type": "Bidirectional", "layer": cfg}
        return {"output": self._prev(inputs) + [cfg]}


# ── GRU ──────────────────────────────────────────────
class GRUNode(LayerNodeBase):
    description = "GRU 순환 레이어."

    def param_schema(self):
        return [
            {"name": "units", "type": "number", "label": "유닛 수", "default": 64},
            {"name": "return_sequences", "type": "boolean", "label": "시퀀스 반환", "default": False},
        ]

    async def _build_output(self, params, inputs, ws_callback):
        return {"output": self._prev(inputs) + [{
            "type": "GRU",
            "units": int(params.get("units", 64)),
            "return_sequences": params.get("return_sequences", False),
        }]}


# ── Reshape ──────────────────────────────────────────
class ReshapeNode(LayerNodeBase):
    description = "텐서 shape을 변환합니다."

    def param_schema(self):
        return [
            {"name": "target_shape", "type": "string", "label": "목표 shape", "default": "-1, 1"},
        ]

    async def _build_output(self, params, inputs, ws_callback):
        raw = params.get("target_shape", "-1, 1")
        if isinstance(raw, (list, tuple)):
            shape = tuple(int(s) for s in raw)
        else:
            shape = tuple(int(s.strip()) for s in str(raw).split(","))
        return {"output": self._prev(inputs) + [{
            "type": "Reshape",
            "target_shape": shape,
        }]}


# ── Add (Skip Connection / Residual) ────────────────
class AddNode(BaseNode):
    category = "layer"
    description = "두 텐서를 원소별 더하기 (Residual 연결)."

    def input_ports(self):
        return [
            {"name": "input_a", "type": "layer_config"},
            {"name": "input_b", "type": "layer_config"},
        ]

    def output_ports(self):
        return [{"name": "output", "type": "layer_config"}]

    async def run(self, params, inputs, ws_callback):
        a = inputs.get("input_a", [])
        b = inputs.get("input_b", [])
        if not isinstance(a, list):
            a = [] if a is None else [a]
        if not isinstance(b, list):
            b = [] if b is None else [b]
        return {"output": [{"type": "Add", "branches": [a, b]}]}


# ── Concat (Merge) ───────────────────────────────────
class ConcatNode(BaseNode):
    category = "layer"
    description = "두 텐서를 연결 (Concatenate)."

    def param_schema(self):
        return [
            {"name": "axis", "type": "number", "label": "연결 축", "default": -1},
        ]

    def input_ports(self):
        return [
            {"name": "input_a", "type": "layer_config"},
            {"name": "input_b", "type": "layer_config"},
        ]

    def output_ports(self):
        return [{"name": "output", "type": "layer_config"}]

    async def run(self, params, inputs, ws_callback):
        a = inputs.get("input_a", [])
        b = inputs.get("input_b", [])
        if not isinstance(a, list):
            a = [] if a is None else [a]
        if not isinstance(b, list):
            b = [] if b is None else [b]
        return {"output": [{
            "type": "Concatenate",
            "axis": int(params.get("axis", -1)),
            "branches": [a, b],
        }]}


# ── Pretrained Model ─────────────────────────────────
class PretrainedModelNode(BaseNode):
    category = "layer"
    description = "사전학습 모델 (ResNet, MobileNet, EfficientNet 등)."

    def param_schema(self):
        return [
            {"name": "model_name", "type": "select", "label": "모델",
             "options": ["MobileNetV2", "ResNet50", "EfficientNetB0", "VGG16", "InceptionV3"],
             "default": "MobileNetV2"},
            {"name": "trainable", "type": "boolean", "label": "파인튜닝", "default": False},
            {"name": "include_top", "type": "boolean", "label": "헤드 포함", "default": False},
        ]

    def input_ports(self):
        return [{"name": "input", "type": "layer_config"}]

    def output_ports(self):
        return [{"name": "output", "type": "layer_config"}]

    async def run(self, params, inputs, ws_callback):
        prev = inputs.get("input", [])
        if not isinstance(prev, list):
            if prev is None:
                prev = []
            else:
                prev = [prev]
        return {"output": prev + [{
            "type": "PretrainedModel",
            "model_name": params.get("model_name", "MobileNetV2"),
            "trainable": params.get("trainable", False),
            "include_top": params.get("include_top", False),
        }]}


# ── MultiHeadAttention ───────────────────────────────
class MultiHeadAttentionNode(LayerNodeBase):
    description = "멀티 헤드 어텐션 레이어."

    def param_schema(self):
        return [
            {"name": "num_heads", "type": "number", "label": "헤드 수", "default": 8},
            {"name": "key_dim", "type": "number", "label": "키 차원", "default": 64},
        ]

    async def _build_output(self, params, inputs, ws_callback):
        return {"output": self._prev(inputs) + [{
            "type": "MultiHeadAttention",
            "num_heads": int(params.get("num_heads", 8)),
            "key_dim": int(params.get("key_dim", 64)),
        }]}


# ── Transformer Block ────────────────────────────────
class TransformerBlockNode(LayerNodeBase):
    description = "Transformer 블록 (MHA + FFN + LayerNorm)."

    def param_schema(self):
        return [
            {"name": "num_heads", "type": "number", "label": "헤드 수", "default": 4},
            {"name": "ff_dim", "type": "number", "label": "FFN 차원", "default": 128},
            {"name": "dropout", "type": "number", "label": "드롭아웃", "default": 0.1},
        ]

    async def _build_output(self, params, inputs, ws_callback):
        return {"output": self._prev(inputs) + [{
            "type": "TransformerBlock",
            "num_heads": int(params.get("num_heads", 4)),
            "ff_dim": int(params.get("ff_dim", 128)),
            "dropout": float(params.get("dropout", 0.1)),
        }]}
