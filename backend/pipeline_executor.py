"""Pipeline Executor — 노드 그래프를 토폴로지 정렬 후 순서대로 실행."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Callable, Coroutine

import numpy as np


def _summarize_value(val: Any) -> dict | None:
    """값을 직렬화 가능한 요약으로 변환."""
    if val is None:
        return None
    # numpy array
    if isinstance(val, np.ndarray):
        info: dict[str, Any] = {
            "type": "ndarray",
            "shape": list(val.shape),
            "dtype": str(val.dtype),
        }
        if val.size > 0 and np.issubdtype(val.dtype, np.number):
            info["min"] = float(np.nanmin(val))
            info["max"] = float(np.nanmax(val))
            info["mean"] = float(np.nanmean(val))
        elif val.size > 0:
            unique = np.unique(val)
            info["unique_count"] = int(len(unique))
            if len(unique) <= 10:
                info["unique_values"] = [str(v) for v in unique]
        return info
    # torch tensor
    try:
        import torch
        if isinstance(val, torch.Tensor):
            return {
                "type": "tensor",
                "shape": list(val.shape),
                "dtype": str(val.dtype),
            }
    except ImportError:
        pass
    # pandas DataFrame
    try:
        import pandas as pd
        if isinstance(val, pd.DataFrame):
            return {
                "type": "DataFrame",
                "shape": list(val.shape),
                "columns": list(val.columns[:20]),
            }
    except ImportError:
        pass
    # base64 이미지 문자열 — 플래그만 전송 (실제 이미지는 VISUALIZATION 메시지로 별도 전송)
    if isinstance(val, str) and len(val) > 200:
        if val.startswith("iVBOR") or val.startswith("/9j/"):
            return {"type": "image_b64", "has_image": True, "length": len(val)}
    # 레이어 config 리스트 (layer nodes)
    if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict) and "type" in val[0]:
        layers = []
        for item in val:
            if isinstance(item, dict) and "type" in item:
                desc = item["type"]
                details = {k: v for k, v in item.items()
                           if k != "type" and not isinstance(v, (list, dict))}
                if details:
                    desc += "(" + ", ".join(f"{k}={v}" for k, v in details.items()) + ")"
                layers.append(desc)
            else:
                layers.append(str(type(item).__name__))
        return {"type": "layers", "count": len(layers), "layers": layers}
    # list (기타)
    if isinstance(val, list):
        return {"type": "list", "length": len(val)}
    # dict (config 등)
    if isinstance(val, dict):
        return {"type": "config", "keys": list(val.keys())[:20], "preview": {
            k: (str(v)[:80] if not isinstance(v, (dict, list, np.ndarray)) else type(v).__name__)
            for k, v in list(val.items())[:10]
        }}
    # scalar
    if isinstance(val, (int, float, bool, str)):
        return {"type": type(val).__name__, "value": val}
    # fallback
    return {"type": type(val).__name__}


def _summarize_outputs(outputs: Any) -> dict:
    """노드 출력 dict를 포트별 요약으로 변환."""
    if not isinstance(outputs, dict):
        return {}
    summary = {}
    for port_name, val in outputs.items():
        if port_name.startswith("_"):
            continue  # 내부 필드 제외
        s = _summarize_value(val)
        if s is not None:
            summary[port_name] = s
    return summary


class PipelineExecutor:
    """노드 그래프를 토폴로지 정렬(Kahn's algorithm)하고 순서대로 실행."""

    def __init__(
        self,
        nodes: list[dict],
        edges: list[dict],
        ws_callback: Callable[[dict], Coroutine] | None,
    ):
        self.nodes = {n["id"]: n for n in nodes}
        self.edges = edges
        self.ws_callback = ws_callback
        self.context: dict[str, Any] = {}  # node_id → outputs

    # ── 토폴로지 정렬 ────────────────────────────────
    def topological_sort(self) -> list[str]:
        in_degree: dict[str, int] = defaultdict(int)
        adj: dict[str, list[str]] = defaultdict(list)

        for nid in self.nodes:
            in_degree.setdefault(nid, 0)

        for edge in self.edges:
            src = edge["source"]
            tgt = edge["target"]
            adj[src].append(tgt)
            in_degree[tgt] += 1

        queue: deque[str] = deque(
            nid for nid, deg in in_degree.items() if deg == 0
        )
        order: list[str] = []

        while queue:
            nid = queue.popleft()
            order.append(nid)
            for neighbor in adj[nid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.nodes):
            raise ValueError("Cycle detected in pipeline graph")

        return order

    # ── 포트명 별칭 매핑 (예제 호환용) ──────────────────
    _PORT_ALIASES = {
        "dataset": ["features", "output"],
        "data": ["features", "output"],
    }

    # ── 입력 수집 ─────────────────────────────────────
    def _gather_inputs(self, node_id: str) -> dict[str, Any]:
        inputs: dict[str, Any] = {}
        for edge in self.edges:
            if edge["target"] == node_id:
                source_id = edge["source"]
                source_port = edge.get("sourceHandle", "output")
                target_port = edge.get("targetHandle", "input")
                source_output = self.context.get(source_id, {})
                if isinstance(source_output, dict):
                    # 포트 키가 존재하는지 확인 (None 값도 유효한 포트 출력)
                    has_port = source_port in source_output
                    value = source_output.get(source_port)
                    # 포트명이 없으면 별칭으로 재시도
                    if not has_port and source_port in self._PORT_ALIASES:
                        for alias in self._PORT_ALIASES[source_port]:
                            if alias in source_output:
                                value = source_output[alias]
                                has_port = True
                                break
                    # 여전히 포트가 없으면 전체 dict 전달
                    if not has_port and value is None:
                        value = source_output
                    inputs[target_port] = value

                    # ── labels 자동 전파 ──────────────────
                    # 데이터 노드 → 레이어 노드 연결 시 labels 전파
                    if target_port == "input" and "labels" in source_output:
                        if "_source_labels" not in inputs:
                            inputs["_source_labels"] = source_output["labels"]

                    # 레이어 노드 출력의 _source_labels 전파
                    if "_source_labels" in source_output and "_source_labels" not in inputs:
                        inputs["_source_labels"] = source_output["_source_labels"]

                    # train_features/val_features 연결 시 대응하는 labels 자동 전파
                    if target_port in ("train_features", "val_features", "test_features"):
                        label_port = target_port.replace("features", "labels")
                        if label_port not in inputs and label_port in source_output:
                            label_val = source_output[label_port]
                            if label_val is not None:
                                inputs[label_port] = label_val
                else:
                    inputs[target_port] = source_output
        return inputs

    # ── 실행 ──────────────────────────────────────────
    async def execute(self):
        from nodes.registry import NODE_REGISTRY

        order = self.topological_sort()

        if self.ws_callback:
            await self.ws_callback(
                {
                    "type": "PIPELINE_START",
                    "execution_order": order,
                    "total_nodes": len(order),
                }
            )

        for idx, node_id in enumerate(order):
            node = self.nodes[node_id]
            node_type = node["type"]

            handler = NODE_REGISTRY.get(node_type)
            if handler is None:
                raise ValueError(f"Unknown node type: {node_type}")

            if self.ws_callback:
                await self.ws_callback(
                    {
                        "type": "NODE_START",
                        "node_id": node_id,
                        "node_type": node_type,
                        "index": idx,
                    }
                )

            inputs = self._gather_inputs(node_id)
            params = node.get("params", {})

            try:
                outputs = await handler.run(params, inputs, self.ws_callback)
            except Exception as e:
                # 노드 ID와 타입을 포함한 상세 에러 메시지
                import traceback
                detail = traceback.format_exc()
                raise RuntimeError(
                    f"Node '{node_id}' (type={node_type}) 실행 중 에러: {e}\n{detail}"
                ) from e

            self.context[node_id] = outputs

            if self.ws_callback:
                await self.ws_callback(
                    {
                        "type": "NODE_COMPLETE",
                        "node_id": node_id,
                        "node_type": node_type,
                        "index": idx,
                        "output_summary": _summarize_outputs(outputs),
                    }
                )

        if self.ws_callback:
            await self.ws_callback({"type": "PIPELINE_COMPLETE"})
