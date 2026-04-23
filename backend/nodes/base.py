"""Base class for all node handlers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine


class BaseNode(ABC):
    """모든 노드 핸들러의 추상 기반 클래스."""

    category: str = "misc"
    description: str = ""

    @abstractmethod
    async def run(
        self,
        params: dict[str, Any],
        inputs: dict[str, Any],
        ws_callback: Callable[[dict], Coroutine] | None,
    ) -> dict[str, Any]:
        """노드 실행. 결과를 dict(port_name → value)로 반환."""
        ...

    def param_schema(self) -> list[dict]:
        """프론트엔드 UI 렌더링용 파라미터 스키마."""
        return []

    def input_ports(self) -> list[dict]:
        return [{"name": "input", "type": "any"}]

    def output_ports(self) -> list[dict]:
        return [{"name": "output", "type": "any"}]
