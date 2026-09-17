"""정적 호스팅(GitHub Pages)용 API 스냅샷 생성.

백엔드 없이도 노드 카탈로그와 예제를 볼 수 있도록
/api/node-catalog, /api/examples, /api/examples/all-details, /api/examples/{id}
응답을 JSON 파일로 내보낸다. (main.py 의 해당 엔드포인트와 동일한 결과)

사용법:
    cd backend
    python scripts/export_static_api.py [출력 디렉터리]
    # 기본 출력: ../frontend-web/public/static-api
"""

import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

EXAMPLES_DIR = BACKEND_DIR / "examples"
DEFAULT_OUT = BACKEND_DIR.parent / "frontend-web" / "public" / "static-api"


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def export_node_catalog(out: Path) -> int:
    from nodes.registry import NODE_REGISTRY

    catalog = [
        {
            "type": node_type,
            "category": handler.category,
            "description": handler.description,
            "params": handler.param_schema(),
            "inputs": handler.input_ports(),
            "outputs": handler.output_ports(),
        }
        for node_type, handler in NODE_REGISTRY.items()
    ]
    write_json(out / "node-catalog.json", catalog)
    return len(catalog)


def export_examples(out: Path) -> int:
    index_path = EXAMPLES_DIR / "_index.json"
    index = []
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
    write_json(out / "examples.json", index)

    details = []
    for json_file in sorted(EXAMPLES_DIR.rglob("*.json")):
        if json_file.name == "_index.json":
            continue
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {json_file.name}: {e}")
            continue
        details.append(data)
        prefix = json_file.name.split("_", 1)[0]
        if prefix.isdigit():
            write_json(out / "examples" / f"{int(prefix)}.json", data)

    details.sort(key=lambda x: x.get("meta", {}).get("id", 999))
    write_json(out / "examples-all-details.json", details)
    return len(details)


def main() -> None:
    out = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_OUT
    n_nodes = export_node_catalog(out)
    n_examples = export_examples(out)
    print(f"Exported {n_nodes} node types, {n_examples} examples -> {out}")


if __name__ == "__main__":
    main()
