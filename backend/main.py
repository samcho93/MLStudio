"""ML Node Studio — FastAPI Backend (공통 서버)"""

from __future__ import annotations

import asyncio
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from pipeline_executor import PipelineExecutor

app = FastAPI(title="ML Node Studio Backend", version="0.1.0")

# ── 학습된 모델 저장소 (in-memory) ───────────────────
_trained_model_store: dict[str, Any] = {
    "model": None,           # 학습된 모델 객체
    "feature_names": None,   # 피처 컬럼 이름들
    "class_names": None,     # 클래스 이름들 (분류)
    "is_regression": False,  # 회귀 여부
    "is_nlp": False,         # NLP 여부
    "input_shape": None,     # 입력 shape
    "scaler": None,          # scaler 객체
    "vocab": None,           # NLP vocab dict
    "max_length": None,      # NLP max_length
    "framework": None,       # tensorflow / pytorch
    "layer_config": None,    # layer config list (for PyTorch rebuild)
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 파일 업로드 디렉토리 ─────────────────────────────
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
EXAMPLES_DIR = Path("examples")
EXAMPLES_DIR.mkdir(exist_ok=True)


# ── REST endpoints ───────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/node-catalog")
async def node_catalog():
    """사용 가능한 노드 타입 목록 반환."""
    from nodes.registry import NODE_REGISTRY

    catalog: list[dict[str, Any]] = []
    for node_type, handler in NODE_REGISTRY.items():
        catalog.append(
            {
                "type": node_type,
                "category": handler.category,
                "description": handler.description,
                "params": handler.param_schema(),
                "inputs": handler.input_ports(),
                "outputs": handler.output_ports(),
            }
        )
    return catalog


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        f.write(await file.read())
    return {"path": str(dest), "filename": file.filename}


@app.post("/api/pipeline/validate")
async def validate_pipeline(payload: dict):
    """파이프라인 JSON 유효성 검사 (실행 전 프리체크)."""
    try:
        executor = PipelineExecutor(
            payload.get("nodes", []),
            payload.get("edges", []),
            ws_callback=None,
        )
        order = executor.topological_sort()
        return {"valid": True, "execution_order": order}
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"valid": False, "error": str(e)},
        )


# ── 파일 미리보기 API ────────────────────────────────
@app.get("/api/preview")
async def preview_file(path: str):
    """데이터 노드의 파일/폴더 내용을 미리보기.

    - CSV/TSV: 컬럼명, 행 수, 샘플 5행
    - 이미지 폴더: 클래스별 이미지 수, 총 파일 수
    - 기타 파일: 파일 크기, 타입 정보
    """
    from nodes.data_nodes import _resolve_path

    try:
        resolved = _resolve_path(path)
        if not resolved.exists():
            return JSONResponse(
                status_code=404,
                content={"error": f"경로를 찾을 수 없습니다: {path}"},
            )

        # ── 폴더 (ImageFolder 용) ──
        if resolved.is_dir():
            classes: list[dict[str, Any]] = []
            total_images = 0
            image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}
            for sub in sorted(resolved.iterdir()):
                if sub.is_dir():
                    imgs = sorted(
                        [f for f in sub.iterdir() if f.suffix.lower() in image_exts],
                        key=lambda f: f.name,
                    )
                    # 클래스별 샘플 이미지 최대 6장 경로
                    sample_paths = [str(f) for f in imgs[:6]]
                    classes.append({
                        "name": sub.name,
                        "count": len(imgs),
                        "samples": sample_paths,
                    })
                    total_images += len(imgs)
            # 루트에 직접 이미지가 있는 경우
            root_imgs = sorted(
                [f for f in resolved.iterdir() if f.is_file() and f.suffix.lower() in image_exts],
                key=lambda f: f.name,
            )
            root_sample_paths = [str(f) for f in root_imgs[:6]]
            return {
                "type": "folder",
                "path": str(resolved),
                "num_classes": len(classes),
                "total_images": total_images + len(root_imgs),
                "root_images": len(root_imgs),
                "root_samples": root_sample_paths,
                "classes": classes[:50],  # 최대 50개 클래스
            }

        # ── CSV / TSV 파일 ──
        if resolved.suffix.lower() in {".csv", ".tsv", ".txt"}:
            sep = "\t" if resolved.suffix.lower() == ".tsv" else ","
            try:
                df = pd.read_csv(resolved, sep=sep, nrows=10)
                df_full_count = sum(1 for _ in open(resolved, encoding="utf-8")) - 1
            except Exception:
                df = pd.read_csv(resolved, sep=sep, nrows=10, encoding="latin-1")
                df_full_count = sum(1 for _ in open(resolved, encoding="latin-1")) - 1

            sample = df.head(5).to_dict(orient="records")
            # NaN → None
            for row in sample:
                for k, v in row.items():
                    if isinstance(v, float) and (v != v):  # NaN check
                        row[k] = None

            return {
                "type": "csv",
                "path": str(resolved),
                "num_rows": max(df_full_count, len(df)),
                "num_columns": len(df.columns),
                "columns": [
                    {"name": col, "dtype": str(df[col].dtype)}
                    for col in df.columns
                ],
                "sample": sample,
            }

        # ── NumPy 파일 ──
        if resolved.suffix.lower() in {".npy", ".npz"}:
            if resolved.suffix.lower() == ".npy":
                arr = np.load(str(resolved), allow_pickle=False)
                return {
                    "type": "numpy",
                    "path": str(resolved),
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "size_mb": round(arr.nbytes / 1024 / 1024, 2),
                }
            else:
                data = np.load(str(resolved), allow_pickle=False)
                arrays = {k: {"shape": list(data[k].shape), "dtype": str(data[k].dtype)} for k in data.files}
                return {
                    "type": "numpy_archive",
                    "path": str(resolved),
                    "arrays": arrays,
                }

        # ── 기타 파일 ──
        size_bytes = resolved.stat().st_size
        return {
            "type": "file",
            "path": str(resolved),
            "filename": resolved.name,
            "extension": resolved.suffix,
            "size_mb": round(size_bytes / 1024 / 1024, 2),
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


@app.get("/api/preview/csv-full")
async def preview_csv_full(path: str, page: int = 1, page_size: int = 100):
    """CSV 파일의 전체 데이터를 페이지네이션으로 반환."""
    from nodes.data_nodes import _resolve_path

    try:
        resolved = _resolve_path(path)
        if not resolved.exists():
            return JSONResponse(
                status_code=404,
                content={"error": f"파일을 찾을 수 없습니다: {path}"},
            )
        if resolved.suffix.lower() not in {".csv", ".tsv", ".txt"}:
            return JSONResponse(
                status_code=400,
                content={"error": "CSV/TSV 파일이 아닙니다"},
            )

        sep = "\t" if resolved.suffix.lower() == ".tsv" else ","
        try:
            df = pd.read_csv(resolved, sep=sep)
        except Exception:
            df = pd.read_csv(resolved, sep=sep, encoding="latin-1")

        total_rows = len(df)
        total_pages = max(1, (total_rows + page_size - 1) // page_size)
        page = max(1, min(page, total_pages))

        start = (page - 1) * page_size
        end = min(start + page_size, total_rows)
        page_df = df.iloc[start:end]

        rows = page_df.to_dict(orient="records")
        for row in rows:
            for k, v in row.items():
                if isinstance(v, float) and (v != v):
                    row[k] = None

        return {
            "columns": [
                {"name": col, "dtype": str(df[col].dtype)}
                for col in df.columns
            ],
            "rows": rows,
            "page": page,
            "page_size": page_size,
            "total_rows": total_rows,
            "total_pages": total_pages,
            "start_row": start + 1,
            "end_row": end,
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


@app.get("/api/preview/image")
async def preview_image(path: str):
    """이미지 파일을 직접 서빙 (썸네일 미리보기용)."""
    from nodes.data_nodes import _resolve_path

    resolved = _resolve_path(path)
    if not resolved.exists() or not resolved.is_file():
        return JSONResponse(status_code=404, content={"error": "이미지를 찾을 수 없습니다"})

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}
    if resolved.suffix.lower() not in image_exts:
        return JSONResponse(status_code=400, content={"error": "이미지 파일이 아닙니다"})

    # MIME 타입 매핑
    mime_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".bmp": "image/bmp",
        ".gif": "image/gif", ".webp": "image/webp",
        ".tiff": "image/tiff",
    }
    media_type = mime_map.get(resolved.suffix.lower(), "image/jpeg")
    return FileResponse(str(resolved), media_type=media_type)


@app.post("/api/pipeline/save")
async def save_pipeline(payload: dict):
    name = payload.get("name", "untitled")
    path = OUTPUT_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return {"saved": str(path)}


@app.get("/api/pipeline/list")
async def list_pipelines():
    files = sorted(OUTPUT_DIR.glob("*.json"))
    return [{"name": f.stem, "path": str(f)} for f in files]


@app.get("/api/pipeline/load/{name}")
async def load_pipeline(name: str):
    path = OUTPUT_DIR / f"{name}.json"
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "Not found"})
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Examples API ─────────────────────────────────────
@app.get("/api/examples")
async def list_examples(category: str = None, difficulty: str = None):
    """예제 목록 반환 (인덱스 기반)."""
    index_path = EXAMPLES_DIR / "_index.json"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            examples = json.load(f)
    else:
        examples = []

    if category:
        examples = [e for e in examples if e["category"] == category]
    if difficulty:
        examples = [e for e in examples if e["difficulty"] == difficulty]

    return examples


@app.get("/api/examples/categories")
async def example_categories():
    """예제 카테고리 목록."""
    index_path = EXAMPLES_DIR / "_index.json"
    if not index_path.exists():
        return []

    with open(index_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    cat_counts: dict[str, int] = {}
    for e in examples:
        cat = e["category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    labels = {
        "tutorial": ("Tutorial", "튜토리얼"),
        "classification": ("Classification", "분류"),
        "regression": ("Regression", "회귀"),
        "unsupervised": ("Unsupervised", "비지도/클러스터링"),
        "image": ("Image", "이미지 심화"),
        "nlp": ("NLP", "자연어처리"),
        "timeseries": ("Time Series", "시계열"),
        "advanced": ("Advanced", "고급 파이프라인"),
    }

    return [
        {
            "id": cat,
            "label": labels.get(cat, (cat, cat))[0],
            "label_ko": labels.get(cat, (cat, cat))[1],
            "count": count,
        }
        for cat, count in cat_counts.items()
    ]


@app.get("/api/examples/all-details")
async def all_example_details():
    """모든 예제의 전체 데이터(meta + nodes + edges) 반환."""
    results = []
    for json_file in sorted(EXAMPLES_DIR.rglob("*.json")):
        if json_file.name == "_index.json":
            continue
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            results.append(data)
        except Exception:
            pass
    results.sort(key=lambda x: x.get("meta", {}).get("id", 999))
    return results


@app.get("/api/examples/{example_id}")
async def load_example(example_id: int):
    """특정 예제 파이프라인 로드."""
    for json_file in EXAMPLES_DIR.rglob("*.json"):
        if json_file.name == "_index.json":
            continue
        if json_file.name.startswith(f"{example_id:03d}_"):
            with open(json_file, "r", encoding="utf-8") as f:
                return json.load(f)

    return JSONResponse(status_code=404, content={"error": f"Example {example_id} not found"})


# ── 모델 메타데이터 추출 ──────────────────────────────
def _extract_model_info(executor: PipelineExecutor, nodes: list):
    """파이프라인 실행 후 모델과 메타데이터를 추출하여 저장."""
    ctx = executor.context

    # 모델 찾기 (Trainer/KFoldTrainer 노드 결과에서)
    model = None
    for node_id, output in ctx.items():
        if isinstance(output, dict) and "model" in output:
            m = output["model"]
            if m is not None and (hasattr(m, 'predict') or hasattr(m, 'parameters')):
                model = m

    if model is None:
        return

    _trained_model_store["model"] = model

    # 피처 이름, 클래스 이름 추출
    feature_names = None
    class_names = None
    is_regression = False
    is_nlp = False
    input_shape = None
    scaler = None

    for node in nodes:
        ntype = node.get("type", "")
        nid = node.get("id", "")
        params = node.get("params", {})
        output = ctx.get(nid, {})

        # CSV 데이터에서 피처 이름 추출
        if ntype == "CSVLoader" and isinstance(output, dict):
            df = output.get("dataframe")
            if df is not None:
                import pandas as pd
                if isinstance(df, pd.DataFrame):
                    target_col = params.get("target_column", "")
                    if params.get("tokenize"):
                        is_nlp = True
                    else:
                        cols = [c for c in df.columns if c != target_col]
                        if not target_col:
                            cols = list(df.columns[:-1])
                        feature_names = cols

                    # 클래스 이름 추출 (분류)
                    label_col = target_col or params.get("label_column", "")
                    if not label_col and len(df.columns) >= 2:
                        label_col = df.columns[-1]
                    if label_col and label_col in df.columns:
                        unique = df[label_col].unique()
                        if len(unique) <= 50:  # 분류로 간주
                            class_names = sorted([str(u) for u in unique])

            features = output.get("features")
            if features is not None:
                import numpy as np
                input_shape = list(features.shape[1:])

        # NumpyInput에서 shape 추출
        if ntype == "NumpyInput" and isinstance(output, dict):
            tf = output.get("train_features")
            if tf is not None:
                import numpy as np
                input_shape = list(tf.shape[1:])

        # Trainer에서 회귀 여부
        if ntype in ("Trainer", "KFoldTrainer") and isinstance(output, dict):
            loss_type = params.get("loss", "")
            if loss_type in ("mse", "mae", "mean_squared_error", "mean_absolute_error"):
                is_regression = True

    # Scaler 추출
    for node in nodes:
        ntype = node.get("type", "")
        nid = node.get("id", "")
        output = ctx.get(nid, {})
        if ntype in ("StandardScaler", "MinMaxScaler"):
            # scaler is not stored in output currently, but note the type
            scaler = {"type": ntype}

    # NLP vocab/max_length 추출 (CSVLoader tokenize 모드)
    vocab = None
    max_length = None
    for node in nodes:
        ntype = node.get("type", "")
        nid = node.get("id", "")
        params = node.get("params", {})
        output = ctx.get(nid, {})
        if ntype == "CSVLoader" and isinstance(output, dict):
            if output.get("vocab"):
                vocab = output["vocab"]
            if output.get("max_length"):
                max_length = output["max_length"]
            if params.get("max_length"):
                max_length = int(params["max_length"])

    # Framework 추출
    framework = "tensorflow"
    for node in nodes:
        ntype = node.get("type", "")
        params = node.get("params", {})
        if ntype in ("Trainer", "KFoldTrainer"):
            framework = params.get("framework", "tensorflow")

    _trained_model_store["feature_names"] = feature_names
    _trained_model_store["class_names"] = class_names
    _trained_model_store["is_regression"] = is_regression
    _trained_model_store["is_nlp"] = is_nlp
    _trained_model_store["input_shape"] = input_shape
    # Layer config 추출 (PyTorch 모델 재빌드용)
    layer_config = None
    for node in nodes:
        ntype = node.get("type", "")
        nid = node.get("id", "")
        output = ctx.get(nid, {})
        if ntype in ("Dense", "Conv2D", "Conv1D", "Flatten", "Dropout",
                      "BatchNorm", "LSTM", "GRU", "Embedding", "MaxPooling2D",
                      "GlobalAveragePooling2D", "GlobalAveragePooling1D",
                      "Reshape", "Add", "Concat", "Conv2DTranspose",
                      "TransformerBlock", "PretrainedModel", "MultiHeadAttention"):
            if layer_config is None:
                layer_config = []
            # Output of layer nodes is typically a dict with layer config
            if isinstance(output, dict) and "type" in output:
                layer_config.append(output)
            elif isinstance(output, list):
                layer_config.extend(
                    item for item in output if isinstance(item, dict) and "type" in item
                )

    # If layer_config empty, try to extract from Trainer inputs
    if not layer_config:
        for node in nodes:
            ntype = node.get("type", "")
            nid = node.get("id", "")
            output = ctx.get(nid, {})
            if ntype in ("Trainer", "KFoldTrainer") and isinstance(output, dict):
                # Check if layers were passed through the pipeline context
                pass

    _trained_model_store["scaler"] = scaler
    _trained_model_store["vocab"] = vocab
    _trained_model_store["max_length"] = max_length
    _trained_model_store["framework"] = framework
    _trained_model_store["layer_config"] = layer_config


# ── Predict API ──────────────────────────────────────
@app.post("/api/predict")
async def predict(payload: dict):
    """학습된 모델로 예측."""
    import numpy as np

    model = _trained_model_store.get("model")
    if model is None:
        return JSONResponse(status_code=400, content={"error": "학습된 모델이 없습니다. 먼저 파이프라인을 실행하세요."})

    input_values = payload.get("input", [])
    if not input_values:
        return JSONResponse(status_code=400, content={"error": "입력 데이터가 없습니다."})

    try:
        X = np.array([input_values], dtype=np.float32)

        # PyTorch 모델
        if hasattr(model, 'parameters'):
            import torch
            model.eval()
            with torch.no_grad():
                tensor = torch.FloatTensor(X)
                device = next(model.parameters()).device
                tensor = tensor.to(device)
                output = model(tensor)
                probs = output.cpu().numpy()
        # TensorFlow 모델
        elif hasattr(model, 'predict'):
            probs = model.predict(X, verbose=0)
        else:
            return JSONResponse(status_code=400, content={"error": "지원하지 않는 모델 형식입니다."})

        probs = probs[0]  # 첫 번째 (유일한) 샘플
        is_regression = _trained_model_store.get("is_regression", False)
        class_names = _trained_model_store.get("class_names")

        if is_regression:
            return {
                "type": "regression",
                "prediction": float(probs[0]) if probs.size > 0 else float(probs),
            }
        else:
            # 분류
            if probs.size == 1:
                # 이진 분류 (sigmoid)
                prob = float(probs)
                predicted_class = 1 if prob > 0.5 else 0
                confidences = [1 - prob, prob]
            else:
                predicted_class = int(np.argmax(probs))
                confidences = probs.tolist()

            label = class_names[predicted_class] if class_names and predicted_class < len(class_names) else str(predicted_class)

            return {
                "type": "classification",
                "predicted_class": predicted_class,
                "predicted_label": label,
                "confidences": confidences,
                "class_names": class_names,
            }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"예측 실패: {str(e)}"})


@app.get("/api/model-info")
async def model_info():
    """현재 로드된 모델 정보."""
    if _trained_model_store.get("model") is None:
        return {"ready": False}
    return {
        "ready": True,
        "feature_names": _trained_model_store.get("feature_names"),
        "class_names": _trained_model_store.get("class_names"),
        "is_regression": _trained_model_store.get("is_regression"),
        "is_nlp": _trained_model_store.get("is_nlp"),
        "input_shape": _trained_model_store.get("input_shape"),
    }


# ── Model Save/Load/List/Delete API ──────────────────
SAVED_MODELS_DIR = Path("outputs/saved_models")
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/api/model/save")
async def save_model(body: dict):
    """Save the currently trained model to disk."""
    import pickle
    from datetime import datetime

    model = _trained_model_store.get("model")
    if model is None:
        return JSONResponse(
            status_code=400,
            content={"error": "No trained model in memory. Train a pipeline first."},
        )

    name = body.get("name", "untitled")
    # Sanitize name
    name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)
    if not name:
        name = "untitled"

    model_dir = SAVED_MODELS_DIR / name
    model_dir.mkdir(parents=True, exist_ok=True)

    framework = _trained_model_store.get("framework", "tensorflow")

    # Save model file
    try:
        if framework == "tensorflow" or hasattr(model, "save"):
            model_path = model_dir / "model.keras"
            model.save(str(model_path))
            print(f"[OK] TF model saved to {model_path}")
        elif hasattr(model, "state_dict"):
            import torch

            model_path = model_dir / "model.pth"
            torch.save(model.state_dict(), str(model_path))
            print(f"[OK] PyTorch model saved to {model_path}")

            # Save layer config for rebuilding
            layer_config = _trained_model_store.get("layer_config")
            input_shape = _trained_model_store.get("input_shape")
            class_json = {
                "layers": layer_config if layer_config else [],
                "input_shape": input_shape if input_shape else [],
            }
            class_path = model_dir / "model_class.json"
            with open(class_path, "w", encoding="utf-8") as f:
                json.dump(class_json, f, ensure_ascii=False, indent=2)
            print(f"[OK] PyTorch model_class.json saved to {class_path}")
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "Unsupported model type for saving."},
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to save model: {str(e)}"},
        )

    # Save scaler if available
    scaler = _trained_model_store.get("scaler")
    scaler_path_str = None
    if scaler is not None and not isinstance(scaler, dict):
        try:
            scaler_file = model_dir / "scaler.pkl"
            with open(scaler_file, "wb") as f:
                pickle.dump(scaler, f)
            scaler_path_str = str(scaler_file)
            print(f"[OK] Scaler saved to {scaler_file}")
        except Exception as e:
            print(f"[X] Failed to save scaler: {e}")

    # Save metadata
    metadata = {
        "name": name,
        "framework": framework,
        "feature_names": _trained_model_store.get("feature_names"),
        "class_names": _trained_model_store.get("class_names"),
        "is_regression": _trained_model_store.get("is_regression", False),
        "is_nlp": _trained_model_store.get("is_nlp", False),
        "input_shape": _trained_model_store.get("input_shape"),
        "scaler_path": scaler_path_str,
        "vocab": _trained_model_store.get("vocab"),
        "max_length": _trained_model_store.get("max_length"),
        "saved_at": datetime.now().isoformat(),
    }
    meta_path = model_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[OK] Model '{name}' saved to {model_dir}")
    return {"saved": True, "name": name, "path": str(model_dir), "metadata": metadata}


@app.get("/api/model/list")
async def list_models():
    """List all saved models."""
    if not SAVED_MODELS_DIR.exists():
        return []

    models = []
    for d in sorted(SAVED_MODELS_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            models.append({
                "name": meta.get("name", d.name),
                "framework": meta.get("framework", "unknown"),
                "saved_at": meta.get("saved_at", ""),
                "is_regression": meta.get("is_regression", False),
                "class_names": meta.get("class_names"),
                "input_shape": meta.get("input_shape"),
            })
        except Exception:
            pass

    return models


@app.post("/api/model/load/{name}")
async def load_model(name: str):
    """Load a saved model into memory."""
    model_dir = SAVED_MODELS_DIR / name
    meta_path = model_dir / "metadata.json"

    if not meta_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": f"Model '{name}' not found."},
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    framework = metadata.get("framework", "tensorflow")

    # Load model
    try:
        if framework == "tensorflow":
            import tensorflow as tf

            model_path = model_dir / "model.keras"
            if not model_path.exists():
                model_path = model_dir / "model.h5"
            if not model_path.exists():
                return JSONResponse(
                    status_code=404,
                    content={"error": f"TF model file not found in {model_dir}"},
                )
            model = tf.keras.models.load_model(str(model_path))
            print(f"[OK] TF model loaded from {model_path}")

        elif framework == "pytorch":
            import torch
            import torch.nn as nn

            model_path = model_dir / "model.pth"
            class_json_path = model_dir / "model_class.json"

            if not model_path.exists():
                return JSONResponse(
                    status_code=404,
                    content={"error": f"PyTorch model file not found in {model_dir}"},
                )
            if not class_json_path.exists():
                return JSONResponse(
                    status_code=404,
                    content={"error": f"model_class.json not found in {model_dir}"},
                )

            with open(class_json_path, "r", encoding="utf-8") as f:
                class_info = json.load(f)

            layers = class_info.get("layers", [])
            input_shape = class_info.get("input_shape", metadata.get("input_shape", []))
            is_regression = metadata.get("is_regression", False)

            from frameworks.torch_backend import TorchTrainer

            trainer = TorchTrainer()
            model = trainer._build_model(nn, layers, tuple(input_shape), is_regression)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            state_dict = torch.load(str(model_path), map_location=device, weights_only=True)
            model._modules.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()
            print(f"[OK] PyTorch model loaded from {model_path}")

        else:
            return JSONResponse(
                status_code=400,
                content={"error": f"Unknown framework: {framework}"},
            )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to load model: {str(e)}"},
        )

    # Load scaler
    scaler = None
    scaler_path = model_dir / "scaler.pkl"
    if scaler_path.exists():
        import pickle

        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            print(f"[OK] Scaler loaded from {scaler_path}")
        except Exception as e:
            print(f"[X] Failed to load scaler: {e}")

    # Populate in-memory store
    _trained_model_store["model"] = model
    _trained_model_store["feature_names"] = metadata.get("feature_names")
    _trained_model_store["class_names"] = metadata.get("class_names")
    _trained_model_store["is_regression"] = metadata.get("is_regression", False)
    _trained_model_store["is_nlp"] = metadata.get("is_nlp", False)
    _trained_model_store["input_shape"] = metadata.get("input_shape")
    _trained_model_store["scaler"] = scaler
    _trained_model_store["vocab"] = metadata.get("vocab")
    _trained_model_store["max_length"] = metadata.get("max_length")
    _trained_model_store["framework"] = framework

    return {
        "loaded": True,
        "name": name,
        "framework": framework,
        "feature_names": metadata.get("feature_names"),
        "class_names": metadata.get("class_names"),
        "is_regression": metadata.get("is_regression", False),
        "is_nlp": metadata.get("is_nlp", False),
        "input_shape": metadata.get("input_shape"),
    }


@app.delete("/api/model/delete/{name}")
async def delete_model(name: str):
    """Delete a saved model."""
    import shutil

    model_dir = SAVED_MODELS_DIR / name

    if not model_dir.exists():
        return JSONResponse(
            status_code=404,
            content={"error": f"Model '{name}' not found."},
        )

    try:
        shutil.rmtree(str(model_dir))
        print(f"[OK] Model '{name}' deleted from {model_dir}")
        return {"deleted": True, "name": name}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to delete model: {str(e)}"},
        )


# ── WebSocket — 학습 실행 ────────────────────────────
@app.websocket("/ws/train")
async def ws_train(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            if msg.get("type") == "START_TRAINING":
                pipeline = msg.get("pipeline", {})
                nodes = pipeline.get("nodes", [])
                edges = pipeline.get("edges", [])

                async def ws_callback(data: dict):
                    await ws.send_json(data)

                executor = PipelineExecutor(nodes, edges, ws_callback)
                try:
                    await executor.execute()
                    # 학습 완료 후 모델+메타데이터 저장
                    _extract_model_info(executor, nodes)
                    # 모델 메타 정보를 프론트엔드에 전송
                    if _trained_model_store["model"] is not None:
                        await ws.send_json({
                            "type": "MODEL_READY",
                            "feature_names": _trained_model_store["feature_names"],
                            "class_names": _trained_model_store["class_names"],
                            "is_regression": _trained_model_store["is_regression"],
                            "is_nlp": _trained_model_store["is_nlp"],
                            "input_shape": _trained_model_store["input_shape"],
                        })
                except Exception as e:
                    await ws.send_json(
                        {
                            "type": "ERROR",
                            "message": str(e),
                            "traceback": traceback.format_exc(),
                        }
                    )

            elif msg.get("type") == "STOP_TRAINING":
                # TODO: 학습 중단 로직
                await ws.send_json({"type": "TRAINING_STOPPED"})

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
