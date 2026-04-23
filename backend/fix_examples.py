"""예제 JSON 파일의 노드 타입 및 에지 문제 일괄 수정."""
import json
import os

# 노드 타입 매핑 (잘못된 이름 → 올바른 이름)
TYPE_MAP = {
    "CSV Loader": "CSVLoader",
    "csv_loader": "CSVLoader",
    "Numpy Input": "NumpyInput",
    "numpy_input": "NumpyInput",
    "Image Folder": "ImageFolder",
    "Train/Val Split": "TrainValSplit",
    "Data Inspector": "DataInspector",
    "Standard Scaler": "StandardScaler",
    "MinMax Scaler": "MinMaxScaler",
    "Label Encoder": "LabelEncoder",
    "OneHot Encoder": "OneHotEncoder",
    "One-Hot Encoder": "OneHotEncoder",
    "Loss Function": "LossFunction",
    "Early Stopping": "EarlyStopping",
    "Model Checkpoint": "ModelCheckpoint",
    "LR Scheduler": "LRScheduler",
    "Model Save": "ModelSave",
    "Loss Curve": "LossCurve",
    "Accuracy Curve": "AccuracyCurve",
    "Model Summary": "ModelSummary",
    "Classification Metrics": "ClassificationMetrics",
    "Confusion Matrix": "ConfusionMatrix",
    "Pretrained Model": "PretrainedModel",
    "Multi Head Attention": "MultiHeadAttention",
    "MultiHead Attention": "MultiHeadAttention",
    "Transformer Block": "TransformerBlock",
    "Batch Norm": "BatchNorm",
    "Global Average Pooling 2D": "GlobalAveragePooling2D",
    "Global Average Pooling 1D": "GlobalAveragePooling1D",
    "Max Pooling 2D": "MaxPooling2D",
    "tSNE Viewer": "tSNEViewer",
    "t-SNE Viewer": "tSNEViewer",
    "UMAP Viewer": "UMAPViewer",
    "Feature Map Viewer": "FeatureMapViewer",
    "Grad-CAM Viewer": "GradCAMViewer",
    "GradCAM Viewer": "GradCAMViewer",
    "ROC Curve": "ROCCurve",
    "Precision-Recall Curve": "PrecisionRecallCurve",
    "Prediction Viewer": "PredictionViewer",
    "Data Distribution": "DataDistribution",
    "Regression Metrics": "RegressionMetrics",
    "ONNX Export": "ONNXExport",
    "TFLite Export": "TFLiteExport",
    "Custom Code Node": "CustomCodeNode",
}

VALID_DATASETS = {"mnist", "fashion_mnist", "cifar10", "boston", "iris", "custom"}

changes_total = 0


def fix_node_type(node_type):
    return TYPE_MAP.get(node_type, node_type)


def fix_example(filepath):
    global changes_total

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    changes = []
    id_map = {}

    for node in nodes:
        old_type = node["type"]
        new_type = fix_node_type(old_type)

        if old_type != new_type:
            old_id = node["id"]
            # ID 접두사 변경
            parts = old_id.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                new_id = new_type + "_" + parts[1]
            else:
                new_id = old_id
            id_map[old_id] = new_id
            node["type"] = new_type
            node["id"] = new_id
            changes.append(f"  type: '{old_type}' -> '{new_type}' (id: {old_id} -> {new_id})")

        if node["type"] == "NumpyInput":
            ds = node.get("params", {}).get("dataset", "mnist")
            if ds not in VALID_DATASETS:
                node["params"]["dataset"] = "mnist"
                changes.append(f"  NumpyInput dataset: '{ds}' -> 'mnist'")

        if node["type"] == "CSVLoader":
            params = node.get("params", {})
            tc = params.get("target_column")
            if tc is None:
                params["target_column"] = ""
                changes.append(f"  CSVLoader target_column: null -> '' (auto-detect)")

    for edge in edges:
        if edge.get("source") in id_map:
            old = edge["source"]
            edge["source"] = id_map[old]
        if edge.get("target") in id_map:
            old = edge["target"]
            edge["target"] = id_map[old]

    meta = data.get("meta", {})
    for step in meta.get("guide_steps", []):
        if step.get("node") in id_map:
            step["node"] = id_map[step["node"]]

    if changes:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        changes_total += len(changes)
        fname = os.path.basename(filepath)
        print(f"{fname} ({len(changes)} changes):")
        for c in changes:
            print(c)

    return len(changes)


def main():
    total_files = 0
    changed_files = 0
    for root, dirs, files in os.walk("examples"):
        for fname in sorted(files):
            if not fname.endswith(".json") or fname == "_index.json":
                continue
            total_files += 1
            n = fix_example(os.path.join(root, fname))
            if n > 0:
                changed_files += 1
    print(f"\nTotal: {total_files}, Changed: {changed_files}, Changes: {changes_total}")


if __name__ == "__main__":
    main()
