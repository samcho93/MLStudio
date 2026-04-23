#!/usr/bin/env python3
"""
Validate and fix edges in all ML Node Studio example JSON files.

Checks for:
1. Disconnected nodes (no edges connecting to/from them)
2. Invalid sourceHandle/targetHandle names
3. Duplicate edges
4. Missing essential edges (e.g., data flow to Trainer)

Fixes are applied based on the actual node port definitions from the backend code.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

# ============================================================
# Valid handles per node type (from backend node definitions)
# ============================================================

# Format: { "NodeType": { "inputs": [...], "outputs": [...] } }
NODE_PORTS = {
    # ── Data nodes ──────────────────────────────────────
    "CSVLoader": {
        "inputs": [],
        "outputs": ["features", "labels", "dataframe"],
    },
    "NumpyInput": {
        "inputs": [],
        "outputs": ["train_features", "train_labels", "test_features", "test_labels"],
    },
    "ImageFolder": {
        "inputs": [],
        "outputs": ["features", "labels", "class_names"],
    },
    "TrainValSplit": {
        "inputs": ["features", "labels"],
        "outputs": ["train_features", "train_labels", "val_features", "val_labels",
                     "test_features", "test_labels"],
    },
    "DataInspector": {
        "inputs": ["input"],
        "outputs": ["output"],
    },
    "StandardScaler": {
        "inputs": ["input"],
        "outputs": ["output"],
    },
    "MinMaxScaler": {
        "inputs": ["input"],
        "outputs": ["output"],
    },
    "LabelEncoder": {
        "inputs": ["input"],
        "outputs": ["output"],
    },
    "OneHotEncoder": {
        "inputs": ["input"],
        "outputs": ["output"],
    },

    # ── Phase 2 data nodes ──────────────────────────────
    "PCA": {
        "inputs": ["input"],
        "outputs": ["output"],
    },
    "Augmentation": {
        "inputs": ["input"],
        "outputs": ["output"],
    },
    "HDF5Loader": {
        "inputs": [],
        "outputs": ["features", "labels"],
    },

    # ── Layer nodes (all use input/output) ──────────────
    "Dense": {
        "inputs": ["input"],
        "outputs": ["output"],
    },
    "Conv2D": {
        "inputs": ["input"],
        "outputs": ["output"],
    },
    "MaxPooling2D": {
        "inputs": ["input"],
        "outputs": ["output"],
    },
    "Flatten": {
        "inputs": ["input"],
        "outputs": ["output"],
    },
    "GlobalAveragePooling2D": {
        "inputs": ["input"],
        "outputs": ["output"],
    },
    "BatchNorm": {
        "inputs": ["input"],
        "outputs": ["output"],
    },
    "Dropout": {
        "inputs": ["input"],
        "outputs": ["output"],
    },
    "LSTM": {
        "inputs": ["input"],
        "outputs": ["output"],
    },
    "GRU": {
        "inputs": ["input"],
        "outputs": ["output"],
    },
    "Embedding": {
        "inputs": ["input"],
        "outputs": ["output"],
    },
    "Concat": {
        "inputs": ["input_a", "input_b"],
        "outputs": ["output"],
    },
    "Add": {
        "inputs": ["input_a", "input_b"],
        "outputs": ["output"],
    },
    "Reshape": {
        "inputs": ["input"],
        "outputs": ["output"],
    },
    "Conv1D": {
        "inputs": ["input"],
        "outputs": ["output"],
    },
    "Conv2DTranspose": {
        "inputs": ["input"],
        "outputs": ["output"],
    },
    "GlobalAveragePooling1D": {
        "inputs": ["input"],
        "outputs": ["output"],
    },
    "MultiHeadAttention": {
        "inputs": ["input"],
        "outputs": ["output"],
    },
    "TransformerBlock": {
        "inputs": ["input"],
        "outputs": ["output"],
    },
    "PretrainedModel": {
        "inputs": ["input"],
        "outputs": ["output", "model"],
    },
    "Bidirectional": {
        "inputs": ["input"],
        "outputs": ["output"],
    },

    # ── Training nodes ──────────────────────────────────
    "Optimizer": {
        "inputs": [],
        "outputs": ["optimizer_config"],
    },
    "LossFunction": {
        "inputs": [],
        "outputs": ["loss_config"],
    },
    "EarlyStopping": {
        "inputs": [],
        "outputs": ["callback_config"],
    },
    "ModelCheckpoint": {
        "inputs": [],
        "outputs": ["callback_config"],
    },
    "LRScheduler": {
        "inputs": [],
        "outputs": ["callback_config"],
    },
    "GradientClipping": {
        "inputs": [],
        "outputs": ["callback_config"],
    },
    "MixedPrecision": {
        "inputs": [],
        "outputs": ["callback_config"],
    },
    "CustomCallback": {
        "inputs": [],
        "outputs": ["callback_config"],
    },
    "Trainer": {
        "inputs": ["layers", "train_features", "train_labels",
                    "val_features", "val_labels",
                    "optimizer_config", "loss_config", "callback_config"],
        "outputs": ["model", "history", "model_path"],
    },

    # ── Output nodes ────────────────────────────────────
    "ModelSave": {
        "inputs": ["model"],
        "outputs": ["model_path"],
    },
    "ONNXExport": {
        "inputs": ["model"],
        "outputs": ["onnx_path"],
    },
    "TFLiteExport": {
        "inputs": ["model"],
        "outputs": ["tflite_path"],
    },
    "Quantization": {
        "inputs": ["model"],
        "outputs": ["model"],
    },

    # ── Visualization nodes ─────────────────────────────
    "LossCurve": {
        "inputs": ["history"],
        "outputs": ["image_b64"],
    },
    "AccuracyCurve": {
        "inputs": ["history"],
        "outputs": ["image_b64"],
    },
    "ModelSummary": {
        "inputs": ["model"],
        "outputs": ["summary"],
    },
    "FeatureMapViewer": {
        "inputs": ["model", "images"],
        "outputs": ["image_b64"],
    },
    "GradCAMViewer": {
        "inputs": ["model", "images"],
        "outputs": ["image_b64"],
    },
    "tSNEViewer": {
        "inputs": ["input"],
        "outputs": ["image_b64"],
    },
    "UMAPViewer": {
        "inputs": ["input"],
        "outputs": ["image_b64"],
    },
    "DataDistribution": {
        "inputs": ["input"],
        "outputs": ["image_b64"],
    },

    # ── Evaluation nodes ────────────────────────────────
    "ClassificationMetrics": {
        "inputs": ["model", "test_features", "test_labels"],
        "outputs": ["metrics"],
    },
    "ConfusionMatrix": {
        "inputs": ["model", "test_features", "test_labels"],
        "outputs": ["image_b64"],
    },
    "RegressionMetrics": {
        "inputs": ["model", "test_features", "test_labels"],
        "outputs": ["metrics"],
    },
    "ROCCurve": {
        "inputs": ["model", "test_features", "test_labels"],
        "outputs": ["image_b64"],
    },
    "PrecisionRecallCurve": {
        "inputs": ["model", "test_features", "test_labels"],
        "outputs": ["image_b64"],
    },
    "PredictionViewer": {
        "inputs": ["model", "test_features", "test_labels"],
        "outputs": ["predictions"],
    },

    # ── Special advanced nodes ──────────────────────────
    "CustomCodeNode": {
        "inputs": ["input"],
        "outputs": ["output"],
    },
    "KFoldTrainer": {
        "inputs": ["layers", "features", "labels",
                    "optimizer_config", "loss_config"],
        "outputs": ["model", "history", "fold_metrics"],
    },
}

# Auxiliary types that feed into Trainer but are not data flow
AUXILIARY_TYPES = {
    "Optimizer", "LossFunction", "EarlyStopping", "ModelCheckpoint",
    "LRScheduler", "GradientClipping", "MixedPrecision", "CustomCallback",
}

# Layer types that chain via input/output
LAYER_TYPES = {
    "Dense", "Conv2D", "MaxPooling2D", "Flatten", "GlobalAveragePooling2D",
    "BatchNorm", "Dropout", "LSTM", "GRU", "Embedding", "Reshape",
    "Conv1D", "Conv2DTranspose", "GlobalAveragePooling1D",
    "MultiHeadAttention", "TransformerBlock", "Bidirectional",
    "Concat", "Add",
}

# Scaler/preprocessor types
SCALER_TYPES = {
    "StandardScaler", "MinMaxScaler", "LabelEncoder", "OneHotEncoder",
    "PCA", "Augmentation", "DataInspector",
}

# Post-trainer output/viz types
POST_TRAINER_TYPES = {
    "LossCurve", "AccuracyCurve", "ModelSummary", "ConfusionMatrix",
    "ClassificationMetrics", "RegressionMetrics", "ROCCurve",
    "PrecisionRecallCurve", "PredictionViewer", "FeatureMapViewer",
    "GradCAMViewer", "tSNEViewer", "UMAPViewer", "DataDistribution",
    "ModelSave", "ONNXExport", "TFLiteExport", "Quantization",
}

# Data source types
DATA_SOURCE_TYPES = {
    "CSVLoader", "NumpyInput", "ImageFolder", "HDF5Loader",
}


# ============================================================
# Handle mapping corrections
# ============================================================

def get_valid_outputs(node_type):
    """Get valid output handle names for a node type."""
    ports = NODE_PORTS.get(node_type, {})
    return ports.get("outputs", [])


def get_valid_inputs(node_type):
    """Get valid input handle names for a node type."""
    ports = NODE_PORTS.get(node_type, {})
    return ports.get("inputs", [])


def fix_source_handle(source_type, current_handle, target_type, target_handle):
    """Fix an invalid sourceHandle to the correct one."""
    valid = get_valid_outputs(source_type)
    if not valid:
        return current_handle

    # If already valid, keep it
    if current_handle in valid:
        return current_handle

    # Common mismatches and their corrections
    corrections = {
        # NumpyInput mismatches - context-dependent
        ("NumpyInput", "dataset"): None,  # needs context
        ("NumpyInput", "data"): "train_features",
        ("NumpyInput", "features"): "train_features",
        ("NumpyInput", "labels"): "train_labels",

        # TrainValSplit mismatches
        ("TrainValSplit", "train"): None,  # needs context
        ("TrainValSplit", "val"): None,  # needs context
        ("TrainValSplit", "test"): None,  # needs context
        ("TrainValSplit", "train_data"): None,
        ("TrainValSplit", "val_data"): None,
        ("TrainValSplit", "test_data"): None,

        # Trainer mismatches
        ("Trainer", "predictions"): "model",
        ("Trainer", "predictions_a"): "model",
        ("Trainer", "predictions_b"): "model",
        ("Trainer", "predictions_c"): "model",
        ("Trainer", "result"): "model",
        ("Trainer", "trained_model"): "model",
        ("Trainer", "output"): "model",
        ("Trainer", "teacher_model"): "model",
        ("Trainer", "fold_results"): "model",
        ("Trainer", "fold_histories"): "history",
        ("Trainer", "saved_path"): "model_path",
        ("Trainer", "backbone"): "model",

        # KFoldTrainer mismatches
        ("KFoldTrainer", "predictions"): "model",
        ("KFoldTrainer", "fold_results"): "model",
        ("KFoldTrainer", "fold_histories"): "history",
        ("KFoldTrainer", "result"): "model",

        # PretrainedModel mismatches
        ("PretrainedModel", "features"): "output",
        ("PretrainedModel", "feature"): "output",
        ("PretrainedModel", "backbone"): "output",

        # EarlyStopping/ModelCheckpoint/LRScheduler
        ("EarlyStopping", "config"): "callback_config",
        ("ModelCheckpoint", "config"): "callback_config",
        ("LRScheduler", "config"): "callback_config",
        ("LRScheduler", "scheduler_config"): "callback_config",
    }

    key = (source_type, current_handle)
    if key in corrections:
        corrected = corrections[key]
        if corrected is not None:
            return corrected

        # Context-dependent TrainValSplit fixes
        if source_type == "TrainValSplit":
            return _fix_train_val_split_source(current_handle, target_type, target_handle)

        # Context-dependent NumpyInput fixes
        if source_type == "NumpyInput":
            if target_type == "TrainValSplit":
                if target_handle == "labels":
                    return "train_labels"
                return "train_features"
            if target_type in LAYER_TYPES or target_type in SCALER_TYPES:
                return "train_features"
            if target_handle == "train_features":
                return "train_features"
            if target_handle == "train_labels":
                return "train_labels"
            if target_handle == "val_features":
                return "test_features"
            if target_handle == "val_labels":
                return "test_labels"
            return "train_features"

    # For layer/scaler nodes, output is always "output"
    if source_type in LAYER_TYPES or source_type in SCALER_TYPES:
        if "output" in valid:
            return "output"

    # Fallback: return first valid output
    return valid[0] if valid else current_handle


def _fix_train_val_split_source(current_handle, target_type, target_handle):
    """Fix TrainValSplit source handle based on context."""
    # Map target handle to appropriate source handle
    handle_map = {
        "train_features": "train_features",
        "train_labels": "train_labels",
        "val_features": "val_features",
        "val_labels": "val_labels",
        "test_features": "test_features",
        "test_labels": "test_labels",
    }

    if target_handle in handle_map:
        return handle_map[target_handle]

    # Generic "train" -> train_features (for layer input)
    if current_handle in ("train", "train_data"):
        if target_type in LAYER_TYPES:
            return "train_features"
        if target_handle == "train_data":
            return "train_features"
        return "train_features"

    if current_handle in ("val", "val_data"):
        return "val_features"

    if current_handle in ("test", "test_data"):
        return "test_features"

    return "train_features"


def fix_target_handle(target_type, current_handle, source_type, source_handle):
    """Fix an invalid targetHandle to the correct one."""
    valid = get_valid_inputs(target_type)
    if not valid:
        return current_handle

    # If already valid, keep it
    if current_handle in valid:
        return current_handle

    # Common mismatches
    corrections = {
        # Trainer mismatches
        ("Trainer", "train_data"): None,  # needs context
        ("Trainer", "val_data"): None,
        ("Trainer", "test_data"): None,
        ("Trainer", "callbacks"): "callback_config",
        ("Trainer", "early_stopping"): "callback_config",
        ("Trainer", "checkpoint_callback"): "callback_config",
        ("Trainer", "lr_scheduler"): "callback_config",
        ("Trainer", "scheduler_config"): "callback_config",
        ("Trainer", "data"): "train_features",
        ("Trainer", "input"): None,  # needs context
        ("Trainer", "model"): None,  # needs context
        ("Trainer", "backbone"): None,  # needs context
        ("Trainer", "teacher_model"): None,  # needs context
        ("Trainer", "resume_model"): None,  # needs context
        ("Trainer", "dataset"): None,  # needs context

        # KFoldTrainer mismatches
        ("KFoldTrainer", "train_data"): "features",
        ("KFoldTrainer", "data"): "features",
        ("KFoldTrainer", "input"): "features",
        ("KFoldTrainer", "callbacks"): "optimizer_config",

        # TrainValSplit mismatches
        ("TrainValSplit", "input"): "features",
        ("TrainValSplit", "data"): "features",
        ("TrainValSplit", "dataset"): "features",

        # Evaluation/viz node mismatches
        ("ClassificationMetrics", "predictions"): "model",
        ("ClassificationMetrics", "predictions_a"): "model",
        ("ClassificationMetrics", "predictions_b"): "model",
        ("ClassificationMetrics", "predictions_c"): "model",
        ("ClassificationMetrics", "history"): "model",
        ("ConfusionMatrix", "predictions"): "model",
        ("ConfusionMatrix", "predictions_a"): "model",
        ("ConfusionMatrix", "predictions_b"): "model",
        ("ConfusionMatrix", "predictions_c"): "model",
        ("ConfusionMatrix", "history"): "model",
        ("RegressionMetrics", "predictions"): "model",
        ("ROCCurve", "predictions"): "model",
        ("PrecisionRecallCurve", "predictions"): "model",
        ("PredictionViewer", "predictions"): "model",
        ("ModelSummary", "history"): "model",

        # tSNE/UMAP viewer
        ("tSNEViewer", "features"): "input",
        ("tSNEViewer", "data"): "input",
        ("tSNEViewer", "embeddings"): "input",
        ("tSNEViewer", "labels"): "input",
        ("UMAPViewer", "features"): "input",
        ("UMAPViewer", "data"): "input",
        ("UMAPViewer", "embeddings"): "input",

        # ModelSave mismatches
        ("ModelSave", "model_path"): "model",
        ("ModelSave", "trained_model"): "model",

        # ONNXExport mismatches
        ("ONNXExport", "model_path"): "model",
        ("ONNXExport", "trained_model"): "model",
        ("ONNXExport", "onnx_model"): "model",
    }

    key = (target_type, current_handle)
    if key in corrections:
        corrected = corrections[key]
        if corrected is not None:
            return corrected

        # Context-dependent fixes
        if target_type in ("Trainer", "KFoldTrainer"):
            return _fix_trainer_target(current_handle, source_type, source_handle)

    # For layer/scaler nodes, input is always "input"
    if target_type in LAYER_TYPES or target_type in SCALER_TYPES:
        if "input" in valid:
            return "input"

    # For Trainer, use source context to determine handle
    if target_type in ("Trainer", "KFoldTrainer"):
        return _fix_trainer_target(current_handle, source_type, source_handle)

    # Fallback: first valid input
    return valid[0] if valid else current_handle


def _fix_trainer_target(current_handle, source_type, source_handle):
    """Fix Trainer target handle based on source context."""
    # If source is a layer node or PretrainedModel, target should be "layers"
    if source_type in LAYER_TYPES or source_type == "PretrainedModel":
        return "layers"

    # If source is an auxiliary type (Optimizer, Loss, EarlyStopping, etc.)
    if source_type in AUXILIARY_TYPES:
        ports = NODE_PORTS.get(source_type, {})
        output = ports.get("outputs", ["callback_config"])[0]
        # output will be "optimizer_config", "loss_config", or "callback_config"
        return output

    # If source is TrainValSplit, map based on source handle
    if source_type == "TrainValSplit":
        tvs_map = {
            "train_features": "train_features",
            "train_labels": "train_labels",
            "val_features": "val_features",
            "val_labels": "val_labels",
            "test_features": "train_features",  # fallback
        }
        return tvs_map.get(source_handle, "train_features")

    # NumpyInput outputs map directly
    if source_type == "NumpyInput":
        np_map = {
            "train_features": "train_features",
            "train_labels": "train_labels",
            "test_features": "val_features",
            "test_labels": "val_labels",
        }
        return np_map.get(source_handle, "train_features")

    # If source is a Trainer (e.g., teacher model feeding student trainer)
    if source_type == "Trainer":
        if source_handle in ("model", "trained_model"):
            return "layers"  # model being used as input
        if source_handle == "history":
            return "layers"  # unlikely but fallback
        return "layers"

    # Generic handle-name-based fixes
    if current_handle in ("train_data", "dataset"):
        return "train_features"
    if current_handle == "val_data":
        return "val_features"
    if current_handle == "test_data":
        return "train_features"
    if current_handle in ("model", "backbone", "teacher_model", "resume_model"):
        return "layers"
    if current_handle in ("input",):
        return "layers"

    return "train_features"


# ============================================================
# Pipeline analysis for adding missing edges
# ============================================================

def find_trainer_nodes(nodes):
    """Find all Trainer/KFoldTrainer nodes."""
    return [n for n in nodes if n["type"] in ("Trainer", "KFoldTrainer")]


def find_data_source_nodes(nodes):
    """Find data source nodes (CSVLoader, NumpyInput, ImageFolder)."""
    return [n for n in nodes if n["type"] in DATA_SOURCE_TYPES]


def find_layer_chain_end(node_id, edges, node_map):
    """Find the last layer in a chain starting from node_id."""
    current = node_id
    visited = set()
    while current not in visited:
        visited.add(current)
        found_next = False
        for e in edges:
            if e["source"] == current:
                tgt = e["target"]
                if tgt in node_map and node_map[tgt]["type"] in LAYER_TYPES:
                    current = tgt
                    found_next = True
                    break
        if not found_next:
            break
    return current


def get_connected_nodes(edges):
    """Get set of all node IDs that appear in at least one edge."""
    connected = set()
    for e in edges:
        connected.add(e["source"])
        connected.add(e["target"])
    return connected


def edge_key(e):
    """Unique key for deduplication."""
    return (e["source"], e["target"], e.get("sourceHandle", ""), e.get("targetHandle", ""))


# ============================================================
# Main validation and fixing logic
# ============================================================

class EdgeFixer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.issues = []
        self.fixes = []

        with open(filepath, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.nodes = self.data.get("nodes", [])
        self.edges = self.data.get("edges", [])
        self.node_map = {n["id"]: n for n in self.nodes}

    def validate_and_fix(self):
        """Run all validations and fixes. Returns True if any changes were made."""
        changed = False

        # 1. Remove duplicate edges
        changed |= self._remove_duplicates()

        # 2. Fix invalid handles
        changed |= self._fix_invalid_handles()

        # 3. Fix disconnected nodes
        changed |= self._fix_disconnected_nodes()

        # 4. Remove duplicate edges again (fixing may have created duplicates)
        changed |= self._remove_duplicates()

        # Write back
        if changed:
            self.data["edges"] = self.edges
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
                f.write("\n")

        return changed

    def _remove_duplicates(self):
        """Remove duplicate edges."""
        seen = set()
        new_edges = []
        removed = 0
        for e in self.edges:
            k = edge_key(e)
            if k not in seen:
                seen.add(k)
                new_edges.append(e)
            else:
                removed += 1
                self.issues.append(f"  DUPLICATE edge: {e['source']}:{e.get('sourceHandle','')} -> {e['target']}:{e.get('targetHandle','')}")
                self.fixes.append(f"  Removed duplicate edge")

        if removed > 0:
            self.edges = new_edges
            return True
        return False

    def _fix_invalid_handles(self):
        """Fix sourceHandle and targetHandle values that don't match node ports."""
        changed = False
        for e in self.edges:
            src_id = e["source"]
            tgt_id = e["target"]

            src_node = self.node_map.get(src_id)
            tgt_node = self.node_map.get(tgt_id)

            if not src_node or not tgt_node:
                continue

            src_type = src_node["type"]
            tgt_type = tgt_node["type"]

            # Skip unknown node types
            if src_type not in NODE_PORTS or tgt_type not in NODE_PORTS:
                continue

            # Check sourceHandle
            old_sh = e.get("sourceHandle", "")
            valid_outputs = get_valid_outputs(src_type)
            if valid_outputs and old_sh not in valid_outputs:
                new_sh = fix_source_handle(src_type, old_sh, tgt_type, e.get("targetHandle", ""))
                if new_sh != old_sh:
                    self.issues.append(
                        f"  INVALID sourceHandle: {src_id}({src_type}).{old_sh} "
                        f"(valid: {valid_outputs})"
                    )
                    self.fixes.append(f"  Fixed: {src_id}.{old_sh} -> {src_id}.{new_sh}")
                    e["sourceHandle"] = new_sh
                    changed = True

            # Check targetHandle
            old_th = e.get("targetHandle", "")
            valid_inputs = get_valid_inputs(tgt_type)
            if valid_inputs and old_th not in valid_inputs:
                # Re-read sourceHandle in case it was just fixed
                new_th = fix_target_handle(tgt_type, old_th, src_type, e.get("sourceHandle", ""))
                if new_th != old_th:
                    self.issues.append(
                        f"  INVALID targetHandle: {tgt_id}({tgt_type}).{old_th} "
                        f"(valid: {valid_inputs})"
                    )
                    self.fixes.append(f"  Fixed: {tgt_id}.{old_th} -> {tgt_id}.{new_th}")
                    e["targetHandle"] = new_th
                    changed = True

        return changed

    def _fix_disconnected_nodes(self):
        """Find and connect disconnected nodes."""
        connected = get_connected_nodes(self.edges)
        all_ids = set(self.node_map.keys())
        disconnected = all_ids - connected

        if not disconnected:
            return False

        changed = False
        for node_id in sorted(disconnected):
            node = self.node_map[node_id]
            node_type = node["type"]

            self.issues.append(f"  DISCONNECTED node: {node_id} ({node_type})")

            new_edges = self._connect_node(node_id, node_type)
            if new_edges:
                for ne in new_edges:
                    self.fixes.append(
                        f"  Added edge: {ne['source']}:{ne['sourceHandle']} -> "
                        f"{ne['target']}:{ne['targetHandle']}"
                    )
                self.edges.extend(new_edges)
                changed = True
            else:
                self.fixes.append(f"  WARNING: Could not auto-connect {node_id}")

        return changed

    def _connect_node(self, node_id, node_type):
        """Try to connect a disconnected node based on its type and the pipeline context."""
        edges_to_add = []

        if node_type in DATA_SOURCE_TYPES:
            # Data source should feed into a scaler, TrainValSplit, or layers
            edges_to_add = self._connect_data_source(node_id, node_type)

        elif node_type in SCALER_TYPES:
            # Scaler should be between data source and TrainValSplit or layers
            edges_to_add = self._connect_scaler(node_id, node_type)

        elif node_type == "TrainValSplit":
            edges_to_add = self._connect_train_val_split(node_id)

        elif node_type in LAYER_TYPES:
            edges_to_add = self._connect_layer(node_id, node_type)

        elif node_type in AUXILIARY_TYPES:
            edges_to_add = self._connect_auxiliary(node_id, node_type)

        elif node_type in ("Trainer", "KFoldTrainer"):
            edges_to_add = self._connect_trainer(node_id, node_type)

        elif node_type in POST_TRAINER_TYPES:
            edges_to_add = self._connect_post_trainer(node_id, node_type)

        return edges_to_add

    def _connect_data_source(self, node_id, node_type):
        """Connect a data source to the nearest compatible downstream node."""
        edges = []
        # Find TrainValSplit or scaler nodes that have no data input
        for n in self.nodes:
            if n["id"] == node_id:
                continue
            if n["type"] == "TrainValSplit":
                # Check if features input is already connected
                has_features = any(
                    e["target"] == n["id"] and e.get("targetHandle") == "features"
                    for e in self.edges
                )
                if not has_features:
                    edges.append({
                        "source": node_id, "target": n["id"],
                        "sourceHandle": "features", "targetHandle": "features"
                    })
                has_labels = any(
                    e["target"] == n["id"] and e.get("targetHandle") == "labels"
                    for e in self.edges
                )
                if not has_labels and node_type in ("CSVLoader", "ImageFolder"):
                    edges.append({
                        "source": node_id, "target": n["id"],
                        "sourceHandle": "labels", "targetHandle": "labels"
                    })
                if edges:
                    return edges

            if n["type"] in SCALER_TYPES:
                has_input = any(
                    e["target"] == n["id"] and e.get("targetHandle") == "input"
                    for e in self.edges
                )
                if not has_input:
                    edges.append({
                        "source": node_id, "target": n["id"],
                        "sourceHandle": "features", "targetHandle": "input"
                    })
                    return edges

        return edges

    def _connect_scaler(self, node_id, node_type):
        """Connect a scaler between data and TrainValSplit."""
        edges = []
        # Find upstream data source
        for n in self.nodes:
            if n["type"] in DATA_SOURCE_TYPES:
                edges.append({
                    "source": n["id"], "target": node_id,
                    "sourceHandle": "features", "targetHandle": "input"
                })
                break

        # Find downstream TrainValSplit or layer
        for n in self.nodes:
            if n["type"] == "TrainValSplit":
                edges.append({
                    "source": node_id, "target": n["id"],
                    "sourceHandle": "output", "targetHandle": "features"
                })
                break
            if n["type"] in LAYER_TYPES:
                has_input = any(
                    e["target"] == n["id"] for e in self.edges
                )
                if not has_input:
                    edges.append({
                        "source": node_id, "target": n["id"],
                        "sourceHandle": "output", "targetHandle": "input"
                    })
                    break

        return edges

    def _connect_train_val_split(self, node_id):
        """Connect TrainValSplit to data sources and Trainer."""
        edges = []
        # Find upstream data
        for n in self.nodes:
            if n["type"] in DATA_SOURCE_TYPES or n["type"] in SCALER_TYPES:
                if n["type"] in SCALER_TYPES:
                    src_handle = "output"
                else:
                    src_handle = "features"
                edges.append({
                    "source": n["id"], "target": node_id,
                    "sourceHandle": src_handle, "targetHandle": "features"
                })
                break

        # Connect to Trainer
        for n in self.nodes:
            if n["type"] == "Trainer":
                for handle in ["train_features", "train_labels", "val_features", "val_labels"]:
                    has_it = any(
                        e["target"] == n["id"] and e.get("targetHandle") == handle
                        for e in self.edges
                    )
                    if not has_it:
                        edges.append({
                            "source": node_id, "target": n["id"],
                            "sourceHandle": handle, "targetHandle": handle
                        })
                break

        return edges

    def _connect_layer(self, node_id, node_type):
        """Connect a layer node into the layer chain."""
        edges = []
        node_pos = self.node_map[node_id].get("position", {"x": 0, "y": 0})
        node_x = node_pos.get("x", 0)

        # Find the nearest layer node by x position to the left
        best_prev = None
        best_dist = float("inf")
        for n in self.nodes:
            if n["id"] == node_id:
                continue
            if n["type"] in LAYER_TYPES:
                nx = n.get("position", {}).get("x", 0)
                if nx < node_x:
                    dist = node_x - nx
                    if dist < best_dist:
                        best_dist = dist
                        best_prev = n

        if best_prev:
            edges.append({
                "source": best_prev["id"], "target": node_id,
                "sourceHandle": "output", "targetHandle": "input"
            })

        # Find the nearest layer node to the right, or Trainer
        best_next = None
        best_dist = float("inf")
        for n in self.nodes:
            if n["id"] == node_id:
                continue
            if n["type"] in LAYER_TYPES or n["type"] == "Trainer":
                nx = n.get("position", {}).get("x", 0)
                if nx > node_x:
                    dist = nx - node_x
                    if dist < best_dist:
                        best_dist = dist
                        best_next = n

        if best_next:
            if best_next["type"] == "Trainer":
                edges.append({
                    "source": node_id, "target": best_next["id"],
                    "sourceHandle": "output", "targetHandle": "layers"
                })
            else:
                # Only connect if the next node doesn't already have input
                has_input = any(
                    e["target"] == best_next["id"] and e.get("targetHandle") == "input"
                    for e in self.edges
                )
                if not has_input:
                    edges.append({
                        "source": node_id, "target": best_next["id"],
                        "sourceHandle": "output", "targetHandle": "input"
                    })

        return edges

    def _connect_auxiliary(self, node_id, node_type):
        """Connect auxiliary node (Optimizer, Loss, etc.) to the nearest Trainer."""
        edges = []
        ports = NODE_PORTS.get(node_type, {})
        output_handle = ports.get("outputs", ["callback_config"])[0]

        # Find the nearest Trainer
        node_pos = self.node_map[node_id].get("position", {"x": 0, "y": 0})
        best_trainer = None
        best_dist = float("inf")

        for n in self.nodes:
            if n["type"] in ("Trainer", "KFoldTrainer"):
                nx = n.get("position", {}).get("x", 0)
                ny = n.get("position", {}).get("y", 0)
                dist = abs(nx - node_pos.get("x", 0)) + abs(ny - node_pos.get("y", 0))
                if dist < best_dist:
                    best_dist = dist
                    best_trainer = n

        if best_trainer:
            # Determine target handle
            target_handle = output_handle  # optimizer_config, loss_config, callback_config
            edges.append({
                "source": node_id, "target": best_trainer["id"],
                "sourceHandle": output_handle, "targetHandle": target_handle
            })

        return edges

    def _connect_trainer(self, node_id, node_type):
        """Connect a Trainer to missing inputs."""
        edges = []
        # Check which inputs are missing
        existing_target_handles = {
            e.get("targetHandle") for e in self.edges if e["target"] == node_id
        }

        # Find layer chain endpoint
        if "layers" not in existing_target_handles:
            for n in self.nodes:
                if n["type"] in LAYER_TYPES:
                    # Check if this layer outputs to a Trainer
                    is_end = not any(
                        e["source"] == n["id"] and
                        self.node_map.get(e["target"], {}).get("type") in LAYER_TYPES
                        for e in self.edges
                    )
                    if is_end:
                        edges.append({
                            "source": n["id"], "target": node_id,
                            "sourceHandle": "output", "targetHandle": "layers"
                        })
                        break

        return edges

    def _connect_post_trainer(self, node_id, node_type):
        """Connect a post-trainer node to the nearest Trainer."""
        edges = []
        ports = NODE_PORTS.get(node_type, {})
        input_handles = ports.get("inputs", [])

        if not input_handles:
            return edges

        node_pos = self.node_map[node_id].get("position", {"x": 0, "y": 0})

        # Find nearest Trainer
        best_trainer = None
        best_dist = float("inf")
        for n in self.nodes:
            if n["type"] in ("Trainer", "KFoldTrainer"):
                nx = n.get("position", {}).get("x", 0)
                ny = n.get("position", {}).get("y", 0)
                dist = abs(nx - node_pos.get("x", 0)) + abs(ny - node_pos.get("y", 0))
                if dist < best_dist:
                    best_dist = dist
                    best_trainer = n

        if best_trainer:
            # Viz nodes that take "history"
            if node_type in ("LossCurve", "AccuracyCurve"):
                edges.append({
                    "source": best_trainer["id"], "target": node_id,
                    "sourceHandle": "history", "targetHandle": "history"
                })
            # Nodes that take "model"
            elif node_type in ("ModelSave", "ONNXExport", "TFLiteExport",
                               "Quantization", "ModelSummary"):
                edges.append({
                    "source": best_trainer["id"], "target": node_id,
                    "sourceHandle": "model", "targetHandle": "model"
                })
            # Evaluation nodes that take model + test data
            elif node_type in ("ClassificationMetrics", "ConfusionMatrix",
                               "RegressionMetrics", "ROCCurve",
                               "PrecisionRecallCurve", "PredictionViewer"):
                edges.append({
                    "source": best_trainer["id"], "target": node_id,
                    "sourceHandle": "model", "targetHandle": "model"
                })
                # Also need test_features/test_labels from TrainValSplit or NumpyInput
                self._connect_test_data_to_eval(node_id, node_type, edges)
            # Feature/Grad-CAM viewers
            elif node_type in ("FeatureMapViewer", "GradCAMViewer"):
                edges.append({
                    "source": best_trainer["id"], "target": node_id,
                    "sourceHandle": "model", "targetHandle": "model"
                })
            # tSNE/UMAP viewers take input data
            elif node_type in ("tSNEViewer", "UMAPViewer", "DataDistribution"):
                # These usually connect to a data node, not trainer
                for n in self.nodes:
                    if n["type"] in SCALER_TYPES or n["type"] == "PCA":
                        edges.append({
                            "source": n["id"], "target": node_id,
                            "sourceHandle": "output", "targetHandle": "input"
                        })
                        return edges
                # Fallback: connect to data source
                for n in self.nodes:
                    if n["type"] in DATA_SOURCE_TYPES:
                        edges.append({
                            "source": n["id"], "target": node_id,
                            "sourceHandle": "features", "targetHandle": "input"
                        })
                        return edges

        return edges

    def _connect_test_data_to_eval(self, node_id, node_type, edges):
        """Connect test data to evaluation nodes."""
        # Find TrainValSplit or NumpyInput
        for n in self.nodes:
            if n["type"] == "TrainValSplit":
                edges.append({
                    "source": n["id"], "target": node_id,
                    "sourceHandle": "test_features", "targetHandle": "test_features"
                })
                edges.append({
                    "source": n["id"], "target": node_id,
                    "sourceHandle": "test_labels", "targetHandle": "test_labels"
                })
                return
            if n["type"] == "NumpyInput":
                edges.append({
                    "source": n["id"], "target": node_id,
                    "sourceHandle": "test_features", "targetHandle": "test_features"
                })
                edges.append({
                    "source": n["id"], "target": node_id,
                    "sourceHandle": "test_labels", "targetHandle": "test_labels"
                })
                return


# ============================================================
# Main
# ============================================================

def process_file(filepath):
    """Process a single JSON file."""
    fixer = EdgeFixer(filepath)
    changed = fixer.validate_and_fix()
    return changed, fixer.issues, fixer.fixes


def main():
    examples_dir = Path(__file__).parent

    subdirs = [
        "classification", "regression", "unsupervised",
        "image", "nlp", "timeseries", "advanced"
    ]

    total_files = 0
    modified_files = 0
    total_issues = 0
    total_fixes = 0
    all_reports = []
    errors = []

    for subdir in subdirs:
        dir_path = examples_dir / subdir
        if not dir_path.exists():
            print(f"  [SKIP] Directory not found: {subdir}/")
            continue

        json_files = sorted(dir_path.glob("*.json"))
        for filepath in json_files:
            total_files += 1
            try:
                changed, issues, fixes = process_file(str(filepath))
                if issues:
                    total_issues += len(issues)
                    total_fixes += len(fixes)
                    all_reports.append((f"{subdir}/{filepath.name}", issues, fixes))
                if changed:
                    modified_files += 1
                    print(f"  [FIXED] {subdir}/{filepath.name} ({len(issues)} issues)")
                else:
                    if issues:
                        print(f"  [WARN]  {subdir}/{filepath.name} ({len(issues)} issues, no changes needed)")
                    else:
                        print(f"  [OK]    {subdir}/{filepath.name}")
            except Exception as e:
                errors.append((f"{subdir}/{filepath.name}", str(e)))
                print(f"  [ERROR] {subdir}/{filepath.name}: {e}")

    # Print detailed report
    print("\n" + "=" * 80)
    print("DETAILED REPORT")
    print("=" * 80)

    for fname, issues, fixes in all_reports:
        print(f"\n--- {fname} ---")
        print("Issues found:")
        for issue in issues:
            print(f"  {issue}")
        print("Fixes applied:")
        for fix in fixes:
            print(f"  {fix}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Total files scanned:  {total_files}")
    print(f"  Files modified:       {modified_files}")
    print(f"  Files with no issues: {total_files - len(all_reports) - len(errors)}")
    print(f"  Total issues found:   {total_issues}")
    print(f"  Total fixes applied:  {total_fixes}")
    print(f"  Errors:               {len(errors)}")

    if errors:
        print("\nErrors:")
        for fname, err in errors:
            print(f"  {fname}: {err}")

    print("=" * 80)

    # Now run the position fixer
    print("\n\nRunning position fixer (fix_positions.py)...")
    import importlib.util
    pos_fixer_path = examples_dir / "fix_positions.py"
    if pos_fixer_path.exists():
        spec = importlib.util.spec_from_file_location("fix_positions", str(pos_fixer_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()
    else:
        print("  fix_positions.py not found, skipping position fix.")


if __name__ == "__main__":
    main()
