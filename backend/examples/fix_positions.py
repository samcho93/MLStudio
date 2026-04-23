#!/usr/bin/env python3
"""
Fix node positions in all 100 ML Node Studio example JSON files.

Uses topological sorting to assign layer-based positions with proper spacing.
Handles parallel branches, auxiliary nodes (Optimizer, LossFunction, EarlyStopping),
and visualization/metrics nodes.

Spacing rules:
  - X (horizontal between layers): 300px
  - Y (vertical between nodes in same layer): 200px
  - Starting position: x=50, y=100
  - Auxiliary nodes feeding into Trainer: positioned below the Trainer in the
    same column as the Trainer
  - Post-trainer nodes: positioned one column after Trainer, spread vertically
  - Parallel branches: 250px gap between branch baselines
"""

import json
from collections import defaultdict, deque
from pathlib import Path

# Node types considered "auxiliary" - they feed into Trainer but are not
# part of the main data flow.
AUXILIARY_TYPES = {
    "Optimizer", "LossFunction", "EarlyStopping", "ModelCheckpoint",
    "LRScheduler", "GradientClipping", "MixedPrecision", "CustomCallback",
}

# Node types that are post-training outputs (visualization, metrics, export)
POST_TRAINER_TYPES = {
    "LossCurve", "AccuracyCurve", "ModelSummary", "ConfusionMatrix",
    "ClassificationMetrics", "RegressionMetrics", "ROCCurve",
    "PrecisionRecallCurve", "PredictionViewer", "FeatureMapViewer",
    "GradCAMViewer", "tSNEViewer", "UMAPViewer", "DataDistribution",
    "ModelSave", "ONNXExport", "TFLiteExport", "Quantization",
}

X_SPACING = 300
Y_SPACING = 200
START_X = 50
START_Y = 100
BRANCH_GAP = 250


def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def compute_positions(nodes, edges):
    """Compute new positions for all nodes based on topological analysis."""
    node_map = {n["id"]: n for n in nodes}
    node_ids = set(node_map.keys())

    # Classify nodes
    auxiliary_ids = {n["id"] for n in nodes if n["type"] in AUXILIARY_TYPES}
    post_trainer_ids = {n["id"] for n in nodes if n["type"] in POST_TRAINER_TYPES}
    main_ids = node_ids - auxiliary_ids - post_trainer_ids

    # Build adjacency using unique (src, tgt) pairs (ignore multiple handles)
    unique_edges = set()
    for edge in edges:
        src, tgt = edge["source"], edge["target"]
        if src in node_ids and tgt in node_ids:
            unique_edges.add((src, tgt))

    children_all = defaultdict(set)
    parents_all = defaultdict(set)
    for src, tgt in unique_edges:
        children_all[src].add(tgt)
        parents_all[tgt].add(src)

    # Main-flow adjacency only
    main_children = defaultdict(set)
    main_parents = defaultdict(set)
    for src, tgt in unique_edges:
        if src in main_ids and tgt in main_ids:
            main_children[src].add(tgt)
            main_parents[tgt].add(src)

    # --- Compute layers via longest-path on main-flow graph ---
    main_in_degree = defaultdict(int)
    for src, tgt in unique_edges:
        if src in main_ids and tgt in main_ids:
            main_in_degree[tgt] += 1

    main_layer = {}
    queue = deque()
    for nid in main_ids:
        if main_in_degree[nid] == 0:
            main_layer[nid] = 0
            queue.append(nid)

    processed = defaultdict(int)
    while queue:
        nid = queue.popleft()
        for child in main_children[nid]:
            candidate = main_layer[nid] + 1
            if child not in main_layer or candidate > main_layer[child]:
                main_layer[child] = candidate
            processed[child] += 1
            if processed[child] == main_in_degree[child]:
                queue.append(child)

    for nid in main_ids:
        if nid not in main_layer:
            main_layer[nid] = 0

    # --- Identify parallel branches ---
    topo_order = sorted(main_ids, key=lambda x: (main_layer.get(x, 0), x))

    branch_id = {}
    next_branch = 0

    # Assign initial branches to roots
    roots = sorted([nid for nid in main_ids if main_in_degree[nid] == 0])
    for nid in roots:
        branch_id[nid] = next_branch
        next_branch += 1

    # Forward pass: propagate and split at divergence points
    for nid in topo_order:
        if nid not in branch_id:
            # Inherit from first parent (by branch order, then alphabetically)
            candidates = [(branch_id[p], p) for p in main_parents[nid] if p in branch_id]
            if candidates:
                candidates.sort()
                branch_id[nid] = candidates[0][0]
            else:
                branch_id[nid] = next_branch
                next_branch += 1

        # At divergence points, assign different branches to children
        mc = sorted(main_children[nid])
        if len(mc) > 1:
            for i, child in enumerate(mc):
                if i == 0:
                    _propagate_branch(child, branch_id[nid], branch_id,
                                      main_children, main_parents)
                else:
                    _propagate_branch(child, next_branch, branch_id,
                                      main_children, main_parents)
                    next_branch += 1

    # --- Position main-flow nodes ---
    # Collect unique branches and assign vertical rank
    all_branches = sorted(set(branch_id.get(nid, 0) for nid in main_ids))
    branch_rank = {bid: i for i, bid in enumerate(all_branches)}

    positions = {}

    # Group main nodes by layer
    layer_groups = defaultdict(list)
    for nid in main_ids:
        layer_groups[main_layer[nid]].append(nid)
    for l in layer_groups:
        layer_groups[l].sort(key=lambda x: (branch_id.get(x, 0), x))

    if len(all_branches) <= 1:
        # Single-branch (linear) pipeline
        for l in sorted(layer_groups.keys()):
            for row_idx, nid in enumerate(layer_groups[l]):
                positions[nid] = {
                    "x": START_X + l * X_SPACING,
                    "y": START_Y + row_idx * Y_SPACING
                }
    else:
        # Multi-branch pipeline
        for l in sorted(layer_groups.keys()):
            by_branch = defaultdict(list)
            for nid in layer_groups[l]:
                by_branch[branch_id.get(nid, 0)].append(nid)

            for bid in sorted(by_branch.keys()):
                rank = branch_rank[bid]
                base_y = START_Y + rank * BRANCH_GAP
                for row_idx, nid in enumerate(by_branch[bid]):
                    positions[nid] = {
                        "x": START_X + main_layer[nid] * X_SPACING,
                        "y": base_y + row_idx * Y_SPACING
                    }

    # --- Position auxiliary nodes ---
    # Place in the same column as their Trainer target, offset below
    aux_count_per_target = defaultdict(int)
    for nid in sorted(auxiliary_ids):
        target = None
        for child in sorted(children_all[nid]):
            if child in positions:
                target = child
                break
        if target:
            tpos = positions[target]
            idx = aux_count_per_target[target]
            aux_count_per_target[target] += 1
            positions[nid] = {
                "x": tpos["x"],
                "y": tpos["y"] + Y_SPACING * (idx + 1)
            }
        else:
            # Fallback: place after all positioned nodes
            max_y = max((p["y"] for p in positions.values()), default=START_Y)
            positions[nid] = {
                "x": START_X,
                "y": max_y + Y_SPACING
            }

    # --- Position post-trainer nodes ---
    # Place one column after their source, starting at the same y as source
    post_count_per_source = defaultdict(int)
    for nid in sorted(post_trainer_ids):
        source = None
        for p in sorted(parents_all[nid]):
            if p in positions:
                source = p
                break
        if source:
            spos = positions[source]
            idx = post_count_per_source[source]
            post_count_per_source[source] += 1
            positions[nid] = {
                "x": spos["x"] + X_SPACING,
                "y": spos["y"] + idx * Y_SPACING
            }
        else:
            max_y = max((p["y"] for p in positions.values()), default=START_Y)
            positions[nid] = {
                "x": START_X,
                "y": max_y + Y_SPACING
            }

    # --- Fix overlaps ---
    _fix_overlaps(positions)

    # --- Ensure no negative y values ---
    min_y = min(p["y"] for p in positions.values())
    if min_y < START_Y:
        offset = START_Y - min_y
        for nid in positions:
            positions[nid]["y"] += offset

    return positions


def _propagate_branch(start, bid, branch_id, main_children, main_parents):
    """Propagate branch ID through a linear chain (stops at divergence or merge)."""
    q = deque([start])
    while q:
        nid = q.popleft()
        branch_id[nid] = bid
        mc = sorted(main_children[nid])
        if len(mc) == 1:
            child = mc[0]
            if len(main_parents[child]) <= 1:
                q.append(child)
            else:
                # Merge point - set only if not already assigned
                if child not in branch_id:
                    branch_id[child] = bid


def _fix_overlaps(positions):
    """Ensure no two nodes are too close (within Y_SPACING vertically at same x)."""
    # Group by x coordinate
    by_x = defaultdict(list)
    for nid, pos in positions.items():
        by_x[pos["x"]].append(nid)

    for x_val in sorted(by_x.keys()):
        nodes_at_x = by_x[x_val]
        # Sort by current y
        nodes_at_x.sort(key=lambda nid: positions[nid]["y"])

        for i in range(1, len(nodes_at_x)):
            prev_nid = nodes_at_x[i - 1]
            curr_nid = nodes_at_x[i]
            prev_y = positions[prev_nid]["y"]
            curr_y = positions[curr_nid]["y"]
            if curr_y - prev_y < Y_SPACING:
                positions[curr_nid]["y"] = prev_y + Y_SPACING


def process_file(filepath):
    """Process a single JSON file and fix node positions. Returns True if modified."""
    data = load_json(filepath)
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    if not nodes:
        return False

    new_positions = compute_positions(nodes, edges)

    changed = False
    for node in nodes:
        nid = node["id"]
        if nid in new_positions:
            old_pos = node.get("position", {})
            new_pos = new_positions[nid]
            if old_pos.get("x") != new_pos["x"] or old_pos.get("y") != new_pos["y"]:
                changed = True
                node["position"] = new_pos

    if changed:
        save_json(filepath, data)

    return changed


def main():
    examples_dir = Path(__file__).parent

    subdirs = [
        "classification", "regression", "unsupervised",
        "image", "nlp", "timeseries", "advanced"
    ]

    total_files = 0
    modified_files = 0
    skipped_files = 0
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
                was_modified = process_file(str(filepath))
                if was_modified:
                    modified_files += 1
                    print(f"  [FIXED] {subdir}/{filepath.name}")
                else:
                    skipped_files += 1
                    print(f"  [OK]    {subdir}/{filepath.name}")
            except Exception as e:
                errors.append((f"{subdir}/{filepath.name}", str(e)))
                print(f"  [ERROR] {subdir}/{filepath.name}: {e}")

    print("\n" + "=" * 60)
    print(f"Summary:")
    print(f"  Total files:    {total_files}")
    print(f"  Modified:       {modified_files}")
    print(f"  Already OK:     {skipped_files}")
    print(f"  Errors:         {len(errors)}")
    if errors:
        print("\nErrors:")
        for fname, err in errors:
            print(f"  {fname}: {err}")
    print("=" * 60)


if __name__ == "__main__":
    main()
