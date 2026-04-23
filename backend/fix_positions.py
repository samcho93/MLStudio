"""
fix_positions.py
Adjusts node positions in ALL example JSON files to prevent overlap.
Uses topological sort to determine DAG depth, then assigns positions
based on node category and depth.

Rules:
- Left-to-right flow: data -> preprocessing -> layers -> training -> visualization
- X gap = 280px, Y gap = 180px minimum
- Data nodes start at x=50
- Sequential layer nodes stacked vertically
- Trainer + config nodes (Optimizer, LossFunction, EarlyStopping, ModelCheckpoint, LRScheduler) grouped
- Visualization/evaluation/export nodes at rightmost position
"""

import json
import os
import glob
from collections import defaultdict, deque

# ── Constants ──
X_START = 50
X_GAP = 280
Y_GAP = 180
Y_START = 80

# ── Node category classification ──
DATA_NODES = {
    'CSVLoader', 'CSV Loader', 'NumpyInput', 'ImageFolder', 'HDF5Loader',
}
PREPROCESSING_NODES = {
    'StandardScaler', 'MinMaxScaler', 'LabelEncoder', 'OneHotEncoder',
    'PCA', 'Augmentation', 'DataInspector', 'TrainValSplit',
}
LAYER_NODES = {
    'Dense', 'Conv2D', 'Conv1D', 'MaxPooling2D', 'Flatten',
    'GlobalAveragePooling2D', 'BatchNorm', 'Dropout',
    'LSTM', 'GRU', 'Embedding', 'Concat', 'Add', 'Reshape',
    'MultiHeadAttention', 'TransformerBlock', 'PretrainedModel',
    'Conv2DTranspose',
}
TRAINER_NODES = {'Trainer'}
TRAINER_CONFIG_NODES = {
    'Optimizer', 'LossFunction', 'LRScheduler',
    'EarlyStopping', 'ModelCheckpoint', 'GradientClipping',
    'MixedPrecision', 'CustomCallback',
}
VIZ_EVAL_NODES = {
    'LossCurve', 'AccuracyCurve', 'ModelSummary',
    'FeatureMapViewer', 'GradCAMViewer', 'tSNEViewer', 'UMAPViewer',
    'DataDistribution', 'ClassificationMetrics', 'RegressionMetrics',
    'ConfusionMatrix', 'ROCCurve', 'PrecisionRecallCurve',
    'PredictionViewer',
}
EXPORT_NODES = {
    'ModelSave', 'ONNXExport', 'TFLiteExport', 'Quantization',
    'CustomCodeNode',
}


def get_node_category(node_type):
    """Return a category string for the node type."""
    if node_type in DATA_NODES:
        return 'data'
    if node_type in PREPROCESSING_NODES:
        return 'preprocess'
    if node_type in LAYER_NODES:
        return 'layer'
    if node_type in TRAINER_NODES:
        return 'trainer'
    if node_type in TRAINER_CONFIG_NODES:
        return 'trainer_config'
    if node_type in VIZ_EVAL_NODES:
        return 'viz'
    if node_type in EXPORT_NODES:
        return 'export'
    return 'layer'  # default unknown nodes to layer


def topological_sort_with_depth(nodes, edges):
    """
    Returns dict: node_id -> depth (0-based).
    Uses Kahn's algorithm. Depth = longest path from any root.
    """
    node_ids = {n['id'] for n in nodes}
    adj = defaultdict(list)      # forward adjacency
    in_degree = defaultdict(int)

    for e in edges:
        src, tgt = e['source'], e['target']
        if src in node_ids and tgt in node_ids:
            adj[src].append(tgt)
            in_degree[tgt] += 1

    # Initialize all nodes
    for n in nodes:
        if n['id'] not in in_degree:
            in_degree[n['id']] = 0

    # BFS for longest path (depth)
    depth = {}
    queue = deque()
    for nid in node_ids:
        if in_degree[nid] == 0:
            queue.append(nid)
            depth[nid] = 0

    while queue:
        nid = queue.popleft()
        for child in adj[nid]:
            # longest path
            new_depth = depth[nid] + 1
            if child not in depth or new_depth > depth[child]:
                depth[child] = new_depth
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    # Handle any nodes not reached (disconnected)
    for n in nodes:
        if n['id'] not in depth:
            depth[n['id']] = 0

    return depth


def assign_positions(nodes, edges):
    """
    Assign x,y positions to nodes based on DAG depth and category.

    Strategy:
    1. Compute DAG depth via topological sort (longest path).
    2. Group nodes by depth.
    3. For trainer_config nodes, place them at the same x as the Trainer
       but stacked below it.
    4. For viz/eval/export nodes, place them one column after the Trainer.
    5. Export nodes that come after viz nodes get an extra column.
    """
    if not nodes:
        return

    node_map = {n['id']: n for n in nodes}
    depth = topological_sort_with_depth(nodes, edges)

    # Build adjacency for finding trainer's downstream
    children = defaultdict(set)
    parents = defaultdict(set)
    for e in edges:
        children[e['source']].add(e['target'])
        parents[e['target']].add(e['source'])

    # Find all trainer nodes
    trainer_ids = [n['id'] for n in nodes if get_node_category(n['type']) == 'trainer']

    # Find trainer_config nodes (nodes that feed INTO a trainer)
    config_for_trainer = defaultdict(list)  # trainer_id -> [config_node_ids]
    for tid in trainer_ids:
        for p in parents[tid]:
            if get_node_category(node_map[p]['type']) == 'trainer_config':
                config_for_trainer[tid].append(p)

    # Find viz/eval nodes (nodes that a trainer feeds INTO)
    viz_after_trainer = defaultdict(list)  # trainer_id -> [viz_node_ids]
    for tid in trainer_ids:
        for c in children[tid]:
            cat = get_node_category(node_map[c]['type'])
            if cat in ('viz', 'export'):
                viz_after_trainer[tid].append(c)

    # Find export nodes that come after other nodes (not directly from trainer)
    export_after_viz = set()
    for n in nodes:
        if get_node_category(n['type']) == 'export':
            # Check if any parent is a viz or export node (not trainer)
            for p in parents[n['id']]:
                pcat = get_node_category(node_map[p]['type'])
                if pcat in ('viz', 'export'):
                    export_after_viz.add(n['id'])

    # Collect all config and viz node IDs that are handled specially
    specially_handled = set()
    for tid in trainer_ids:
        specially_handled.update(config_for_trainer[tid])
        specially_handled.update(viz_after_trainer[tid])
    specially_handled.update(export_after_viz)

    # Group remaining nodes by depth
    depth_groups = defaultdict(list)
    for n in nodes:
        if n['id'] not in specially_handled:
            depth_groups[depth[n['id']]].append(n)

    # Sort depths
    sorted_depths = sorted(depth_groups.keys())

    # Assign x based on column index, y based on row within column
    col = 0
    node_positions = {}  # node_id -> (x, y)

    for d in sorted_depths:
        group = depth_groups[d]
        # Sort nodes within same depth for consistent ordering:
        # data first, then preprocess, then layers, then trainer
        category_order = {'data': 0, 'preprocess': 1, 'layer': 2, 'trainer': 3, 'trainer_config': 4, 'viz': 5, 'export': 6}
        group.sort(key=lambda n: (category_order.get(get_node_category(n['type']), 3), n['id']))

        x = X_START + col * X_GAP
        for row, n in enumerate(group):
            y = Y_START + row * Y_GAP
            node_positions[n['id']] = (x, y)

        col += 1

    # Now handle trainer config nodes: same x as trainer, stacked below
    for tid in trainer_ids:
        if tid not in node_positions:
            continue
        tx, ty = node_positions[tid]
        configs = config_for_trainer[tid]
        # Sort configs for consistency
        config_order = {'Optimizer': 0, 'LossFunction': 1, 'LRScheduler': 2,
                        'EarlyStopping': 3, 'ModelCheckpoint': 4, 'GradientClipping': 5,
                        'MixedPrecision': 6, 'CustomCallback': 7}
        configs.sort(key=lambda cid: (config_order.get(node_map[cid]['type'], 99), cid))

        for i, cid in enumerate(configs):
            node_positions[cid] = (tx, ty + (i + 1) * Y_GAP)

    # Handle viz/eval nodes: one column after trainer
    for tid in trainer_ids:
        if tid not in node_positions:
            continue
        tx, ty = node_positions[tid]
        viz_x = tx + X_GAP

        viz_nodes = viz_after_trainer[tid]
        # Sort viz nodes for consistency
        viz_nodes.sort(key=lambda vid: (node_map[vid]['type'], vid))

        for i, vid in enumerate(viz_nodes):
            if vid not in export_after_viz:
                node_positions[vid] = (viz_x, Y_START + i * Y_GAP)

    # Handle export nodes after viz: two columns after trainer
    if export_after_viz:
        for tid in trainer_ids:
            if tid not in node_positions:
                continue
            tx, ty = node_positions[tid]
            export_x = tx + 2 * X_GAP

            export_list = sorted(export_after_viz, key=lambda eid: eid)
            for i, eid in enumerate(export_list):
                node_positions[eid] = (export_x, Y_START + i * Y_GAP)

    # Handle any remaining viz/export nodes not connected to a trainer
    # (e.g., in unsupervised pipelines with no trainer)
    unplaced = [n for n in nodes if n['id'] not in node_positions]
    if unplaced:
        # Find max x already used
        max_x = max((pos[0] for pos in node_positions.values()), default=X_START)
        next_x = max_x + X_GAP

        # Group by category
        unplaced_viz = [n for n in unplaced if get_node_category(n['type']) in ('viz', 'export')]
        unplaced_other = [n for n in unplaced if get_node_category(n['type']) not in ('viz', 'export')]

        for i, n in enumerate(unplaced_other):
            node_positions[n['id']] = (next_x, Y_START + i * Y_GAP)

        viz_x = next_x + X_GAP if unplaced_other else next_x
        for i, n in enumerate(unplaced_viz):
            node_positions[n['id']] = (viz_x, Y_START + i * Y_GAP)

    # Apply positions to nodes
    for n in nodes:
        if n['id'] in node_positions:
            x, y = node_positions[n['id']]
            n['position']['x'] = x
            n['position']['y'] = y


def process_file(filepath):
    """Process a single JSON example file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"  SKIP (parse error): {filepath} - {e}")
        return False

    if 'nodes' not in data or 'edges' not in data:
        print(f"  SKIP (no nodes/edges): {filepath}")
        return False

    nodes = data['nodes']
    edges = data['edges']

    if not nodes:
        print(f"  SKIP (empty nodes): {filepath}")
        return False

    assign_positions(nodes, edges)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return True


def main():
    examples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples')

    if not os.path.isdir(examples_dir):
        print(f"ERROR: examples directory not found: {examples_dir}")
        return

    # Find all JSON files recursively, excluding _index.json
    json_files = []
    for root, dirs, files in os.walk(examples_dir):
        for f in files:
            if f.endswith('.json') and f != '_index.json':
                json_files.append(os.path.join(root, f))

    json_files.sort()

    print(f"Found {len(json_files)} example JSON files.")
    print()

    success = 0
    skipped = 0

    for filepath in json_files:
        rel = os.path.relpath(filepath, examples_dir)
        if process_file(filepath):
            print(f"  OK: {rel}")
            success += 1
        else:
            skipped += 1

    print()
    print(f"Done. {success} files updated, {skipped} skipped.")


if __name__ == '__main__':
    main()
