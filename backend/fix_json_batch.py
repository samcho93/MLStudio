"""Batch fix JSON examples."""
import json

# Fix 93: wine_quality uses comma separator, not semicolon
with open('examples/advanced/093_hyperparameter_search.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
if 'separator' in data['nodes'][0]['params']:
    del data['nodes'][0]['params']['separator']
data['edges'] = [e for e in data['edges'] if not (e.get('sourceHandle')=='test_features' and e.get('targetHandle')=='train_features')]
with open('examples/advanced/093_hyperparameter_search.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print('Fixed 93')

# Fix 98: KFoldTrainer needs data edges
with open('examples/advanced/098_kfold_cross_validation.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
data['edges'].extend([
    {'source': 'CSVLoader_1', 'sourceHandle': 'features', 'target': 'KFoldTrainer_1', 'targetHandle': 'train_features', 'id': 'fix_data_KFoldTrainer'},
    {'source': 'CSVLoader_1', 'sourceHandle': 'labels', 'target': 'KFoldTrainer_1', 'targetHandle': 'train_labels', 'id': 'fix_labels_KFoldTrainer'},
])
for e in data['edges']:
    if e['source'] == 'EarlyStopping_1' and e.get('targetHandle') == 'layers':
        e['targetHandle'] = 'callback_config'
data['edges'] = [e for e in data['edges'] if not (e['source']=='StandardScaler_1' and e['target']=='KFoldTrainer_1' and e.get('targetHandle')=='layers')]
with open('examples/advanced/098_kfold_cross_validation.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print('Fixed 98')

# Fix 100: Add missing edge from CSVLoader to LabelEncoder
with open('examples/advanced/100_full_pipeline_comprehensive.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
data['edges'].append({
    'source': 'CSVLoader_1', 'sourceHandle': 'features', 'target': 'LabelEncoder_1', 'targetHandle': 'input',
    'id': 'fix_CSVLoader_LabelEncoder'
})
data['edges'] = [e for e in data['edges'] if not (e.get('sourceHandle')=='test_features' and e.get('targetHandle')=='train_features')]
with open('examples/advanced/100_full_pipeline_comprehensive.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print('Fixed 100')

# Fix 75: Add data edges to trainer
with open('examples/nlp/075_sentence_similarity_prediction.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
data['edges'].extend([
    {'source': 'csv_loader_01', 'sourceHandle': 'features', 'target': 'trainer_01', 'targetHandle': 'train_features', 'id': 'fix_data_75'},
    {'source': 'csv_loader_01', 'sourceHandle': 'labels', 'target': 'trainer_01', 'targetHandle': 'train_labels', 'id': 'fix_labels_75'},
])
with open('examples/nlp/075_sentence_similarity_prediction.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print('Fixed 75')

# Fix 76: NER - simplify to standard classification
with open('examples/nlp/076_named_entity_recognition.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
for n in data['nodes']:
    if n['id'] == 'bilstm_01':
        n['params']['return_sequences'] = False
data['edges'].extend([
    {'source': 'csv_loader_01', 'sourceHandle': 'features', 'target': 'trainer_01', 'targetHandle': 'train_features', 'id': 'fix_data_76'},
    {'source': 'csv_loader_01', 'sourceHandle': 'labels', 'target': 'trainer_01', 'targetHandle': 'train_labels', 'id': 'fix_labels_76'},
])
with open('examples/nlp/076_named_entity_recognition.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print('Fixed 76')

# Fix 63: Siamese - simplify to single-branch CNN
with open('examples/image/063_siamese_network_similarity.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
nodes_to_keep = {'ImageFolder_1', 'Conv2D_1', 'MaxPooling2D_1', 'Conv2D_2', 'GlobalAveragePooling2D_1',
                 'Dense_1', 'Dense_2', 'Optimizer_1', 'LossFunction_1', 'Trainer_1', 'LossCurve_1', 'AccuracyCurve_1'}
data['nodes'] = [n for n in data['nodes'] if n['id'] in nodes_to_keep]
data['edges'] = [
    {'source': 'ImageFolder_1', 'target': 'Conv2D_1', 'sourceHandle': 'features', 'targetHandle': 'input'},
    {'source': 'Conv2D_1', 'target': 'MaxPooling2D_1', 'sourceHandle': 'output', 'targetHandle': 'input'},
    {'source': 'MaxPooling2D_1', 'target': 'Conv2D_2', 'sourceHandle': 'output', 'targetHandle': 'input'},
    {'source': 'Conv2D_2', 'target': 'GlobalAveragePooling2D_1', 'sourceHandle': 'output', 'targetHandle': 'input'},
    {'source': 'GlobalAveragePooling2D_1', 'target': 'Dense_1', 'sourceHandle': 'output', 'targetHandle': 'input'},
    {'source': 'Dense_1', 'target': 'Dense_2', 'sourceHandle': 'output', 'targetHandle': 'input'},
    {'source': 'Dense_2', 'target': 'Trainer_1', 'sourceHandle': 'output', 'targetHandle': 'layers'},
    {'source': 'ImageFolder_1', 'target': 'Trainer_1', 'sourceHandle': 'features', 'targetHandle': 'train_features'},
    {'source': 'ImageFolder_1', 'target': 'Trainer_1', 'sourceHandle': 'labels', 'targetHandle': 'train_labels'},
    {'source': 'Optimizer_1', 'target': 'Trainer_1', 'sourceHandle': 'optimizer_config', 'targetHandle': 'optimizer_config'},
    {'source': 'LossFunction_1', 'target': 'Trainer_1', 'sourceHandle': 'loss_config', 'targetHandle': 'loss_config'},
    {'source': 'Trainer_1', 'target': 'LossCurve_1', 'sourceHandle': 'history', 'targetHandle': 'history'},
    {'source': 'Trainer_1', 'target': 'AccuracyCurve_1', 'sourceHandle': 'history', 'targetHandle': 'history'},
]
with open('examples/image/063_siamese_network_similarity.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print('Fixed 63')

# Fix 94: Mixed input - add data edges from CSV
with open('examples/advanced/094_mixed_input_model.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
csv_id = trainer_id = None
for n in data['nodes']:
    if n['type'] == 'CSVLoader': csv_id = n['id']
    if n['type'] == 'Trainer': trainer_id = n['id']
if csv_id and trainer_id:
    data['edges'].extend([
        {'source': csv_id, 'sourceHandle': 'features', 'target': trainer_id, 'targetHandle': 'train_features', 'id': 'fix_data_94'},
        {'source': csv_id, 'sourceHandle': 'labels', 'target': trainer_id, 'targetHandle': 'train_labels', 'id': 'fix_labels_94'},
    ])
with open('examples/advanced/094_mixed_input_model.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print('Fixed 94')

print('All JSON fixes applied')
