"""
30개 튜토리얼 예제 JSON 생성기.
각 예제는 ML Node Studio의 개별 기능을 익히기 위한 간단한 파이프라인.
"""
import json, os

DIR = os.path.dirname(os.path.abspath(__file__))

# ── 노드/엣지 헬퍼 ────────────────────────────────────
def N(nid, ntype, x, y, params=None):
    return {"id": nid, "type": ntype, "position": {"x": x, "y": y}, "params": params or {}}

def E(src, tgt, sh="output", th="input"):
    return {"source": src, "target": tgt, "sourceHandle": sh, "targetHandle": th}

# ── 공통 레이아웃 x좌표 ──────────────────────────────
X1, X2, X3, X4, X5, X6 = 50, 330, 610, 890, 1170, 1450
Y1, Y2, Y3 = 80, 480, 880

TUTORIALS = []

# ============================================================
# 101: CSV 로드 + 데이터 살펴보기
# ============================================================
TUTORIALS.append({
    "meta": {
        "id": 101, "title": "Load CSV & Inspect Data",
        "title_ko": "CSV 파일 로드 & 데이터 확인",
        "category": "tutorial", "difficulty": "beginner",
        "description": "Load a CSV file and inspect its shape, dtypes, and sample rows.",
        "description_ko": "CSV 파일을 불러오고 DataInspector로 데이터의 형태, 타입, 샘플을 확인합니다.",
        "tags": ["csv", "data-inspector", "기초"],
        "estimated_time": "30초",
        "learning_objectives": [
            "CSVLoader 노드에서 파일 경로와 타겟 컬럼 설정법",
            "DataInspector로 shape, dtype, 샘플 미리보기"
        ],
        "guide_steps": [
            {"step": 1, "node": "CSVLoader_1", "text": "CSV 파일을 로드합니다. file_path에 데이터 경로, target_column에 예측할 컬럼명을 입력하세요."},
            {"step": 2, "node": "DataInspector_1", "text": "로드된 데이터의 shape, dtype, 첫 5행을 확인합니다. Run 버튼을 누르세요."}
        ],
        "requires_phase": 1
    },
    "nodes": [
        N("CSVLoader_1", "CSVLoader", X1, Y1, {"file_path": "datasets/iris.csv", "target_column": "species"}),
        N("DataInspector_1", "DataInspector", X2, Y1),
    ],
    "edges": [
        E("CSVLoader_1", "DataInspector_1", "features", "input"),
    ]
})

# ============================================================
# 102: 노드 연결 기초 — 3개 노드 직렬 연결
# ============================================================
TUTORIALS.append({
    "meta": {
        "id": 102, "title": "Connect Nodes in Series",
        "title_ko": "노드 직렬 연결 기초",
        "category": "tutorial", "difficulty": "beginner",
        "description": "Learn to connect nodes: CSVLoader → StandardScaler → DataInspector.",
        "description_ko": "CSVLoader → StandardScaler → DataInspector 순서로 노드를 연결하는 기본 패턴을 익힙니다.",
        "tags": ["연결", "scaler", "기초"],
        "estimated_time": "30초",
        "learning_objectives": [
            "노드 출력 포트 → 입력 포트 연결 방법",
            "StandardScaler가 데이터를 어떻게 변환하는지 확인"
        ],
        "guide_steps": [
            {"step": 1, "node": "CSVLoader_1", "text": "Iris 데이터를 로드합니다."},
            {"step": 2, "node": "StandardScaler_1", "text": "평균=0, 분산=1로 정규화합니다."},
            {"step": 3, "node": "DataInspector_1", "text": "정규화된 결과를 확인합니다. 값이 -2~2 범위로 바뀌었는지 보세요."}
        ],
        "requires_phase": 1
    },
    "nodes": [
        N("CSVLoader_1", "CSVLoader", X1, Y1, {"file_path": "datasets/iris.csv", "target_column": "species"}),
        N("StandardScaler_1", "StandardScaler", X2, Y1),
        N("DataInspector_1", "DataInspector", X3, Y1),
    ],
    "edges": [
        E("CSVLoader_1", "StandardScaler_1", "features", "input"),
        E("StandardScaler_1", "DataInspector_1"),
    ]
})

# ============================================================
# 103: Train/Val 분할
# ============================================================
TUTORIALS.append({
    "meta": {
        "id": 103, "title": "Split Data into Train/Val",
        "title_ko": "학습/검증 데이터 분할",
        "category": "tutorial", "difficulty": "beginner",
        "description": "Split a dataset into training and validation sets using TrainValSplit.",
        "description_ko": "TrainValSplit 노드로 데이터를 학습용과 검증용으로 분할합니다.",
        "tags": ["split", "train", "validation", "기초"],
        "estimated_time": "30초",
        "learning_objectives": [
            "val_ratio로 분할 비율 설정",
            "학습/검증 분할의 중요성 이해"
        ],
        "guide_steps": [
            {"step": 1, "node": "CSVLoader_1", "text": "Iris 데이터 150개 샘플을 로드합니다."},
            {"step": 2, "node": "TrainValSplit_1", "text": "80:20으로 분할합니다. 학습 120개, 검증 30개로 나뉩니다."}
        ],
        "requires_phase": 1
    },
    "nodes": [
        N("CSVLoader_1", "CSVLoader", X1, Y1, {"file_path": "datasets/iris.csv", "target_column": "species"}),
        N("TrainValSplit_1", "TrainValSplit", X2, Y1, {"val_ratio": 0.2, "test_ratio": 0.0, "random_seed": 42}),
    ],
    "edges": [
        E("CSVLoader_1", "TrainValSplit_1", "features", "features"),
        E("CSVLoader_1", "TrainValSplit_1", "labels", "labels"),
    ]
})

# ============================================================
# 104: Dense 레이어 1개로 학습
# ============================================================
TUTORIALS.append({
    "meta": {
        "id": 104, "title": "Single Dense Layer Training",
        "title_ko": "Dense 레이어 1개로 학습하기",
        "category": "tutorial", "difficulty": "beginner",
        "description": "Train a minimal model with just one Dense layer.",
        "description_ko": "Dense 레이어 1개만으로 Iris 분류 모델을 학습합니다. 최소 구성의 학습 파이프라인입니다.",
        "tags": ["dense", "minimal", "학습기초"],
        "estimated_time": "1분",
        "learning_objectives": [
            "최소 학습 파이프라인 구성 (데이터→레이어→학습기)",
            "Optimizer, LossFunction 노드의 역할",
            "Run 버튼으로 학습 실행하기"
        ],
        "guide_steps": [
            {"step": 1, "node": "CSVLoader_1", "text": "Iris 데이터를 로드합니다."},
            {"step": 2, "node": "TrainValSplit_1", "text": "학습/검증 데이터를 분할합니다."},
            {"step": 3, "node": "Dense_1", "text": "3개 뉴런, softmax. 3종 분류를 위한 출력층입니다."},
            {"step": 4, "node": "Optimizer_1", "text": "Adam 옵티마이저. 학습률 0.01로 설정합니다."},
            {"step": 5, "node": "LossFunction_1", "text": "분류 문제의 표준 손실 함수입니다."},
            {"step": 6, "node": "Trainer_1", "text": "Run을 눌러 학습을 시작하세요!"}
        ],
        "requires_phase": 1
    },
    "nodes": [
        N("CSVLoader_1", "CSVLoader", X1, Y1, {"file_path": "datasets/iris.csv", "target_column": "species"}),
        N("StandardScaler_1", "StandardScaler", X2, Y1),
        N("TrainValSplit_1", "TrainValSplit", X3, Y1, {"val_ratio": 0.2, "random_seed": 42}),
        N("Dense_1", "Dense", X1, Y2, {"units": 3, "activation": "softmax"}),
        N("Optimizer_1", "Optimizer", X2, Y2, {"type": "adam", "learning_rate": 0.01}),
        N("LossFunction_1", "LossFunction", X3, Y2, {"type": "sparse_categorical_crossentropy"}),
        N("Trainer_1", "Trainer", X4, Y1, {"epochs": 20, "batch_size": 16, "framework": "pytorch"}),
    ],
    "edges": [
        E("CSVLoader_1", "StandardScaler_1", "features", "input"),
        E("StandardScaler_1", "TrainValSplit_1", "output", "features"),
        E("CSVLoader_1", "TrainValSplit_1", "labels", "labels"),
        E("Dense_1", "Trainer_1", "output", "layers"),
        E("TrainValSplit_1", "Trainer_1", "train_features", "train_features"),
        E("TrainValSplit_1", "Trainer_1", "train_labels", "train_labels"),
        E("TrainValSplit_1", "Trainer_1", "val_features", "val_features"),
        E("TrainValSplit_1", "Trainer_1", "val_labels", "val_labels"),
        E("Optimizer_1", "Trainer_1", "optimizer_config", "optimizer_config"),
        E("LossFunction_1", "Trainer_1", "loss_config", "loss_config"),
    ]
})

# ============================================================
# 105: Loss Curve 보기
# ============================================================
TUTORIALS.append({
    "meta": {
        "id": 105, "title": "View Loss Curve",
        "title_ko": "Loss 곡선 확인하기",
        "category": "tutorial", "difficulty": "beginner",
        "description": "Train a model and observe the loss curve in real-time.",
        "description_ko": "학습 후 LossCurve 노드로 Train/Val Loss 그래프를 확인합니다.",
        "tags": ["loss-curve", "시각화", "학습기초"],
        "estimated_time": "1분",
        "learning_objectives": [
            "LossCurve 노드를 Trainer에 연결하는 방법",
            "Train Loss와 Val Loss 곡선 해석",
            "Loss가 줄어드는 것이 학습이 진행되고 있음을 의미"
        ],
        "guide_steps": [
            {"step": 1, "node": "Trainer_1", "text": "학습을 실행합니다. epochs=30으로 충분한 학습량을 줍니다."},
            {"step": 2, "node": "LossCurve_1", "text": "학습 완료 후 Loss 곡선이 표시됩니다. 두 곡선이 함께 내려가면 정상입니다."}
        ],
        "requires_phase": 1
    },
    "nodes": [
        N("CSVLoader_1", "CSVLoader", X1, Y1, {"file_path": "datasets/iris.csv", "target_column": "species"}),
        N("StandardScaler_1", "StandardScaler", X2, Y1),
        N("TrainValSplit_1", "TrainValSplit", X3, Y1, {"val_ratio": 0.2, "random_seed": 42}),
        N("Dense_1", "Dense", X1, Y2, {"units": 32, "activation": "relu"}),
        N("Dense_2", "Dense", X2, Y2, {"units": 3, "activation": "softmax"}),
        N("Optimizer_1", "Optimizer", X3, Y2, {"type": "adam", "learning_rate": 0.001}),
        N("LossFunction_1", "LossFunction", X3, Y3, {"type": "sparse_categorical_crossentropy"}),
        N("Trainer_1", "Trainer", X4, Y1, {"epochs": 30, "batch_size": 16, "framework": "pytorch"}),
        N("LossCurve_1", "LossCurve", X5, Y1),
    ],
    "edges": [
        E("CSVLoader_1", "StandardScaler_1", "features", "input"),
        E("StandardScaler_1", "TrainValSplit_1", "output", "features"),
        E("CSVLoader_1", "TrainValSplit_1", "labels", "labels"),
        E("Dense_1", "Dense_2", "output", "input"),
        E("Dense_2", "Trainer_1", "output", "layers"),
        E("TrainValSplit_1", "Trainer_1", "train_features", "train_features"),
        E("TrainValSplit_1", "Trainer_1", "train_labels", "train_labels"),
        E("TrainValSplit_1", "Trainer_1", "val_features", "val_features"),
        E("TrainValSplit_1", "Trainer_1", "val_labels", "val_labels"),
        E("Optimizer_1", "Trainer_1", "optimizer_config", "optimizer_config"),
        E("LossFunction_1", "Trainer_1", "loss_config", "loss_config"),
        E("Trainer_1", "LossCurve_1", "history", "history"),
    ]
})

# ============================================================
# 106: Accuracy Curve 보기
# ============================================================
TUTORIALS.append({
    "meta": {
        "id": 106, "title": "View Accuracy Curve",
        "title_ko": "Accuracy 곡선 확인하기",
        "category": "tutorial", "difficulty": "beginner",
        "description": "See training accuracy improve over epochs.",
        "description_ko": "AccuracyCurve 노드로 학습 정확도가 epoch마다 향상되는 과정을 관찰합니다.",
        "tags": ["accuracy", "시각화", "학습기초"],
        "estimated_time": "1분",
        "learning_objectives": [
            "AccuracyCurve 노드 사용법",
            "Accuracy가 1.0에 가까울수록 좋은 모델",
            "Train vs Val Accuracy 비교로 과적합 감지"
        ],
        "guide_steps": [
            {"step": 1, "node": "Trainer_1", "text": "학습을 실행합니다."},
            {"step": 2, "node": "AccuracyCurve_1", "text": "정확도 곡선을 확인합니다. Train Acc가 높은데 Val Acc가 낮으면 과적합입니다."}
        ],
        "requires_phase": 1
    },
    "nodes": [
        N("CSVLoader_1", "CSVLoader", X1, Y1, {"file_path": "datasets/iris.csv", "target_column": "species"}),
        N("StandardScaler_1", "StandardScaler", X2, Y1),
        N("TrainValSplit_1", "TrainValSplit", X3, Y1, {"val_ratio": 0.2, "random_seed": 42}),
        N("Dense_1", "Dense", X1, Y2, {"units": 64, "activation": "relu"}),
        N("Dense_2", "Dense", X2, Y2, {"units": 3, "activation": "softmax"}),
        N("Optimizer_1", "Optimizer", X3, Y2, {"type": "adam", "learning_rate": 0.001}),
        N("LossFunction_1", "LossFunction", X3, Y3, {"type": "sparse_categorical_crossentropy"}),
        N("Trainer_1", "Trainer", X4, Y1, {"epochs": 30, "batch_size": 16, "framework": "pytorch"}),
        N("AccuracyCurve_1", "AccuracyCurve", X5, Y1),
    ],
    "edges": [
        E("CSVLoader_1", "StandardScaler_1", "features", "input"),
        E("StandardScaler_1", "TrainValSplit_1", "output", "features"),
        E("CSVLoader_1", "TrainValSplit_1", "labels", "labels"),
        E("Dense_1", "Dense_2", "output", "input"),
        E("Dense_2", "Trainer_1", "output", "layers"),
        E("TrainValSplit_1", "Trainer_1", "train_features", "train_features"),
        E("TrainValSplit_1", "Trainer_1", "train_labels", "train_labels"),
        E("TrainValSplit_1", "Trainer_1", "val_features", "val_features"),
        E("TrainValSplit_1", "Trainer_1", "val_labels", "val_labels"),
        E("Optimizer_1", "Trainer_1", "optimizer_config", "optimizer_config"),
        E("LossFunction_1", "Trainer_1", "loss_config", "loss_config"),
        E("Trainer_1", "AccuracyCurve_1", "history", "history"),
    ]
})

# ============================================================
# 107: MinMaxScaler 사용법
# ============================================================
TUTORIALS.append({
    "meta": {
        "id": 107, "title": "MinMax Normalization",
        "title_ko": "MinMaxScaler로 0~1 정규화",
        "category": "tutorial", "difficulty": "beginner",
        "description": "Normalize features to [0, 1] range using MinMaxScaler.",
        "description_ko": "MinMaxScaler로 모든 특성을 0~1 범위로 정규화합니다. StandardScaler와 비교해봅니다.",
        "tags": ["minmax", "정규화", "전처리"],
        "estimated_time": "30초",
        "learning_objectives": [
            "MinMaxScaler vs StandardScaler 차이",
            "0~1 범위 정규화가 유용한 경우"
        ],
        "guide_steps": [
            {"step": 1, "node": "CSVLoader_1", "text": "데이터를 로드합니다."},
            {"step": 2, "node": "MinMaxScaler_1", "text": "모든 값이 0~1 사이로 변환됩니다."},
            {"step": 3, "node": "DataInspector_1", "text": "변환 결과를 확인합니다. 최솟값=0, 최댓값=1이 되었는지 보세요."}
        ],
        "requires_phase": 1
    },
    "nodes": [
        N("CSVLoader_1", "CSVLoader", X1, Y1, {"file_path": "datasets/iris.csv", "target_column": "species"}),
        N("MinMaxScaler_1", "MinMaxScaler", X2, Y1),
        N("DataInspector_1", "DataInspector", X3, Y1),
    ],
    "edges": [
        E("CSVLoader_1", "MinMaxScaler_1", "features", "input"),
        E("MinMaxScaler_1", "DataInspector_1"),
    ]
})

# ============================================================
# 108: LabelEncoder 사용법
# ============================================================
TUTORIALS.append({
    "meta": {
        "id": 108, "title": "Encode Text Labels",
        "title_ko": "LabelEncoder로 텍스트→숫자 변환",
        "category": "tutorial", "difficulty": "beginner",
        "description": "Convert text labels (cat, dog) to integers (0, 1) using LabelEncoder.",
        "description_ko": "LabelEncoder로 문자열 레이블을 정수로 변환합니다. 분류 모델에 필수적인 전처리입니다.",
        "tags": ["label-encoder", "전처리", "인코딩"],
        "estimated_time": "30초",
        "learning_objectives": [
            "LabelEncoder가 필요한 이유 (모델은 숫자만 입력 가능)",
            "인코딩 결과 확인"
        ],
        "guide_steps": [
            {"step": 1, "node": "CSVLoader_1", "text": "타이타닉 데이터를 로드합니다. sex, embarked 등 문자열 컬럼이 있습니다."},
            {"step": 2, "node": "LabelEncoder_1", "text": "문자열을 정수로 변환합니다. male→0, female→1 등."},
            {"step": 3, "node": "DataInspector_1", "text": "변환 결과를 확인합니다. 모든 값이 숫자로 바뀌었는지 보세요."}
        ],
        "requires_phase": 1
    },
    "nodes": [
        N("CSVLoader_1", "CSVLoader", X1, Y1, {"file_path": "datasets/titanic.csv", "target_column": "survived"}),
        N("LabelEncoder_1", "LabelEncoder", X2, Y1),
        N("DataInspector_1", "DataInspector", X3, Y1),
    ],
    "edges": [
        E("CSVLoader_1", "LabelEncoder_1", "features", "input"),
        E("LabelEncoder_1", "DataInspector_1"),
    ]
})

# ============================================================
# 109: OneHotEncoder 사용법
# ============================================================
TUTORIALS.append({
    "meta": {
        "id": 109, "title": "One-Hot Encoding",
        "title_ko": "OneHotEncoder로 범주형 변환",
        "category": "tutorial", "difficulty": "beginner",
        "description": "Convert categorical features to one-hot vectors.",
        "description_ko": "OneHotEncoder로 범주형 특성을 원-핫 벡터로 변환합니다.",
        "tags": ["one-hot", "전처리", "인코딩"],
        "estimated_time": "30초",
        "learning_objectives": [
            "One-Hot Encoding이 LabelEncoding보다 나은 경우",
            "원-핫 벡터의 구조 이해"
        ],
        "guide_steps": [
            {"step": 1, "node": "CSVLoader_1", "text": "데이터를 로드합니다."},
            {"step": 2, "node": "OneHotEncoder_1", "text": "범주형 컬럼이 [1,0,0], [0,1,0] 같은 벡터로 변환됩니다."},
            {"step": 3, "node": "DataInspector_1", "text": "변환 결과를 확인합니다. 컬럼 수가 늘어난 것을 볼 수 있습니다."}
        ],
        "requires_phase": 1
    },
    "nodes": [
        N("CSVLoader_1", "CSVLoader", X1, Y1, {"file_path": "datasets/titanic.csv", "target_column": "survived"}),
        N("OneHotEncoder_1", "OneHotEncoder", X2, Y1),
        N("DataInspector_1", "DataInspector", X3, Y1),
    ],
    "edges": [
        E("CSVLoader_1", "OneHotEncoder_1", "features", "input"),
        E("OneHotEncoder_1", "DataInspector_1"),
    ]
})

# ============================================================
# 110: Dense 레이어 쌓기
# ============================================================
TUTORIALS.append({
    "meta": {
        "id": 110, "title": "Stack Multiple Dense Layers",
        "title_ko": "Dense 레이어 여러 개 쌓기",
        "category": "tutorial", "difficulty": "beginner",
        "description": "Build a deeper network by connecting multiple Dense layers.",
        "description_ko": "Dense 레이어를 64→32→3으로 쌓아 더 깊은 네트워크를 만듭니다.",
        "tags": ["dense", "deep-network", "레이어"],
        "estimated_time": "1분",
        "learning_objectives": [
            "레이어를 직렬로 연결하는 방법",
            "은닉층 뉴런 수를 줄여가는 패턴",
            "relu → softmax 활성화 함수 조합"
        ],
        "guide_steps": [
            {"step": 1, "node": "Dense_1", "text": "64개 뉴런의 첫 번째 은닉층. ReLU로 비선형성을 추가합니다."},
            {"step": 2, "node": "Dense_2", "text": "32개로 줄여 점점 추상적인 특징을 추출합니다."},
            {"step": 3, "node": "Dense_3", "text": "3개 출력(3클래스). softmax로 확률 분포를 만듭니다."}
        ],
        "requires_phase": 1
    },
    "nodes": [
        N("CSVLoader_1", "CSVLoader", X1, Y1, {"file_path": "datasets/iris.csv", "target_column": "species"}),
        N("StandardScaler_1", "StandardScaler", X2, Y1),
        N("TrainValSplit_1", "TrainValSplit", X3, Y1, {"val_ratio": 0.2, "random_seed": 42}),
        N("Dense_1", "Dense", X1, Y2, {"units": 64, "activation": "relu"}),
        N("Dense_2", "Dense", X2, Y2, {"units": 32, "activation": "relu"}),
        N("Dense_3", "Dense", X3, Y2, {"units": 3, "activation": "softmax"}),
        N("Optimizer_1", "Optimizer", X4, Y2, {"type": "adam", "learning_rate": 0.001}),
        N("LossFunction_1", "LossFunction", X4, Y3, {"type": "sparse_categorical_crossentropy"}),
        N("Trainer_1", "Trainer", X4, Y1, {"epochs": 30, "batch_size": 16, "framework": "pytorch"}),
        N("LossCurve_1", "LossCurve", X5, Y1),
        N("AccuracyCurve_1", "AccuracyCurve", X5, Y2),
    ],
    "edges": [
        E("CSVLoader_1", "StandardScaler_1", "features", "input"),
        E("StandardScaler_1", "TrainValSplit_1", "output", "features"),
        E("CSVLoader_1", "TrainValSplit_1", "labels", "labels"),
        E("Dense_1", "Dense_2", "output", "input"),
        E("Dense_2", "Dense_3", "output", "input"),
        E("Dense_3", "Trainer_1", "output", "layers"),
        E("TrainValSplit_1", "Trainer_1", "train_features", "train_features"),
        E("TrainValSplit_1", "Trainer_1", "train_labels", "train_labels"),
        E("TrainValSplit_1", "Trainer_1", "val_features", "val_features"),
        E("TrainValSplit_1", "Trainer_1", "val_labels", "val_labels"),
        E("Optimizer_1", "Trainer_1", "optimizer_config", "optimizer_config"),
        E("LossFunction_1", "Trainer_1", "loss_config", "loss_config"),
        E("Trainer_1", "LossCurve_1", "history", "history"),
        E("Trainer_1", "AccuracyCurve_1", "history", "history"),
    ]
})

# ============================================================
# 111: Dropout으로 과적합 방지
# ============================================================
TUTORIALS.append({
    "meta": {
        "id": 111, "title": "Prevent Overfitting with Dropout",
        "title_ko": "Dropout으로 과적합 방지",
        "category": "tutorial", "difficulty": "beginner",
        "description": "Add Dropout layers between Dense layers to reduce overfitting.",
        "description_ko": "Dense 레이어 사이에 Dropout을 추가하여 과적합을 줄입니다.",
        "tags": ["dropout", "과적합", "정규화"],
        "estimated_time": "1분",
        "learning_objectives": [
            "Dropout 노드의 rate 파라미터 의미 (0.3 = 30% 뉴런 비활성화)",
            "Dropout이 과적합을 방지하는 원리",
            "Dropout은 레이어 사이에 삽입"
        ],
        "guide_steps": [
            {"step": 1, "node": "Dense_1", "text": "64개 뉴런의 은닉층입니다."},
            {"step": 2, "node": "Dropout_1", "text": "30%의 뉴런을 랜덤으로 끕니다. 과적합을 방지합니다."},
            {"step": 3, "node": "Dense_2", "text": "3개 출력층입니다."},
            {"step": 4, "node": "LossCurve_1", "text": "Train/Val Loss 차이가 작아지면 Dropout이 효과적입니다."}
        ],
        "requires_phase": 1
    },
    "nodes": [
        N("CSVLoader_1", "CSVLoader", X1, Y1, {"file_path": "datasets/iris.csv", "target_column": "species"}),
        N("StandardScaler_1", "StandardScaler", X2, Y1),
        N("TrainValSplit_1", "TrainValSplit", X3, Y1, {"val_ratio": 0.2, "random_seed": 42}),
        N("Dense_1", "Dense", X1, Y2, {"units": 64, "activation": "relu"}),
        N("Dropout_1", "Dropout", X2, Y2, {"rate": 0.3}),
        N("Dense_2", "Dense", X3, Y2, {"units": 3, "activation": "softmax"}),
        N("Optimizer_1", "Optimizer", X4, Y2, {"type": "adam", "learning_rate": 0.001}),
        N("LossFunction_1", "LossFunction", X4, Y3, {"type": "sparse_categorical_crossentropy"}),
        N("Trainer_1", "Trainer", X4, Y1, {"epochs": 50, "batch_size": 16, "framework": "pytorch"}),
        N("LossCurve_1", "LossCurve", X5, Y1),
    ],
    "edges": [
        E("CSVLoader_1", "StandardScaler_1", "features", "input"),
        E("StandardScaler_1", "TrainValSplit_1", "output", "features"),
        E("CSVLoader_1", "TrainValSplit_1", "labels", "labels"),
        E("Dense_1", "Dropout_1", "output", "input"),
        E("Dropout_1", "Dense_2", "output", "input"),
        E("Dense_2", "Trainer_1", "output", "layers"),
        E("TrainValSplit_1", "Trainer_1", "train_features", "train_features"),
        E("TrainValSplit_1", "Trainer_1", "train_labels", "train_labels"),
        E("TrainValSplit_1", "Trainer_1", "val_features", "val_features"),
        E("TrainValSplit_1", "Trainer_1", "val_labels", "val_labels"),
        E("Optimizer_1", "Trainer_1", "optimizer_config", "optimizer_config"),
        E("LossFunction_1", "Trainer_1", "loss_config", "loss_config"),
        E("Trainer_1", "LossCurve_1", "history", "history"),
    ]
})

# ============================================================
# 112: BatchNorm 사용법
# ============================================================
TUTORIALS.append({
    "meta": {
        "id": 112, "title": "BatchNormalization Layer",
        "title_ko": "BatchNorm으로 학습 안정화",
        "category": "tutorial", "difficulty": "beginner",
        "description": "Use BatchNorm between layers to stabilize and speed up training.",
        "description_ko": "BatchNorm 노드를 Dense 사이에 추가하여 학습을 안정화하고 빠르게 합니다.",
        "tags": ["batchnorm", "안정화", "레이어"],
        "estimated_time": "1분",
        "learning_objectives": [
            "BatchNorm이 학습을 빠르게 하는 이유",
            "Dense → BatchNorm → Activation 패턴",
            "BatchNorm + Dropout 조합"
        ],
        "guide_steps": [
            {"step": 1, "node": "Dense_1", "text": "64개 뉴런. activation은 relu입니다."},
            {"step": 2, "node": "BatchNorm_1", "text": "레이어 출력을 정규화합니다. 학습이 더 안정적으로 됩니다."},
            {"step": 3, "node": "Dense_2", "text": "출력층 3개 뉴런입니다."}
        ],
        "requires_phase": 1
    },
    "nodes": [
        N("CSVLoader_1", "CSVLoader", X1, Y1, {"file_path": "datasets/iris.csv", "target_column": "species"}),
        N("StandardScaler_1", "StandardScaler", X2, Y1),
        N("TrainValSplit_1", "TrainValSplit", X3, Y1, {"val_ratio": 0.2, "random_seed": 42}),
        N("Dense_1", "Dense", X1, Y2, {"units": 64, "activation": "relu"}),
        N("BatchNorm_1", "BatchNorm", X2, Y2),
        N("Dense_2", "Dense", X3, Y2, {"units": 3, "activation": "softmax"}),
        N("Optimizer_1", "Optimizer", X4, Y2, {"type": "adam", "learning_rate": 0.001}),
        N("LossFunction_1", "LossFunction", X4, Y3, {"type": "sparse_categorical_crossentropy"}),
        N("Trainer_1", "Trainer", X4, Y1, {"epochs": 30, "batch_size": 16, "framework": "pytorch"}),
        N("LossCurve_1", "LossCurve", X5, Y1),
    ],
    "edges": [
        E("CSVLoader_1", "StandardScaler_1", "features", "input"),
        E("StandardScaler_1", "TrainValSplit_1", "output", "features"),
        E("CSVLoader_1", "TrainValSplit_1", "labels", "labels"),
        E("Dense_1", "BatchNorm_1", "output", "input"),
        E("BatchNorm_1", "Dense_2", "output", "input"),
        E("Dense_2", "Trainer_1", "output", "layers"),
        E("TrainValSplit_1", "Trainer_1", "train_features", "train_features"),
        E("TrainValSplit_1", "Trainer_1", "train_labels", "train_labels"),
        E("TrainValSplit_1", "Trainer_1", "val_features", "val_features"),
        E("TrainValSplit_1", "Trainer_1", "val_labels", "val_labels"),
        E("Optimizer_1", "Trainer_1", "optimizer_config", "optimizer_config"),
        E("LossFunction_1", "Trainer_1", "loss_config", "loss_config"),
        E("Trainer_1", "LossCurve_1", "history", "history"),
    ]
})

# ============================================================
# 113: Optimizer 비교 (SGD vs Adam)
# ============================================================
TUTORIALS.append({
    "meta": {
        "id": 113, "title": "Compare SGD vs Adam",
        "title_ko": "SGD vs Adam 옵티마이저 비교",
        "category": "tutorial", "difficulty": "beginner",
        "description": "Try SGD optimizer and compare it with Adam.",
        "description_ko": "Optimizer 노드의 type을 sgd로 바꿔 Adam과 학습 속도 차이를 비교합니다.",
        "tags": ["optimizer", "sgd", "adam", "비교"],
        "estimated_time": "1분",
        "learning_objectives": [
            "Optimizer 노드의 type 파라미터 변경법",
            "SGD는 단순하지만 느림, Adam은 적응적으로 빠름",
            "learning_rate 조절의 효과"
        ],
        "guide_steps": [
            {"step": 1, "node": "Optimizer_1", "text": "현재 SGD로 설정되어 있습니다. 학습 후 Loss를 관찰하세요."},
            {"step": 2, "node": "Trainer_1", "text": "학습을 실행합니다."},
            {"step": 3, "node": "LossCurve_1", "text": "Loss가 천천히 내려갑니다. Optimizer를 adam으로 바꿔 다시 실행해보세요!"}
        ],
        "requires_phase": 1
    },
    "nodes": [
        N("CSVLoader_1", "CSVLoader", X1, Y1, {"file_path": "datasets/iris.csv", "target_column": "species"}),
        N("StandardScaler_1", "StandardScaler", X2, Y1),
        N("TrainValSplit_1", "TrainValSplit", X3, Y1, {"val_ratio": 0.2, "random_seed": 42}),
        N("Dense_1", "Dense", X1, Y2, {"units": 64, "activation": "relu"}),
        N("Dense_2", "Dense", X2, Y2, {"units": 3, "activation": "softmax"}),
        N("Optimizer_1", "Optimizer", X3, Y2, {"type": "sgd", "learning_rate": 0.01}),
        N("LossFunction_1", "LossFunction", X3, Y3, {"type": "sparse_categorical_crossentropy"}),
        N("Trainer_1", "Trainer", X4, Y1, {"epochs": 50, "batch_size": 16, "framework": "pytorch"}),
        N("LossCurve_1", "LossCurve", X5, Y1),
    ],
    "edges": [
        E("CSVLoader_1", "StandardScaler_1", "features", "input"),
        E("StandardScaler_1", "TrainValSplit_1", "output", "features"),
        E("CSVLoader_1", "TrainValSplit_1", "labels", "labels"),
        E("Dense_1", "Dense_2", "output", "input"),
        E("Dense_2", "Trainer_1", "output", "layers"),
        E("TrainValSplit_1", "Trainer_1", "train_features", "train_features"),
        E("TrainValSplit_1", "Trainer_1", "train_labels", "train_labels"),
        E("TrainValSplit_1", "Trainer_1", "val_features", "val_features"),
        E("TrainValSplit_1", "Trainer_1", "val_labels", "val_labels"),
        E("Optimizer_1", "Trainer_1", "optimizer_config", "optimizer_config"),
        E("LossFunction_1", "Trainer_1", "loss_config", "loss_config"),
        E("Trainer_1", "LossCurve_1", "history", "history"),
    ]
})

# ============================================================
# 114: EarlyStopping 사용법
# ============================================================
TUTORIALS.append({
    "meta": {
        "id": 114, "title": "Early Stopping",
        "title_ko": "EarlyStopping으로 자동 종료",
        "category": "tutorial", "difficulty": "beginner",
        "description": "Stop training automatically when validation loss stops improving.",
        "description_ko": "EarlyStopping 노드로 검증 Loss가 개선되지 않으면 자동으로 학습을 종료합니다.",
        "tags": ["early-stopping", "콜백", "과적합"],
        "estimated_time": "1분",
        "learning_objectives": [
            "patience: 몇 epoch 동안 개선 없으면 종료할지",
            "monitor: 어떤 지표를 감시할지 (val_loss)",
            "불필요한 학습을 방지하여 시간 절약"
        ],
        "guide_steps": [
            {"step": 1, "node": "EarlyStopping_1", "text": "patience=5: 5 epoch 동안 val_loss가 줄지 않으면 학습을 멈춥니다."},
            {"step": 2, "node": "Trainer_1", "text": "epochs=100이지만 EarlyStopping이 있으므로 일찍 끝날 수 있습니다."},
            {"step": 3, "node": "LossCurve_1", "text": "학습이 자동으로 멈춘 시점을 확인하세요."}
        ],
        "requires_phase": 1
    },
    "nodes": [
        N("CSVLoader_1", "CSVLoader", X1, Y1, {"file_path": "datasets/iris.csv", "target_column": "species"}),
        N("StandardScaler_1", "StandardScaler", X2, Y1),
        N("TrainValSplit_1", "TrainValSplit", X3, Y1, {"val_ratio": 0.2, "random_seed": 42}),
        N("Dense_1", "Dense", X1, Y2, {"units": 64, "activation": "relu"}),
        N("Dense_2", "Dense", X2, Y2, {"units": 3, "activation": "softmax"}),
        N("Optimizer_1", "Optimizer", X3, Y2, {"type": "adam", "learning_rate": 0.001}),
        N("LossFunction_1", "LossFunction", X3, Y3, {"type": "sparse_categorical_crossentropy"}),
        N("EarlyStopping_1", "EarlyStopping", X4, Y2, {"patience": 5, "monitor": "val_loss"}),
        N("Trainer_1", "Trainer", X4, Y1, {"epochs": 100, "batch_size": 16, "framework": "pytorch"}),
        N("LossCurve_1", "LossCurve", X5, Y1),
    ],
    "edges": [
        E("CSVLoader_1", "StandardScaler_1", "features", "input"),
        E("StandardScaler_1", "TrainValSplit_1", "output", "features"),
        E("CSVLoader_1", "TrainValSplit_1", "labels", "labels"),
        E("Dense_1", "Dense_2", "output", "input"),
        E("Dense_2", "Trainer_1", "output", "layers"),
        E("TrainValSplit_1", "Trainer_1", "train_features", "train_features"),
        E("TrainValSplit_1", "Trainer_1", "train_labels", "train_labels"),
        E("TrainValSplit_1", "Trainer_1", "val_features", "val_features"),
        E("TrainValSplit_1", "Trainer_1", "val_labels", "val_labels"),
        E("Optimizer_1", "Trainer_1", "optimizer_config", "optimizer_config"),
        E("LossFunction_1", "Trainer_1", "loss_config", "loss_config"),
        E("EarlyStopping_1", "Trainer_1", "callback_config", "early_stopping"),
        E("Trainer_1", "LossCurve_1", "history", "history"),
    ]
})

# ============================================================
# 115: ModelSummary 보기
# ============================================================
TUTORIALS.append({
    "meta": {
        "id": 115, "title": "View Model Summary",
        "title_ko": "ModelSummary로 모델 구조 확인",
        "category": "tutorial", "difficulty": "beginner",
        "description": "See the model architecture and parameter count using ModelSummary.",
        "description_ko": "ModelSummary 노드로 모델의 레이어 구조와 파라미터 수를 확인합니다.",
        "tags": ["model-summary", "구조", "시각화"],
        "estimated_time": "1분",
        "learning_objectives": [
            "ModelSummary로 총 파라미터 수 확인",
            "각 레이어의 출력 shape 이해",
            "학습 가능 파라미터 vs 고정 파라미터"
        ],
        "guide_steps": [
            {"step": 1, "node": "Trainer_1", "text": "학습을 실행합니다."},
            {"step": 2, "node": "ModelSummary_1", "text": "모델의 레이어 구성, 파라미터 수가 표시됩니다."}
        ],
        "requires_phase": 1
    },
    "nodes": [
        N("CSVLoader_1", "CSVLoader", X1, Y1, {"file_path": "datasets/iris.csv", "target_column": "species"}),
        N("StandardScaler_1", "StandardScaler", X2, Y1),
        N("TrainValSplit_1", "TrainValSplit", X3, Y1, {"val_ratio": 0.2, "random_seed": 42}),
        N("Dense_1", "Dense", X1, Y2, {"units": 64, "activation": "relu"}),
        N("Dense_2", "Dense", X2, Y2, {"units": 32, "activation": "relu"}),
        N("Dense_3", "Dense", X3, Y2, {"units": 3, "activation": "softmax"}),
        N("Optimizer_1", "Optimizer", X4, Y2, {"type": "adam", "learning_rate": 0.001}),
        N("LossFunction_1", "LossFunction", X4, Y3, {"type": "sparse_categorical_crossentropy"}),
        N("Trainer_1", "Trainer", X4, Y1, {"epochs": 10, "batch_size": 16, "framework": "pytorch"}),
        N("ModelSummary_1", "ModelSummary", X5, Y1),
    ],
    "edges": [
        E("CSVLoader_1", "StandardScaler_1", "features", "input"),
        E("StandardScaler_1", "TrainValSplit_1", "output", "features"),
        E("CSVLoader_1", "TrainValSplit_1", "labels", "labels"),
        E("Dense_1", "Dense_2", "output", "input"),
        E("Dense_2", "Dense_3", "output", "input"),
        E("Dense_3", "Trainer_1", "output", "layers"),
        E("TrainValSplit_1", "Trainer_1", "train_features", "train_features"),
        E("TrainValSplit_1", "Trainer_1", "train_labels", "train_labels"),
        E("TrainValSplit_1", "Trainer_1", "val_features", "val_features"),
        E("TrainValSplit_1", "Trainer_1", "val_labels", "val_labels"),
        E("Optimizer_1", "Trainer_1", "optimizer_config", "optimizer_config"),
        E("LossFunction_1", "Trainer_1", "loss_config", "loss_config"),
        E("Trainer_1", "ModelSummary_1", "model", "model"),
    ]
})

# ============================================================
# 116: ConfusionMatrix 보기
# ============================================================
TUTORIALS.append({
    "meta": {
        "id": 116, "title": "View Confusion Matrix",
        "title_ko": "혼동 행렬(Confusion Matrix) 확인",
        "category": "tutorial", "difficulty": "beginner",
        "description": "Visualize classification results with a Confusion Matrix heatmap.",
        "description_ko": "ConfusionMatrix 노드로 어떤 클래스를 잘 맞추고 어디서 실수하는지 확인합니다.",
        "tags": ["confusion-matrix", "평가", "시각화"],
        "estimated_time": "1~2분",
        "learning_objectives": [
            "혼동 행렬 히트맵 읽는 법",
            "대각선이 높으면 정확한 분류",
            "오분류 패턴 분석"
        ],
        "guide_steps": [
            {"step": 1, "node": "Trainer_1", "text": "모델을 학습합니다."},
            {"step": 2, "node": "ConfusionMatrix_1", "text": "학습 후 혼동 행렬이 생성됩니다. 대각선 값이 높을수록 좋습니다."}
        ],
        "requires_phase": 1
    },
    "nodes": [
        N("CSVLoader_1", "CSVLoader", X1, Y1, {"file_path": "datasets/iris.csv", "target_column": "species"}),
        N("StandardScaler_1", "StandardScaler", X2, Y1),
        N("TrainValSplit_1", "TrainValSplit", X3, Y1, {"val_ratio": 0.2, "random_seed": 42}),
        N("Dense_1", "Dense", X1, Y2, {"units": 64, "activation": "relu"}),
        N("Dense_2", "Dense", X2, Y2, {"units": 3, "activation": "softmax"}),
        N("Optimizer_1", "Optimizer", X3, Y2, {"type": "adam", "learning_rate": 0.001}),
        N("LossFunction_1", "LossFunction", X3, Y3, {"type": "sparse_categorical_crossentropy"}),
        N("Trainer_1", "Trainer", X4, Y1, {"epochs": 30, "batch_size": 16, "framework": "pytorch"}),
        N("ConfusionMatrix_1", "ConfusionMatrix", X5, Y1),
    ],
    "edges": [
        E("CSVLoader_1", "StandardScaler_1", "features", "input"),
        E("StandardScaler_1", "TrainValSplit_1", "output", "features"),
        E("CSVLoader_1", "TrainValSplit_1", "labels", "labels"),
        E("Dense_1", "Dense_2", "output", "input"),
        E("Dense_2", "Trainer_1", "output", "layers"),
        E("TrainValSplit_1", "Trainer_1", "train_features", "train_features"),
        E("TrainValSplit_1", "Trainer_1", "train_labels", "train_labels"),
        E("TrainValSplit_1", "Trainer_1", "val_features", "val_features"),
        E("TrainValSplit_1", "Trainer_1", "val_labels", "val_labels"),
        E("Optimizer_1", "Trainer_1", "optimizer_config", "optimizer_config"),
        E("LossFunction_1", "Trainer_1", "loss_config", "loss_config"),
        E("Trainer_1", "ConfusionMatrix_1", "model", "model"),
        E("TrainValSplit_1", "ConfusionMatrix_1", "val_features", "features"),
        E("TrainValSplit_1", "ConfusionMatrix_1", "val_labels", "labels"),
    ]
})

# ============================================================
# 117: Classification Metrics 보기
# ============================================================
TUTORIALS.append({
    "meta": {
        "id": 117, "title": "Classification Metrics",
        "title_ko": "분류 지표(Precision/Recall/F1) 확인",
        "category": "tutorial", "difficulty": "beginner",
        "description": "View Accuracy, Precision, Recall, and F1 scores.",
        "description_ko": "ClassificationMetrics 노드로 Accuracy, Precision, Recall, F1 점수를 확인합니다.",
        "tags": ["metrics", "precision", "recall", "f1"],
        "estimated_time": "1~2분",
        "learning_objectives": [
            "Accuracy: 전체 정확도",
            "Precision: 양성 예측 중 실제 양성 비율",
            "Recall: 실제 양성 중 찾아낸 비율",
            "F1: Precision과 Recall의 조화 평균"
        ],
        "guide_steps": [
            {"step": 1, "node": "Trainer_1", "text": "모델을 학습합니다."},
            {"step": 2, "node": "ClassificationMetrics_1", "text": "Accuracy, Precision, Recall, F1 점수가 표시됩니다."}
        ],
        "requires_phase": 1
    },
    "nodes": [
        N("CSVLoader_1", "CSVLoader", X1, Y1, {"file_path": "datasets/iris.csv", "target_column": "species"}),
        N("StandardScaler_1", "StandardScaler", X2, Y1),
        N("TrainValSplit_1", "TrainValSplit", X3, Y1, {"val_ratio": 0.2, "random_seed": 42}),
        N("Dense_1", "Dense", X1, Y2, {"units": 64, "activation": "relu"}),
        N("Dense_2", "Dense", X2, Y2, {"units": 3, "activation": "softmax"}),
        N("Optimizer_1", "Optimizer", X3, Y2, {"type": "adam", "learning_rate": 0.001}),
        N("LossFunction_1", "LossFunction", X3, Y3, {"type": "sparse_categorical_crossentropy"}),
        N("Trainer_1", "Trainer", X4, Y1, {"epochs": 30, "batch_size": 16, "framework": "pytorch"}),
        N("ClassificationMetrics_1", "ClassificationMetrics", X5, Y1),
    ],
    "edges": [
        E("CSVLoader_1", "StandardScaler_1", "features", "input"),
        E("StandardScaler_1", "TrainValSplit_1", "output", "features"),
        E("CSVLoader_1", "TrainValSplit_1", "labels", "labels"),
        E("Dense_1", "Dense_2", "output", "input"),
        E("Dense_2", "Trainer_1", "output", "layers"),
        E("TrainValSplit_1", "Trainer_1", "train_features", "train_features"),
        E("TrainValSplit_1", "Trainer_1", "train_labels", "train_labels"),
        E("TrainValSplit_1", "Trainer_1", "val_features", "val_features"),
        E("TrainValSplit_1", "Trainer_1", "val_labels", "val_labels"),
        E("Optimizer_1", "Trainer_1", "optimizer_config", "optimizer_config"),
        E("LossFunction_1", "Trainer_1", "loss_config", "loss_config"),
        E("Trainer_1", "ClassificationMetrics_1", "model", "model"),
        E("TrainValSplit_1", "ClassificationMetrics_1", "val_features", "features"),
        E("TrainValSplit_1", "ClassificationMetrics_1", "val_labels", "labels"),
    ]
})

# ============================================================
# 118: 이진 분류 (Binary Classification)
# ============================================================
TUTORIALS.append({
    "meta": {
        "id": 118, "title": "Binary Classification Basics",
        "title_ko": "이진 분류 기초 (생존 예측)",
        "category": "tutorial", "difficulty": "beginner",
        "description": "Build a binary classifier for Titanic survival prediction.",
        "description_ko": "타이타닉 데이터로 생존/사망 이진 분류를 학습합니다. sigmoid 출력을 사용합니다.",
        "tags": ["binary", "sigmoid", "이진분류"],
        "estimated_time": "1~2분",
        "learning_objectives": [
            "이진 분류에서는 출력 뉴런 1개 + sigmoid 사용",
            "binary_crossentropy 손실 함수",
            "LabelEncoder로 문자열 전처리"
        ],
        "guide_steps": [
            {"step": 1, "node": "CSVLoader_1", "text": "타이타닉 데이터를 로드합니다."},
            {"step": 2, "node": "LabelEncoder_1", "text": "문자열 컬럼을 숫자로 변환합니다."},
            {"step": 3, "node": "Dense_2", "text": "출력 1개 + sigmoid: 생존 확률을 0~1로 출력합니다."},
            {"step": 4, "node": "LossFunction_1", "text": "binary_crossentropy: 이진 분류의 표준 손실 함수입니다."}
        ],
        "requires_phase": 1
    },
    "nodes": [
        N("CSVLoader_1", "CSVLoader", X1, Y1, {"file_path": "datasets/titanic.csv", "target_column": "survived"}),
        N("LabelEncoder_1", "LabelEncoder", X2, Y1),
        N("StandardScaler_1", "StandardScaler", X3, Y1),
        N("TrainValSplit_1", "TrainValSplit", X4, Y1, {"val_ratio": 0.2, "random_seed": 42}),
        N("Dense_1", "Dense", X1, Y2, {"units": 32, "activation": "relu"}),
        N("Dense_2", "Dense", X2, Y2, {"units": 1, "activation": "sigmoid"}),
        N("Optimizer_1", "Optimizer", X3, Y2, {"type": "adam", "learning_rate": 0.001}),
        N("LossFunction_1", "LossFunction", X3, Y3, {"type": "binary_crossentropy"}),
        N("Trainer_1", "Trainer", X5, Y1, {"epochs": 30, "batch_size": 16, "framework": "pytorch"}),
        N("LossCurve_1", "LossCurve", X6, Y1),
    ],
    "edges": [
        E("CSVLoader_1", "LabelEncoder_1", "features", "input"),
        E("LabelEncoder_1", "StandardScaler_1", "output", "input"),
        E("StandardScaler_1", "TrainValSplit_1", "output", "features"),
        E("CSVLoader_1", "TrainValSplit_1", "labels", "labels"),
        E("Dense_1", "Dense_2", "output", "input"),
        E("Dense_2", "Trainer_1", "output", "layers"),
        E("TrainValSplit_1", "Trainer_1", "train_features", "train_features"),
        E("TrainValSplit_1", "Trainer_1", "train_labels", "train_labels"),
        E("TrainValSplit_1", "Trainer_1", "val_features", "val_features"),
        E("TrainValSplit_1", "Trainer_1", "val_labels", "val_labels"),
        E("Optimizer_1", "Trainer_1", "optimizer_config", "optimizer_config"),
        E("LossFunction_1", "Trainer_1", "loss_config", "loss_config"),
        E("Trainer_1", "LossCurve_1", "history", "history"),
    ]
})

# ============================================================
# 119: 회귀 문제 기초 (MSE Loss)
# ============================================================
TUTORIALS.append({
    "meta": {
        "id": 119, "title": "Regression Basics (MSE Loss)",
        "title_ko": "회귀 문제 기초 (MSE 손실 함수)",
        "category": "tutorial", "difficulty": "beginner",
        "description": "Use MSE loss and linear output for a regression task.",
        "description_ko": "회귀 문제에서는 MSE 손실 함수와 활성화 없는 출력층을 사용합니다.",
        "tags": ["regression", "mse", "회귀기초"],
        "estimated_time": "1~2분",
        "learning_objectives": [
            "회귀 문제에서는 activation='linear' (없음)",
            "MSE(Mean Squared Error) 손실 함수",
            "분류 vs 회귀의 차이"
        ],
        "guide_steps": [
            {"step": 1, "node": "CSVLoader_1", "text": "주택 가격 데이터를 로드합니다."},
            {"step": 2, "node": "Dense_2", "text": "출력 1개, 활성화 함수 없음(linear). 연속 숫자를 예측합니다."},
            {"step": 3, "node": "LossFunction_1", "text": "MSE: 예측값과 실제값 차이의 제곱 평균입니다."}
        ],
        "requires_phase": 1
    },
    "nodes": [
        N("CSVLoader_1", "CSVLoader", X1, Y1, {"file_path": "datasets/boston_housing.csv", "target_column": "medv"}),
        N("StandardScaler_1", "StandardScaler", X2, Y1),
        N("TrainValSplit_1", "TrainValSplit", X3, Y1, {"val_ratio": 0.2, "random_seed": 42}),
        N("Dense_1", "Dense", X1, Y2, {"units": 64, "activation": "relu"}),
        N("Dense_2", "Dense", X2, Y2, {"units": 1, "activation": "linear"}),
        N("Optimizer_1", "Optimizer", X3, Y2, {"type": "adam", "learning_rate": 0.001}),
        N("LossFunction_1", "LossFunction", X3, Y3, {"type": "mse"}),
        N("Trainer_1", "Trainer", X4, Y1, {"epochs": 50, "batch_size": 16, "framework": "pytorch"}),
        N("LossCurve_1", "LossCurve", X5, Y1),
    ],
    "edges": [
        E("CSVLoader_1", "StandardScaler_1", "features", "input"),
        E("StandardScaler_1", "TrainValSplit_1", "output", "features"),
        E("CSVLoader_1", "TrainValSplit_1", "labels", "labels"),
        E("Dense_1", "Dense_2", "output", "input"),
        E("Dense_2", "Trainer_1", "output", "layers"),
        E("TrainValSplit_1", "Trainer_1", "train_features", "train_features"),
        E("TrainValSplit_1", "Trainer_1", "train_labels", "train_labels"),
        E("TrainValSplit_1", "Trainer_1", "val_features", "val_features"),
        E("TrainValSplit_1", "Trainer_1", "val_labels", "val_labels"),
        E("Optimizer_1", "Trainer_1", "optimizer_config", "optimizer_config"),
        E("LossFunction_1", "Trainer_1", "loss_config", "loss_config"),
        E("Trainer_1", "LossCurve_1", "history", "history"),
    ]
})

# ============================================================
# 120: 이미지 로드 (ImageFolder)
# ============================================================
TUTORIALS.append({
    "meta": {
        "id": 120, "title": "Load Image Dataset",
        "title_ko": "이미지 폴더 데이터 로드",
        "category": "tutorial", "difficulty": "beginner",
        "description": "Load images from folders where each subfolder is a class.",
        "description_ko": "ImageFolder 노드로 폴더 구조의 이미지 데이터셋을 로드합니다.",
        "tags": ["image", "image-folder", "데이터로드"],
        "estimated_time": "30초",
        "learning_objectives": [
            "ImageFolder: 하위폴더명 = 클래스명",
            "image_size로 이미지 크기 통일",
            "이미지 데이터를 DataInspector로 확인"
        ],
        "guide_steps": [
            {"step": 1, "node": "ImageFolder_1", "text": "dogs_vs_cats 폴더를 로드합니다. cat/, dog/ 하위폴더가 각각 하나의 클래스입니다."},
            {"step": 2, "node": "DataInspector_1", "text": "로드된 이미지의 shape, 클래스 수, 샘플 수를 확인합니다."}
        ],
        "requires_phase": 1
    },
    "nodes": [
        N("ImageFolder_1", "ImageFolder", X1, Y1, {"folder_path": "datasets/dogs_vs_cats", "image_size": 64}),
        N("DataInspector_1", "DataInspector", X2, Y1),
    ],
    "edges": [
        E("ImageFolder_1", "DataInspector_1", "features", "input"),
    ]
})

# ============================================================
# 121: Conv2D 기초
# ============================================================
TUTORIALS.append({
    "meta": {
        "id": 121, "title": "First Conv2D Network",
        "title_ko": "Conv2D 첫 사용 (이미지 분류)",
        "category": "tutorial", "difficulty": "intermediate",
        "description": "Build a simple CNN with Conv2D → Flatten → Dense.",
        "description_ko": "Conv2D → Flatten → Dense 구조로 첫 CNN을 만들어 이미지를 분류합니다.",
        "tags": ["conv2d", "cnn", "flatten", "이미지"],
        "estimated_time": "2분",
        "learning_objectives": [
            "Conv2D: 이미지에서 특징(엣지, 패턴)을 추출",
            "Flatten: 2D 특성맵을 1D로 펼침 (Dense 연결 전 필수)",
            "CNN이 Dense보다 이미지에 효과적인 이유"
        ],
        "guide_steps": [
            {"step": 1, "node": "ImageFolder_1", "text": "MNIST 이미지를 로드합니다."},
            {"step": 2, "node": "Conv2D_1", "text": "32개의 3x3 필터로 이미지에서 패턴을 추출합니다."},
            {"step": 3, "node": "Flatten_1", "text": "2D 특성맵을 1D 벡터로 펼칩니다. Dense에 연결하기 위해 필수입니다."},
            {"step": 4, "node": "Dense_1", "text": "10개 출력 (0~9 숫자). softmax로 확률을 출력합니다."}
        ],
        "requires_phase": 1
    },
    "nodes": [
        N("ImageFolder_1", "ImageFolder", X1, Y1, {"folder_path": "datasets/mnist_images", "image_size": 28}),
        N("TrainValSplit_1", "TrainValSplit", X2, Y1, {"val_ratio": 0.2, "random_seed": 42}),
        N("Conv2D_1", "Conv2D", X1, Y2, {"filters": 32, "kernel_size": 3, "activation": "relu"}),
        N("Flatten_1", "Flatten", X2, Y2),
        N("Dense_1", "Dense", X3, Y2, {"units": 10, "activation": "softmax"}),
        N("Optimizer_1", "Optimizer", X4, Y2, {"type": "adam", "learning_rate": 0.001}),
        N("LossFunction_1", "LossFunction", X4, Y3, {"type": "sparse_categorical_crossentropy"}),
        N("Trainer_1", "Trainer", X3, Y1, {"epochs": 5, "batch_size": 32, "framework": "pytorch"}),
        N("LossCurve_1", "LossCurve", X4, Y1),
    ],
    "edges": [
        E("ImageFolder_1", "TrainValSplit_1", "features", "features"),
        E("ImageFolder_1", "TrainValSplit_1", "labels", "labels"),
        E("Conv2D_1", "Flatten_1", "output", "input"),
        E("Flatten_1", "Dense_1", "output", "input"),
        E("Dense_1", "Trainer_1", "output", "layers"),
        E("TrainValSplit_1", "Trainer_1", "train_features", "train_features"),
        E("TrainValSplit_1", "Trainer_1", "train_labels", "train_labels"),
        E("TrainValSplit_1", "Trainer_1", "val_features", "val_features"),
        E("TrainValSplit_1", "Trainer_1", "val_labels", "val_labels"),
        E("Optimizer_1", "Trainer_1", "optimizer_config", "optimizer_config"),
        E("LossFunction_1", "Trainer_1", "loss_config", "loss_config"),
        E("Trainer_1", "LossCurve_1", "history", "history"),
    ]
})

# ============================================================
# 122: MaxPooling2D 사용법
# ============================================================
TUTORIALS.append({
    "meta": {
        "id": 122, "title": "Conv2D + MaxPooling",
        "title_ko": "MaxPooling2D로 특성맵 축소",
        "category": "tutorial", "difficulty": "intermediate",
        "description": "Add MaxPooling2D after Conv2D to reduce spatial dimensions.",
        "description_ko": "Conv2D 뒤에 MaxPooling2D를 추가하여 특성맵 크기를 절반으로 줄입니다.",
        "tags": ["maxpooling", "conv2d", "cnn"],
        "estimated_time": "2분",
        "learning_objectives": [
            "MaxPooling2D: 2x2 영역에서 최댓값만 선택",
            "특성맵 크기를 줄여 계산량 감소",
            "Conv2D → MaxPooling2D 반복 패턴"
        ],
        "guide_steps": [
            {"step": 1, "node": "Conv2D_1", "text": "32개 필터로 특징을 추출합니다."},
            {"step": 2, "node": "MaxPooling2D_1", "text": "2x2 풀링으로 크기를 절반으로 줄입니다. 중요한 특징만 남깁니다."},
            {"step": 3, "node": "Flatten_1", "text": "1D로 펼칩니다."},
            {"step": 4, "node": "Dense_1", "text": "10개 클래스를 분류합니다."}
        ],
        "requires_phase": 1
    },
    "nodes": [
        N("ImageFolder_1", "ImageFolder", X1, Y1, {"folder_path": "datasets/mnist_images", "image_size": 28}),
        N("TrainValSplit_1", "TrainValSplit", X2, Y1, {"val_ratio": 0.2, "random_seed": 42}),
        N("Conv2D_1", "Conv2D", X1, Y2, {"filters": 32, "kernel_size": 3, "activation": "relu"}),
        N("MaxPooling2D_1", "MaxPooling2D", X2, Y2, {"pool_size": 2}),
        N("Flatten_1", "Flatten", X3, Y2),
        N("Dense_1", "Dense", X4, Y2, {"units": 10, "activation": "softmax"}),
        N("Optimizer_1", "Optimizer", X5, Y2, {"type": "adam", "learning_rate": 0.001}),
        N("LossFunction_1", "LossFunction", X5, Y3, {"type": "sparse_categorical_crossentropy"}),
        N("Trainer_1", "Trainer", X3, Y1, {"epochs": 5, "batch_size": 32, "framework": "pytorch"}),
        N("AccuracyCurve_1", "AccuracyCurve", X4, Y1),
    ],
    "edges": [
        E("ImageFolder_1", "TrainValSplit_1", "features", "features"),
        E("ImageFolder_1", "TrainValSplit_1", "labels", "labels"),
        E("Conv2D_1", "MaxPooling2D_1", "output", "input"),
        E("MaxPooling2D_1", "Flatten_1", "output", "input"),
        E("Flatten_1", "Dense_1", "output", "input"),
        E("Dense_1", "Trainer_1", "output", "layers"),
        E("TrainValSplit_1", "Trainer_1", "train_features", "train_features"),
        E("TrainValSplit_1", "Trainer_1", "train_labels", "train_labels"),
        E("TrainValSplit_1", "Trainer_1", "val_features", "val_features"),
        E("TrainValSplit_1", "Trainer_1", "val_labels", "val_labels"),
        E("Optimizer_1", "Trainer_1", "optimizer_config", "optimizer_config"),
        E("LossFunction_1", "Trainer_1", "loss_config", "loss_config"),
        E("Trainer_1", "AccuracyCurve_1", "history", "history"),
    ]
})

# ============================================================
# 123: GlobalAveragePooling2D
# ============================================================
TUTORIALS.append({
    "meta": {
        "id": 123, "title": "GlobalAveragePooling2D",
        "title_ko": "GlobalAvgPool로 Flatten 대체",
        "category": "tutorial", "difficulty": "intermediate",
        "description": "Use GlobalAveragePooling2D instead of Flatten for lighter models.",
        "description_ko": "Flatten 대신 GlobalAveragePooling2D를 사용하여 파라미터 수를 크게 줄입니다.",
        "tags": ["global-avg-pool", "경량화", "cnn"],
        "estimated_time": "2분",
        "learning_objectives": [
            "GlobalAvgPool: 채널별 평균 → 파라미터 대폭 감소",
            "Flatten vs GlobalAvgPool 비교",
            "현대 CNN에서 GlobalAvgPool이 표준"
        ],
        "guide_steps": [
            {"step": 1, "node": "Conv2D_1", "text": "64개 필터로 특징을 추출합니다."},
            {"step": 2, "node": "GlobalAveragePooling2D_1", "text": "각 채널의 평균값만 남깁니다. 64개 특징 → 64차원 벡터."},
            {"step": 3, "node": "Dense_1", "text": "64차원에서 바로 분류합니다. Flatten보다 파라미터가 훨씬 적습니다."}
        ],
        "requires_phase": 1
    },
    "nodes": [
        N("ImageFolder_1", "ImageFolder", X1, Y1, {"folder_path": "datasets/mnist_images", "image_size": 28}),
        N("TrainValSplit_1", "TrainValSplit", X2, Y1, {"val_ratio": 0.2, "random_seed": 42}),
        N("Conv2D_1", "Conv2D", X1, Y2, {"filters": 64, "kernel_size": 3, "activation": "relu"}),
        N("GlobalAveragePooling2D_1", "GlobalAveragePooling2D", X2, Y2),
        N("Dense_1", "Dense", X3, Y2, {"units": 10, "activation": "softmax"}),
        N("Optimizer_1", "Optimizer", X4, Y2, {"type": "adam", "learning_rate": 0.001}),
        N("LossFunction_1", "LossFunction", X4, Y3, {"type": "sparse_categorical_crossentropy"}),
        N("Trainer_1", "Trainer", X3, Y1, {"epochs": 5, "batch_size": 32, "framework": "pytorch"}),
        N("ModelSummary_1", "ModelSummary", X4, Y1),
    ],
    "edges": [
        E("ImageFolder_1", "TrainValSplit_1", "features", "features"),
        E("ImageFolder_1", "TrainValSplit_1", "labels", "labels"),
        E("Conv2D_1", "GlobalAveragePooling2D_1", "output", "input"),
        E("GlobalAveragePooling2D_1", "Dense_1", "output", "input"),
        E("Dense_1", "Trainer_1", "output", "layers"),
        E("TrainValSplit_1", "Trainer_1", "train_features", "train_features"),
        E("TrainValSplit_1", "Trainer_1", "train_labels", "train_labels"),
        E("TrainValSplit_1", "Trainer_1", "val_features", "val_features"),
        E("TrainValSplit_1", "Trainer_1", "val_labels", "val_labels"),
        E("Optimizer_1", "Trainer_1", "optimizer_config", "optimizer_config"),
        E("LossFunction_1", "Trainer_1", "loss_config", "loss_config"),
        E("Trainer_1", "ModelSummary_1", "model", "model"),
    ]
})

# ============================================================
# 124: PCA 차원 축소
# ============================================================
TUTORIALS.append({
    "meta": {
        "id": 124, "title": "PCA Dimensionality Reduction",
        "title_ko": "PCA로 차원 축소하기",
        "category": "tutorial", "difficulty": "intermediate",
        "description": "Reduce feature dimensions with PCA before training.",
        "description_ko": "PCA 노드로 특성 차원을 줄여 학습 효율을 높입니다.",
        "tags": ["pca", "차원축소", "전처리"],
        "estimated_time": "1분",
        "learning_objectives": [
            "PCA: 고차원 데이터를 저차원으로 변환",
            "n_components로 유지할 차원 수 설정",
            "정보 손실 vs 계산 효율의 트레이드오프"
        ],
        "guide_steps": [
            {"step": 1, "node": "CSVLoader_1", "text": "와인 데이터(13개 특성)를 로드합니다."},
            {"step": 2, "node": "StandardScaler_1", "text": "PCA 전에 반드시 정규화해야 합니다."},
            {"step": 3, "node": "PCA_1", "text": "13차원 → 4차원으로 줄입니다. 주요 정보만 남깁니다."},
            {"step": 4, "node": "DataInspector_1", "text": "변환 후 4개 컬럼만 남았는지 확인합니다."}
        ],
        "requires_phase": 1
    },
    "nodes": [
        N("CSVLoader_1", "CSVLoader", X1, Y1, {"file_path": "datasets/wine.csv", "target_column": "class"}),
        N("StandardScaler_1", "StandardScaler", X2, Y1),
        N("PCA_1", "PCA", X3, Y1, {"n_components": 4}),
        N("DataInspector_1", "DataInspector", X4, Y1),
    ],
    "edges": [
        E("CSVLoader_1", "StandardScaler_1", "features", "input"),
        E("StandardScaler_1", "PCA_1", "output", "input"),
        E("PCA_1", "DataInspector_1"),
    ]
})

# ============================================================
# 125: 모델 저장하기
# ============================================================
TUTORIALS.append({
    "meta": {
        "id": 125, "title": "Save Trained Model",
        "title_ko": "학습된 모델 저장하기",
        "category": "tutorial", "difficulty": "beginner",
        "description": "Save a trained model to disk using ModelSave node.",
        "description_ko": "ModelSave 노드로 학습 완료된 모델을 파일로 저장합니다.",
        "tags": ["model-save", "저장", "내보내기"],
        "estimated_time": "1~2분",
        "learning_objectives": [
            "ModelSave로 학습된 가중치를 파일로 저장",
            "저장된 모델을 나중에 불러와 재사용 가능",
            ".pth(PyTorch) / .keras(TensorFlow) 파일 형식"
        ],
        "guide_steps": [
            {"step": 1, "node": "Trainer_1", "text": "모델을 학습합니다."},
            {"step": 2, "node": "ModelSave_1", "text": "학습된 모델이 지정된 경로에 저장됩니다."}
        ],
        "requires_phase": 1
    },
    "nodes": [
        N("CSVLoader_1", "CSVLoader", X1, Y1, {"file_path": "datasets/iris.csv", "target_column": "species"}),
        N("StandardScaler_1", "StandardScaler", X2, Y1),
        N("TrainValSplit_1", "TrainValSplit", X3, Y1, {"val_ratio": 0.2, "random_seed": 42}),
        N("Dense_1", "Dense", X1, Y2, {"units": 32, "activation": "relu"}),
        N("Dense_2", "Dense", X2, Y2, {"units": 3, "activation": "softmax"}),
        N("Optimizer_1", "Optimizer", X3, Y2, {"type": "adam", "learning_rate": 0.001}),
        N("LossFunction_1", "LossFunction", X3, Y3, {"type": "sparse_categorical_crossentropy"}),
        N("Trainer_1", "Trainer", X4, Y1, {"epochs": 20, "batch_size": 16, "framework": "pytorch"}),
        N("ModelSave_1", "ModelSave", X5, Y1, {"save_path": "outputs/tutorial_model"}),
    ],
    "edges": [
        E("CSVLoader_1", "StandardScaler_1", "features", "input"),
        E("StandardScaler_1", "TrainValSplit_1", "output", "features"),
        E("CSVLoader_1", "TrainValSplit_1", "labels", "labels"),
        E("Dense_1", "Dense_2", "output", "input"),
        E("Dense_2", "Trainer_1", "output", "layers"),
        E("TrainValSplit_1", "Trainer_1", "train_features", "train_features"),
        E("TrainValSplit_1", "Trainer_1", "train_labels", "train_labels"),
        E("TrainValSplit_1", "Trainer_1", "val_features", "val_features"),
        E("TrainValSplit_1", "Trainer_1", "val_labels", "val_labels"),
        E("Optimizer_1", "Trainer_1", "optimizer_config", "optimizer_config"),
        E("LossFunction_1", "Trainer_1", "loss_config", "loss_config"),
        E("Trainer_1", "ModelSave_1", "model", "model"),
    ]
})

# ============================================================
# 126: Learning Rate 변경 효과
# ============================================================
TUTORIALS.append({
    "meta": {
        "id": 126, "title": "Learning Rate Effect",
        "title_ko": "Learning Rate 변경 실험",
        "category": "tutorial", "difficulty": "beginner",
        "description": "Experiment with different learning rates and observe the effect on training.",
        "description_ko": "Learning Rate를 바꿔가며 학습 속도와 안정성의 차이를 관찰합니다.",
        "tags": ["learning-rate", "실험", "하이퍼파라미터"],
        "estimated_time": "1~2분",
        "learning_objectives": [
            "높은 lr(0.1): 빠르지만 불안정, 발산 가능",
            "적당한 lr(0.001): 안정적으로 수렴",
            "낮은 lr(0.00001): 매우 느리게 학습",
            "lr 변경 → Optimizer 노드에서 learning_rate 수정"
        ],
        "guide_steps": [
            {"step": 1, "node": "Optimizer_1", "text": "현재 lr=0.1 (높음). 학습이 불안정할 수 있습니다."},
            {"step": 2, "node": "Trainer_1", "text": "학습을 실행합니다."},
            {"step": 3, "node": "LossCurve_1", "text": "Loss가 진동하거나 발산하면 lr이 너무 높은 것입니다. 0.001로 바꿔 다시 실행해보세요."}
        ],
        "requires_phase": 1
    },
    "nodes": [
        N("CSVLoader_1", "CSVLoader", X1, Y1, {"file_path": "datasets/iris.csv", "target_column": "species"}),
        N("StandardScaler_1", "StandardScaler", X2, Y1),
        N("TrainValSplit_1", "TrainValSplit", X3, Y1, {"val_ratio": 0.2, "random_seed": 42}),
        N("Dense_1", "Dense", X1, Y2, {"units": 64, "activation": "relu"}),
        N("Dense_2", "Dense", X2, Y2, {"units": 3, "activation": "softmax"}),
        N("Optimizer_1", "Optimizer", X3, Y2, {"type": "adam", "learning_rate": 0.1}),
        N("LossFunction_1", "LossFunction", X3, Y3, {"type": "sparse_categorical_crossentropy"}),
        N("Trainer_1", "Trainer", X4, Y1, {"epochs": 30, "batch_size": 16, "framework": "pytorch"}),
        N("LossCurve_1", "LossCurve", X5, Y1),
    ],
    "edges": [
        E("CSVLoader_1", "StandardScaler_1", "features", "input"),
        E("StandardScaler_1", "TrainValSplit_1", "output", "features"),
        E("CSVLoader_1", "TrainValSplit_1", "labels", "labels"),
        E("Dense_1", "Dense_2", "output", "input"),
        E("Dense_2", "Trainer_1", "output", "layers"),
        E("TrainValSplit_1", "Trainer_1", "train_features", "train_features"),
        E("TrainValSplit_1", "Trainer_1", "train_labels", "train_labels"),
        E("TrainValSplit_1", "Trainer_1", "val_features", "val_features"),
        E("TrainValSplit_1", "Trainer_1", "val_labels", "val_labels"),
        E("Optimizer_1", "Trainer_1", "optimizer_config", "optimizer_config"),
        E("LossFunction_1", "Trainer_1", "loss_config", "loss_config"),
        E("Trainer_1", "LossCurve_1", "history", "history"),
    ]
})

# ============================================================
# 127: Batch Size 변경 효과
# ============================================================
TUTORIALS.append({
    "meta": {
        "id": 127, "title": "Batch Size Effect",
        "title_ko": "Batch Size 변경 실험",
        "category": "tutorial", "difficulty": "beginner",
        "description": "Try different batch sizes and see how they affect training.",
        "description_ko": "Trainer 노드의 batch_size를 변경하며 학습 속도와 품질 차이를 관찰합니다.",
        "tags": ["batch-size", "실험", "하이퍼파라미터"],
        "estimated_time": "1분",
        "learning_objectives": [
            "작은 batch(4): 노이즈 많지만 일반화 가능성 높음",
            "큰 batch(128): 안정적이지만 메모리 더 필요",
            "일반적으로 16~64가 무난"
        ],
        "guide_steps": [
            {"step": 1, "node": "Trainer_1", "text": "batch_size=4로 설정되어 있습니다. 학습을 실행하세요."},
            {"step": 2, "node": "LossCurve_1", "text": "Loss가 들쑥날쑥할 수 있습니다. batch_size를 32로 바꿔 비교해보세요."}
        ],
        "requires_phase": 1
    },
    "nodes": [
        N("CSVLoader_1", "CSVLoader", X1, Y1, {"file_path": "datasets/iris.csv", "target_column": "species"}),
        N("StandardScaler_1", "StandardScaler", X2, Y1),
        N("TrainValSplit_1", "TrainValSplit", X3, Y1, {"val_ratio": 0.2, "random_seed": 42}),
        N("Dense_1", "Dense", X1, Y2, {"units": 64, "activation": "relu"}),
        N("Dense_2", "Dense", X2, Y2, {"units": 3, "activation": "softmax"}),
        N("Optimizer_1", "Optimizer", X3, Y2, {"type": "adam", "learning_rate": 0.001}),
        N("LossFunction_1", "LossFunction", X3, Y3, {"type": "sparse_categorical_crossentropy"}),
        N("Trainer_1", "Trainer", X4, Y1, {"epochs": 30, "batch_size": 4, "framework": "pytorch"}),
        N("LossCurve_1", "LossCurve", X5, Y1),
    ],
    "edges": [
        E("CSVLoader_1", "StandardScaler_1", "features", "input"),
        E("StandardScaler_1", "TrainValSplit_1", "output", "features"),
        E("CSVLoader_1", "TrainValSplit_1", "labels", "labels"),
        E("Dense_1", "Dense_2", "output", "input"),
        E("Dense_2", "Trainer_1", "output", "layers"),
        E("TrainValSplit_1", "Trainer_1", "train_features", "train_features"),
        E("TrainValSplit_1", "Trainer_1", "train_labels", "train_labels"),
        E("TrainValSplit_1", "Trainer_1", "val_features", "val_features"),
        E("TrainValSplit_1", "Trainer_1", "val_labels", "val_labels"),
        E("Optimizer_1", "Trainer_1", "optimizer_config", "optimizer_config"),
        E("LossFunction_1", "Trainer_1", "loss_config", "loss_config"),
        E("Trainer_1", "LossCurve_1", "history", "history"),
    ]
})

# ============================================================
# 128: Epoch 수 변경 효과
# ============================================================
TUTORIALS.append({
    "meta": {
        "id": 128, "title": "Epoch Count Effect",
        "title_ko": "Epoch 수 변경 실험",
        "category": "tutorial", "difficulty": "beginner",
        "description": "See underfitting (too few epochs) vs overfitting (too many).",
        "description_ko": "Epoch 수를 바꿔 과소적합(너무 적음)과 과적합(너무 많음)의 차이를 관찰합니다.",
        "tags": ["epoch", "과적합", "실험"],
        "estimated_time": "1분",
        "learning_objectives": [
            "epoch=5: 학습 부족 (과소적합)",
            "epoch=30: 적당한 학습",
            "epoch=200: 과적합 위험",
            "Val Loss가 올라가기 시작하면 과적합"
        ],
        "guide_steps": [
            {"step": 1, "node": "Trainer_1", "text": "현재 epochs=200. 과적합이 발생할 수 있습니다."},
            {"step": 2, "node": "LossCurve_1", "text": "Train Loss는 계속 내려가지만 Val Loss가 올라가면 과적합입니다."},
            {"step": 3, "node": "AccuracyCurve_1", "text": "Train Acc와 Val Acc 차이가 벌어지는지 확인하세요."}
        ],
        "requires_phase": 1
    },
    "nodes": [
        N("CSVLoader_1", "CSVLoader", X1, Y1, {"file_path": "datasets/iris.csv", "target_column": "species"}),
        N("StandardScaler_1", "StandardScaler", X2, Y1),
        N("TrainValSplit_1", "TrainValSplit", X3, Y1, {"val_ratio": 0.2, "random_seed": 42}),
        N("Dense_1", "Dense", X1, Y2, {"units": 128, "activation": "relu"}),
        N("Dense_2", "Dense", X2, Y2, {"units": 64, "activation": "relu"}),
        N("Dense_3", "Dense", X3, Y2, {"units": 3, "activation": "softmax"}),
        N("Optimizer_1", "Optimizer", X4, Y2, {"type": "adam", "learning_rate": 0.001}),
        N("LossFunction_1", "LossFunction", X4, Y3, {"type": "sparse_categorical_crossentropy"}),
        N("Trainer_1", "Trainer", X4, Y1, {"epochs": 200, "batch_size": 16, "framework": "pytorch"}),
        N("LossCurve_1", "LossCurve", X5, Y1),
        N("AccuracyCurve_1", "AccuracyCurve", X5, Y2),
    ],
    "edges": [
        E("CSVLoader_1", "StandardScaler_1", "features", "input"),
        E("StandardScaler_1", "TrainValSplit_1", "output", "features"),
        E("CSVLoader_1", "TrainValSplit_1", "labels", "labels"),
        E("Dense_1", "Dense_2", "output", "input"),
        E("Dense_2", "Dense_3", "output", "input"),
        E("Dense_3", "Trainer_1", "output", "layers"),
        E("TrainValSplit_1", "Trainer_1", "train_features", "train_features"),
        E("TrainValSplit_1", "Trainer_1", "train_labels", "train_labels"),
        E("TrainValSplit_1", "Trainer_1", "val_features", "val_features"),
        E("TrainValSplit_1", "Trainer_1", "val_labels", "val_labels"),
        E("Optimizer_1", "Trainer_1", "optimizer_config", "optimizer_config"),
        E("LossFunction_1", "Trainer_1", "loss_config", "loss_config"),
        E("Trainer_1", "LossCurve_1", "history", "history"),
        E("Trainer_1", "AccuracyCurve_1", "history", "history"),
    ]
})

# ============================================================
# 129: DataDistribution 히스토그램
# ============================================================
TUTORIALS.append({
    "meta": {
        "id": 129, "title": "View Data Distribution",
        "title_ko": "데이터 분포 히스토그램 보기",
        "category": "tutorial", "difficulty": "beginner",
        "description": "Visualize feature distributions with histograms and box plots.",
        "description_ko": "DataDistribution 노드로 각 특성의 분포를 히스토그램으로 시각화합니다.",
        "tags": ["distribution", "히스토그램", "시각화"],
        "estimated_time": "30초",
        "learning_objectives": [
            "데이터 분포를 시각적으로 확인하는 방법",
            "정규분포 vs 편향된 분포 구분",
            "이상치(outlier) 시각적 탐지"
        ],
        "guide_steps": [
            {"step": 1, "node": "CSVLoader_1", "text": "데이터를 로드합니다."},
            {"step": 2, "node": "DataDistribution_1", "text": "각 특성의 히스토그램이 표시됩니다. 분포 모양을 확인하세요."}
        ],
        "requires_phase": 1
    },
    "nodes": [
        N("CSVLoader_1", "CSVLoader", X1, Y1, {"file_path": "datasets/iris.csv", "target_column": "species"}),
        N("DataDistribution_1", "DataDistribution", X2, Y1),
    ],
    "edges": [
        E("CSVLoader_1", "DataDistribution_1", "features", "input"),
    ]
})

# ============================================================
# 130: 전체 파이프라인 종합 (입문)
# ============================================================
TUTORIALS.append({
    "meta": {
        "id": 130, "title": "Complete Beginner Pipeline",
        "title_ko": "입문 종합 파이프라인",
        "category": "tutorial", "difficulty": "beginner",
        "description": "A complete pipeline: load → preprocess → split → train → evaluate → visualize.",
        "description_ko": "데이터 로드 → 전처리 → 분할 → 학습 → 평가 → 시각화까지 전체 과정을 한 번에 체험합니다.",
        "tags": ["종합", "전체흐름", "입문"],
        "estimated_time": "2~3분",
        "learning_objectives": [
            "ML 파이프라인의 전체 흐름 이해",
            "각 단계별 노드의 역할 복습",
            "여러 시각화 노드를 동시에 연결"
        ],
        "guide_steps": [
            {"step": 1, "node": "CSVLoader_1", "text": "① 데이터 로드: Iris CSV를 불러옵니다."},
            {"step": 2, "node": "StandardScaler_1", "text": "② 전처리: 특성을 정규화합니다."},
            {"step": 3, "node": "TrainValSplit_1", "text": "③ 분할: 학습 80%, 검증 20%로 나눕니다."},
            {"step": 4, "node": "Dense_1", "text": "④ 모델: 64→32→3 Dense 네트워크입니다."},
            {"step": 5, "node": "Trainer_1", "text": "⑤ 학습: Run 버튼을 클릭하세요!"},
            {"step": 6, "node": "LossCurve_1", "text": "⑥ 시각화: Loss 곡선을 확인합니다."},
            {"step": 7, "node": "AccuracyCurve_1", "text": "⑦ 시각화: 정확도 곡선을 확인합니다."},
            {"step": 8, "node": "ConfusionMatrix_1", "text": "⑧ 평가: 혼동 행렬로 오분류를 분석합니다."},
            {"step": 9, "node": "ClassificationMetrics_1", "text": "⑨ 평가: 정밀도/재현율/F1 점수를 확인합니다."}
        ],
        "requires_phase": 1
    },
    "nodes": [
        N("CSVLoader_1", "CSVLoader", X1, Y1, {"file_path": "datasets/iris.csv", "target_column": "species"}),
        N("StandardScaler_1", "StandardScaler", X2, Y1),
        N("TrainValSplit_1", "TrainValSplit", X3, Y1, {"val_ratio": 0.2, "random_seed": 42}),
        N("Dense_1", "Dense", X1, Y2, {"units": 64, "activation": "relu"}),
        N("Dense_2", "Dense", X2, Y2, {"units": 32, "activation": "relu"}),
        N("Dense_3", "Dense", X3, Y2, {"units": 3, "activation": "softmax"}),
        N("Optimizer_1", "Optimizer", X4, Y2, {"type": "adam", "learning_rate": 0.001}),
        N("LossFunction_1", "LossFunction", X4, Y3, {"type": "sparse_categorical_crossentropy"}),
        N("Trainer_1", "Trainer", X4, Y1, {"epochs": 30, "batch_size": 16, "framework": "pytorch"}),
        N("LossCurve_1", "LossCurve", X5, Y1),
        N("AccuracyCurve_1", "AccuracyCurve", X5, Y2),
        N("ConfusionMatrix_1", "ConfusionMatrix", X6, Y1),
        N("ClassificationMetrics_1", "ClassificationMetrics", X6, Y2),
    ],
    "edges": [
        E("CSVLoader_1", "StandardScaler_1", "features", "input"),
        E("StandardScaler_1", "TrainValSplit_1", "output", "features"),
        E("CSVLoader_1", "TrainValSplit_1", "labels", "labels"),
        E("Dense_1", "Dense_2", "output", "input"),
        E("Dense_2", "Dense_3", "output", "input"),
        E("Dense_3", "Trainer_1", "output", "layers"),
        E("TrainValSplit_1", "Trainer_1", "train_features", "train_features"),
        E("TrainValSplit_1", "Trainer_1", "train_labels", "train_labels"),
        E("TrainValSplit_1", "Trainer_1", "val_features", "val_features"),
        E("TrainValSplit_1", "Trainer_1", "val_labels", "val_labels"),
        E("Optimizer_1", "Trainer_1", "optimizer_config", "optimizer_config"),
        E("LossFunction_1", "Trainer_1", "loss_config", "loss_config"),
        E("Trainer_1", "LossCurve_1", "history", "history"),
        E("Trainer_1", "AccuracyCurve_1", "history", "history"),
        E("Trainer_1", "ConfusionMatrix_1", "model", "model"),
        E("TrainValSplit_1", "ConfusionMatrix_1", "val_features", "features"),
        E("TrainValSplit_1", "ConfusionMatrix_1", "val_labels", "labels"),
        E("Trainer_1", "ClassificationMetrics_1", "model", "model"),
        E("TrainValSplit_1", "ClassificationMetrics_1", "val_features", "features"),
        E("TrainValSplit_1", "ClassificationMetrics_1", "val_labels", "labels"),
    ]
})


# ── 파일 저장 ──────────────────────────────────────
for tut in TUTORIALS:
    eid = tut["meta"]["id"]
    title = tut["meta"]["title"].lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    filename = f"{eid:03d}_{title}.json"
    filepath = os.path.join(DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(tut, f, ensure_ascii=False, indent=2)
    print(f"  Created: {filename}")

print(f"\nTotal: {len(TUTORIALS)} tutorial examples generated.")
