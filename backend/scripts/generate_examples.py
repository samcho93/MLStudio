"""예제 파이프라인 JSON 자동 생성 스크립트.

Usage:
    cd backend
    python scripts/generate_examples.py
"""

from __future__ import annotations

import json
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def pos(col: int, row: int) -> dict:
    """노드 위치 계산 (그리드 기반)."""
    return {"x": 50 + col * 260, "y": 80 + row * 140}


def make_example(
    id: int,
    title: str,
    title_ko: str,
    category: str,
    difficulty: str,
    description: str,
    description_ko: str,
    tags: list[str],
    estimated_time: str,
    dataset_info: dict,
    learning_objectives: list[str],
    guide_steps: list[dict],
    nodes: list[dict],
    edges: list[dict],
    requires_phase: int = 1,
) -> dict:
    return {
        "meta": {
            "id": id,
            "title": title,
            "title_ko": title_ko,
            "category": category,
            "difficulty": difficulty,
            "description": description,
            "description_ko": description_ko,
            "tags": tags,
            "estimated_time": estimated_time,
            "dataset_info": dataset_info,
            "learning_objectives": learning_objectives,
            "guide_steps": guide_steps,
            "requires_phase": requires_phase,
        },
        "nodes": nodes,
        "edges": edges,
    }


# ═══════════════════════════════════════════════════════
# Classification Examples
# ═══════════════════════════════════════════════════════

EXAMPLES: list[dict] = []

# ── 1. Iris ──────────────────────────────────────────
EXAMPLES.append(make_example(
    id=1,
    title="Iris Classification",
    title_ko="붓꽃(Iris) 품종 분류",
    category="classification",
    difficulty="beginner",
    description="Classify iris flowers into 3 species using a simple dense network with StandardScaler preprocessing.",
    description_ko="StandardScaler 전처리 후 Dense 네트워크로 붓꽃 3종을 분류합니다. ML의 Hello World 예제입니다.",
    tags=["tabular", "classification", "dense", "beginner"],
    estimated_time="1~2분",
    dataset_info={"name": "Iris", "source": "sklearn built-in", "samples": 150, "features": 4, "classes": 3},
    learning_objectives=[
        "CSV 데이터를 로드하고 타겟 컬럼을 지정하는 방법",
        "StandardScaler로 특성을 정규화하는 이유",
        "Dense 레이어 연결로 분류 모델 구성",
        "softmax 활성화 함수의 역할 (다중 클래스)",
        "학습 실행 후 Loss/Accuracy 차트 해석",
    ],
    guide_steps=[
        {"step": 1, "node": "CSVLoader_1", "text": "Iris 데이터셋을 로드합니다. 4개의 꽃잎/꽃받침 측정값(features)과 품종(labels)이 포함됩니다."},
        {"step": 2, "node": "StandardScaler_1", "text": "특성값의 스케일을 통일합니다. 평균=0, 분산=1로 변환하여 학습 효율을 높입니다."},
        {"step": 3, "node": "TrainValSplit_1", "text": "데이터를 학습(80%)과 검증(20%)으로 분할합니다. 과적합을 모니터링하기 위해 필수입니다."},
        {"step": 4, "node": "Dense_1", "text": "64개 뉴런의 은닉층. ReLU 활성화로 비선형 패턴을 학습합니다."},
        {"step": 5, "node": "Dense_2", "text": "32개 뉴런의 두 번째 은닉층. 더 추상적인 특징을 추출합니다."},
        {"step": 6, "node": "Dense_3", "text": "3개 뉴런의 출력층. softmax로 각 품종의 확률을 출력합니다."},
        {"step": 7, "node": "Optimizer_1", "text": "Adam 옵티마이저 (lr=0.001). 가장 널리 쓰이는 적응적 학습률 옵티마이저입니다."},
        {"step": 8, "node": "LossFunction_1", "text": "Sparse Categorical Crossentropy: 정수 레이블 다중 클래스 분류의 표준 손실 함수입니다."},
        {"step": 9, "node": "Trainer_1", "text": "Run 버튼을 클릭하세요! epochs=30으로 학습합니다. 우측 차트에서 Loss/Accuracy 변화를 관찰하세요."},
        {"step": 10, "node": "LossCurve_1", "text": "학습 완료 후 Train/Val Loss 그래프가 생성됩니다. 두 곡선이 수렴하면 좋은 학습입니다."},
    ],
    nodes=[
        {"id": "CSVLoader_1", "type": "CSVLoader", "position": pos(0, 1), "params": {"file_path": "datasets/iris.csv", "target_column": "species", "separator": ","}},
        {"id": "StandardScaler_1", "type": "StandardScaler", "position": pos(1, 0), "params": {}},
        {"id": "TrainValSplit_1", "type": "TrainValSplit", "position": pos(2, 1), "params": {"val_ratio": 0.2, "test_ratio": 0.0, "random_seed": 42}},
        {"id": "Dense_1", "type": "Dense", "position": pos(3, 0), "params": {"units": 64, "activation": "relu"}},
        {"id": "Dense_2", "type": "Dense", "position": pos(4, 0), "params": {"units": 32, "activation": "relu"}},
        {"id": "Dense_3", "type": "Dense", "position": pos(5, 0), "params": {"units": 3, "activation": "softmax"}},
        {"id": "Optimizer_1", "type": "Optimizer", "position": pos(4, 2), "params": {"type": "adam", "learning_rate": 0.001}},
        {"id": "LossFunction_1", "type": "LossFunction", "position": pos(5, 2), "params": {"type": "sparse_categorical_crossentropy"}},
        {"id": "Trainer_1", "type": "Trainer", "position": pos(6, 1), "params": {"epochs": 30, "batch_size": 16, "framework": "tensorflow"}},
        {"id": "LossCurve_1", "type": "LossCurve", "position": pos(7, 0), "params": {}},
        {"id": "AccuracyCurve_1", "type": "AccuracyCurve", "position": pos(7, 2), "params": {}},
    ],
    edges=[
        {"source": "CSVLoader_1", "target": "StandardScaler_1", "sourceHandle": "features", "targetHandle": "input"},
        {"source": "StandardScaler_1", "target": "TrainValSplit_1", "sourceHandle": "output", "targetHandle": "features"},
        {"source": "CSVLoader_1", "target": "TrainValSplit_1", "sourceHandle": "labels", "targetHandle": "labels"},
        {"source": "Dense_1", "target": "Dense_2", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "Dense_2", "target": "Dense_3", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "Dense_3", "target": "Trainer_1", "sourceHandle": "output", "targetHandle": "layers"},
        {"source": "TrainValSplit_1", "target": "Trainer_1", "sourceHandle": "train_features", "targetHandle": "train_features"},
        {"source": "TrainValSplit_1", "target": "Trainer_1", "sourceHandle": "train_labels", "targetHandle": "train_labels"},
        {"source": "TrainValSplit_1", "target": "Trainer_1", "sourceHandle": "val_features", "targetHandle": "val_features"},
        {"source": "TrainValSplit_1", "target": "Trainer_1", "sourceHandle": "val_labels", "targetHandle": "val_labels"},
        {"source": "Optimizer_1", "target": "Trainer_1", "sourceHandle": "optimizer_config", "targetHandle": "optimizer_config"},
        {"source": "LossFunction_1", "target": "Trainer_1", "sourceHandle": "loss_config", "targetHandle": "loss_config"},
        {"source": "Trainer_1", "target": "LossCurve_1", "sourceHandle": "history", "targetHandle": "history"},
        {"source": "Trainer_1", "target": "AccuracyCurve_1", "sourceHandle": "history", "targetHandle": "history"},
    ],
))

# ── 2. Titanic ───────────────────────────────────────
EXAMPLES.append(make_example(
    id=2,
    title="Titanic Survival Prediction",
    title_ko="타이타닉 생존 예측",
    category="classification",
    difficulty="beginner",
    description="Predict Titanic passenger survival using label encoding and dense layers.",
    description_ko="범주형 데이터를 LabelEncoder로 변환한 후 Dense 네트워크로 생존 여부를 예측합니다.",
    tags=["tabular", "binary-classification", "label-encoding"],
    estimated_time="2분",
    dataset_info={"name": "Titanic", "source": "Kaggle", "samples": 891, "features": 7, "classes": 2},
    learning_objectives=[
        "범주형 데이터(성별, 탑승지 등)를 LabelEncoder로 변환",
        "이진 분류(binary classification) 파이프라인 구성",
        "binary_crossentropy 손실 함수 사용",
    ],
    guide_steps=[
        {"step": 1, "node": "CSVLoader_1", "text": "타이타닉 데이터를 로드합니다. target_column은 'Survived'입니다."},
        {"step": 2, "node": "LabelEncoder_1", "text": "문자열 특성(Sex, Embarked 등)을 정수로 변환합니다."},
        {"step": 3, "node": "TrainValSplit_1", "text": "80/20으로 분할합니다."},
        {"step": 4, "node": "Dense_1", "text": "128개 뉴런 은닉층 (ReLU)."},
        {"step": 5, "node": "Dense_2", "text": "1개 뉴런 출력층 (sigmoid). 이진 분류에서는 sigmoid를 사용합니다."},
        {"step": 6, "node": "Trainer_1", "text": "학습을 시작합니다. Loss가 안정되면 학습이 수렴한 것입니다."},
    ],
    nodes=[
        {"id": "CSVLoader_1", "type": "CSVLoader", "position": pos(0, 1), "params": {"file_path": "datasets/titanic.csv", "target_column": "Survived"}},
        {"id": "LabelEncoder_1", "type": "LabelEncoder", "position": pos(1, 0), "params": {}},
        {"id": "TrainValSplit_1", "type": "TrainValSplit", "position": pos(2, 1), "params": {"val_ratio": 0.2}},
        {"id": "Dense_1", "type": "Dense", "position": pos(3, 0), "params": {"units": 128, "activation": "relu"}},
        {"id": "Dense_2", "type": "Dense", "position": pos(4, 0), "params": {"units": 1, "activation": "sigmoid"}},
        {"id": "Optimizer_1", "type": "Optimizer", "position": pos(3, 2), "params": {"type": "adam", "learning_rate": 0.001}},
        {"id": "LossFunction_1", "type": "LossFunction", "position": pos(4, 2), "params": {"type": "binary_crossentropy"}},
        {"id": "Trainer_1", "type": "Trainer", "position": pos(5, 1), "params": {"epochs": 30, "batch_size": 32, "framework": "tensorflow"}},
        {"id": "LossCurve_1", "type": "LossCurve", "position": pos(6, 1), "params": {}},
    ],
    edges=[
        {"source": "CSVLoader_1", "target": "LabelEncoder_1", "sourceHandle": "features", "targetHandle": "input"},
        {"source": "LabelEncoder_1", "target": "TrainValSplit_1", "sourceHandle": "output", "targetHandle": "features"},
        {"source": "CSVLoader_1", "target": "TrainValSplit_1", "sourceHandle": "labels", "targetHandle": "labels"},
        {"source": "Dense_1", "target": "Dense_2", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "Dense_2", "target": "Trainer_1", "sourceHandle": "output", "targetHandle": "layers"},
        {"source": "TrainValSplit_1", "target": "Trainer_1", "sourceHandle": "train_features", "targetHandle": "train_features"},
        {"source": "TrainValSplit_1", "target": "Trainer_1", "sourceHandle": "train_labels", "targetHandle": "train_labels"},
        {"source": "TrainValSplit_1", "target": "Trainer_1", "sourceHandle": "val_features", "targetHandle": "val_features"},
        {"source": "TrainValSplit_1", "target": "Trainer_1", "sourceHandle": "val_labels", "targetHandle": "val_labels"},
        {"source": "Optimizer_1", "target": "Trainer_1", "sourceHandle": "optimizer_config", "targetHandle": "optimizer_config"},
        {"source": "LossFunction_1", "target": "Trainer_1", "sourceHandle": "loss_config", "targetHandle": "loss_config"},
        {"source": "Trainer_1", "target": "LossCurve_1", "sourceHandle": "history", "targetHandle": "history"},
    ],
))

# ── 3. MNIST Dense ───────────────────────────────────
EXAMPLES.append(make_example(
    id=3,
    title="MNIST Dense Network",
    title_ko="손글씨(MNIST) 숫자 인식",
    category="classification",
    difficulty="beginner",
    description="Recognize handwritten digits using a simple 3-layer dense network. Classic ML benchmark.",
    description_ko="Flatten → Dense 3층 구조로 0~9 손글씨 숫자를 인식합니다. 딥러닝 입문의 필수 예제입니다.",
    tags=["image", "mnist", "dense", "beginner"],
    estimated_time="3분",
    dataset_info={"name": "MNIST", "source": "tf.keras.datasets", "samples": 70000, "features": 784, "classes": 10},
    learning_objectives=[
        "내장 데이터셋(MNIST) 로드 방법",
        "이미지 데이터를 Flatten으로 1D 변환하는 이유",
        "다층 Dense 네트워크의 구조와 패라미터",
        "10개 클래스 분류에서 softmax의 역할",
    ],
    guide_steps=[
        {"step": 1, "node": "NumpyInput_1", "text": "MNIST 데이터셋 (28x28 손글씨 이미지 70,000장)을 자동으로 다운로드합니다."},
        {"step": 2, "node": "Flatten_1", "text": "28×28=784차원의 1D 벡터로 변환합니다. Dense 레이어에 입력하기 위해 필요합니다."},
        {"step": 3, "node": "Dense_1", "text": "128개 뉴런 은닉층. 784개 입력에서 패턴을 학습합니다."},
        {"step": 4, "node": "Dense_2", "text": "64개 뉴런 은닉층. 더 추상적인 특징을 추출합니다."},
        {"step": 5, "node": "Dense_3", "text": "10개 뉴런 출력층 (0~9 숫자). softmax로 각 숫자의 확률을 출력합니다."},
        {"step": 6, "node": "Trainer_1", "text": "Run! 3~5 에포크만으로도 95%+ 정확도에 도달합니다."},
    ],
    nodes=[
        {"id": "NumpyInput_1", "type": "NumpyInput", "position": pos(0, 1), "params": {"dataset": "mnist"}},
        {"id": "Flatten_1", "type": "Flatten", "position": pos(1, 0), "params": {}},
        {"id": "Dense_1", "type": "Dense", "position": pos(2, 0), "params": {"units": 128, "activation": "relu"}},
        {"id": "Dense_2", "type": "Dense", "position": pos(3, 0), "params": {"units": 64, "activation": "relu"}},
        {"id": "Dense_3", "type": "Dense", "position": pos(4, 0), "params": {"units": 10, "activation": "softmax"}},
        {"id": "Optimizer_1", "type": "Optimizer", "position": pos(3, 2), "params": {"type": "adam", "learning_rate": 0.001}},
        {"id": "LossFunction_1", "type": "LossFunction", "position": pos(4, 2), "params": {"type": "sparse_categorical_crossentropy"}},
        {"id": "Trainer_1", "type": "Trainer", "position": pos(5, 1), "params": {"epochs": 10, "batch_size": 32, "framework": "tensorflow"}},
        {"id": "LossCurve_1", "type": "LossCurve", "position": pos(6, 0), "params": {}},
        {"id": "AccuracyCurve_1", "type": "AccuracyCurve", "position": pos(6, 2), "params": {}},
    ],
    edges=[
        {"source": "Flatten_1", "target": "Dense_1", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "Dense_1", "target": "Dense_2", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "Dense_2", "target": "Dense_3", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "Dense_3", "target": "Trainer_1", "sourceHandle": "output", "targetHandle": "layers"},
        {"source": "NumpyInput_1", "target": "Trainer_1", "sourceHandle": "train_features", "targetHandle": "train_features"},
        {"source": "NumpyInput_1", "target": "Trainer_1", "sourceHandle": "train_labels", "targetHandle": "train_labels"},
        {"source": "NumpyInput_1", "target": "Trainer_1", "sourceHandle": "test_features", "targetHandle": "val_features"},
        {"source": "NumpyInput_1", "target": "Trainer_1", "sourceHandle": "test_labels", "targetHandle": "val_labels"},
        {"source": "Optimizer_1", "target": "Trainer_1", "sourceHandle": "optimizer_config", "targetHandle": "optimizer_config"},
        {"source": "LossFunction_1", "target": "Trainer_1", "sourceHandle": "loss_config", "targetHandle": "loss_config"},
        {"source": "Trainer_1", "target": "LossCurve_1", "sourceHandle": "history", "targetHandle": "history"},
        {"source": "Trainer_1", "target": "AccuracyCurve_1", "sourceHandle": "history", "targetHandle": "history"},
    ],
))

# ── 4. MNIST CNN ─────────────────────────────────────
EXAMPLES.append(make_example(
    id=4,
    title="MNIST CNN Classification",
    title_ko="MNIST CNN 분류",
    category="classification",
    difficulty="beginner",
    description="Classify MNIST digits with Conv2D layers. Learn how CNNs outperform dense networks on image data.",
    description_ko="Conv2D 레이어로 MNIST 숫자를 분류합니다. Dense만 쓸 때보다 CNN이 왜 더 좋은지 비교해보세요.",
    tags=["image", "mnist", "cnn", "conv2d"],
    estimated_time="5분",
    dataset_info={"name": "MNIST", "source": "tf.keras.datasets", "samples": 70000, "features": "28x28x1", "classes": 10},
    learning_objectives=[
        "Conv2D 레이어의 필터, 커널 크기 이해",
        "MaxPooling2D로 공간 크기 줄이기",
        "Conv2D → Flatten → Dense 패턴 (CNN의 표준 구조)",
        "CNN이 Dense보다 이미지 분류에 효과적인 이유",
    ],
    guide_steps=[
        {"step": 1, "node": "NumpyInput_1", "text": "MNIST 데이터셋을 로드합니다. CNN 입력을 위해 (28,28,1) shape이 필요합니다."},
        {"step": 2, "node": "Conv2D_1", "text": "32개의 3×3 필터로 엣지, 코너 등 저수준 특징을 추출합니다."},
        {"step": 3, "node": "MaxPooling2D_1", "text": "2×2 풀링으로 공간 크기를 절반으로 줄입니다 (28→14)."},
        {"step": 4, "node": "Conv2D_2", "text": "64개의 3×3 필터로 더 복잡한 패턴을 인식합니다."},
        {"step": 5, "node": "Flatten_1", "text": "2D 특징맵을 1D 벡터로 변환하여 Dense 레이어에 연결합니다."},
        {"step": 6, "node": "Dense_1", "text": "128개 뉴런으로 추출된 특징을 종합합니다."},
        {"step": 7, "node": "Dense_2", "text": "10개 클래스 출력. 예제 3(Dense only)과 정확도를 비교해보세요!"},
        {"step": 8, "node": "Trainer_1", "text": "5 에포크만으로 99%+ 정확도 달성 가능. Dense 예제(95%)보다 훨씬 좋습니다."},
    ],
    nodes=[
        {"id": "NumpyInput_1", "type": "NumpyInput", "position": pos(0, 1), "params": {"dataset": "mnist"}},
        {"id": "Conv2D_1", "type": "Conv2D", "position": pos(1, 0), "params": {"filters": 32, "kernel_size": 3, "activation": "relu", "padding": "same"}},
        {"id": "MaxPooling2D_1", "type": "MaxPooling2D", "position": pos(2, 0), "params": {"pool_size": 2}},
        {"id": "Conv2D_2", "type": "Conv2D", "position": pos(3, 0), "params": {"filters": 64, "kernel_size": 3, "activation": "relu", "padding": "same"}},
        {"id": "Flatten_1", "type": "Flatten", "position": pos(4, 0), "params": {}},
        {"id": "Dense_1", "type": "Dense", "position": pos(5, 0), "params": {"units": 128, "activation": "relu"}},
        {"id": "Dense_2", "type": "Dense", "position": pos(6, 0), "params": {"units": 10, "activation": "softmax"}},
        {"id": "Optimizer_1", "type": "Optimizer", "position": pos(5, 2), "params": {"type": "adam", "learning_rate": 0.001}},
        {"id": "LossFunction_1", "type": "LossFunction", "position": pos(6, 2), "params": {"type": "sparse_categorical_crossentropy"}},
        {"id": "Trainer_1", "type": "Trainer", "position": pos(7, 1), "params": {"epochs": 5, "batch_size": 64, "framework": "tensorflow"}},
        {"id": "LossCurve_1", "type": "LossCurve", "position": pos(8, 0), "params": {}},
        {"id": "AccuracyCurve_1", "type": "AccuracyCurve", "position": pos(8, 2), "params": {}},
    ],
    edges=[
        {"source": "Conv2D_1", "target": "MaxPooling2D_1", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "MaxPooling2D_1", "target": "Conv2D_2", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "Conv2D_2", "target": "Flatten_1", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "Flatten_1", "target": "Dense_1", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "Dense_1", "target": "Dense_2", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "Dense_2", "target": "Trainer_1", "sourceHandle": "output", "targetHandle": "layers"},
        {"source": "NumpyInput_1", "target": "Trainer_1", "sourceHandle": "train_features", "targetHandle": "train_features"},
        {"source": "NumpyInput_1", "target": "Trainer_1", "sourceHandle": "train_labels", "targetHandle": "train_labels"},
        {"source": "NumpyInput_1", "target": "Trainer_1", "sourceHandle": "test_features", "targetHandle": "val_features"},
        {"source": "NumpyInput_1", "target": "Trainer_1", "sourceHandle": "test_labels", "targetHandle": "val_labels"},
        {"source": "Optimizer_1", "target": "Trainer_1", "sourceHandle": "optimizer_config", "targetHandle": "optimizer_config"},
        {"source": "LossFunction_1", "target": "Trainer_1", "sourceHandle": "loss_config", "targetHandle": "loss_config"},
        {"source": "Trainer_1", "target": "LossCurve_1", "sourceHandle": "history", "targetHandle": "history"},
        {"source": "Trainer_1", "target": "AccuracyCurve_1", "sourceHandle": "history", "targetHandle": "history"},
    ],
))

# ── 5. Fashion MNIST ─────────────────────────────────
EXAMPLES.append(make_example(
    id=5,
    title="Fashion MNIST Classification",
    title_ko="패션 MNIST 의류 분류",
    category="classification",
    difficulty="beginner",
    description="Classify fashion items (T-shirt, trouser, pullover, etc.) using Conv2D + GlobalAveragePooling2D.",
    description_ko="Conv2D와 GlobalAveragePooling2D로 10가지 의류 종류를 분류합니다. Flatten 대신 GAP 사용법을 배웁니다.",
    tags=["image", "fashion", "cnn", "global-avg-pool"],
    estimated_time="5분",
    dataset_info={"name": "Fashion MNIST", "source": "tf.keras.datasets", "samples": 70000, "features": "28x28", "classes": 10},
    learning_objectives=[
        "Fashion MNIST와 MNIST의 차이점",
        "GlobalAveragePooling2D vs Flatten 비교",
        "GAP의 장점: 파라미터 수 감소, 과적합 방지",
    ],
    guide_steps=[
        {"step": 1, "node": "NumpyInput_1", "text": "Fashion MNIST 데이터셋: T-shirt, 바지, 풀오버 등 10종 의류 이미지입니다."},
        {"step": 2, "node": "Conv2D_1", "text": "32개 필터로 의류의 모양, 텍스처 특징을 추출합니다."},
        {"step": 3, "node": "Conv2D_2", "text": "64개 필터로 더 세밀한 패턴을 감지합니다."},
        {"step": 4, "node": "GlobalAveragePooling2D_1", "text": "Flatten 대신 GAP 사용! 각 채널의 평균을 구해 64차원 벡터를 만듭니다."},
        {"step": 5, "node": "Dense_1", "text": "128개 뉴런 은닉층으로 특징을 종합합니다."},
        {"step": 6, "node": "Trainer_1", "text": "학습 후 MNIST보다 어려운 데이터인 만큼 정확도 차이를 관찰하세요."},
    ],
    nodes=[
        {"id": "NumpyInput_1", "type": "NumpyInput", "position": pos(0, 1), "params": {"dataset": "fashion_mnist"}},
        {"id": "Conv2D_1", "type": "Conv2D", "position": pos(1, 0), "params": {"filters": 32, "kernel_size": 3, "activation": "relu", "padding": "same"}},
        {"id": "Conv2D_2", "type": "Conv2D", "position": pos(2, 0), "params": {"filters": 64, "kernel_size": 3, "activation": "relu", "padding": "same"}},
        {"id": "GlobalAveragePooling2D_1", "type": "GlobalAveragePooling2D", "position": pos(3, 0), "params": {}},
        {"id": "Dense_1", "type": "Dense", "position": pos(4, 0), "params": {"units": 128, "activation": "relu"}},
        {"id": "Dense_2", "type": "Dense", "position": pos(5, 0), "params": {"units": 10, "activation": "softmax"}},
        {"id": "Optimizer_1", "type": "Optimizer", "position": pos(4, 2), "params": {"type": "adam", "learning_rate": 0.001}},
        {"id": "LossFunction_1", "type": "LossFunction", "position": pos(5, 2), "params": {"type": "sparse_categorical_crossentropy"}},
        {"id": "Trainer_1", "type": "Trainer", "position": pos(6, 1), "params": {"epochs": 10, "batch_size": 64, "framework": "tensorflow"}},
        {"id": "LossCurve_1", "type": "LossCurve", "position": pos(7, 0), "params": {}},
        {"id": "AccuracyCurve_1", "type": "AccuracyCurve", "position": pos(7, 2), "params": {}},
    ],
    edges=[
        {"source": "Conv2D_1", "target": "Conv2D_2", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "Conv2D_2", "target": "GlobalAveragePooling2D_1", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "GlobalAveragePooling2D_1", "target": "Dense_1", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "Dense_1", "target": "Dense_2", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "Dense_2", "target": "Trainer_1", "sourceHandle": "output", "targetHandle": "layers"},
        {"source": "NumpyInput_1", "target": "Trainer_1", "sourceHandle": "train_features", "targetHandle": "train_features"},
        {"source": "NumpyInput_1", "target": "Trainer_1", "sourceHandle": "train_labels", "targetHandle": "train_labels"},
        {"source": "NumpyInput_1", "target": "Trainer_1", "sourceHandle": "test_features", "targetHandle": "val_features"},
        {"source": "NumpyInput_1", "target": "Trainer_1", "sourceHandle": "test_labels", "targetHandle": "val_labels"},
        {"source": "Optimizer_1", "target": "Trainer_1", "sourceHandle": "optimizer_config", "targetHandle": "optimizer_config"},
        {"source": "LossFunction_1", "target": "Trainer_1", "sourceHandle": "loss_config", "targetHandle": "loss_config"},
        {"source": "Trainer_1", "target": "LossCurve_1", "sourceHandle": "history", "targetHandle": "history"},
        {"source": "Trainer_1", "target": "AccuracyCurve_1", "sourceHandle": "history", "targetHandle": "history"},
    ],
))

# ── 6. CIFAR-10 ──────────────────────────────────────
EXAMPLES.append(make_example(
    id=6,
    title="CIFAR-10 Image Classification",
    title_ko="CIFAR-10 이미지 분류",
    category="classification",
    difficulty="intermediate",
    description="Classify 32x32 color images into 10 categories using a deeper CNN with BatchNorm and Dropout.",
    description_ko="BatchNorm + Dropout을 추가한 깊은 CNN으로 CIFAR-10 컬러 이미지를 분류합니다. 정규화 기법을 배웁니다.",
    tags=["image", "cifar10", "cnn", "batchnorm", "dropout"],
    estimated_time="10분",
    dataset_info={"name": "CIFAR-10", "source": "tf.keras.datasets", "samples": 60000, "features": "32x32x3", "classes": 10},
    learning_objectives=[
        "컬러 이미지(3채널) 처리",
        "BatchNormalization의 효과: 학습 안정화",
        "Dropout의 효과: 과적합 방지",
        "깊은 CNN 구조 설계",
    ],
    guide_steps=[
        {"step": 1, "node": "NumpyInput_1", "text": "CIFAR-10: 비행기, 자동차, 새, 고양이 등 10종 컬러 이미지(32x32x3)."},
        {"step": 2, "node": "Conv2D_1", "text": "32개 필터. MNIST와 달리 3채널(RGB) 입력입니다."},
        {"step": 3, "node": "BatchNorm_1", "text": "배치 정규화: 레이어 출력의 분포를 안정화시킵니다. 학습 속도가 빨라집니다."},
        {"step": 4, "node": "Conv2D_2", "text": "64개 필터로 더 복잡한 패턴을 추출합니다."},
        {"step": 5, "node": "Dropout_1", "text": "30% 드롭아웃: 랜덤으로 뉴런을 비활성화하여 과적합을 방지합니다."},
        {"step": 6, "node": "Conv2D_3", "text": "128개 필터의 세 번째 합성곱층."},
        {"step": 7, "node": "Trainer_1", "text": "CIFAR-10은 MNIST보다 어렵습니다. 10 에포크 후 약 70%+ 정확도를 기대할 수 있습니다."},
    ],
    nodes=[
        {"id": "NumpyInput_1", "type": "NumpyInput", "position": pos(0, 1), "params": {"dataset": "cifar10"}},
        {"id": "Conv2D_1", "type": "Conv2D", "position": pos(1, 0), "params": {"filters": 32, "kernel_size": 3, "activation": "relu", "padding": "same"}},
        {"id": "BatchNorm_1", "type": "BatchNorm", "position": pos(2, 0), "params": {}},
        {"id": "MaxPooling2D_1", "type": "MaxPooling2D", "position": pos(3, 0), "params": {"pool_size": 2}},
        {"id": "Conv2D_2", "type": "Conv2D", "position": pos(4, 0), "params": {"filters": 64, "kernel_size": 3, "activation": "relu", "padding": "same"}},
        {"id": "BatchNorm_2", "type": "BatchNorm", "position": pos(5, 0), "params": {}},
        {"id": "MaxPooling2D_2", "type": "MaxPooling2D", "position": pos(6, 0), "params": {"pool_size": 2}},
        {"id": "Dropout_1", "type": "Dropout", "position": pos(7, 0), "params": {"rate": 0.3}},
        {"id": "Conv2D_3", "type": "Conv2D", "position": pos(1, 3), "params": {"filters": 128, "kernel_size": 3, "activation": "relu", "padding": "same"}},
        {"id": "GlobalAveragePooling2D_1", "type": "GlobalAveragePooling2D", "position": pos(2, 3), "params": {}},
        {"id": "Dense_1", "type": "Dense", "position": pos(3, 3), "params": {"units": 128, "activation": "relu"}},
        {"id": "Dropout_2", "type": "Dropout", "position": pos(4, 3), "params": {"rate": 0.5}},
        {"id": "Dense_2", "type": "Dense", "position": pos(5, 3), "params": {"units": 10, "activation": "softmax"}},
        {"id": "Optimizer_1", "type": "Optimizer", "position": pos(6, 3), "params": {"type": "adam", "learning_rate": 0.001}},
        {"id": "LossFunction_1", "type": "LossFunction", "position": pos(6, 4), "params": {"type": "sparse_categorical_crossentropy"}},
        {"id": "Trainer_1", "type": "Trainer", "position": pos(7, 3), "params": {"epochs": 15, "batch_size": 64, "framework": "tensorflow"}},
        {"id": "LossCurve_1", "type": "LossCurve", "position": pos(8, 2), "params": {}},
        {"id": "AccuracyCurve_1", "type": "AccuracyCurve", "position": pos(8, 4), "params": {}},
    ],
    edges=[
        {"source": "Conv2D_1", "target": "BatchNorm_1", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "BatchNorm_1", "target": "MaxPooling2D_1", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "MaxPooling2D_1", "target": "Conv2D_2", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "Conv2D_2", "target": "BatchNorm_2", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "BatchNorm_2", "target": "MaxPooling2D_2", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "MaxPooling2D_2", "target": "Dropout_1", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "Dropout_1", "target": "Conv2D_3", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "Conv2D_3", "target": "GlobalAveragePooling2D_1", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "GlobalAveragePooling2D_1", "target": "Dense_1", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "Dense_1", "target": "Dropout_2", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "Dropout_2", "target": "Dense_2", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "Dense_2", "target": "Trainer_1", "sourceHandle": "output", "targetHandle": "layers"},
        {"source": "NumpyInput_1", "target": "Trainer_1", "sourceHandle": "train_features", "targetHandle": "train_features"},
        {"source": "NumpyInput_1", "target": "Trainer_1", "sourceHandle": "train_labels", "targetHandle": "train_labels"},
        {"source": "NumpyInput_1", "target": "Trainer_1", "sourceHandle": "test_features", "targetHandle": "val_features"},
        {"source": "NumpyInput_1", "target": "Trainer_1", "sourceHandle": "test_labels", "targetHandle": "val_labels"},
        {"source": "Optimizer_1", "target": "Trainer_1", "sourceHandle": "optimizer_config", "targetHandle": "optimizer_config"},
        {"source": "LossFunction_1", "target": "Trainer_1", "sourceHandle": "loss_config", "targetHandle": "loss_config"},
        {"source": "Trainer_1", "target": "LossCurve_1", "sourceHandle": "history", "targetHandle": "history"},
        {"source": "Trainer_1", "target": "AccuracyCurve_1", "sourceHandle": "history", "targetHandle": "history"},
    ],
))

# ── 11. Wine Quality ─────────────────────────────────
EXAMPLES.append(make_example(
    id=11,
    title="Wine Quality Classification",
    title_ko="와인 품질 분류",
    category="classification",
    difficulty="beginner",
    description="Classify wine quality levels using chemical properties with a 3-layer dense network.",
    description_ko="와인의 화학적 특성(산도, 알코올 등)으로 품질 등급을 분류합니다. 3층 Dense 네트워크 활용.",
    tags=["tabular", "classification", "dense", "scaler"],
    estimated_time="2분",
    dataset_info={"name": "Wine Quality", "source": "UCI ML Repository", "samples": 1599, "features": 11, "classes": 6},
    learning_objectives=["다층 Dense 네트워크 설계", "StandardScaler 적용", "다중 클래스 분류"],
    guide_steps=[
        {"step": 1, "node": "CSVLoader_1", "text": "와인 품질 데이터를 로드합니다. 11개 화학 특성과 품질 등급(3~8)이 있습니다."},
        {"step": 2, "node": "StandardScaler_1", "text": "특성 스케일을 통일합니다. 알코올(10~15)과 산도(0~1)의 범위가 다르기 때문입니다."},
        {"step": 3, "node": "Dense_1", "text": "128개 뉴런의 첫 번째 은닉층."},
        {"step": 4, "node": "Trainer_1", "text": "학습을 시작합니다."},
    ],
    nodes=[
        {"id": "CSVLoader_1", "type": "CSVLoader", "position": pos(0, 1), "params": {"file_path": "datasets/wine_quality.csv", "target_column": "quality"}},
        {"id": "StandardScaler_1", "type": "StandardScaler", "position": pos(1, 0), "params": {}},
        {"id": "TrainValSplit_1", "type": "TrainValSplit", "position": pos(2, 1), "params": {"val_ratio": 0.2}},
        {"id": "Dense_1", "type": "Dense", "position": pos(3, 0), "params": {"units": 128, "activation": "relu"}},
        {"id": "Dense_2", "type": "Dense", "position": pos(4, 0), "params": {"units": 64, "activation": "relu"}},
        {"id": "Dense_3", "type": "Dense", "position": pos(5, 0), "params": {"units": 32, "activation": "relu"}},
        {"id": "Dense_4", "type": "Dense", "position": pos(6, 0), "params": {"units": 6, "activation": "softmax"}},
        {"id": "Optimizer_1", "type": "Optimizer", "position": pos(5, 2), "params": {"type": "adam", "learning_rate": 0.001}},
        {"id": "LossFunction_1", "type": "LossFunction", "position": pos(6, 2), "params": {"type": "sparse_categorical_crossentropy"}},
        {"id": "Trainer_1", "type": "Trainer", "position": pos(7, 1), "params": {"epochs": 50, "batch_size": 32, "framework": "tensorflow"}},
        {"id": "LossCurve_1", "type": "LossCurve", "position": pos(8, 0), "params": {}},
        {"id": "AccuracyCurve_1", "type": "AccuracyCurve", "position": pos(8, 2), "params": {}},
    ],
    edges=[
        {"source": "CSVLoader_1", "target": "StandardScaler_1", "sourceHandle": "features", "targetHandle": "input"},
        {"source": "StandardScaler_1", "target": "TrainValSplit_1", "sourceHandle": "output", "targetHandle": "features"},
        {"source": "CSVLoader_1", "target": "TrainValSplit_1", "sourceHandle": "labels", "targetHandle": "labels"},
        {"source": "Dense_1", "target": "Dense_2", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "Dense_2", "target": "Dense_3", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "Dense_3", "target": "Dense_4", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "Dense_4", "target": "Trainer_1", "sourceHandle": "output", "targetHandle": "layers"},
        {"source": "TrainValSplit_1", "target": "Trainer_1", "sourceHandle": "train_features", "targetHandle": "train_features"},
        {"source": "TrainValSplit_1", "target": "Trainer_1", "sourceHandle": "train_labels", "targetHandle": "train_labels"},
        {"source": "TrainValSplit_1", "target": "Trainer_1", "sourceHandle": "val_features", "targetHandle": "val_features"},
        {"source": "TrainValSplit_1", "target": "Trainer_1", "sourceHandle": "val_labels", "targetHandle": "val_labels"},
        {"source": "Optimizer_1", "target": "Trainer_1", "sourceHandle": "optimizer_config", "targetHandle": "optimizer_config"},
        {"source": "LossFunction_1", "target": "Trainer_1", "sourceHandle": "loss_config", "targetHandle": "loss_config"},
        {"source": "Trainer_1", "target": "LossCurve_1", "sourceHandle": "history", "targetHandle": "history"},
        {"source": "Trainer_1", "target": "AccuracyCurve_1", "sourceHandle": "history", "targetHandle": "history"},
    ],
))

# ── 12. Breast Cancer ────────────────────────────────
EXAMPLES.append(make_example(
    id=12,
    title="Breast Cancer Classification",
    title_ko="유방암 양성/악성 분류",
    category="classification",
    difficulty="beginner",
    description="Binary classification of breast tumors as malignant or benign using standardized features.",
    description_ko="30개 세포 특성으로 유방 종양의 양성/악성을 이진 분류합니다. 의료 AI의 기초 예제입니다.",
    tags=["tabular", "binary-classification", "medical"],
    estimated_time="2분",
    dataset_info={"name": "Breast Cancer Wisconsin", "source": "sklearn.datasets", "samples": 569, "features": 30, "classes": 2},
    learning_objectives=["의료 데이터 이진 분류", "30차원 고차원 특성 처리", "EarlyStopping 사용법"],
    guide_steps=[
        {"step": 1, "node": "CSVLoader_1", "text": "유방암 데이터: 세포핵의 반지름, 텍스처, 둘레 등 30개 특성."},
        {"step": 2, "node": "EarlyStopping_1", "text": "val_loss가 5 에포크 동안 개선 없으면 자동 중단합니다. 과적합 방지!"},
        {"step": 3, "node": "Trainer_1", "text": "이진 분류이므로 sigmoid + binary_crossentropy 조합을 사용합니다."},
    ],
    nodes=[
        {"id": "CSVLoader_1", "type": "CSVLoader", "position": pos(0, 1), "params": {"file_path": "datasets/breast_cancer.csv", "target_column": "diagnosis"}},
        {"id": "StandardScaler_1", "type": "StandardScaler", "position": pos(1, 0), "params": {}},
        {"id": "TrainValSplit_1", "type": "TrainValSplit", "position": pos(2, 1), "params": {"val_ratio": 0.2}},
        {"id": "Dense_1", "type": "Dense", "position": pos(3, 0), "params": {"units": 64, "activation": "relu"}},
        {"id": "Dense_2", "type": "Dense", "position": pos(4, 0), "params": {"units": 32, "activation": "relu"}},
        {"id": "Dense_3", "type": "Dense", "position": pos(5, 0), "params": {"units": 1, "activation": "sigmoid"}},
        {"id": "Optimizer_1", "type": "Optimizer", "position": pos(4, 2), "params": {"type": "adam", "learning_rate": 0.001}},
        {"id": "LossFunction_1", "type": "LossFunction", "position": pos(5, 2), "params": {"type": "binary_crossentropy"}},
        {"id": "EarlyStopping_1", "type": "EarlyStopping", "position": pos(5, 3), "params": {"patience": 5, "monitor": "val_loss"}},
        {"id": "Trainer_1", "type": "Trainer", "position": pos(6, 1), "params": {"epochs": 100, "batch_size": 32, "framework": "tensorflow"}},
        {"id": "LossCurve_1", "type": "LossCurve", "position": pos(7, 1), "params": {}},
    ],
    edges=[
        {"source": "CSVLoader_1", "target": "StandardScaler_1", "sourceHandle": "features", "targetHandle": "input"},
        {"source": "StandardScaler_1", "target": "TrainValSplit_1", "sourceHandle": "output", "targetHandle": "features"},
        {"source": "CSVLoader_1", "target": "TrainValSplit_1", "sourceHandle": "labels", "targetHandle": "labels"},
        {"source": "Dense_1", "target": "Dense_2", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "Dense_2", "target": "Dense_3", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "Dense_3", "target": "Trainer_1", "sourceHandle": "output", "targetHandle": "layers"},
        {"source": "TrainValSplit_1", "target": "Trainer_1", "sourceHandle": "train_features", "targetHandle": "train_features"},
        {"source": "TrainValSplit_1", "target": "Trainer_1", "sourceHandle": "train_labels", "targetHandle": "train_labels"},
        {"source": "TrainValSplit_1", "target": "Trainer_1", "sourceHandle": "val_features", "targetHandle": "val_features"},
        {"source": "TrainValSplit_1", "target": "Trainer_1", "sourceHandle": "val_labels", "targetHandle": "val_labels"},
        {"source": "Optimizer_1", "target": "Trainer_1", "sourceHandle": "optimizer_config", "targetHandle": "optimizer_config"},
        {"source": "LossFunction_1", "target": "Trainer_1", "sourceHandle": "loss_config", "targetHandle": "loss_config"},
        {"source": "EarlyStopping_1", "target": "Trainer_1", "sourceHandle": "callback_config", "targetHandle": "callback_config"},
        {"source": "Trainer_1", "target": "LossCurve_1", "sourceHandle": "history", "targetHandle": "history"},
    ],
))

# ── 13. Diabetes Prediction ─────────────────────────
EXAMPLES.append(make_example(
    id=13,
    title="Diabetes Prediction",
    title_ko="당뇨병 예측",
    category="classification",
    difficulty="beginner",
    description="Predict diabetes using MinMaxScaler with Dropout regularization.",
    description_ko="MinMaxScaler + Dropout 조합으로 당뇨병 발병을 예측합니다. 정규화 기법 비교 예제.",
    tags=["tabular", "binary-classification", "dropout", "minmax"],
    estimated_time="2분",
    dataset_info={"name": "Pima Indians Diabetes", "source": "Kaggle", "samples": 768, "features": 8, "classes": 2},
    learning_objectives=["MinMaxScaler vs StandardScaler 차이", "Dropout으로 과적합 방지", "의료 이진 분류"],
    guide_steps=[
        {"step": 1, "node": "CSVLoader_1", "text": "당뇨병 데이터: 혈당, BMI, 나이 등 8개 건강 지표."},
        {"step": 2, "node": "MinMaxScaler_1", "text": "모든 특성을 0~1 범위로 스케일링합니다. StandardScaler와 비교해보세요."},
        {"step": 3, "node": "Dropout_1", "text": "50% 드롭아웃: 과적합 방지의 핵심. 학습 시에만 작동합니다."},
        {"step": 4, "node": "Trainer_1", "text": "학습 후 Train vs Val Accuracy 차이를 관찰하세요."},
    ],
    nodes=[
        {"id": "CSVLoader_1", "type": "CSVLoader", "position": pos(0, 1), "params": {"file_path": "datasets/diabetes.csv", "target_column": "Outcome"}},
        {"id": "MinMaxScaler_1", "type": "MinMaxScaler", "position": pos(1, 0), "params": {}},
        {"id": "TrainValSplit_1", "type": "TrainValSplit", "position": pos(2, 1), "params": {"val_ratio": 0.2}},
        {"id": "Dense_1", "type": "Dense", "position": pos(3, 0), "params": {"units": 64, "activation": "relu"}},
        {"id": "Dropout_1", "type": "Dropout", "position": pos(4, 0), "params": {"rate": 0.5}},
        {"id": "Dense_2", "type": "Dense", "position": pos(5, 0), "params": {"units": 32, "activation": "relu"}},
        {"id": "Dropout_2", "type": "Dropout", "position": pos(6, 0), "params": {"rate": 0.3}},
        {"id": "Dense_3", "type": "Dense", "position": pos(7, 0), "params": {"units": 1, "activation": "sigmoid"}},
        {"id": "Optimizer_1", "type": "Optimizer", "position": pos(6, 2), "params": {"type": "adam", "learning_rate": 0.001}},
        {"id": "LossFunction_1", "type": "LossFunction", "position": pos(7, 2), "params": {"type": "binary_crossentropy"}},
        {"id": "Trainer_1", "type": "Trainer", "position": pos(8, 1), "params": {"epochs": 50, "batch_size": 32, "framework": "tensorflow"}},
        {"id": "LossCurve_1", "type": "LossCurve", "position": pos(9, 0), "params": {}},
        {"id": "AccuracyCurve_1", "type": "AccuracyCurve", "position": pos(9, 2), "params": {}},
    ],
    edges=[
        {"source": "CSVLoader_1", "target": "MinMaxScaler_1", "sourceHandle": "features", "targetHandle": "input"},
        {"source": "MinMaxScaler_1", "target": "TrainValSplit_1", "sourceHandle": "output", "targetHandle": "features"},
        {"source": "CSVLoader_1", "target": "TrainValSplit_1", "sourceHandle": "labels", "targetHandle": "labels"},
        {"source": "Dense_1", "target": "Dropout_1", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "Dropout_1", "target": "Dense_2", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "Dense_2", "target": "Dropout_2", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "Dropout_2", "target": "Dense_3", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "Dense_3", "target": "Trainer_1", "sourceHandle": "output", "targetHandle": "layers"},
        {"source": "TrainValSplit_1", "target": "Trainer_1", "sourceHandle": "train_features", "targetHandle": "train_features"},
        {"source": "TrainValSplit_1", "target": "Trainer_1", "sourceHandle": "train_labels", "targetHandle": "train_labels"},
        {"source": "TrainValSplit_1", "target": "Trainer_1", "sourceHandle": "val_features", "targetHandle": "val_features"},
        {"source": "TrainValSplit_1", "target": "Trainer_1", "sourceHandle": "val_labels", "targetHandle": "val_labels"},
        {"source": "Optimizer_1", "target": "Trainer_1", "sourceHandle": "optimizer_config", "targetHandle": "optimizer_config"},
        {"source": "LossFunction_1", "target": "Trainer_1", "sourceHandle": "loss_config", "targetHandle": "loss_config"},
        {"source": "Trainer_1", "target": "LossCurve_1", "sourceHandle": "history", "targetHandle": "history"},
        {"source": "Trainer_1", "target": "AccuracyCurve_1", "sourceHandle": "history", "targetHandle": "history"},
    ],
))

# ── 26. Boston Housing (Regression) ──────────────────
EXAMPLES.append(make_example(
    id=26,
    title="Boston Housing Price Prediction",
    title_ko="보스턴 주택 가격 예측",
    category="regression",
    difficulty="beginner",
    description="Predict house prices using a dense regression network with MSE loss.",
    description_ko="회귀(Regression) 입문 예제. 13개 주택 특성으로 가격을 예측합니다. 분류와의 차이를 배웁니다.",
    tags=["tabular", "regression", "dense", "mse"],
    estimated_time="2분",
    dataset_info={"name": "Boston Housing", "source": "sklearn.datasets", "samples": 506, "features": 13},
    learning_objectives=[
        "회귀 vs 분류의 차이 (연속값 예측)",
        "MSE 손실 함수 사용",
        "출력층에 activation='linear' 사용하는 이유",
        "회귀 모델의 Loss 해석",
    ],
    guide_steps=[
        {"step": 1, "node": "CSVLoader_1", "text": "보스턴 주택 데이터: 범죄율, 방 수, 학생-교사 비율 등 13개 특성."},
        {"step": 2, "node": "StandardScaler_1", "text": "회귀에서도 스케일링은 필수! 특성 범위가 다르면 학습이 불안정합니다."},
        {"step": 3, "node": "Dense_3", "text": "출력 뉴런 1개, activation=linear. 분류의 softmax/sigmoid와 다릅니다!"},
        {"step": 4, "node": "LossFunction_1", "text": "MSE(Mean Squared Error): 회귀의 표준 손실 함수입니다."},
        {"step": 5, "node": "Trainer_1", "text": "학습 후 Loss가 충분히 낮아지는지 확인하세요."},
    ],
    nodes=[
        {"id": "CSVLoader_1", "type": "CSVLoader", "position": pos(0, 1), "params": {"file_path": "datasets/boston_housing.csv", "target_column": "MEDV"}},
        {"id": "StandardScaler_1", "type": "StandardScaler", "position": pos(1, 0), "params": {}},
        {"id": "TrainValSplit_1", "type": "TrainValSplit", "position": pos(2, 1), "params": {"val_ratio": 0.2}},
        {"id": "Dense_1", "type": "Dense", "position": pos(3, 0), "params": {"units": 64, "activation": "relu"}},
        {"id": "Dense_2", "type": "Dense", "position": pos(4, 0), "params": {"units": 32, "activation": "relu"}},
        {"id": "Dense_3", "type": "Dense", "position": pos(5, 0), "params": {"units": 1, "activation": "linear"}},
        {"id": "Optimizer_1", "type": "Optimizer", "position": pos(4, 2), "params": {"type": "adam", "learning_rate": 0.001}},
        {"id": "LossFunction_1", "type": "LossFunction", "position": pos(5, 2), "params": {"type": "mse"}},
        {"id": "Trainer_1", "type": "Trainer", "position": pos(6, 1), "params": {"epochs": 100, "batch_size": 32, "framework": "tensorflow"}},
        {"id": "LossCurve_1", "type": "LossCurve", "position": pos(7, 1), "params": {}},
    ],
    edges=[
        {"source": "CSVLoader_1", "target": "StandardScaler_1", "sourceHandle": "features", "targetHandle": "input"},
        {"source": "StandardScaler_1", "target": "TrainValSplit_1", "sourceHandle": "output", "targetHandle": "features"},
        {"source": "CSVLoader_1", "target": "TrainValSplit_1", "sourceHandle": "labels", "targetHandle": "labels"},
        {"source": "Dense_1", "target": "Dense_2", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "Dense_2", "target": "Dense_3", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "Dense_3", "target": "Trainer_1", "sourceHandle": "output", "targetHandle": "layers"},
        {"source": "TrainValSplit_1", "target": "Trainer_1", "sourceHandle": "train_features", "targetHandle": "train_features"},
        {"source": "TrainValSplit_1", "target": "Trainer_1", "sourceHandle": "train_labels", "targetHandle": "train_labels"},
        {"source": "TrainValSplit_1", "target": "Trainer_1", "sourceHandle": "val_features", "targetHandle": "val_features"},
        {"source": "TrainValSplit_1", "target": "Trainer_1", "sourceHandle": "val_labels", "targetHandle": "val_labels"},
        {"source": "Optimizer_1", "target": "Trainer_1", "sourceHandle": "optimizer_config", "targetHandle": "optimizer_config"},
        {"source": "LossFunction_1", "target": "Trainer_1", "sourceHandle": "loss_config", "targetHandle": "loss_config"},
        {"source": "Trainer_1", "target": "LossCurve_1", "sourceHandle": "history", "targetHandle": "history"},
    ],
))

# ── 27. Car MPG Prediction (Regression) ──────────────
EXAMPLES.append(make_example(
    id=27,
    title="Car MPG Prediction",
    title_ko="자동차 연비(MPG) 예측",
    category="regression",
    difficulty="beginner",
    description="Predict fuel efficiency using MinMaxScaler and a 2-layer dense network.",
    description_ko="MinMaxScaler + 2층 Dense로 자동차 연비를 예측합니다. 회귀 파이프라인의 기본 패턴.",
    tags=["tabular", "regression", "minmax"],
    estimated_time="2분",
    dataset_info={"name": "Auto MPG", "source": "UCI ML Repository", "samples": 398, "features": 7},
    learning_objectives=["MinMaxScaler 회귀 적용", "MAE vs MSE 손실 비교"],
    guide_steps=[
        {"step": 1, "node": "CSVLoader_1", "text": "자동차 데이터: 실린더 수, 배기량, 마력, 무게 등으로 연비를 예측합니다."},
        {"step": 2, "node": "MinMaxScaler_1", "text": "0~1 범위로 정규화. 회귀에서는 MinMaxScaler도 자주 사용됩니다."},
        {"step": 3, "node": "Trainer_1", "text": "MSE 손실로 학습합니다. MAE로 바꿔서 결과를 비교해보세요."},
    ],
    nodes=[
        {"id": "CSVLoader_1", "type": "CSVLoader", "position": pos(0, 1), "params": {"file_path": "datasets/auto_mpg.csv", "target_column": "mpg"}},
        {"id": "MinMaxScaler_1", "type": "MinMaxScaler", "position": pos(1, 0), "params": {}},
        {"id": "TrainValSplit_1", "type": "TrainValSplit", "position": pos(2, 1), "params": {"val_ratio": 0.2}},
        {"id": "Dense_1", "type": "Dense", "position": pos(3, 0), "params": {"units": 64, "activation": "relu"}},
        {"id": "Dense_2", "type": "Dense", "position": pos(4, 0), "params": {"units": 32, "activation": "relu"}},
        {"id": "Dense_3", "type": "Dense", "position": pos(5, 0), "params": {"units": 1, "activation": "linear"}},
        {"id": "Optimizer_1", "type": "Optimizer", "position": pos(4, 2), "params": {"type": "adam", "learning_rate": 0.001}},
        {"id": "LossFunction_1", "type": "LossFunction", "position": pos(5, 2), "params": {"type": "mse"}},
        {"id": "Trainer_1", "type": "Trainer", "position": pos(6, 1), "params": {"epochs": 100, "batch_size": 32, "framework": "tensorflow"}},
        {"id": "LossCurve_1", "type": "LossCurve", "position": pos(7, 1), "params": {}},
    ],
    edges=[
        {"source": "CSVLoader_1", "target": "MinMaxScaler_1", "sourceHandle": "features", "targetHandle": "input"},
        {"source": "MinMaxScaler_1", "target": "TrainValSplit_1", "sourceHandle": "output", "targetHandle": "features"},
        {"source": "CSVLoader_1", "target": "TrainValSplit_1", "sourceHandle": "labels", "targetHandle": "labels"},
        {"source": "Dense_1", "target": "Dense_2", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "Dense_2", "target": "Dense_3", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "Dense_3", "target": "Trainer_1", "sourceHandle": "output", "targetHandle": "layers"},
        {"source": "TrainValSplit_1", "target": "Trainer_1", "sourceHandle": "train_features", "targetHandle": "train_features"},
        {"source": "TrainValSplit_1", "target": "Trainer_1", "sourceHandle": "train_labels", "targetHandle": "train_labels"},
        {"source": "TrainValSplit_1", "target": "Trainer_1", "sourceHandle": "val_features", "targetHandle": "val_features"},
        {"source": "TrainValSplit_1", "target": "Trainer_1", "sourceHandle": "val_labels", "targetHandle": "val_labels"},
        {"source": "Optimizer_1", "target": "Trainer_1", "sourceHandle": "optimizer_config", "targetHandle": "optimizer_config"},
        {"source": "LossFunction_1", "target": "Trainer_1", "sourceHandle": "loss_config", "targetHandle": "loss_config"},
        {"source": "Trainer_1", "target": "LossCurve_1", "sourceHandle": "history", "targetHandle": "history"},
    ],
))


# ═══════════════════════════════════════════════════════
# Generate files
# ═══════════════════════════════════════════════════════

def generate():
    index = []

    for ex in EXAMPLES:
        meta = ex["meta"]
        cat = meta["category"]
        eid = meta["id"]
        slug = meta["title"].lower().replace(" ", "_").replace("/", "_")
        filename = f"{eid:03d}_{slug}.json"

        cat_dir = EXAMPLES_DIR / cat
        cat_dir.mkdir(parents=True, exist_ok=True)

        filepath = cat_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(ex, f, ensure_ascii=False, indent=2)

        index.append({
            "id": eid,
            "title": meta["title"],
            "title_ko": meta["title_ko"],
            "category": cat,
            "difficulty": meta["difficulty"],
            "tags": meta["tags"],
            "estimated_time": meta["estimated_time"],
            "requires_phase": meta.get("requires_phase", 1),
        })

        print(f"  Generated: {filepath}")

    # _index.json
    index.sort(key=lambda x: x["id"])
    index_path = EXAMPLES_DIR / "_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"\n  Index: {index_path} ({len(index)} examples)")


if __name__ == "__main__":
    print("Generating example pipelines...")
    generate()
    print("Done!")
