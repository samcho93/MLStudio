# CLAUDE.md — Node-Based ML Training Program

## Project Overview

노드 기반의 머신러닝 학습 파이프라인 설계/실행 툴.
사용자가 노드를 드래그앤드롭으로 연결해 데이터 전처리 → 모델 정의 → 학습 → 평가 → 시각화까지의
전체 ML 워크플로를 시각적으로 구성하고 실행할 수 있는 애플리케이션.

---

## Architecture Decision

> **두 버전을 동시에 개발한다.** Python 백엔드는 100% 공유, 프론트엔드만 분리.

### 공통 백엔드

```
[FastAPI Python Server — 공통]
        ↕ WebSocket / REST API
[TensorFlow / PyTorch 학습 엔진]
        ↕
[SQLite / File System (모델/결과 저장)]
```

### 버전 A: Web Frontend

```
[Browser: React + React Flow]
        ↕ WebSocket / REST API
[FastAPI Python Server — 공통]
```

### 버전 B: C# WPF Frontend

```
[WPF: Nodify 노드 에디터 + LiveCharts2]
        ↕ HttpClient + WebSocket Client
[FastAPI Python Server — 공통]
```

### 동시 개발 전략

| 항목 | 내용 |
|------|------|
| 공유 코드 | Python 백엔드 전체 (노드 실행 엔진, WebSocket 프로토콜) |
| 분리 코드 | Web: React/TypeScript, WPF: C#/.NET 8 |
| API 계약 | OpenAPI 스펙으로 양쪽 클라이언트 동기화 |
| 우선 개발 | Web 버전 → WPF 버전 순 (UI 검증 후 포팅) |

---

## Tech Stack

### 공통 백엔드 (Python)
| 역할 | 라이브러리 |
|------|-----------|
| API 서버 | `FastAPI` |
| 실시간 통신 | `WebSocket` (FastAPI 내장) |
| ML 프레임워크 | `TensorFlow 2.x` / `PyTorch` |
| 데이터 처리 | `NumPy`, `Pandas`, `scikit-learn` |
| 시각화 데이터 | `Matplotlib` (base64 이미지 변환) |
| ONNX 내보내기 | `onnx`, `onnxruntime` |

### 버전 A — Web Frontend
| 역할 | 라이브러리 |
|------|-----------|
| 노드 에디터 | `React Flow` (xyflow) |
| UI 프레임워크 | React + TypeScript |
| 스타일 | Tailwind CSS |
| 학습 차트 | `Recharts` |
| 상태 관리 | Zustand |

### 버전 B — C# WPF Frontend
| 역할 | 라이브러리 |
|------|-----------|
| 노드 에디터 | `Nodify` (NuGet) |
| UI 프레임워크 | WPF + .NET 8 |
| 학습 차트 | `LiveCharts2` |
| HTTP/WS 클라이언트 | `HttpClient` + `ClientWebSocket` |
| MVVM | `CommunityToolkit.Mvvm` |

---

## ML Framework: Essential Functions

### TensorFlow / Keras

```python
# ── 데이터 파이프라인 ──────────────────────────────
tf.data.Dataset.from_tensor_slices(data)   # 텐서 → Dataset
dataset.batch(batch_size)                  # 배치 분할
dataset.shuffle(buffer_size)               # 셔플
dataset.map(preprocess_fn)                 # 전처리 매핑
dataset.prefetch(tf.data.AUTOTUNE)         # 비동기 프리패치

# ── 모델 정의 ─────────────────────────────────────
model = tf.keras.Sequential([...])         # Sequential API
model = tf.keras.Model(inputs, outputs)    # Functional API
tf.keras.layers.Dense(units, activation)
tf.keras.layers.Conv2D(filters, kernel_size)
tf.keras.layers.LSTM(units)
tf.keras.layers.Dropout(rate)
tf.keras.layers.BatchNormalization()

# ── 컴파일 & 학습 ──────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(
    train_ds, epochs=N,
    validation_data=val_ds,
    callbacks=[...],
    verbose=1
)

# ── 콜백 (학습 모니터링) ───────────────────────────
tf.keras.callbacks.EarlyStopping(patience=5)
tf.keras.callbacks.ModelCheckpoint(filepath)
tf.keras.callbacks.ReduceLROnPlateau()
tf.keras.callbacks.TensorBoard(log_dir)
tf.keras.callbacks.LambdaCallback(on_epoch_end=fn)  # WebSocket 전송용

# ── 평가 & 추론 ────────────────────────────────────
model.evaluate(test_ds)
model.predict(x)
model.save('model.keras')
tf.keras.models.load_model('model.keras')

# ── 전이학습 ──────────────────────────────────────
base = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)
base.trainable = False
```

### PyTorch

```python
# ── 데이터 파이프라인 ──────────────────────────────
from torch.utils.data import Dataset, DataLoader
class CustomDataset(Dataset):
    def __len__(self): ...
    def __getitem__(self, idx): ...

loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# ── 모델 정의 ─────────────────────────────────────
import torch.nn as nn
class MyModel(nn.Module):
    def __init__(self): super().__init__(); self.fc = nn.Linear(in, out)
    def forward(self, x): return self.fc(x)

nn.Conv2d(in_channels, out_channels, kernel_size)
nn.LSTM(input_size, hidden_size, num_layers)
nn.BatchNorm2d(num_features)
nn.Dropout(p)

# ── 학습 루프 (핵심) ───────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in loader:
        optimizer.zero_grad()          # ① 그래디언트 초기화
        output = model(batch_x)        # ② 순전파
        loss = criterion(output, batch_y)  # ③ 손실 계산
        loss.backward()                # ④ 역전파
        optimizer.step()               # ⑤ 파라미터 업데이트
    scheduler.step()                   # LR 스케줄링

    model.eval()
    with torch.no_grad():              # 평가 (그래디언트 비활성)
        for batch_x, batch_y in val_loader: ...

# ── 저장 & 로드 ────────────────────────────────────
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))

# ── GPU 이동 ───────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
batch_x = batch_x.to(device)

# ── 전이학습 ──────────────────────────────────────
import torchvision.models as models
backbone = models.resnet50(weights='IMAGENET1K_V1')
for param in backbone.parameters():
    param.requires_grad = False
backbone.fc = nn.Linear(2048, num_classes)
```

---

## Visualization: Functions & Libraries

### Python Backend — 데이터 생성
```python
# Matplotlib — 이미지로 변환해 프론트에 전송
import matplotlib
matplotlib.use('Agg')  # GUI 없이 렌더링
import matplotlib.pyplot as plt
import io, base64

def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    return base64.b64encode(buf.getvalue()).decode()

# Loss / Accuracy 곡선
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')

# Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d')

# Feature Map 시각화
# Grad-CAM
# t-SNE / UMAP 임베딩
from sklearn.manifold import TSNE
from umap import UMAP
```

### Frontend — 실시간 차트
```typescript
// Recharts — Loss/Accuracy 실시간 그래프
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

// Chart.js — 유연한 커스터마이징
import { Chart } from 'chart.js';

// WebSocket 수신 → 차트 업데이트
const ws = new WebSocket('ws://localhost:8000/ws/train');
ws.onmessage = (e) => {
  const { epoch, loss, val_loss, accuracy } = JSON.parse(e.data);
  setChartData(prev => [...prev, { epoch, loss, val_loss, accuracy }]);
};
```

### WPF (.NET) — 차트
```csharp
// LiveCharts2
using LiveChartsCore;
using LiveChartsCore.SkiaSharpView;

// Series 정의
SeriesCollection = new ISeries[]
{
    new LineSeries<double> { Values = LossValues, Name = "Train Loss" },
    new LineSeries<double> { Values = ValLossValues, Name = "Val Loss" }
};

// 실시간 추가
LossValues.Add(newLoss);  // ObservableCollection
```

---

## Node Types (노드 카탈로그)

> 우선순위: 🔴 즉시 필요 (Phase 1) / 🟡 Phase 2 / 🟢 Phase 3

---

### 📂 데이터 노드

| 노드명 | 기능 | 출력 포트 | 우선순위 |
|--------|------|-----------|----------|
| `CSV Loader` | CSV 파일 로드 | Dataset | 🔴 |
| `Numpy Input` | ndarray / 메모리 배열 직접 입력 | Tensor | 🔴 |
| `Image Folder` | 폴더 이미지 로드 (subdir=class) | Dataset | 🔴 |
| `HDF5 Loader` | HDF5 / Parquet 대용량 데이터셋 | Dataset | 🟡 |
| `Train/Val Split` | 학습/검증/테스트 분리 | Train DS, Val DS, Test DS | 🔴 |
| `Data Inspector` | shape, dtype, 샘플 미리보기 | (통과) | 🔴 |
| `StandardScaler` | Z-score 정규화 (fit/transform 분리) | Dataset | 🔴 |
| `MinMaxScaler` | Min-Max 정규화 | Dataset | 🔴 |
| `Label Encoder` | 정수 인코딩 (문자 → 숫자) | Dataset | 🔴 |
| `One-Hot Encoder` | 범주형 → One-Hot 벡터 | Dataset | 🔴 |
| `PCA` | 주성분 분석, 차원 축소 | Dataset | 🟡 |
| `Augmentation` | 이미지 증강 (flip, rotate, crop 등) | Dataset | 🟡 |

---

### 🧱 레이어 노드

| 노드명 | 기능 | 입출력 | 우선순위 |
|--------|------|--------|----------|
| `Dense` | 완전연결층 | Tensor → Tensor | 🔴 |
| `Conv2D` | 2D 합성곱층 | Tensor → Tensor | 🔴 |
| `MaxPooling2D` | 최대 풀링 | Tensor → Tensor | 🔴 |
| `Flatten` | 다차원 → 1D 변환 (Conv→Dense 연결 필수) | Tensor → Tensor | 🔴 |
| `GlobalAveragePooling2D` | 공간 평균 풀링 (CNN 헤드) | Tensor → Tensor | 🔴 |
| `BatchNorm` | 배치정규화 | Tensor → Tensor | 🔴 |
| `Dropout` | 드롭아웃 | Tensor → Tensor | 🔴 |
| `LSTM` | Long Short-Term Memory 순환층 | Sequence → Tensor | 🟡 |
| `GRU` | Gated Recurrent Unit | Sequence → Tensor | 🟡 |
| `Embedding` | 정수 인덱스 → 벡터 (NLP 필수) | Index → Tensor | 🟡 |
| `Concat (Merge)` | 두 텐서 연결 (axis 지정) | Tensor×2 → Tensor | 🟡 |
| `Add (Merge)` | Skip connection, Residual 연결 | Tensor×2 → Tensor | 🟡 |
| `Reshape` | 텐서 shape 변환 | Tensor → Tensor | 🟡 |
| `MultiHeadAttention` | Transformer Attention 블록 | Tensor → Tensor | 🟢 |
| `Transformer Block` | MHA + FFN + LayerNorm 통합 | Tensor → Tensor | 🟢 |
| `Pretrained Model` | ResNet / MobileNet / EfficientNet 등 | Tensor → Feature | 🟡 |

---

### ⚙️ 학습 노드

| 노드명 | 기능 | 주요 파라미터 | 우선순위 |
|--------|------|--------------|----------|
| `Optimizer` | Adam / SGD / RMSProp / AdamW 선택 | lr, momentum, weight_decay | 🔴 |
| `Loss Function` | CE / MSE / BCE / Focal Loss 선택 | - | 🔴 |
| `LR Scheduler` | StepLR / CosineAnnealing / ReduceOnPlateau | step, gamma | 🟡 |
| `Gradient Clipping` | 그래디언트 클리핑 (RNN 계열 필수) | max_norm | 🟡 |
| `Trainer` | 학습 루프 실행 (epoch, batch 관리) | epochs, batch_size, device | 🔴 |
| `Early Stopping` | val_loss 기준 조기 종료 | patience, monitor, min_delta | 🔴 |
| `Model Checkpoint` | 최적 모델 자동 저장 | filepath, monitor | 🔴 |
| `Mixed Precision` | FP16 학습 설정 (GPU 메모리 절감) | dtype | 🟢 |
| `Custom Callback` | 사용자 Python 코드 삽입 포인트 | code snippet | 🟢 |

---

### 📊 평가 노드

| 노드명 | 기능 | 출력 | 우선순위 |
|--------|------|------|----------|
| `Classification Metrics` | Accuracy / Precision / Recall / F1 / AUC | 수치 + 표 | 🔴 |
| `Regression Metrics` | MAE / RMSE / R² | 수치 + 표 | 🟡 |
| `Confusion Matrix` | 혼동 행렬 히트맵 | 이미지 | 🔴 |
| `ROC Curve` | FPR-TPR 곡선 + AUC | 차트 | 🟡 |
| `Precision-Recall Curve` | PR 곡선 (불균형 데이터에 유용) | 차트 | 🟡 |
| `Prediction Viewer` | 샘플별 예측값 vs 정답 테이블 | 테이블 | 🟡 |

---

### 📈 시각화 노드

| 노드명 | 기능 | 우선순위 |
|--------|------|----------|
| `Loss Curve` | epoch별 train/val loss 실시간 표시 | 🔴 |
| `Accuracy Curve` | epoch별 정확도 추이 | 🔴 |
| `Model Summary` | 레이어 구조 / 파라미터 수 텍스트 출력 | 🔴 |
| `Feature Map Viewer` | Conv2D 레이어 활성화 맵 시각화 | 🟡 |
| `Grad-CAM Viewer` | 입력 이미지의 판단 근거 히트맵 | 🟢 |
| `t-SNE Viewer` | 고차원 임베딩 2D/3D 분포 | 🟡 |
| `UMAP Viewer` | t-SNE 대비 빠른 임베딩 시각화 | 🟢 |
| `Data Distribution` | 입력 데이터 히스토그램 / 박스플롯 | 🟡 |

---

### 💾 출력 / 내보내기 노드

| 노드명 | 기능 | 우선순위 |
|--------|------|----------|
| `Model Save` | `.keras` / `.pth` 저장 | 🔴 |
| `ONNX Export` | ONNX 포맷 변환 (C# OnnxRuntime 연동) | 🟡 |
| `TFLite Export` | TensorFlow Lite 변환 (모바일 배포) | 🟢 |
| `Quantization` | INT8 / FP16 경량화 | 🟢 |
| `Custom Code Node` | Python 코드 직접 삽입 (파워유저용) | 🟢 |

---

### Phase별 구현 목표 요약

| Phase | 포함 노드 수 | 커버 가능한 작업 |
|-------|------------|----------------|
| **Phase 1** (MVP) | ~18개 | CSV/이미지 → Dense/CNN → 학습 → Loss 확인 |
| **Phase 2** | ~32개 | RNN/NLP, 전이학습, 평가 전체, 시각화 확장 |
| **Phase 3** | ~42개 | Transformer, 배포 내보내기, 파워유저 기능 |

---

## WebSocket Protocol (Frontend ↔ Backend)

```jsonc
// Client → Server: 파이프라인 실행 요청
{
  "type": "START_TRAINING",
  "pipeline": {
    "nodes": [...],   // 노드 목록 (id, type, params)
    "edges": [...]    // 연결 정보 (source, target, port)
  }
}

// Server → Client: 실시간 학습 진행
{
  "type": "EPOCH_UPDATE",
  "epoch": 5,
  "total_epochs": 50,
  "loss": 0.342,
  "val_loss": 0.401,
  "accuracy": 0.876,
  "val_accuracy": 0.851,
  "lr": 0.0003,
  "elapsed_sec": 12.4
}

// Server → Client: 완료
{
  "type": "TRAINING_COMPLETE",
  "model_path": "outputs/model_20250314.keras",
  "metrics": { "test_acc": 0.912 },
  "confusion_matrix_b64": "iVBORw0KGgo..."
}

// Server → Client: 에러
{
  "type": "ERROR",
  "message": "CUDA out of memory",
  "node_id": "trainer_01"
}
```

---

## Pipeline Execution Engine (Python)

```python
# pipeline_executor.py
class PipelineExecutor:
    """노드 그래프를 토폴로지 정렬 후 순서대로 실행"""
    
    def __init__(self, nodes: list, edges: list, ws_callback):
        self.nodes = {n['id']: n for n in nodes}
        self.edges = edges
        self.ws_callback = ws_callback  # WebSocket 전송 함수
        self.context = {}               # 노드 간 데이터 전달 dict
    
    def topological_sort(self) -> list:
        """Kahn's algorithm으로 실행 순서 결정"""
        ...
    
    async def execute(self):
        order = self.topological_sort()
        for node_id in order:
            node = self.nodes[node_id]
            handler = NODE_REGISTRY[node['type']]
            inputs = self._gather_inputs(node_id)
            outputs = await handler.run(node['params'], inputs, self.ws_callback)
            self.context[node_id] = outputs

# 노드 핸들러 등록
NODE_REGISTRY = {
    'CSV Loader':    CSVLoaderNode(),
    'Dense Layer':   DenseLayerNode(),
    'Trainer':       TrainerNode(),
    'Loss Curve':    LossCurveNode(),
    ...
}
```

---

## Project File Structure

```
ml-node-studio/
├── backend/                         # Python FastAPI (두 버전 공통)
│   ├── main.py
│   ├── pipeline_executor.py
│   ├── nodes/
│   │   ├── data_nodes.py
│   │   ├── layer_nodes.py
│   │   ├── trainer_nodes.py
│   │   └── viz_nodes.py
│   ├── frameworks/
│   │   ├── tf_backend.py
│   │   └── torch_backend.py
│   └── requirements.txt
│
├── frontend-web/                    # 버전 A: React + React Flow
│   ├── src/
│   │   ├── nodes/
│   │   ├── charts/
│   │   ├── store/
│   │   └── App.tsx
│   ├── package.json
│   └── vite.config.ts
│
├── frontend-wpf/                    # 버전 B: C# WPF
│   ├── MLNodeStudio.sln
│   ├── Views/
│   │   ├── MainWindow.xaml
│   │   └── NodeCanvas.xaml
│   ├── ViewModels/
│   ├── Nodes/
│   ├── Services/
│   │   └── BackendApiService.cs      # WebSocket + REST 클라이언트
│   └── MLNodeStudio.csproj
│
├── docs/
│   ├── TEXTBOOK.md                  # 학습 교재 원본 (Markdown)
│   └── TEXTBOOK.pdf                 # 학습 교재 PDF (빌드 산출물)
│
└── CLAUDE.md
```

---

## Development Phases

### Phase 1 — MVP (4~6주)
- [ ] FastAPI 서버 + WebSocket 기본 구조
- [ ] **[Web]** React Flow 캔버스 + 기본 노드 (CSV Loader, Dense, Trainer)
- [ ] **[WPF]** Nodify 캔버스 + 동일 기본 노드 포팅
- [ ] TensorFlow 학습 루프 + 실시간 Loss 전송
- [ ] Loss / Accuracy 차트 실시간 표시 (Web: Recharts, WPF: LiveCharts2)
- [ ] 파이프라인 JSON 저장/불러오기

### Phase 2 — 노드 확장 (4주)
- [ ] PyTorch 백엔드 추가 (프레임워크 선택 옵션)
- [ ] Conv2D, LSTM, Dropout, BatchNorm, Flatten, GlobalAveragePooling 노드
- [ ] 이미지 데이터 로더 + 증강 노드
- [ ] 평가 노드 전체 (Metrics, ROC Curve, Confusion Matrix)
- [ ] StandardScaler, LabelEncoder, One-Hot 노드

### Phase 3 — 고도화 (4주)
- [ ] 전이학습 노드 (ResNet, MobileNet, EfficientNet)
- [ ] Transformer Block, MultiHeadAttention 노드
- [ ] t-SNE / UMAP 임베딩 시각화
- [ ] ONNX Export 노드 (C# OnnxRuntime 연동)
- [ ] 하이퍼파라미터 튜닝 노드
- [ ] 모델 비교 대시보드

---

## 학습 예제 100가지

> `docs/TEXTBOOK.md` 및 `docs/TEXTBOOK.pdf`에 전체 수록.
> 아래는 카테고리별 분류 목록.

### 분류 (Classification) — 예제 1~25
| # | 예제 | 사용 노드 |
|---|------|----------|
| 1 | 붓꽃(Iris) 품종 분류 | CSV → StandardScaler → Dense×2 → Trainer |
| 2 | 타이타닉 생존 예측 | CSV → LabelEncoder → Dense → Trainer |
| 3 | 손글씨(MNIST) 숫자 인식 | NumpyInput → Flatten → Dense×3 → Trainer |
| 4 | MNIST CNN 분류 | NumpyInput → Conv2D×2 → Flatten → Dense → Trainer |
| 5 | 패션 MNIST 의류 분류 | NumpyInput → Conv2D → GlobalAvgPool → Dense → Trainer |
| 6 | CIFAR-10 이미지 분류 | ImageFolder → Conv2D×3 → Dense → Trainer |
| 7 | 개/고양이 이진 분류 | ImageFolder → Conv2D×3 → GlobalAvgPool → Dense → Trainer |
| 8 | 꽃 10종 분류 (전이학습) | ImageFolder → PretrainedModel(MobileNetV2) → Dense → Trainer |
| 9 | 스팸 메일 분류 | CSV → Embedding → LSTM → Dense → Trainer |
| 10 | 감성 분석 (긍정/부정) | CSV → Embedding → LSTM → Dense → Trainer |
| 11 | 와인 품질 분류 | CSV → StandardScaler → Dense×3 → Trainer |
| 12 | 유방암 양성/악성 분류 | CSV → StandardScaler → Dense×2 → Trainer |
| 13 | 당뇨병 예측 | CSV → MinMaxScaler → Dense×3 → Dropout → Trainer |
| 14 | 신용카드 사기 탐지 (불균형) | CSV → StandardScaler → Dense → Trainer + ClassWeight |
| 15 | 손동작 인식 | CSV → Dense×3 → Trainer |
| 16 | 얼굴 표정 분류 | ImageFolder → Conv2D×4 → Dense → Trainer |
| 17 | 피부 병변 분류 (전이학습) | ImageFolder → PretrainedModel(EfficientNet) → Dense → Trainer |
| 18 | 토양 유형 분류 | CSV → PCA → Dense×2 → Trainer |
| 19 | 뇌 MRI 종양 분류 | ImageFolder → Conv2D×3 → GlobalAvgPool → Dense → Trainer |
| 20 | 음성 명령 분류 | CSV(MFCC) → Dense×3 → Trainer |
| 21 | 자동차 고장 유형 분류 | CSV → StandardScaler → Dense×3 → Trainer |
| 22 | 식물 질병 분류 (전이학습) | ImageFolder → PretrainedModel(ResNet50) → Dense → Trainer |
| 23 | 문서 주제 분류 | CSV → Embedding → LSTM → Dense → Trainer |
| 24 | 교통 표지판 분류 | ImageFolder → Conv2D×3 → BatchNorm → Dense → Trainer |
| 25 | 다중 레이블 분류 (영화 장르) | CSV → Embedding → LSTM → Dense(sigmoid) → Trainer |

### 회귀 (Regression) — 예제 26~45
| # | 예제 | 사용 노드 |
|---|------|----------|
| 26 | 보스턴 주택 가격 예측 | CSV → StandardScaler → Dense×3 → Trainer |
| 27 | 자동차 연비(MPG) 예측 | CSV → MinMaxScaler → Dense×2 → Trainer |
| 28 | 기온 예측 (시계열) | CSV → MinMaxScaler → LSTM → Dense → Trainer |
| 29 | 주식 가격 예측 | CSV → MinMaxScaler → LSTM×2 → Dense → Trainer |
| 30 | 전력 소비량 예측 | CSV → StandardScaler → GRU → Dense → Trainer |
| 31 | 나이 예측 (얼굴 이미지) | ImageFolder → Conv2D×3 → Dense → Trainer |
| 32 | 부동산 가격 예측 | CSV → StandardScaler → Dense×4 → Dropout → Trainer |
| 33 | CO2 배출량 예측 | CSV → StandardScaler → Dense×2 → Trainer |
| 34 | 태양광 발전량 예측 | CSV → MinMaxScaler → LSTM → Dense → Trainer |
| 35 | 의료비 예측 | CSV → OneHot → Dense×3 → Trainer |
| 36 | 음식 칼로리 예측 | CSV → StandardScaler → Dense×2 → Trainer |
| 37 | 풍속 예측 (멀티스텝) | CSV → MinMaxScaler → LSTM×2 → Dense → Trainer |
| 38 | 교통량 예측 | CSV → StandardScaler → GRU×2 → Dense → Trainer |
| 39 | 수질 오염 지수 예측 | CSV → StandardScaler → Dense×3 → Trainer |
| 40 | 건물 에너지 효율 예측 | CSV → MinMaxScaler → Dense×3 → Dropout → Trainer |
| 41 | 뇌파(EEG) 강도 예측 | CSV → StandardScaler → LSTM → Dense → Trainer |
| 42 | 제품 판매량 예측 | CSV → MinMaxScaler → LSTM → Dense → Trainer |
| 43 | 환율 예측 | CSV → MinMaxScaler → GRU → Dense → Trainer |
| 44 | 강수량 예측 | CSV → StandardScaler → LSTM → Dense → Trainer |
| 45 | 자동차 가격 예측 | CSV → OneHot → StandardScaler → Dense×4 → Trainer |

### 비지도 / 클러스터링 — 예제 46~55
| # | 예제 | 사용 노드 |
|---|------|----------|
| 46 | 고객 세분화 (K-Means 시각화) | CSV → StandardScaler → PCA → tSNE Viewer |
| 47 | 이상치 탐지 (Autoencoder) | CSV → StandardScaler → Dense(enc) → Dense(dec) → Trainer |
| 48 | 제조 결함 탐지 (Autoencoder) | CSV → StandardScaler → Dense(enc/dec) → Trainer |
| 49 | 이미지 압축 (Autoencoder) | NumpyInput → Conv2D(enc) → Conv2DTranspose(dec) → Trainer |
| 50 | 차원 축소 시각화 (PCA + t-SNE) | CSV → StandardScaler → PCA → tSNE Viewer |
| 51 | 노래 장르 클러스터링 | CSV → StandardScaler → PCA → UMAP Viewer |
| 52 | 뉴스 기사 클러스터링 | CSV → Embedding → LSTM → tSNE Viewer |
| 53 | 센서 이상 감지 | CSV → MinMaxScaler → LSTM(Autoencoder) → Trainer |
| 54 | 얼굴 임베딩 시각화 | ImageFolder → PretrainedModel → UMAP Viewer |
| 55 | 텍스트 유사도 시각화 | CSV → Embedding → LSTM → tSNE Viewer |

### 이미지 심화 — 예제 56~70
| # | 예제 | 사용 노드 |
|---|------|----------|
| 56 | ResNet50 전이학습 파인튜닝 | ImageFolder → PretrainedModel(ResNet50, trainable) → Dense → Trainer |
| 57 | EfficientNetB0 전이학습 | ImageFolder → PretrainedModel(EfficientNetB0) → GlobalAvgPool → Dense → Trainer |
| 58 | 데이터 증강 효과 비교 | ImageFolder → Augmentation → Conv2D×2 → Trainer |
| 59 | Grad-CAM 시각화 | 학습된 CNN 모델 → GradCAM Viewer |
| 60 | Feature Map 시각화 | 학습된 CNN 모델 → FeatureMap Viewer |
| 61 | 이미지 세그멘테이션 (U-Net) | ImageFolder → Conv2D(enc) → Add → Conv2DTranspose(dec) → Trainer |
| 62 | 객체 경계 박스 회귀 | ImageFolder → Conv2D×3 → GlobalAvgPool → Dense(4출력) → Trainer |
| 63 | 시암 네트워크 유사도 비교 | ImageFolder×2 → 공유Conv2D → Add → Dense → Trainer |
| 64 | 의료 X-ray 이상 탐지 | ImageFolder → PretrainedModel → GlobalAvgPool → Dense → Trainer |
| 65 | 위성 이미지 토지 분류 | ImageFolder → Conv2D×4 → BatchNorm → Dense → Trainer |
| 66 | 문서 레이아웃 분류 | ImageFolder → Conv2D×3 → GlobalAvgPool → Dense → Trainer |
| 67 | 손 제스처 인식 | ImageFolder → Conv2D×3 → Flatten → Dense → Trainer |
| 68 | 초해상도 이미지 (SRCNN) | NumpyInput → Conv2D×3 → Trainer(MSE Loss) |
| 69 | 스타일 전이 특성 추출 | ImageFolder → PretrainedModel(VGG, feature) → tSNE Viewer |
| 70 | 결함 이미지 분류 (소량 데이터) | ImageFolder → Augmentation → PretrainedModel → Dense → Trainer |

### 자연어처리 (NLP) — 예제 71~82
| # | 예제 | 사용 노드 |
|---|------|----------|
| 71 | 영화 리뷰 감성 분류 | CSV → Embedding → LSTM → Dense → Trainer |
| 72 | 텍스트 독성 분류 | CSV → Embedding → GRU → Dense → Trainer |
| 73 | 뉴스 카테고리 분류 | CSV → Embedding → LSTM×2 → Dense → Trainer |
| 74 | 기계 번역 품질 예측 | CSV → Embedding → Transformer Block → Dense → Trainer |
| 75 | 문장 유사도 예측 | CSV → Embedding×2 → LSTM → Add → Dense → Trainer |
| 76 | 개체명 인식 (NER) | CSV → Embedding → Bidirectional LSTM → Dense → Trainer |
| 77 | 다음 단어 예측 (언어 모델) | CSV → Embedding → LSTM×2 → Dense → Trainer |
| 78 | 키워드 추출 점수 예측 | CSV → Embedding → GRU → Dense → Trainer |
| 79 | 질문-답변 유형 분류 | CSV → Embedding → Transformer Block → Dense → Trainer |
| 80 | 챗봇 의도 분류 | CSV → Embedding → LSTM → Dense → Trainer |
| 81 | 코드 언어 분류 | CSV → Embedding → Conv1D → GlobalAvgPool → Dense → Trainer |
| 82 | 감성 강도 회귀 | CSV → Embedding → LSTM → Dense(regression) → Trainer |

### 시계열 심화 — 예제 83~91
| # | 예제 | 사용 노드 |
|---|------|----------|
| 83 | 다변량 시계열 분류 | CSV → StandardScaler → LSTM → Dense → Trainer |
| 84 | 멀티스텝 예측 | CSV → MinMaxScaler → LSTM×2 → Dense(N출력) → Trainer |
| 85 | CNN+LSTM 하이브리드 예측 | CSV → Conv1D → LSTM → Dense → Trainer |
| 86 | 심전도(ECG) 이상 감지 | CSV → StandardScaler → LSTM(Autoencoder) → Trainer |
| 87 | 진동 센서 고장 예측 | CSV → StandardScaler → GRU → Dense → Trainer |
| 88 | 날씨 다음 날 예측 | CSV → MinMaxScaler → Transformer Block → Dense → Trainer |
| 89 | IoT 센서 이상 탐지 | CSV → MinMaxScaler → LSTM → Dense → Trainer |
| 90 | 멀티채널 시계열 분류 | CSV → Reshape → LSTM → Dense → Trainer |
| 91 | Seq2Seq 인코더-디코더 예측 | CSV → MinMaxScaler → LSTM(Encoder) → LSTM(Decoder) → Dense → Trainer |

### 고급 파이프라인 — 예제 92~100
| # | 예제 | 사용 노드 |
|---|------|----------|
| 92 | 앙상블 모델 비교 | 동일 데이터 → 3개 모델 병렬 학습 → Metrics 비교 |
| 93 | 하이퍼파라미터 탐색 | CSV → StandardScaler → Dense → Trainer(Grid Search) |
| 94 | 혼합 입력 모델 (이미지+수치) | ImageFolder + CSV → Conv2D + Dense → Concat → Dense → Trainer |
| 95 | 점진적 전이학습 (단계별 해동) | ImageFolder → PretrainedModel(frozen→unfreeze) → Dense → Trainer |
| 96 | Knowledge Distillation | Teacher Model → Student Model 학습 → Metrics 비교 |
| 97 | 클래스 불균형 처리 비교 | CSV → ClassWeight/Oversample/FocalLoss 세 갈래 → Trainer 비교 |
| 98 | 교차 검증 (K-Fold) | CSV → StandardScaler → Dense → KFoldTrainer → Metrics |
| 99 | ONNX 내보내기 + C# 추론 | 학습된 모델 → ONNX Export → (C# OnnxRuntime 연동 코드 생성) |
| 100 | 전체 파이프라인 종합 예제 | CSV+Image → 전처리 → CNN+Dense 혼합 → 학습 → 평가 → ONNX Export |

---

## 학습 교재 (PDF)

> 파일 위치: `docs/TEXTBOOK.pdf`  
> 원본 Markdown: `docs/TEXTBOOK.md`

### 교재 구성

| 장 | 제목 | 내용 |
|----|------|------|
| 1장 | 프로그램 소개 | ML Node Studio 개요, 두 플랫폼 비교 |
| 2장 | 설치 및 실행 | Web/WPF 설치, Python 백엔드 설정, GPU 설정 |
| 3장 | UI 사용법 | 캔버스 조작, 노드 추가/연결, 단축키, 파이프라인 저장 |
| 4장 | 노드 카탈로그 | 42개 노드 전체 파라미터 및 기능 설명 |
| 5장 | 노드 연결 패턴 | 데이터 흐름 원칙, 자주 쓰는 연결 패턴 10가지 |
| 6장 | 기본 예제 5가지 | 따라하기 형식의 단계별 스크린샷 예제 |
| 7장 | 학습 예제 100가지 | 분류/회귀/비지도/이미지/NLP/시계열/고급 전체 수록 |

### 교재 빌드 명령
```bash
# Markdown → PDF 변환
cd docs
pip install reportlab
python build_pdf.py       # TEXTBOOK.md → TEXTBOOK.pdf 생성
```

---

## Key Constraints & Notes

- **Python 백엔드는 Web/WPF 공통** → 기능 추가 시 한 번만 작성
- **WPF는 Web 검증 후 포팅** → React Flow 노드 구조를 Nodify로 1:1 대응
- **Python 학습 프로세스는 서브프로세스로 분리** → 프론트 UI 블록 방지
- **GPU 사용 여부 자동 감지** → `torch.cuda.is_available()` / `tf.config.list_physical_devices('GPU')`
- **노드 파라미터 유효성 검사**는 프론트와 백엔드 양쪽에서 수행
- **대용량 모델 가중치**는 WebSocket 전송 금지 → 파일시스템 경로 전달
- **ONNX Export 노드**는 WPF 배포(C# OnnxRuntime)와 직접 연결되는 핵심 노드
- **교재 PDF**는 `docs/TEXTBOOK.md`를 단일 소스로 관리, `build_pdf.py`로 빌드
- **프레임워크 추상화**: `BaseTrainer` 인터페이스로 TF/PyTorch 구현체를 교환 가능하게 설계

---

## Commands

```bash
# ── 백엔드 (공통) ──────────────────────────────────
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# ── Web 프론트엔드 ──────────────────────────────────
cd frontend-web
npm install
npm run dev        # http://localhost:5173

# ── WPF 프론트엔드 ─────────────────────────────────
# Visual Studio 2022에서 frontend-wpf/MLNodeStudio.sln 열고 F5

# ── 교재 PDF 빌드 ──────────────────────────────────
cd docs
pip install reportlab
python build_pdf.py    # TEXTBOOK.md → TEXTBOOK.pdf

# ── 주요 의존성 ────────────────────────────────────
# backend/requirements.txt:
fastapi uvicorn websockets
tensorflow torch torchvision
numpy pandas scikit-learn
matplotlib seaborn umap-learn
onnx onnxruntime
Pillow reportlab
```
