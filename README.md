# ML Node Studio

노드 기반의 머신러닝 학습 파이프라인 설계 / 실행 툴.
사용자가 노드를 드래그앤드롭으로 연결해 **데이터 전처리 → 모델 정의 → 학습 → 평가 → 시각화**
까지의 전체 ML 워크플로를 시각적으로 구성하고 실행할 수 있는 웹 애플리케이션이다.

---

## 프로그램 개요

- **대상**: 코드를 직접 작성하지 않고도 TensorFlow / PyTorch 기반 모델 학습을 실습하고 싶은 학습자 및 엔지니어
- **핵심 기능**
  - React Flow 기반 노드 캔버스에서 파이프라인을 시각적으로 구성
  - FastAPI + WebSocket을 통한 실시간 학습 진행 상황 전송 (epoch / loss / accuracy)
  - TensorFlow 2.x와 PyTorch 두 프레임워크를 노드 단위에서 선택 가능
  - 학습된 모델의 저장 / 불러오기 / 예측 / ONNX 내보내기까지 한 화면에서 처리
  - 예제 파이프라인 브라우저 (분류 / 회귀 / 비지도 / 이미지 / NLP / 시계열 등 100종)

---

## 프로젝트 구조

```
MLStudio/
├── backend/                         # FastAPI 기반 공통 백엔드
│   ├── main.py                      # REST / WebSocket 엔드포인트 진입점
│   ├── pipeline_executor.py         # 노드 그래프 토폴로지 정렬 + 실행 엔진
│   ├── nodes/                       # 노드 핸들러 구현
│   │   ├── base.py                  #   - BaseNode 추상 클래스
│   │   ├── registry.py              #   - NODE_REGISTRY (타입 → 핸들러)
│   │   ├── data_nodes.py            #   - CSV / 이미지 / 스케일러 / 인코더 등
│   │   ├── layer_nodes.py           #   - Dense / Conv2D / LSTM / Attention 등
│   │   ├── trainer_nodes.py         #   - Optimizer / Loss / Trainer / Callback
│   │   ├── model_io_nodes.py        #   - 모델 로더 / 테스터
│   │   └── viz_nodes.py             #   - Loss / Accuracy / Confusion Matrix 등
│   ├── frameworks/
│   │   ├── tf_backend.py            #   - TensorFlow 백엔드 어댑터
│   │   └── torch_backend.py         #   - PyTorch 백엔드 어댑터
│   ├── examples/                    # 예제 파이프라인 JSON 저장소
│   ├── scripts/                     # 예제 생성 유틸
│   └── requirements.txt
│
├── frontend-web/                    # React + React Flow 프런트엔드
│   ├── src/
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   ├── nodes/                   # MLNode 커스텀 노드 + 타입 정의
│   │   ├── components/              # Sidebar / Toolbar / Properties / Panels
│   │   ├── charts/                  # Recharts 기반 실시간 학습 차트
│   │   └── store/                   # Zustand 상태 저장소
│   ├── vite.config.ts               # /api, /ws 프록시 설정
│   ├── tailwind.config.js
│   └── package.json
│
├── start.bat                        # 백엔드 + 프런트엔드 동시 실행 런처
├── stop.bat                         # 실행 중인 서비스 종료
├── vite.config.js                   # 루트 레벨 vite 보조 설정
├── CLAUDE.md                        # 설계 문서 / 노드 카탈로그 원본
└── README.md                        # (이 문서)
```

---

## 기술 스택

### 백엔드 (Python)

| 역할              | 라이브러리                                     |
| ----------------- | ---------------------------------------------- |
| API 서버          | FastAPI 0.115, Uvicorn                         |
| 실시간 통신       | WebSocket (FastAPI 내장), websockets 12        |
| ML 프레임워크     | TensorFlow 2.16, PyTorch 2.3, TorchVision 0.18 |
| 데이터 처리       | NumPy 1.26, Pandas 2.2, scikit-learn 1.5       |
| 시각화 데이터생성 | Matplotlib 3.9, Seaborn 0.13                   |
| 모델 내보내기     | ONNX 1.16, ONNX Runtime 1.18                   |
| 이미지 처리       | Pillow 10.4                                    |

### 프런트엔드 (Web)

| 역할         | 라이브러리                |
| ------------ | ------------------------- |
| 번들러 / 개발 서버 | Vite 5                    |
| UI 프레임워크    | React 18 + TypeScript 5   |
| 노드 에디터     | @xyflow/react (React Flow) 12 |
| 차트         | Recharts 2                |
| 상태 관리      | Zustand 4                 |
| 스타일        | Tailwind CSS 3, PostCSS, Autoprefixer |
| 보고서 내보내기  | html-to-image, html2pdf.js, jsPDF |

### 아키텍처

```
[Browser: React + React Flow]
        ↕  WebSocket (/ws)  +  REST (/api)
[FastAPI Backend  :8000]
        ↕
[TensorFlow / PyTorch 학습 엔진]
        ↕
[파일시스템 (uploads / outputs / examples)]
```

---

## 로컬 실행 방법

### 1. 사전 준비

- **Python 3.9** (권장 — `requirements.txt`가 TF 2.16 / Torch 2.3에 묶여 있음)
- **Node.js 18+** (Vite 5 요구사항)
- (선택) NVIDIA GPU + CUDA — 설치된 경우 자동 감지

### 2. 백엔드 의존성 설치

```bash
cd backend
pip install -r requirements.txt
```

### 3. 프런트엔드 의존성 설치

```bash
cd frontend-web
npm install
```

### 4. `start.bat` 수정

현재 `start.bat`은 작성자의 개발 경로(`D:\Work\WebSharp\...`)와 Python 경로가 하드코딩되어
있어 **그대로 실행하면 동작하지 않는다.** 자기 환경에 맞게 다음 3줄을 수정해야 한다.

```bat
:: 수정 전 (원본)
set "PYTHON=%USERPROFILE%\AppData\Local\Programs\Python\Python39\python.exe"
set "BACKEND=D:\Work\WebSharp\backend"
set "FRONTEND=D:\Work\WebSharp\frontend-web"
```

| 변수       | 바꿀 값                                                                  |
| ---------- | ----------------------------------------------------------------------- |
| `PYTHON`   | 실제 Python 3.9 실행파일 경로. 예) `C:\Python39\python.exe`, 또는 venv 내부 |
| `BACKEND`  | 이 프로젝트의 `backend` 폴더 절대경로. 예) `D:\Work\MLStudio\backend`         |
| `FRONTEND` | 이 프로젝트의 `frontend-web` 폴더 절대경로. 예) `D:\Work\MLStudio\frontend-web` |

> 💡 **팁**: 프로젝트를 옮겨 다닐 계획이라면 배치 파일이 위치한 폴더를 기준으로 상대화할 수 있다.
>
> ```bat
> set "PYTHON=python"
> set "BACKEND=%~dp0backend"
> set "FRONTEND=%~dp0frontend-web"
> ```
>
> `%~dp0`은 해당 `.bat` 파일이 있는 폴더 경로로 치환되므로, 어디에 복사해 놓아도 바로 동작한다.
> 단 `python` 명령이 PATH에 등록되어 있어야 하며, 버전이 맞지 않으면 위와 같이 절대경로를 지정한다.

### 5. 실행

```bat
start.bat
```

실행되면 다음이 자동으로 수행된다.

1. `ML-Backend` 창 — `python main.py` (FastAPI, `http://localhost:8000`)
2. `ML-Frontend` 창 — `npm run dev -- --host` (Vite, `http://localhost:5173`)
3. 기본 브라우저에서 `http://localhost:5173` 자동 오픈

접속 가능한 URL:

- Local : `http://localhost:5173`
- Backend : `http://localhost:8000` (헬스체크: `/api/health`)
- Network : 같은 LAN 내 다른 기기에서 `http://<PC의 IPv4>:5173`

### 6. 종료

런처 창에서 아무 키나 눌러 종료하거나, 따로 띄워 둔 경우 `stop.bat`를 실행한다.

```bat
stop.bat
```

---

## GitHub Pages 배포 (웹 데모)

**접속 주소: https://samcho93.github.io/MLStudio/**

빌드 결과물은 `gh-pages` 브랜치로 배포한다. 코드를 수정한 뒤 다음 명령으로 다시 배포한다.

```bat
cd frontend-web
set PYTHON=%USERPROFILE%\AppData\Local\Programs\Python\Python39\python.exe
npm run deploy:pages
```

`scripts/deploy-pages.mjs` 가 수행하는 작업:

1. `backend/scripts/export_static_api.py` — 노드 카탈로그 / 예제 130종을 `frontend-web/public/static-api/*.json` 으로 내보냄
2. `VITE_BASE_PATH=/MLStudio/`, `VITE_STATIC_HOST=1` 로 `npm run build`
3. `frontend-web/dist` 를 `gh-pages` 브랜치에 force push

> 최초 1회: 저장소 **Settings → Pages → Build and deployment** 에서 Source 를 **Deploy from a branch**, Branch 를 **gh-pages / (root)** 로 지정해야 한다.

### 브라우저 모드 (백엔드 없이 학습)

GitHub Pages 는 정적 호스팅이라 Python 백엔드가 없다.
이 경우 **TensorFlow.js 로 브라우저에서 직접 학습**한다 (`frontend-web/src/local/`).

- 학습 데이터: `backend/datasets/*.csv` (저장소에 포함, 빌드 시 `/datasets/` 로 복사됨)
- 브라우저에서 지원하는 노드
  - 데이터: `CSVLoader`, `TrainValSplit`, `StandardScaler`, `MinMaxScaler`, `LabelEncoder`, `OneHotEncoder`, `DataInspector`
  - 레이어: `Dense`, `Dropout`, `BatchNorm`, `Flatten`
  - 학습: `Optimizer`, `LossFunction`, `EarlyStopping`, `LRScheduler`, `ModelCheckpoint`, `Trainer`
  - 평가/시각화: `ClassificationMetrics`, `RegressionMetrics`, `ConfusionMatrix`, `ModelSummary`, `LossCurve`, `AccuracyCurve`
- 예제 130개 중 **45개**(표 데이터 + Dense 계열)가 브라우저에서 그대로 학습된다.
  지원하지 않는 노드가 있으면 실행 시 해당 노드 이름을 알려준다.

| 백엔드 없이 가능 | 백엔드 필요 |
| --- | --- |
| 노드 편집, 예제 불러오기, 도움말, 보고서(PDF), **표 데이터 학습 + 평가/시각화** | CNN·LSTM·Transformer·전이학습, 이미지 데이터셋, 예측 패널, 모델 저장/불러오기, 파이프라인 Save/Load, ONNX 내보내기 |

모든 노드를 사용하려면 로컬 PC에서 백엔드를 실행하고 **HTTPS 터널**로 공개한 뒤,
툴바 오른쪽 **[브라우저 모드]** 버튼에 주소를 입력한다. (Pages 는 HTTPS 이므로 `http://` 원격 주소는 브라우저가 차단한다)

```bat
cd backend
python -X utf8 main.py
ngrok http 8000
```

주소는 URL 쿼리로도 지정할 수 있으며 브라우저에 저장된다: `https://samcho93.github.io/MLStudio/?api=https://xxxx.ngrok-free.app`
(초기화: `?api=`)

---

## 노드 카탈로그 요약

| 분류       | 대표 노드                                                                  |
| ---------- | ------------------------------------------------------------------------- |
| 데이터     | CSV Loader, Image Folder, Numpy Input, Train/Val Split, StandardScaler 등 |
| 레이어     | Dense, Conv2D, MaxPooling2D, LSTM, GRU, Embedding, Transformer Block 등   |
| 학습       | Optimizer, Loss Function, LR Scheduler, Trainer, Early Stopping           |
| 평가       | Classification Metrics, Confusion Matrix, ROC Curve, Prediction Viewer    |
| 시각화     | Loss Curve, Accuracy Curve, Feature Map, t-SNE, Grad-CAM                  |
| 입출력     | Model Save / Load, ONNX Export, TFLite Export                             |

노드별 입출력 포트와 파라미터는 프런트엔드의 Properties 패널,
또는 `GET /api/node-catalog` 엔드포인트에서 동적으로 확인할 수 있다.

---

## WebSocket 프로토콜 (요약)

```jsonc
// Client → Server
{ "type": "START_TRAINING", "pipeline": { "nodes": [...], "edges": [...] } }

// Server → Client (에폭마다)
{ "type": "EPOCH_UPDATE", "epoch": 5, "loss": 0.34, "val_loss": 0.40,
  "accuracy": 0.87, "val_accuracy": 0.85, "lr": 3e-4 }

// Server → Client (완료)
{ "type": "TRAINING_COMPLETE", "model_path": "...", "metrics": {...} }

// Server → Client (오류)
{ "type": "ERROR", "message": "...", "node_id": "..." }
```

자세한 설계 배경과 Phase 계획, 100종 예제 분류는 `CLAUDE.md`에 정리되어 있다.
