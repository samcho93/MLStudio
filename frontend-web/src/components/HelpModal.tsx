import React, { useState, useEffect, useMemo } from 'react';
import { useStore } from '../store/useStore';

/* ───────────────────────────── 타입 ─────────────────────────────── */
interface ExampleFull {
  meta: {
    id: number;
    title: string;
    title_ko: string;
    category: string;
    difficulty: string;
    description: string;
    description_ko: string;
    tags: string[];
    estimated_time: string;
    dataset_info?: {
      name: string;
      source: string;
      samples: number;
      features: number | string;
      classes?: number;
      target?: string;
    };
    learning_objectives: string[];
    guide_steps: { step: number; node: string; text: string }[];
  };
  nodes: { id: string; type: string; position: { x: number; y: number }; params: Record<string, any> }[];
  edges: { source: string; target: string; sourceHandle?: string; targetHandle?: string }[];
}

/* ───────────────────────── 노드 카테고리 색상 ────────────────────── */
const CAT_COLORS: Record<string, string> = {
  data: '#3b82f6', layer: '#8b5cf6', training: '#f59e0b',
  visualization: '#10b981', evaluation: '#10b981', output: '#ef4444',
};
const CAT_LABELS: Record<string, string> = {
  data: 'DATA', layer: 'LAYER', training: 'TRAIN',
  visualization: 'VIZ', evaluation: 'EVAL', output: 'OUT',
};

const DIFF_COLORS: Record<string, string> = {
  beginner: '#22c55e', intermediate: '#eab308', advanced: '#ef4444',
};
const DIFF_LABELS: Record<string, string> = {
  beginner: 'Beginner', intermediate: 'Intermediate', advanced: 'Advanced',
};

const CATEGORY_INFO: Record<string, { label: string; label_ko: string; icon: string }> = {
  classification: { label: 'Classification', label_ko: '분류', icon: '🏷️' },
  regression: { label: 'Regression', label_ko: '회귀', icon: '📈' },
  unsupervised: { label: 'Unsupervised', label_ko: '비지도/클러스터링', icon: '🔍' },
  image: { label: 'Image', label_ko: '이미지 심화', icon: '🖼️' },
  nlp: { label: 'NLP', label_ko: '자연어처리', icon: '💬' },
  timeseries: { label: 'Time Series', label_ko: '시계열', icon: '📊' },
  advanced: { label: 'Advanced', label_ko: '고급 파이프라인', icon: '⚡' },
};

/* ──────────────────── 미니 노드 컴포넌트 (비주얼) ────────────────── */
function MiniNode({ type, category, params, compact }: {
  type: string; category: string; params?: Record<string, any>; compact?: boolean;
}) {
  const color = CAT_COLORS[category] || '#6b7280';
  const catLabel = CAT_LABELS[category] || 'MISC';
  const displayParams = params ? Object.entries(params).filter(([k]) => !k.startsWith('_') && k !== 'file_path' && k !== 'target_column').slice(0, compact ? 2 : 5) : [];

  return (
    <div className="inline-block rounded-lg shadow-lg" style={{
      border: `2px solid ${color}`,
      background: '#1e1e2e',
      minWidth: compact ? 100 : 160,
    }}>
      <div className="px-2 py-1 rounded-t-md flex items-center gap-1.5" style={{ background: color + '22' }}>
        <span className="text-[8px] font-bold px-1 py-0.5 rounded" style={{ background: color, color: '#fff' }}>
          {catLabel}
        </span>
        <span className="text-[10px] font-semibold text-gray-200">{type}</span>
      </div>
      {displayParams.length > 0 && (
        <div className="px-2 py-1 text-[9px] text-gray-500" style={{ borderTop: '1px solid #333' }}>
          {displayParams.map(([k, v]) => (
            <div key={k} className="flex justify-between">
              <span>{k}</span>
              <span className="text-gray-400 ml-2">{String(v).substring(0, 15)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ─────────────── 파이프라인 다이어그램 (노드+엣지) ──────────────── */
function PipelineDiagram({ nodes, edges, catalog }: {
  nodes: ExampleFull['nodes']; edges: ExampleFull['edges'];
  catalog: { type: string; category: string }[];
}) {
  // 토폴로지 정렬하여 레벨별 그룹핑
  const catMap = useMemo(() => {
    const m: Record<string, string> = {};
    catalog.forEach(c => { m[c.type] = c.category; });
    return m;
  }, [catalog]);

  // y좌표 기준으로 그룹, x좌표 기준으로 정렬
  const sorted = [...nodes].sort((a, b) => a.position.x - b.position.x);

  // 엣지 연결 맵
  const edgeMap: Record<string, string[]> = {};
  edges.forEach(e => {
    if (!edgeMap[e.source]) edgeMap[e.source] = [];
    edgeMap[e.source].push(e.target);
  });

  return (
    <div className="overflow-x-auto pb-2">
      <div className="flex items-start gap-2 min-w-max">
        {sorted.map((node, i) => (
          <React.Fragment key={node.id}>
            <div className="flex flex-col items-center">
              <MiniNode type={node.type} category={catMap[node.type] || 'misc'} params={node.params} compact />
              <span className="text-[8px] text-gray-600 mt-0.5">{node.id}</span>
            </div>
            {i < sorted.length - 1 && (
              <div className="flex items-center self-center text-gray-600 text-xs mt-2">→</div>
            )}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

/* ──────────────────── 노드 상세 설명 데이터 ────────────────────── */
export const NODE_DESCRIPTIONS: Record<string, { desc_ko: string; detail: string; tips: string }> = {
  CSVLoader: { desc_ko: 'CSV 파일 로드', detail: 'CSV 파일을 읽어 특성(features)과 레이블(labels)로 분리합니다. target_column을 지정하면 해당 컬럼이 레이블이 되고, 나머지가 특성이 됩니다. NLP 모드에서는 텍스트를 토큰화하여 정수 시퀀스로 변환합니다.', tips: '문자열 컬럼은 자동으로 LabelEncoder가 적용됩니다.' },
  NumpyInput: { desc_ko: '내장 데이터셋 로드', detail: 'MNIST, Fashion-MNIST, CIFAR-10, Iris, Boston 등 내장 데이터셋을 로드합니다. 별도의 파일 없이 바로 학습에 사용할 수 있습니다.', tips: '이미지 데이터셋은 자동으로 0~1 범위로 정규화됩니다.' },
  ImageFolder: { desc_ko: '이미지 폴더 로드', detail: '폴더 구조(각 서브폴더=클래스)로 이미지를 로드합니다. 자동으로 리사이즈와 정규화가 적용됩니다.', tips: '폴더 구조: root/class_a/img1.jpg, root/class_b/img2.jpg' },
  TrainValSplit: { desc_ko: '데이터 분할', detail: '데이터를 학습/검증/테스트 세트로 분할합니다. stratified 분할을 지원하여 클래스 비율을 유지합니다.', tips: 'random_seed를 고정하면 재현 가능한 분할이 됩니다.' },
  StandardScaler: { desc_ko: 'Z-score 정규화', detail: '각 특성의 평균을 0, 표준편차를 1로 변환합니다. 특성의 스케일이 다를 때 필수적입니다.', tips: 'Dense 레이어 전에 사용하면 학습 속도가 빨라집니다.' },
  MinMaxScaler: { desc_ko: 'Min-Max 정규화', detail: '각 특성을 0~1 범위로 변환합니다. 값의 범위가 정해진 경우에 유용합니다.', tips: '시계열 데이터에 주로 사용됩니다.' },
  LabelEncoder: { desc_ko: '레이블 인코딩', detail: '문자열 레이블을 정수로 변환합니다 (예: cat→0, dog→1).', tips: '분류 문제에서 타겟 변수가 문자열일 때 사용합니다.' },
  OneHotEncoder: { desc_ko: 'One-Hot 인코딩', detail: '범주형 변수를 이진 벡터로 변환합니다 (예: [1,0,0], [0,1,0]).', tips: '범주가 많은 경우 차원이 크게 증가할 수 있습니다.' },
  PCA: { desc_ko: '주성분 분석', detail: '고차원 데이터를 저차원으로 축소합니다. 데이터의 분산을 최대한 보존하는 방향을 찾습니다.', tips: '시각화(2D/3D)나 노이즈 제거에 활용됩니다.' },
  Augmentation: { desc_ko: '이미지 증강', detail: '회전, 뒤집기, 크롭 등으로 학습 데이터를 인위적으로 늘립니다. 과적합을 방지합니다.', tips: '소량 데이터셋에서 특히 효과적입니다.' },
  Dense: { desc_ko: '완전연결층 (Fully Connected)', detail: '모든 입력 뉴런과 출력 뉴런이 연결된 레이어입니다. units는 출력 뉴런 수, activation은 활성화 함수를 지정합니다.', tips: '마지막 레이어: 분류=softmax(다중)/sigmoid(이진), 회귀=linear' },
  Conv2D: { desc_ko: '2D 합성곱층', detail: '이미지에서 공간적 특성(엣지, 텍스처, 패턴)을 추출합니다. filters는 필터 수, kernel_size는 필터 크기입니다.', tips: '여러 Conv2D를 쌓아 저수준→고수준 특성을 순차적으로 학습합니다.' },
  Conv1D: { desc_ko: '1D 합성곱층', detail: '시퀀스 데이터에서 지역적 패턴을 추출합니다. NLP나 시계열에 사용됩니다.', tips: 'LSTM/GRU와 결합하여 하이브리드 모델을 만들 수 있습니다.' },
  Conv2DTranspose: { desc_ko: '전치 합성곱 (업샘플링)', detail: '특성 맵의 크기를 키웁니다. 오토인코더의 디코더나 U-Net의 확장 경로에 사용됩니다.', tips: 'stride를 2로 설정하면 크기가 2배로 증가합니다.' },
  MaxPooling2D: { desc_ko: '최대 풀링', detail: '특성 맵의 공간 크기를 줄이면서 가장 강한 활성화를 유지합니다.', tips: 'pool_size=2이면 가로·세로 각각 절반으로 줄어듭니다.' },
  Flatten: { desc_ko: 'Flatten', detail: '다차원 텐서를 1D 벡터로 펼칩니다. Conv → Dense 연결 시 필수입니다.', tips: 'GlobalAveragePooling2D가 더 적은 파라미터를 사용합니다.' },
  GlobalAveragePooling2D: { desc_ko: '글로벌 평균 풀링 (2D)', detail: '각 특성 맵의 공간 차원을 평균값 하나로 요약합니다. Flatten 대비 파라미터가 적어 과적합에 강합니다.', tips: '전이학습의 헤드 레이어로 자주 사용됩니다.' },
  GlobalAveragePooling1D: { desc_ko: '글로벌 평균 풀링 (1D)', detail: '시퀀스 차원을 평균값으로 요약합니다. NLP에서 문장 벡터를 만들 때 사용합니다.', tips: 'Conv1D 뒤에 사용하여 고정 길이 벡터를 만듭니다.' },
  BatchNorm: { desc_ko: '배치 정규화', detail: '각 미니배치의 활성화를 정규화하여 학습을 안정화합니다. 더 높은 학습률을 사용할 수 있게 합니다.', tips: 'Dense/Conv 뒤, Activation 앞에 배치합니다.' },
  Dropout: { desc_ko: '드롭아웃', detail: '학습 중 랜덤하게 뉴런을 비활성화하여 과적합을 방지합니다.', tips: 'rate=0.5면 50%의 뉴런이 꺼집니다. 일반적으로 0.2~0.5를 사용합니다.' },
  Embedding: { desc_ko: '임베딩', detail: '정수 인덱스를 밀집 벡터로 변환합니다. 단어를 의미 공간의 점으로 매핑합니다.', tips: 'NLP 파이프라인의 첫 레이어로 사용합니다.' },
  LSTM: { desc_ko: 'LSTM (장단기 메모리)', detail: '시퀀스 데이터의 장기 의존성을 학습합니다. 게이트 메커니즘으로 중요한 정보를 선별적으로 기억합니다.', tips: 'return_sequences=true면 각 타임스텝의 출력을 모두 반환합니다.' },
  GRU: { desc_ko: 'GRU (게이트 순환 유닛)', detail: 'LSTM의 간소화 버전으로, 더 적은 파라미터로 비슷한 성능을 냅니다.', tips: '데이터가 적을 때 LSTM 대비 유리할 수 있습니다.' },
  Reshape: { desc_ko: 'Reshape', detail: '텐서의 shape을 변환합니다. 데이터의 총 원소 수는 유지됩니다.', tips: '시계열 데이터를 LSTM 입력 형태로 바꿀 때 사용합니다.' },
  Add: { desc_ko: 'Add (잔차 연결)', detail: '두 텐서를 원소별로 더합니다. ResNet의 Skip Connection에 사용됩니다.', tips: '두 입력의 shape이 동일해야 합니다.' },
  Concat: { desc_ko: 'Concatenate (연결)', detail: '두 텐서를 지정한 축으로 이어 붙입니다. 다중 입력 모델의 병합에 사용됩니다.', tips: 'axis=-1(마지막 축)이 가장 일반적입니다.' },
  PretrainedModel: { desc_ko: '사전학습 모델', detail: 'ImageNet으로 학습된 모델(MobileNetV2, ResNet50 등)을 백본으로 사용합니다. trainable=false면 특성 추출기로, true면 파인튜닝됩니다.', tips: '소량 데이터에서 처음부터 학습하는 것보다 훨씬 좋은 성능을 냅니다.' },
  TransformerBlock: { desc_ko: 'Transformer 블록', detail: 'Multi-Head Attention + Feed-Forward Network + Layer Normalization으로 구성됩니다. 자연어 처리의 핵심 구조입니다.', tips: 'BERT, GPT 등 최신 모델의 기반이 됩니다.' },
  MultiHeadAttention: { desc_ko: '멀티 헤드 어텐션', detail: '여러 개의 어텐션 헤드가 독립적으로 입력의 다양한 부분에 주목합니다.', tips: 'num_heads×key_dim이 입력 차원과 일치해야 합니다.' },
  Optimizer: { desc_ko: '옵티마이저', detail: '모델 파라미터를 업데이트하는 알고리즘을 선택합니다. Adam이 가장 범용적입니다.', tips: 'learning_rate: 보통 0.001~0.0001 범위를 사용합니다.' },
  LossFunction: { desc_ko: '손실 함수', detail: '모델의 예측과 정답 사이의 차이를 측정합니다. 분류: CrossEntropy, 회귀: MSE/MAE', tips: '분류 문제에서 원-핫이 아닌 정수 레이블이면 sparse_categorical을 사용합니다.' },
  EarlyStopping: { desc_ko: '조기 종료', detail: '검증 손실이 patience 에폭 동안 개선되지 않으면 학습을 중단합니다.', tips: '과적합을 방지하는 가장 효과적인 방법 중 하나입니다.' },
  LRScheduler: { desc_ko: '학습률 스케줄러', detail: '학습 중 학습률을 동적으로 조절합니다. CosineAnnealing이 안정적인 성능을 보입니다.', tips: 'ReduceOnPlateau: 검증 손실이 정체되면 학습률을 줄입니다.' },
  Trainer: { desc_ko: '학습 실행기', detail: '모델을 학습합니다. 배치 단위로 순전파→손실 계산→역전파→파라미터 업데이트를 반복합니다.', tips: '프레임워크를 pytorch 또는 tensorflow로 선택할 수 있습니다.' },
  KFoldTrainer: { desc_ko: 'K-Fold 교차 검증', detail: '데이터를 K개로 나누어 각각 학습/검증하여 안정적인 성능 추정치를 얻습니다.', tips: '소량 데이터에서 단일 분할보다 신뢰할 수 있는 평가를 제공합니다.' },
  LossCurve: { desc_ko: 'Loss 곡선', detail: '에폭별 학습/검증 손실의 추이를 시각화합니다.', tips: '학습 손실↓, 검증 손실↑이면 과적합 신호입니다.' },
  AccuracyCurve: { desc_ko: 'Accuracy 곡선', detail: '에폭별 학습/검증 정확도의 추이를 시각화합니다.', tips: '검증 정확도가 더 이상 오르지 않으면 학습을 중단할 시점입니다.' },
  ModelSummary: { desc_ko: '모델 구조 요약', detail: '레이어 구조, 파라미터 수, 출력 shape을 텍스트로 표시합니다.', tips: '전체 파라미터 수로 모델 복잡도를 파악할 수 있습니다.' },
  ClassificationMetrics: { desc_ko: '분류 메트릭', detail: 'Accuracy, Precision, Recall, F1 Score, AUC 등 분류 성능 지표를 계산합니다.', tips: '불균형 데이터에서는 Accuracy보다 F1이 더 신뢰할 수 있습니다.' },
  ConfusionMatrix: { desc_ko: '혼동 행렬', detail: '각 클래스별 예측/정답의 교차 테이블을 히트맵으로 시각화합니다.', tips: '대각선이 진할수록 정확한 예측입니다.' },
  RegressionMetrics: { desc_ko: '회귀 메트릭', detail: 'MAE, RMSE, R² 등 회귀 성능 지표를 계산합니다.', tips: 'R²이 1에 가까울수록 좋은 모델입니다.' },
  ROCCurve: { desc_ko: 'ROC 곡선', detail: 'FPR-TPR 곡선과 AUC를 시각화합니다. 이진 분류 성능 평가에 사용됩니다.', tips: 'AUC가 0.5면 랜덤, 1.0이면 완벽한 분류입니다.' },
  ModelSave: { desc_ko: '모델 저장', detail: '학습된 모델을 파일로 저장합니다. PyTorch: .pth, TensorFlow: .keras', tips: 'ONNX Export 노드와 연계하여 다양한 포맷으로 변환할 수 있습니다.' },
  ModelLoader: { desc_ko: '저장된 모델 로드', detail: 'Model Manager에서 저장한 모델을 다시 불러옵니다. 모델 파일(.keras/.pth)과 메타데이터(feature_names, class_names, scaler 등)를 함께 복원합니다. TensorFlow와 PyTorch 모두 지원합니다.', tips: 'ModelTester와 연결하여 새 데이터로 테스트하거나, 다른 파이프라인에서 전이학습의 시작점으로 활용할 수 있습니다.' },
  ModelTester: { desc_ko: '모델 테스트', detail: '학습된 모델에 테스트 데이터를 넣어 예측을 수행하고 성능을 평가합니다. 분류: Accuracy, Precision, Recall, F1, Confusion Matrix를 계산합니다. 회귀: MSE, RMSE, MAE, R2 Score를 계산합니다. 샘플별 예측 vs 정답 비교 테이블도 생성합니다.', tips: 'Trainer의 model 출력 또는 ModelLoader의 model 출력을 연결하세요. test_labels를 연결하면 메트릭을 계산하고, 생략하면 예측만 수행합니다.' },
  DataInspector: { desc_ko: '데이터 미리보기', detail: '데이터의 shape, dtype, 샘플 값을 확인합니다. 디버깅에 유용합니다.', tips: '파이프라인 중간에 삽입하여 데이터 흐름을 확인할 수 있습니다.' },
  ModelCheckpoint: { desc_ko: '모델 체크포인트', detail: '학습 중 최적 모델을 자동 저장합니다. 검증 손실이 가장 낮은 시점의 가중치를 보존합니다.', tips: 'EarlyStopping과 함께 사용하면 최적 모델을 확보할 수 있습니다.' },
  ONNXExport: { desc_ko: 'ONNX 내보내기', detail: '학습된 모델을 ONNX 포맷으로 변환합니다. C# OnnxRuntime 등 다양한 런타임에서 추론할 수 있습니다.', tips: 'WPF 버전과 연동하여 데스크톱 앱에서 모델을 사용할 수 있습니다.' },
  TFLiteExport: { desc_ko: 'TFLite 내보내기', detail: 'TensorFlow 모델을 TFLite로 변환합니다. 모바일/IoT 기기에서 경량 추론이 가능합니다.', tips: 'INT8 양자화를 적용하면 모델 크기가 크게 줄어듭니다.' },
  PredictionViewer: { desc_ko: '예측 뷰어', detail: '샘플별 예측값과 정답을 테이블로 비교합니다.', tips: '오분류된 샘플을 분석하여 모델 개선 방향을 찾을 수 있습니다.' },
  PrecisionRecallCurve: { desc_ko: 'Precision-Recall 곡선', detail: 'Precision과 Recall의 트레이드오프를 시각화합니다. 불균형 데이터에서 ROC보다 유용합니다.', tips: 'AP(Average Precision)가 높을수록 좋은 모델입니다.' },
  DataDistribution: { desc_ko: '데이터 분포', detail: '입력 데이터의 히스토그램/박스플롯을 표시합니다. 이상치나 분포 편향을 확인합니다.', tips: '전처리 전후의 분포를 비교하여 정규화 효과를 확인할 수 있습니다.' },
  FeatureMapViewer: { desc_ko: 'Feature Map 뷰어', detail: 'CNN 레이어의 활성화 맵을 시각화합니다. 각 필터가 어떤 패턴을 감지하는지 볼 수 있습니다.', tips: '초기 레이어는 엣지/텍스처를, 깊은 레이어는 고수준 패턴을 감지합니다.' },
  GradCAMViewer: { desc_ko: 'Grad-CAM 뷰어', detail: '모델이 분류 판단에 사용한 이미지 영역을 히트맵으로 시각화합니다.', tips: '모델의 판단 근거를 설명하여 신뢰성을 검증할 수 있습니다.' },
  tSNEViewer: { desc_ko: 't-SNE 시각화', detail: '고차원 임베딩을 2D/3D 공간으로 투영하여 클러스터 구조를 시각화합니다.', tips: 'perplexity 파라미터를 조절하여 지역/전역 구조 균형을 맞춥니다.' },
  UMAPViewer: { desc_ko: 'UMAP 시각화', detail: 't-SNE보다 빠르고 전역 구조를 더 잘 보존하는 차원 축소 시각화입니다.', tips: '대규모 데이터셋에서 t-SNE 대신 사용하면 속도가 크게 개선됩니다.' },
  CustomCodeNode: { desc_ko: '커스텀 코드', detail: '파이썬 코드를 직접 작성하여 실행합니다. 내장 노드로 해결되지 않는 전처리나 후처리에 사용합니다.', tips: '입력 데이터는 inputs 딕셔너리로 접근하고, 결과는 return으로 반환합니다.' },
};

/* ────────────────────────── Help Modal ──────────────────────────── */
type Section = 'overview' | 'nodes' | 'techstack' | 'architecture' | 'tutorial' | 'features' | 'examples';

export function HelpModal({ onClose }: { onClose: () => void }) {
  const [section, setSection] = useState<Section>('overview');
  const [examples, setExamples] = useState<ExampleFull[]>([]);
  const [selectedExample, setSelectedExample] = useState<number | null>(null);
  const [exampleCategory, setExampleCategory] = useState<string>('all');
  const [nodeFilter, setNodeFilter] = useState<string>('all');
  const catalog = useStore((s) => s.catalog);

  useEffect(() => {
    fetch('/api/examples/all-details')
      .then(r => r.json())
      .then(data => setExamples(data))
      .catch(() => {});
  }, []);

  const filteredExamples = exampleCategory === 'all'
    ? examples
    : examples.filter(e => e.meta.category === exampleCategory);

  const filteredCatalog = nodeFilter === 'all'
    ? catalog
    : catalog.filter(n => n.category === nodeFilter);

  const categories = [...new Set(examples.map(e => e.meta.category))];

  const navItems: { id: Section; label: string; icon: string }[] = [
    { id: 'overview', label: '프로그램 개요', icon: '🏠' },
    { id: 'features', label: '주요 기능 안내', icon: '🚀' },
    { id: 'nodes', label: '노드 카탈로그', icon: '🧩' },
    { id: 'techstack', label: '기술 스택', icon: '⚙️' },
    { id: 'architecture', label: '프로그램 구조', icon: '🏗️' },
    { id: 'tutorial', label: '튜토리얼', icon: '📖' },
    { id: 'examples', label: '예제 (100개)', icon: '📚' },
  ];

  return (
    <div className="fixed inset-0 bg-black/70 z-50 flex">
      {/* 사이드 네비게이션 */}
      <div className="w-52 bg-gray-900 border-r border-gray-700 flex flex-col">
        <div className="p-4 border-b border-gray-700">
          <h2 className="text-lg font-bold text-white">📘 Help</h2>
          <p className="text-[10px] text-gray-500 mt-1">ML Node Studio Guide</p>
        </div>
        <nav className="flex-1 p-2 space-y-0.5 overflow-y-auto">
          {navItems.map(item => (
            <button
              key={item.id}
              onClick={() => { setSection(item.id); setSelectedExample(null); }}
              className={`w-full text-left px-3 py-2 rounded text-sm transition-colors ${
                section === item.id
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-400 hover:bg-gray-800 hover:text-white'
              }`}
            >
              {item.icon} {item.label}
            </button>
          ))}
        </nav>
        <div className="p-3 border-t border-gray-700">
          <button onClick={onClose} className="w-full px-3 py-1.5 text-sm rounded bg-gray-700 hover:bg-gray-600 text-gray-300">
            ✕ Close
          </button>
        </div>
      </div>

      {/* 메인 컨텐츠 */}
      <div className="flex-1 overflow-y-auto bg-gray-850" style={{ background: '#111827' }}>
        <div className="max-w-5xl mx-auto p-8">
          {section === 'overview' && <OverviewSection />}
          {section === 'features' && <FeaturesSection />}
          {section === 'nodes' && (
            <NodesSection catalog={filteredCatalog} filter={nodeFilter} onFilterChange={setNodeFilter} />
          )}
          {section === 'techstack' && <TechStackSection />}
          {section === 'architecture' && <ArchitectureSection />}
          {section === 'tutorial' && <TutorialSection catalog={catalog} />}
          {section === 'examples' && (
            <ExamplesSection
              examples={filteredExamples}
              allExamples={examples}
              catalog={catalog}
              category={exampleCategory}
              onCategoryChange={setExampleCategory}
              categories={categories}
              selectedId={selectedExample}
              onSelect={setSelectedExample}
            />
          )}
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   섹션 컴포넌트들
   ═══════════════════════════════════════════════════════════════════ */

function SectionTitle({ children }: { children: React.ReactNode }) {
  return <h2 className="text-2xl font-bold text-white mb-6 pb-3 border-b border-gray-700">{children}</h2>;
}

/* ── 1. 프로그램 개요 ──────────────────────────────────── */
function OverviewSection() {
  return (
    <div>
      <SectionTitle>🏠 ML Node Studio</SectionTitle>

      <div className="bg-gradient-to-r from-blue-900/40 to-purple-900/40 rounded-xl p-6 mb-8 border border-blue-800/30">
        <h3 className="text-xl font-bold text-white mb-3">노드 기반 머신러닝 학습 플랫폼</h3>
        <p className="text-gray-300 leading-relaxed">
          드래그 앤 드롭으로 ML 파이프라인을 구성하고 실행할 수 있는 시각적 도구입니다.
          코드를 작성하지 않고도 데이터 전처리 → 모델 정의 → 학습 → 평가 → 시각화까지의
          전체 머신러닝 워크플로를 구성할 수 있습니다.
        </p>
      </div>

      {/* 핵심 기능 */}
      <div className="grid grid-cols-3 gap-4 mb-8">
        {[
          { icon: '🎨', title: '시각적 파이프라인', desc: '노드를 연결하여 ML 파이프라인을 직관적으로 설계' },
          { icon: '⚡', title: '실시간 학습 모니터링', desc: 'Loss/Accuracy 차트를 실시간으로 확인' },
          { icon: '🧪', title: '즉시 테스트', desc: '학습 완료 후 바로 예측 테스트 가능' },
          { icon: '💾', title: '모델 저장/로드', desc: '학습된 모델을 저장하고 나중에 불러와 테스트' },
          { icon: '🔧', title: '듀얼 프레임워크', desc: 'PyTorch와 TensorFlow 중 선택 가능' },
          { icon: '📊', title: '100개 예제', desc: '분류, 회귀, 이미지, NLP 등 다양한 예제 제공' },
          { icon: '🤖', title: 'AI 모델 추천', desc: '데이터 유형/목적에 맞는 최적 파이프라인 자동 추천' },
          { icon: '📄', title: 'Report / PDF', desc: '파이프라인 구조와 학습 결과를 보고서로 내보내기' },
          { icon: '✂️', title: 'Edit 기능', desc: 'Undo/Redo, Cut/Copy/Paste, 키보드 단축키 지원' },
        ].map(f => (
          <div key={f.title} className="bg-gray-800/50 rounded-lg p-4 border border-gray-700/50">
            <span className="text-2xl">{f.icon}</span>
            <h4 className="text-sm font-bold text-white mt-2">{f.title}</h4>
            <p className="text-xs text-gray-400 mt-1">{f.desc}</p>
          </div>
        ))}
      </div>

      {/* 워크플로 다이어그램 */}
      <h3 className="text-lg font-bold text-white mb-4">📐 기본 워크플로</h3>
      <div className="flex items-center gap-3 justify-center mb-8 flex-wrap">
        {[
          { type: 'CSVLoader', cat: 'data' },
          { type: 'StandardScaler', cat: 'data' },
          { type: 'Dense', cat: 'layer' },
          { type: 'Dropout', cat: 'layer' },
          { type: 'Dense', cat: 'layer' },
          { type: 'Trainer', cat: 'training' },
          { type: 'LossCurve', cat: 'visualization' },
        ].map((n, i) => (
          <React.Fragment key={i}>
            <MiniNode type={n.type} category={n.cat} compact />
            {i < 6 && <span className="text-gray-600 text-lg">→</span>}
          </React.Fragment>
        ))}
      </div>

      {/* 사용법 */}
      <h3 className="text-lg font-bold text-white mb-4">🚀 빠른 시작</h3>
      <div className="space-y-3">
        {[
          { step: 1, text: '왼쪽 사이드바에서 노드를 캔버스로 드래그합니다.', detail: '또는 Examples 탭에서 예제를 선택하거나, 🤖 추천 버튼으로 데이터에 맞는 파이프라인을 추천받을 수 있습니다.' },
          { step: 2, text: '노드의 출력 포트(오른쪽)를 다른 노드의 입력 포트(왼쪽)로 드래그하여 연결합니다.', detail: '데이터 → 전처리 → 레이어 → 학습 순서로 연결합니다.' },
          { step: 3, text: '노드를 클릭하여 속성 패널에서 파라미터를 설정합니다.', detail: '예: Dense의 유닛 수, Trainer의 에폭 수 등' },
          { step: 4, text: '상단의 Run 버튼을 눌러 파이프라인을 실행합니다.', detail: '실시간으로 학습 진행 상황을 Output 패널과 차트에서 확인할 수 있습니다.' },
          { step: 5, text: '학습 완료 후 Test 패널에서 직접 예측을 테스트합니다.', detail: '값을 입력하고 Predict 버튼을 눌러 결과를 확인합니다.' },
          { step: 6, text: 'Models 버튼으로 학습된 모델을 저장합니다.', detail: '저장된 모델은 ModelLoader 노드로 불러와 다른 데이터로 테스트할 수 있습니다.' },
          { step: 7, text: 'Report 버튼으로 파이프라인 보고서를 생성하고 PDF로 저장합니다.', detail: '파이프라인 구조, 학습 결과, 메트릭을 한눈에 볼 수 있는 보고서를 만듭니다.' },
        ].map(s => (
          <div key={s.step} className="flex gap-3 items-start">
            <div className="w-7 h-7 rounded-full bg-blue-600 flex items-center justify-center text-sm font-bold text-white flex-shrink-0">
              {s.step}
            </div>
            <div>
              <p className="text-sm text-white">{s.text}</p>
              <p className="text-xs text-gray-500">{s.detail}</p>
            </div>
          </div>
        ))}
      </div>

      {/* 툴바 버튼 가이드 */}
      <h3 className="text-lg font-bold text-white mt-8 mb-4">🖥️ 툴바 버튼 안내</h3>
      <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700/50">
        <div className="space-y-2">
          {[
            { btn: 'New', color: 'bg-gray-600', desc: '캔버스를 초기화하고 새 파이프라인을 시작합니다.' },
            { btn: 'Save', color: 'bg-gray-600', desc: '현재 파이프라인을 JSON으로 서버에 저장합니다.' },
            { btn: 'Load', color: 'bg-gray-600', desc: '저장된 파이프라인을 불러옵니다.' },
            { btn: 'Edit ▾', color: 'bg-gray-600', desc: 'Undo/Redo, Cut/Copy/Paste, Delete, Select All 등 편집 기능 메뉴.' },
            { btn: 'Models', color: 'bg-gray-600', desc: '학습된 모델의 저장/관리/로드/삭제 및 Predict 테스트.' },
            { btn: '🤖 추천', color: 'bg-indigo-600', desc: '데이터 유형 → ML 방법 → 데이터 크기를 선택하면 최적 파이프라인을 추천합니다.' },
            { btn: 'Run', color: 'bg-green-600', desc: '파이프라인을 실행하여 학습을 시작합니다.' },
            { btn: 'Stop', color: 'bg-red-600', desc: '실행 중인 학습을 중단합니다.' },
            { btn: 'Report', color: 'bg-gray-600', desc: '파이프라인 보고서를 미리보기하고 PDF로 다운로드합니다.' },
            { btn: '📘 Help', color: 'bg-gray-600', desc: '프로그램 도움말 (현재 화면).' },
            { btn: 'Output', color: 'bg-gray-600', desc: '학습 로그, 에러 메시지, 메트릭 등의 실시간 로그 패널을 열고 닫습니다.' },
          ].map(item => (
            <div key={item.btn} className="flex items-center gap-3">
              <span className={`${item.color} text-white text-[10px] px-2 py-0.5 rounded font-medium min-w-[80px] text-center`}>{item.btn}</span>
              <span className="text-xs text-gray-400">{item.desc}</span>
            </div>
          ))}
        </div>
      </div>

      {/* 키보드 단축키 */}
      <h3 className="text-lg font-bold text-white mt-8 mb-4">⌨️ 키보드 단축키</h3>
      <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700/50">
        <div className="grid grid-cols-2 gap-x-8 gap-y-1.5">
          {[
            { key: 'Ctrl + Z', action: '되돌리기 (Undo)' },
            { key: 'Ctrl + Shift + Z', action: '다시 실행 (Redo)' },
            { key: 'Ctrl + X', action: '잘라내기 (Cut)' },
            { key: 'Ctrl + C', action: '복사 (Copy)' },
            { key: 'Ctrl + V', action: '붙여넣기 (Paste)' },
            { key: 'Delete', action: '선택된 노드/엣지 삭제' },
            { key: 'Ctrl + A', action: '전체 선택 (Select All)' },
            { key: 'Backspace', action: '선택된 노드/엣지 삭제' },
          ].map(s => (
            <div key={s.key} className="flex items-center gap-3">
              <kbd className="text-[10px] font-mono bg-gray-700 text-gray-300 px-2 py-0.5 rounded border border-gray-600 min-w-[130px] text-center">{s.key}</kbd>
              <span className="text-xs text-gray-400">{s.action}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ── 1.5. 주요 기능 안내 ──────────────────────────────── */
function FeaturesSection() {
  return (
    <div>
      <SectionTitle>🚀 주요 기능 안내</SectionTitle>

      {/* ── 🤖 모델 추천 ── */}
      <div className="bg-gradient-to-r from-indigo-900/30 to-purple-900/30 rounded-xl p-6 border border-indigo-800/30 mb-8">
        <div className="flex items-center gap-3 mb-4">
          <span className="text-3xl">🤖</span>
          <div>
            <h3 className="text-lg font-bold text-white">AI 모델 추천 마법사</h3>
            <p className="text-xs text-gray-400">툴바의 <span className="bg-indigo-600 text-white text-[10px] px-2 py-0.5 rounded font-medium">🤖 추천</span> 버튼</p>
          </div>
        </div>
        <p className="text-sm text-gray-300 mb-4">
          데이터 유형과 목적에 맞는 최적의 ML 파이프라인을 자동으로 추천합니다.
          3단계 선택만으로 바로 예제를 로드할 수 있습니다.
        </p>

        {/* 4단계 설명 */}
        <div className="grid grid-cols-4 gap-3 mb-4">
          {[
            { step: 1, title: '데이터 유형', desc: 'CSV/테이블, 이미지, 텍스트, 시계열, 오디오 중 선택', icon: '📁', color: 'blue' },
            { step: 2, title: 'ML 방법', desc: '데이터에 따라 7~11가지 세부 방법 표시 (분류, 회귀, 이상탐지 등)', icon: '🧠', color: 'purple' },
            { step: 3, title: '데이터 크기', desc: 'Small / Medium / Large 규모 선택', icon: '📏', color: 'green' },
            { step: 4, title: '추천 결과', desc: '조건에 맞는 예제 파이프라인 목록 표시, 바로 로드 가능', icon: '✨', color: 'yellow' },
          ].map(s => (
            <div key={s.step} className="bg-gray-800/50 rounded-lg p-3 border border-gray-700/50 text-center">
              <span className="text-xl">{s.icon}</span>
              <div className="text-[10px] text-gray-500 mt-1">Step {s.step}</div>
              <div className="text-xs font-bold text-white mt-0.5">{s.title}</div>
              <div className="text-[10px] text-gray-400 mt-1">{s.desc}</div>
            </div>
          ))}
        </div>

        <div className="bg-gray-900/50 rounded-lg p-3">
          <h4 className="text-xs font-bold text-indigo-400 mb-2">데이터 유형별 ML 방법 예시</h4>
          <div className="grid grid-cols-2 gap-3 text-[11px]">
            <div>
              <span className="text-blue-400 font-bold">CSV/테이블:</span>
              <span className="text-gray-400 ml-1">다중분류, 이진분류, 불균형분류, 회귀, 이상탐지, 클러스터링, K-Fold, 앙상블</span>
            </div>
            <div>
              <span className="text-green-400 font-bold">이미지:</span>
              <span className="text-gray-400 ml-1">분류, 전이학습, 회귀, 세그멘테이션, 유사도, 초해상도, 이상탐지, 해석</span>
            </div>
            <div>
              <span className="text-purple-400 font-bold">텍스트:</span>
              <span className="text-gray-400 ml-1">분류, 다중레이블, NER, 회귀, 생성, 유사도, 클러스터링</span>
            </div>
            <div>
              <span className="text-amber-400 font-bold">시계열:</span>
              <span className="text-gray-400 ml-1">분류, 단일스텝예측, 멀티스텝예측, 이상탐지, CNN+RNN, Transformer, Seq2Seq</span>
            </div>
          </div>
        </div>
      </div>

      {/* ── 📄 Report / PDF ── */}
      <div className="bg-gradient-to-r from-emerald-900/30 to-teal-900/30 rounded-xl p-6 border border-emerald-800/30 mb-8">
        <div className="flex items-center gap-3 mb-4">
          <span className="text-3xl">📄</span>
          <div>
            <h3 className="text-lg font-bold text-white">Report / PDF 내보내기</h3>
            <p className="text-xs text-gray-400">툴바의 <span className="bg-gray-600 text-white text-[10px] px-2 py-0.5 rounded font-medium">Report</span> 버튼</p>
          </div>
        </div>
        <p className="text-sm text-gray-300 mb-4">
          현재 파이프라인의 구조, 노드 설정, 학습 결과를 한눈에 볼 수 있는 보고서를 생성합니다.
          미리보기 후 PDF로 다운로드할 수 있습니다.
        </p>

        <div className="space-y-2">
          {[
            { icon: '📸', title: '파이프라인 캡처', desc: '캔버스의 노드/엣지 구조를 이미지로 캡처하여 보고서에 포함' },
            { icon: '📋', title: '노드 설정 요약', desc: '각 노드의 파라미터 설정값을 표로 정리' },
            { icon: '📊', title: '학습 결과', desc: 'Loss/Accuracy 그래프, 메트릭, 모델 경로 등 학습 결과 포함' },
            { icon: '📥', title: 'PDF 다운로드', desc: 'A4 용지에 맞게 자동 페이지 분할하여 PDF로 저장' },
          ].map(item => (
            <div key={item.title} className="flex gap-3 items-start bg-gray-800/30 rounded-lg p-3">
              <span className="text-lg">{item.icon}</span>
              <div>
                <div className="text-sm font-bold text-white">{item.title}</div>
                <div className="text-xs text-gray-400">{item.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── ✂️ Edit 기능 ── */}
      <div className="bg-gradient-to-r from-amber-900/30 to-orange-900/30 rounded-xl p-6 border border-amber-800/30 mb-8">
        <div className="flex items-center gap-3 mb-4">
          <span className="text-3xl">✂️</span>
          <div>
            <h3 className="text-lg font-bold text-white">Edit 기능 & 키보드 단축키</h3>
            <p className="text-xs text-gray-400">툴바의 <span className="bg-gray-600 text-white text-[10px] px-2 py-0.5 rounded font-medium">Edit ▾</span> 드롭다운 메뉴</p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          {/* 편집 기능 */}
          <div>
            <h4 className="text-sm font-bold text-amber-400 mb-2">편집 기능</h4>
            <div className="space-y-1.5">
              {[
                { action: 'Undo / Redo', desc: '노드 추가/삭제/이동, 엣지 연결 등 모든 작업을 되돌리거나 다시 실행' },
                { action: 'Cut / Copy / Paste', desc: '선택된 노드를 잘라내기, 복사, 붙여넣기 (파라미터 포함)' },
                { action: 'Delete', desc: '선택된 노드 또는 엣지를 삭제' },
                { action: 'Select All', desc: '캔버스의 모든 노드를 선택' },
              ].map(item => (
                <div key={item.action} className="bg-gray-800/50 rounded p-2">
                  <div className="text-xs font-bold text-white">{item.action}</div>
                  <div className="text-[10px] text-gray-400">{item.desc}</div>
                </div>
              ))}
            </div>
          </div>

          {/* 단축키 */}
          <div>
            <h4 className="text-sm font-bold text-amber-400 mb-2">키보드 단축키</h4>
            <div className="space-y-1">
              {[
                { key: 'Ctrl + Z', action: 'Undo (되돌리기)' },
                { key: 'Ctrl + Shift + Z', action: 'Redo (다시 실행)' },
                { key: 'Ctrl + X', action: 'Cut (잘라내기)' },
                { key: 'Ctrl + C', action: 'Copy (복사)' },
                { key: 'Ctrl + V', action: 'Paste (붙여넣기)' },
                { key: 'Delete / Backspace', action: '삭제' },
                { key: 'Ctrl + A', action: '전체 선택' },
              ].map(s => (
                <div key={s.key} className="flex items-center gap-2 bg-gray-800/50 rounded px-2 py-1">
                  <kbd className="text-[10px] font-mono bg-gray-700 text-gray-300 px-1.5 py-0.5 rounded border border-gray-600 min-w-[120px] text-center">{s.key}</kbd>
                  <span className="text-[11px] text-gray-400">{s.action}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* ── 📟 Output 로그 패널 ── */}
      <div className="bg-gradient-to-r from-cyan-900/30 to-blue-900/30 rounded-xl p-6 border border-cyan-800/30 mb-8">
        <div className="flex items-center gap-3 mb-4">
          <span className="text-3xl">📟</span>
          <div>
            <h3 className="text-lg font-bold text-white">Output 로그 패널</h3>
            <p className="text-xs text-gray-400">툴바 오른쪽의 <span className="bg-gray-600 text-white text-[10px] px-2 py-0.5 rounded font-medium">Output</span> 버튼</p>
          </div>
        </div>
        <p className="text-sm text-gray-300 mb-4">
          파이프라인 실행 중 발생하는 모든 이벤트를 실시간으로 표시합니다.
          에러 발생 시 버튼이 빨간색으로 변하여 알려줍니다.
        </p>

        <div className="grid grid-cols-2 gap-3">
          {[
            { icon: '🚀', label: 'Pipeline Start', desc: '파이프라인 시작, 실행 순서 표시', color: 'text-blue-400' },
            { icon: '▶', label: 'Node Start/Complete', desc: '각 노드의 시작/완료 상태', color: 'text-green-400' },
            { icon: '📊', label: 'Epoch Update', desc: 'Loss, Accuracy 등 학습 메트릭', color: 'text-yellow-400' },
            { icon: '🖥️', label: 'Device Info', desc: 'GPU/CPU 장치 정보', color: 'text-cyan-400' },
            { icon: '📈', label: 'Visualization', desc: '차트/이미지 생성 알림', color: 'text-purple-400' },
            { icon: '❌', label: 'Error', desc: '에러 메시지 및 트레이스백', color: 'text-red-400' },
          ].map(item => (
            <div key={item.label} className="bg-gray-800/50 rounded p-2 flex items-start gap-2">
              <span className={`text-sm ${item.color}`}>{item.icon}</span>
              <div>
                <div className="text-xs font-bold text-white">{item.label}</div>
                <div className="text-[10px] text-gray-400">{item.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── 💾 파이프라인 저장/불러오기 ── */}
      <div className="bg-gradient-to-r from-gray-800/50 to-gray-700/30 rounded-xl p-6 border border-gray-700/50 mb-8">
        <div className="flex items-center gap-3 mb-4">
          <span className="text-3xl">💾</span>
          <div>
            <h3 className="text-lg font-bold text-white">파이프라인 저장 / 불러오기</h3>
            <p className="text-xs text-gray-400">
              <span className="bg-gray-600 text-white text-[10px] px-2 py-0.5 rounded font-medium mr-1">New</span>
              <span className="bg-gray-600 text-white text-[10px] px-2 py-0.5 rounded font-medium mr-1">Save</span>
              <span className="bg-gray-600 text-white text-[10px] px-2 py-0.5 rounded font-medium">Load</span>
            </p>
          </div>
        </div>

        <div className="space-y-2">
          {[
            { btn: 'New', desc: '현재 캔버스를 초기화하고 빈 파이프라인으로 시작합니다. 기존 노드가 있으면 확인 대화상자가 표시됩니다.' },
            { btn: 'Save', desc: '현재 파이프라인의 노드, 엣지, 파라미터, 위치 정보를 JSON으로 서버에 저장합니다. 이름을 지정하여 여러 파이프라인을 관리할 수 있습니다.' },
            { btn: 'Load', desc: '서버에 저장된 파이프라인 목록에서 선택하여 불러옵니다. 노드와 엣지가 원래 위치에 복원됩니다.' },
          ].map(item => (
            <div key={item.btn} className="flex gap-3 items-start bg-gray-800/30 rounded-lg p-3">
              <span className="bg-gray-600 text-white text-xs px-2 py-0.5 rounded font-medium min-w-[50px] text-center">{item.btn}</span>
              <span className="text-xs text-gray-400">{item.desc}</span>
            </div>
          ))}
        </div>
      </div>

      {/* ── 🖱️ 캔버스 조작법 ── */}
      <div className="bg-gradient-to-r from-rose-900/30 to-pink-900/30 rounded-xl p-6 border border-rose-800/30">
        <div className="flex items-center gap-3 mb-4">
          <span className="text-3xl">🖱️</span>
          <h3 className="text-lg font-bold text-white">캔버스 조작법</h3>
        </div>

        <div className="grid grid-cols-2 gap-3">
          {[
            { action: '노드 추가', how: '왼쪽 사이드바에서 노드를 캔버스로 드래그 & 드롭' },
            { action: '노드 연결', how: '출력 포트(오른쪽 동그라미)를 입력 포트(왼쪽 동그라미)로 드래그' },
            { action: '노드 이동', how: '노드를 클릭 후 드래그' },
            { action: '캔버스 이동', how: '빈 공간에서 마우스 드래그 또는 휠 누르고 드래그' },
            { action: '확대/축소', how: '마우스 휠 스크롤' },
            { action: '노드 선택', how: '노드 클릭 (Shift+클릭으로 다중 선택)' },
            { action: '영역 선택', how: '빈 공간에서 Shift+드래그로 선택 박스' },
            { action: '속성 편집', how: '노드 클릭 → 오른쪽 속성 패널에서 파라미터 수정' },
            { action: '노드 설명', how: '노드 우클릭 → "Description" 메뉴로 상세 설명 확인' },
            { action: '엣지 삭제', how: '엣지 클릭 후 Delete 키' },
          ].map(item => (
            <div key={item.action} className="bg-gray-800/50 rounded p-2">
              <div className="text-xs font-bold text-white">{item.action}</div>
              <div className="text-[10px] text-gray-400">{item.how}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ── 2. 노드 카탈로그 ──────────────────────────────────── */
function NodesSection({ catalog, filter, onFilterChange }: {
  catalog: { type: string; category: string; description: string; params: any[]; inputs: any[]; outputs: any[] }[];
  filter: string;
  onFilterChange: (f: string) => void;
}) {
  const categories = ['all', 'data', 'layer', 'training', 'visualization', 'evaluation', 'output'];
  const catNames: Record<string, string> = {
    all: '전체', data: '데이터', layer: '레이어', training: '학습',
    visualization: '시각화', evaluation: '평가', output: '출력',
  };

  return (
    <div>
      <SectionTitle>🧩 노드 카탈로그 ({catalog.length}개)</SectionTitle>

      {/* 카테고리 필터 */}
      <div className="flex gap-2 mb-6 flex-wrap">
        {categories.map(cat => (
          <button
            key={cat}
            onClick={() => onFilterChange(cat)}
            className={`px-3 py-1 text-xs rounded-full transition-colors ${
              filter === cat
                ? 'text-white font-bold'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
            style={filter === cat ? { background: CAT_COLORS[cat] || '#3b82f6' } : {}}
          >
            {catNames[cat]} {cat !== 'all' && `(${catalog.filter(n => cat === 'all' || n.category === cat).length})`}
          </button>
        ))}
      </div>

      {/* 노드 목록 */}
      <div className="space-y-4">
        {catalog.map(node => {
          const info = NODE_DESCRIPTIONS[node.type];
          return (
            <div key={node.type} className="bg-gray-800/50 rounded-lg p-4 border border-gray-700/50 flex gap-4">
              {/* 노드 비주얼 */}
              <div className="flex-shrink-0">
                <MiniNode
                  type={node.type}
                  category={node.category}
                  params={Object.fromEntries(node.params.filter((p: any) => p.default !== undefined).map((p: any) => [p.name, p.default]))}
                />
              </div>

              {/* 설명 */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <h4 className="text-sm font-bold text-white">{node.type}</h4>
                  <span className="text-[10px] px-1.5 py-0.5 rounded" style={{
                    background: (CAT_COLORS[node.category] || '#6b7280') + '33',
                    color: CAT_COLORS[node.category] || '#6b7280',
                  }}>
                    {node.category}
                  </span>
                </div>
                <p className="text-xs text-gray-400 mb-2">{info?.desc_ko || node.description}</p>
                {info && <p className="text-xs text-gray-500 leading-relaxed mb-2">{info.detail}</p>}

                {/* 포트 정보 */}
                <div className="flex gap-4 text-[10px]">
                  {node.inputs.length > 0 && (
                    <div>
                      <span className="text-gray-600">입력: </span>
                      {node.inputs.map((p: any) => (
                        <span key={p.name} className="text-blue-400 mr-1">{p.name}</span>
                      ))}
                    </div>
                  )}
                  {node.outputs.length > 0 && (
                    <div>
                      <span className="text-gray-600">출력: </span>
                      {node.outputs.map((p: any) => (
                        <span key={p.name} className="text-green-400 mr-1">{p.name}</span>
                      ))}
                    </div>
                  )}
                </div>

                {info?.tips && (
                  <div className="mt-2 text-[10px] text-yellow-500/70 bg-yellow-900/10 rounded px-2 py-1">
                    💡 {info.tips}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ── 3. 기술 스택 ──────────────────────────────────────── */
function TechStackSection() {
  const stacks = [
    {
      title: 'Backend (Python)',
      items: [
        { name: 'FastAPI', role: 'API 서버 + WebSocket', color: '#009688' },
        { name: 'PyTorch', role: 'ML 프레임워크 (기본)', color: '#ee4c2c' },
        { name: 'TensorFlow', role: 'ML 프레임워크 (선택)', color: '#ff6f00' },
        { name: 'NumPy / Pandas', role: '데이터 처리', color: '#4dabcf' },
        { name: 'scikit-learn', role: '전처리 / 메트릭', color: '#f89939' },
        { name: 'Matplotlib', role: '시각화 (base64 이미지)', color: '#11557c' },
      ],
    },
    {
      title: 'Frontend (Web)',
      items: [
        { name: 'React', role: 'UI 프레임워크', color: '#61dafb' },
        { name: 'React Flow', role: '노드 에디터', color: '#ff0072' },
        { name: 'TypeScript', role: '타입 안전성', color: '#3178c6' },
        { name: 'Tailwind CSS', role: '스타일링', color: '#06b6d4' },
        { name: 'Recharts', role: '실시간 차트', color: '#8884d8' },
        { name: 'Zustand', role: '상태 관리', color: '#764abc' },
      ],
    },
    {
      title: 'Communication',
      items: [
        { name: 'WebSocket', role: '실시간 학습 진행 전송', color: '#4caf50' },
        { name: 'REST API', role: '예제/모델/파이프라인 관리', color: '#2196f3' },
        { name: 'JSON', role: '파이프라인 직렬화', color: '#fdd835' },
      ],
    },
  ];

  return (
    <div>
      <SectionTitle>⚙️ 기술 스택</SectionTitle>
      <div className="space-y-6">
        {stacks.map(stack => (
          <div key={stack.title}>
            <h3 className="text-lg font-bold text-white mb-3">{stack.title}</h3>
            <div className="grid grid-cols-2 gap-3">
              {stack.items.map(item => (
                <div key={item.name} className="bg-gray-800/50 rounded-lg p-3 border border-gray-700/50 flex items-center gap-3">
                  <div className="w-2 h-8 rounded-full" style={{ background: item.color }} />
                  <div>
                    <div className="text-sm font-bold text-white">{item.name}</div>
                    <div className="text-xs text-gray-500">{item.role}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── 4. 프로그램 구조 ──────────────────────────────────── */
function ArchitectureSection() {
  return (
    <div>
      <SectionTitle>🏗️ 프로그램 구조</SectionTitle>

      {/* 아키텍처 다이어그램 */}
      <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700/50 mb-8">
        <div className="flex flex-col items-center gap-4">
          {/* 프론트엔드 */}
          <div className="w-full bg-blue-900/20 rounded-lg p-4 border border-blue-800/30">
            <div className="text-center text-sm font-bold text-blue-400 mb-3">🖥️ Frontend (React)</div>
            <div className="flex justify-center gap-3">
              {['React Flow\n(캔버스)', 'Sidebar\n(노드 목록)', 'Properties\n(파라미터)', 'Training\n(차트/테스트)'].map(m => (
                <div key={m} className="bg-blue-900/30 rounded px-3 py-2 text-[10px] text-blue-300 text-center whitespace-pre-line border border-blue-800/30">
                  {m}
                </div>
              ))}
            </div>
          </div>

          <div className="text-gray-600 text-sm">↕️ WebSocket + REST API</div>

          {/* 백엔드 */}
          <div className="w-full bg-green-900/20 rounded-lg p-4 border border-green-800/30">
            <div className="text-center text-sm font-bold text-green-400 mb-3">⚙️ Backend (Python FastAPI)</div>
            <div className="flex justify-center gap-3">
              {['Pipeline\nExecutor', 'Node\nRegistry', 'Torch\nBackend', 'TF\nBackend'].map(m => (
                <div key={m} className="bg-green-900/30 rounded px-3 py-2 text-[10px] text-green-300 text-center whitespace-pre-line border border-green-800/30">
                  {m}
                </div>
              ))}
            </div>
          </div>

          <div className="text-gray-600 text-sm">↕️</div>

          {/* 실행 엔진 */}
          <div className="w-full bg-amber-900/20 rounded-lg p-4 border border-amber-800/30">
            <div className="text-center text-sm font-bold text-amber-400 mb-3">🔥 ML Framework</div>
            <div className="flex justify-center gap-3">
              {['PyTorch\n(GPU: CUDA)', 'TensorFlow\n(CPU/GPU)', '데이터셋\n(CSV/이미지)', '모델 저장\n(.pth/.keras)'].map(m => (
                <div key={m} className="bg-amber-900/30 rounded px-3 py-2 text-[10px] text-amber-300 text-center whitespace-pre-line border border-amber-800/30">
                  {m}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* 실행 흐름 */}
      <h3 className="text-lg font-bold text-white mb-4">🔄 파이프라인 실행 흐름</h3>
      <div className="space-y-2">
        {[
          { step: 'Run 클릭', desc: '프론트엔드에서 노드/엣지 정보를 WebSocket으로 전송', icon: '▶️' },
          { step: '토폴로지 정렬', desc: "Kahn's Algorithm으로 노드 실행 순서 결정 (DAG 기반)", icon: '📐' },
          { step: '순차 실행', desc: '정렬된 순서대로 각 노드의 run() 메서드 실행', icon: '⚡' },
          { step: '데이터 전달', desc: '이전 노드의 출력을 다음 노드의 입력으로 전달 (context dict)', icon: '🔗' },
          { step: '실시간 전송', desc: 'EPOCH_UPDATE, VISUALIZATION 등의 메시지를 WebSocket으로 전송', icon: '📡' },
          { step: '학습 완료', desc: '모델 저장, 메트릭 전송, 테스트 패널 활성화', icon: '✅' },
        ].map((s, i) => (
          <div key={i} className="flex gap-3 items-start bg-gray-800/30 rounded-lg p-3">
            <span className="text-lg">{s.icon}</span>
            <div>
              <div className="text-sm font-bold text-white">{s.step}</div>
              <div className="text-xs text-gray-400">{s.desc}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── 5. 튜토리얼 ──────────────────────────────────────── */
function TutorialSection({ catalog }: { catalog: any[] }) {
  return (
    <div>
      <SectionTitle>📖 튜토리얼</SectionTitle>

      {/* 튜토리얼 1: Iris 분류 */}
      <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700/50 mb-8">
        <h3 className="text-lg font-bold text-white mb-2">튜토리얼 1: 붓꽃(Iris) 분류</h3>
        <p className="text-xs text-gray-400 mb-4">4개의 꽃 특성으로 3종의 붓꽃을 분류하는 가장 기본적인 파이프라인</p>

        {/* 파이프라인 다이어그램 */}
        <div className="bg-gray-900/50 rounded-lg p-4 mb-4">
          <div className="flex items-center gap-3 justify-center flex-wrap">
            {[
              { type: 'CSVLoader', cat: 'data', params: { file_path: 'iris.csv', target: 'species' } },
              { type: 'StandardScaler', cat: 'data', params: {} },
              { type: 'Dense', cat: 'layer', params: { units: 64, activation: 'relu' } },
              { type: 'Dense', cat: 'layer', params: { units: 32, activation: 'relu' } },
              { type: 'Dense', cat: 'layer', params: { units: 3, activation: 'softmax' } },
              { type: 'Trainer', cat: 'training', params: { epochs: 30, batch_size: 16 } },
            ].map((n, i) => (
              <React.Fragment key={i}>
                <MiniNode type={n.type} category={n.cat} params={n.params} />
                {i < 5 && <span className="text-gray-600 text-xl">→</span>}
              </React.Fragment>
            ))}
          </div>
        </div>

        <div className="space-y-3 text-sm">
          {[
            { step: 1, title: 'CSVLoader', desc: 'iris.csv를 로드합니다. target_column="species"로 설정하여 꽃 종류를 레이블로 사용합니다. 4개 특성: sepal_length, sepal_width, petal_length, petal_width' },
            { step: 2, title: 'StandardScaler', desc: '4개 특성의 스케일을 맞춥니다 (평균=0, 표준편차=1). 학습 속도와 성능이 개선됩니다.' },
            { step: 3, title: 'Dense (64, relu)', desc: '첫 번째 은닉층. 64개 뉴런으로 특성 간의 비선형 관계를 학습합니다.' },
            { step: 4, title: 'Dense (32, relu)', desc: '두 번째 은닉층. 더 추상적인 패턴을 학습합니다.' },
            { step: 5, title: 'Dense (3, softmax)', desc: '출력층. 3개 클래스에 대한 확률을 출력합니다.' },
            { step: 6, title: 'Trainer', desc: '30 에폭, 배치 크기 16으로 학습합니다. loss, accuracy를 실시간 모니터링합니다.' },
          ].map(s => (
            <div key={s.step} className="flex gap-3">
              <div className="w-6 h-6 rounded-full bg-blue-600/30 flex items-center justify-center text-xs text-blue-400 flex-shrink-0">
                {s.step}
              </div>
              <div>
                <span className="font-bold text-white">{s.title}</span>
                <span className="text-gray-400 ml-2">{s.desc}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 튜토리얼 2: CNN 이미지 분류 */}
      <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700/50 mb-8">
        <h3 className="text-lg font-bold text-white mb-2">튜토리얼 2: MNIST CNN 분류</h3>
        <p className="text-xs text-gray-400 mb-4">합성곱 신경망으로 손글씨 숫자를 인식합니다</p>

        <div className="bg-gray-900/50 rounded-lg p-4 mb-4">
          <div className="flex items-center gap-2 justify-center flex-wrap">
            {[
              { type: 'NumpyInput', cat: 'data', params: { dataset: 'mnist' } },
              { type: 'Conv2D', cat: 'layer', params: { filters: 32, kernel: 3 } },
              { type: 'MaxPooling2D', cat: 'layer', params: { pool_size: 2 } },
              { type: 'Conv2D', cat: 'layer', params: { filters: 64, kernel: 3 } },
              { type: 'Flatten', cat: 'layer', params: {} },
              { type: 'Dense', cat: 'layer', params: { units: 10, act: 'softmax' } },
              { type: 'Trainer', cat: 'training', params: { epochs: 5 } },
            ].map((n, i) => (
              <React.Fragment key={i}>
                <MiniNode type={n.type} category={n.cat} params={n.params} compact />
                {i < 6 && <span className="text-gray-600">→</span>}
              </React.Fragment>
            ))}
          </div>
        </div>

        <div className="text-xs text-gray-400 space-y-1">
          <p><span className="text-white font-bold">핵심 개념:</span> Conv2D가 이미지에서 엣지, 텍스처 등 공간적 특성을 추출합니다.</p>
          <p><span className="text-white font-bold">데이터 흐름:</span> 28×28 이미지 → Conv2D(32필터) → 풀링(14×14) → Conv2D(64필터) → Flatten → Dense(10클래스)</p>
          <p><span className="text-white font-bold">기대 성능:</span> 5 에폭만으로 99%+ 정확도 달성</p>
        </div>
      </div>

      {/* 튜토리얼 3: 시계열 예측 */}
      <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700/50 mb-8">
        <h3 className="text-lg font-bold text-white mb-2">튜토리얼 3: LSTM 시계열 예측</h3>
        <p className="text-xs text-gray-400 mb-4">과거 온도 데이터로 미래 기온을 예측합니다</p>

        <div className="bg-gray-900/50 rounded-lg p-4 mb-4">
          <div className="flex items-center gap-2 justify-center flex-wrap">
            {[
              { type: 'CSVLoader', cat: 'data', params: { file: 'temp.csv' } },
              { type: 'MinMaxScaler', cat: 'data', params: {} },
              { type: 'LSTM', cat: 'layer', params: { units: 64 } },
              { type: 'Dense', cat: 'layer', params: { units: 1, act: 'linear' } },
              { type: 'Trainer', cat: 'training', params: { loss: 'mse' } },
            ].map((n, i) => (
              <React.Fragment key={i}>
                <MiniNode type={n.type} category={n.cat} params={n.params} compact />
                {i < 4 && <span className="text-gray-600">→</span>}
              </React.Fragment>
            ))}
          </div>
        </div>

        <div className="text-xs text-gray-400 space-y-1">
          <p><span className="text-white font-bold">핵심 개념:</span> LSTM이 시간적 패턴(계절성, 추세)을 학습합니다.</p>
          <p><span className="text-white font-bold">회귀 문제:</span> 출력 activation은 linear, 손실 함수는 MSE를 사용합니다.</p>
          <p><span className="text-white font-bold">데이터 전처리:</span> MinMaxScaler로 0~1 범위로 정규화합니다.</p>
        </div>
      </div>

      {/* 튜토리얼 4: 모델 저장 & 로드 & 테스트 */}
      <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700/50 mb-8">
        <h3 className="text-lg font-bold text-white mb-2">튜토리얼 4: 모델 저장 / 불러오기 / 테스트</h3>
        <p className="text-xs text-gray-400 mb-4">학습한 모델을 저장하고, 나중에 불러와 새 데이터로 테스트합니다</p>

        {/* Step A: 학습 & 저장 */}
        <div className="mb-5">
          <h4 className="text-sm font-bold text-yellow-400 mb-2">A. 모델 학습 후 저장</h4>
          <div className="bg-gray-900/50 rounded-lg p-4 mb-3">
            <div className="flex items-center gap-3 justify-center flex-wrap">
              {[
                { type: 'CSVLoader', cat: 'data', params: { file: 'iris.csv' } },
                { type: 'StandardScaler', cat: 'data', params: {} },
                { type: 'Dense', cat: 'layer', params: { units: 64 } },
                { type: 'Dense', cat: 'layer', params: { units: 3, act: 'softmax' } },
                { type: 'Trainer', cat: 'training', params: { epochs: 30 } },
              ].map((n, i) => (
                <React.Fragment key={i}>
                  <MiniNode type={n.type} category={n.cat} params={n.params} compact />
                  {i < 4 && <span className="text-gray-600">{'>'}</span>}
                </React.Fragment>
              ))}
            </div>
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex gap-3">
              <div className="w-6 h-6 rounded-full bg-yellow-600/30 flex items-center justify-center text-xs text-yellow-400 flex-shrink-0">1</div>
              <div>
                <span className="font-bold text-white">파이프라인 학습</span>
                <span className="text-gray-400 ml-2">평소처럼 파이프라인을 구성하고 Run 버튼으로 학습합니다.</span>
              </div>
            </div>
            <div className="flex gap-3">
              <div className="w-6 h-6 rounded-full bg-yellow-600/30 flex items-center justify-center text-xs text-yellow-400 flex-shrink-0">2</div>
              <div>
                <span className="font-bold text-white">모델 저장</span>
                <span className="text-gray-400 ml-2">학습 완료 후 툴바의 <span className="text-blue-400 font-mono text-xs px-1 py-0.5 bg-blue-900/30 rounded">Save Model</span> 버튼 또는 <span className="text-white font-mono text-xs px-1 py-0.5 bg-gray-700 rounded">Models</span> 버튼을 클릭합니다.</span>
              </div>
            </div>
            <div className="flex gap-3">
              <div className="w-6 h-6 rounded-full bg-yellow-600/30 flex items-center justify-center text-xs text-yellow-400 flex-shrink-0">3</div>
              <div>
                <span className="font-bold text-white">이름 지정</span>
                <span className="text-gray-400 ml-2">Save 탭에서 모델 이름(예: "iris-classifier")을 입력하고 Save 버튼을 누릅니다.</span>
              </div>
            </div>
          </div>
        </div>

        {/* Step B: 불러와서 테스트 */}
        <div className="mb-5">
          <h4 className="text-sm font-bold text-green-400 mb-2">B. 저장된 모델로 테스트</h4>
          <div className="bg-gray-900/50 rounded-lg p-4 mb-3">
            <div className="flex items-center gap-2 justify-center flex-wrap">
              <div className="flex flex-col items-center gap-1">
                <MiniNode type="ModelLoader" category="output" params={{ model: 'iris-classifier' }} compact />
                <span className="text-[9px] text-gray-500">model</span>
              </div>
              <span className="text-gray-600 text-lg self-start mt-3">{'\\>'}</span>
              <div className="flex flex-col items-center gap-1">
                <MiniNode type="ModelTester" category="evaluation" params={{ samples: 10 }} />
                <span className="text-[9px] text-gray-500">metrics + samples</span>
              </div>
              <span className="text-gray-600 text-lg self-start mt-3">{'<'}</span>
              <div className="flex flex-col items-center gap-1">
                <div className="flex items-center gap-1">
                  <MiniNode type="CSVLoader" category="data" params={{ file: 'new_data.csv' }} compact />
                  <span className="text-gray-600">{'>'}</span>
                  <MiniNode type="TrainValSplit" category="data" params={{ test: 1.0 }} compact />
                </div>
                <span className="text-[9px] text-gray-500">test_features + test_labels</span>
              </div>
            </div>
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex gap-3">
              <div className="w-6 h-6 rounded-full bg-green-600/30 flex items-center justify-center text-xs text-green-400 flex-shrink-0">1</div>
              <div>
                <span className="font-bold text-white">ModelLoader 배치</span>
                <span className="text-gray-400 ml-2">사이드바 Output 카테고리에서 ModelLoader를 추가하고, model_name에서 저장된 모델을 선택합니다.</span>
              </div>
            </div>
            <div className="flex gap-3">
              <div className="w-6 h-6 rounded-full bg-green-600/30 flex items-center justify-center text-xs text-green-400 flex-shrink-0">2</div>
              <div>
                <span className="font-bold text-white">테스트 데이터 준비</span>
                <span className="text-gray-400 ml-2">CSVLoader + TrainValSplit(test_ratio=1.0)으로 테스트 전용 데이터를 만듭니다. 또는 NumpyInput의 test_features/test_labels 출력을 바로 연결합니다.</span>
              </div>
            </div>
            <div className="flex gap-3">
              <div className="w-6 h-6 rounded-full bg-green-600/30 flex items-center justify-center text-xs text-green-400 flex-shrink-0">3</div>
              <div>
                <span className="font-bold text-white">ModelTester 연결</span>
                <span className="text-gray-400 ml-2">ModelLoader의 <span className="text-blue-400 font-mono text-xs">model</span> 출력 → ModelTester의 <span className="text-blue-400 font-mono text-xs">model</span> 입력으로 연결. 데이터의 <span className="text-blue-400 font-mono text-xs">test_features</span>, <span className="text-blue-400 font-mono text-xs">test_labels</span>도 연결합니다.</span>
              </div>
            </div>
            <div className="flex gap-3">
              <div className="w-6 h-6 rounded-full bg-green-600/30 flex items-center justify-center text-xs text-green-400 flex-shrink-0">4</div>
              <div>
                <span className="font-bold text-white">Run</span>
                <span className="text-gray-400 ml-2">실행하면 테스트 결과 패널에 Accuracy, F1 Score, Confusion Matrix, 샘플별 비교 테이블이 표시됩니다.</span>
              </div>
            </div>
          </div>
        </div>

        {/* 포트 연결 참고 */}
        <div className="bg-blue-900/20 rounded-lg p-3 border border-blue-800/30">
          <h4 className="text-xs font-bold text-blue-400 mb-2">포트 연결 가이드</h4>
          <div className="text-[11px] text-gray-400 space-y-1.5">
            <div className="flex items-center gap-2">
              <span className="text-red-400 font-mono text-[10px] bg-red-900/30 px-1.5 py-0.5 rounded w-28 text-center">ModelLoader</span>
              <span className="text-gray-600">model →</span>
              <span className="text-green-400 font-mono text-[10px] bg-green-900/30 px-1.5 py-0.5 rounded w-28 text-center">ModelTester</span>
              <span className="text-gray-500 ml-1">모델 객체 전달</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-blue-400 font-mono text-[10px] bg-blue-900/30 px-1.5 py-0.5 rounded w-28 text-center">TrainValSplit</span>
              <span className="text-gray-600">test_features →</span>
              <span className="text-green-400 font-mono text-[10px] bg-green-900/30 px-1.5 py-0.5 rounded w-28 text-center">ModelTester</span>
              <span className="text-gray-500 ml-1">테스트 입력 데이터</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-blue-400 font-mono text-[10px] bg-blue-900/30 px-1.5 py-0.5 rounded w-28 text-center">TrainValSplit</span>
              <span className="text-gray-600">test_labels →</span>
              <span className="text-green-400 font-mono text-[10px] bg-green-900/30 px-1.5 py-0.5 rounded w-28 text-center">ModelTester</span>
              <span className="text-gray-500 ml-1">정답 레이블 (선택사항)</span>
            </div>
          </div>
        </div>
      </div>

      {/* 튜토리얼 5: Model Manager 사용법 */}
      <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700/50">
        <h3 className="text-lg font-bold text-white mb-2">튜토리얼 5: Model Manager 활용</h3>
        <p className="text-xs text-gray-400 mb-4">모델의 저장/관리/삭제 및 Predict 패널 활용법</p>

        <div className="space-y-4">
          <div>
            <h4 className="text-sm font-bold text-purple-400 mb-2">Model Manager 열기</h4>
            <div className="text-xs text-gray-400 space-y-1">
              <p>{'>'} 툴바의 <span className="font-mono text-white bg-gray-700 px-1.5 py-0.5 rounded">Models</span> 버튼을 클릭합니다.</p>
              <p>{'>'} 또는 학습 완료 후 나타나는 <span className="font-mono text-blue-400 bg-blue-900/30 px-1.5 py-0.5 rounded">Save Model</span> 버튼을 클릭합니다.</p>
            </div>
          </div>

          <div>
            <h4 className="text-sm font-bold text-purple-400 mb-2">Save 탭</h4>
            <div className="text-xs text-gray-400 space-y-1">
              <p>{'>'} 현재 메모리에 학습된 모델이 있어야 활성화됩니다.</p>
              <p>{'>'} 모델 이름을 입력하고 Save 버튼을 클릭합니다.</p>
              <p>{'>'} 모델 파일 + 메타데이터 + Scaler가 <span className="font-mono text-yellow-400">outputs/saved_models/</span> 폴더에 저장됩니다.</p>
            </div>
          </div>

          <div>
            <h4 className="text-sm font-bold text-purple-400 mb-2">Saved Models 탭</h4>
            <div className="text-xs text-gray-400 space-y-1">
              <p>{'>'} 저장된 모든 모델이 카드 형태로 표시됩니다.</p>
              <p>{'>'} 각 카드에는 프레임워크(TF/PyTorch), 태스크 유형(분류/회귀), 클래스 수, 저장 시간이 표시됩니다.</p>
              <p>{'>'} <span className="text-green-400 font-bold">Load</span> 버튼: 모델을 메모리에 로드하여 Predict 패널에서 바로 테스트할 수 있습니다.</p>
              <p>{'>'} <span className="text-red-400 font-bold">Del</span> 버튼: 모델을 디스크에서 삭제합니다.</p>
            </div>
          </div>

          <div>
            <h4 className="text-sm font-bold text-purple-400 mb-2">Trainer 직접 연결 vs ModelLoader 노드</h4>
            <div className="bg-gray-900/50 rounded-lg p-3">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-gray-500">
                    <th className="text-left py-1 pr-3">방법</th>
                    <th className="text-left py-1 pr-3">사용 시점</th>
                    <th className="text-left py-1">장점</th>
                  </tr>
                </thead>
                <tbody className="text-gray-400">
                  <tr className="border-t border-gray-700/50">
                    <td className="py-1.5 pr-3 text-white font-bold">Trainer → ModelTester</td>
                    <td className="py-1.5 pr-3">학습 직후 바로 테스트</td>
                    <td className="py-1.5">한 파이프라인에서 학습+테스트 완료</td>
                  </tr>
                  <tr className="border-t border-gray-700/50">
                    <td className="py-1.5 pr-3 text-white font-bold">ModelLoader → ModelTester</td>
                    <td className="py-1.5 pr-3">나중에 다른 데이터로 테스트</td>
                    <td className="py-1.5">모델 재학습 없이 다양한 데이터로 검증</td>
                  </tr>
                  <tr className="border-t border-gray-700/50">
                    <td className="py-1.5 pr-3 text-white font-bold">Model Manager Load</td>
                    <td className="py-1.5 pr-3">간단한 예측 테스트</td>
                    <td className="py-1.5">노드 없이 Predict 패널에서 바로 테스트</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── 6. 예제 목록 ──────────────────────────────────────── */
function ExamplesSection({ examples, allExamples, catalog, category, onCategoryChange, categories, selectedId, onSelect }: {
  examples: ExampleFull[]; allExamples: ExampleFull[];
  catalog: any[]; category: string;
  onCategoryChange: (c: string) => void;
  categories: string[];
  selectedId: number | null;
  onSelect: (id: number | null) => void;
}) {
  const selected = selectedId !== null ? allExamples.find(e => e.meta.id === selectedId) : null;

  if (selected) {
    return <ExampleDetail example={selected} catalog={catalog} onBack={() => onSelect(null)} />;
  }

  return (
    <div>
      <SectionTitle>📚 학습 예제 ({allExamples.length}개)</SectionTitle>

      {/* 카테고리 필터 */}
      <div className="flex gap-2 mb-6 flex-wrap">
        <button
          onClick={() => onCategoryChange('all')}
          className={`px-3 py-1.5 text-xs rounded-full ${category === 'all' ? 'bg-blue-600 text-white font-bold' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}`}
        >
          전체 ({allExamples.length})
        </button>
        {categories.map(cat => {
          const info = CATEGORY_INFO[cat] || { label: cat, label_ko: cat, icon: '📄' };
          const count = allExamples.filter(e => e.meta.category === cat).length;
          return (
            <button
              key={cat}
              onClick={() => onCategoryChange(cat)}
              className={`px-3 py-1.5 text-xs rounded-full ${category === cat ? 'bg-blue-600 text-white font-bold' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}`}
            >
              {info.icon} {info.label_ko} ({count})
            </button>
          );
        })}
      </div>

      {/* 예제 그리드 */}
      <div className="grid grid-cols-2 gap-3">
        {examples.map(ex => {
          const m = ex.meta;
          const info = CATEGORY_INFO[m.category] || { icon: '📄', label_ko: m.category };
          return (
            <button
              key={m.id}
              onClick={() => onSelect(m.id)}
              className="text-left bg-gray-800/50 rounded-lg p-4 border border-gray-700/50 hover:border-blue-600/50 hover:bg-gray-800 transition-colors"
            >
              <div className="flex items-start justify-between mb-2">
                <span className="text-lg">{info.icon}</span>
                <div className="flex gap-1">
                  <span className="text-[9px] px-1.5 py-0.5 rounded" style={{
                    background: (DIFF_COLORS[m.difficulty] || '#888') + '22',
                    color: DIFF_COLORS[m.difficulty] || '#888',
                  }}>
                    {DIFF_LABELS[m.difficulty] || m.difficulty}
                  </span>
                  <span className="text-[9px] px-1.5 py-0.5 rounded bg-gray-700 text-gray-400">
                    {m.estimated_time}
                  </span>
                </div>
              </div>
              <h4 className="text-sm font-bold text-white mb-0.5">
                #{m.id} {m.title_ko}
              </h4>
              <p className="text-[10px] text-gray-500 mb-2 line-clamp-2">{m.description_ko}</p>
              <div className="flex gap-1 flex-wrap">
                {m.tags.slice(0, 4).map(tag => (
                  <span key={tag} className="text-[8px] px-1 py-0.5 rounded bg-gray-700/50 text-gray-500">{tag}</span>
                ))}
              </div>
              <div className="mt-2 text-[9px] text-gray-600">
                {ex.nodes.length} nodes · {ex.edges.length} edges
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

/* ── 예제 상세 ──────────────────────────────────────────── */
function ExampleDetail({ example, catalog, onBack }: {
  example: ExampleFull; catalog: any[]; onBack: () => void;
}) {
  const m = example.meta;
  const info = CATEGORY_INFO[m.category] || { icon: '📄', label_ko: m.category };
  const loadExample = useStore(s => s.loadExample);

  const handleLoadExample = () => {
    loadExample(m.id);
    // HelpModal will be closed by parent
  };

  return (
    <div>
      <button onClick={onBack} className="text-sm text-blue-400 hover:text-blue-300 mb-4 flex items-center gap-1">
        ← 예제 목록으로
      </button>

      {/* 헤더 */}
      <div className="bg-gradient-to-r from-gray-800 to-gray-800/50 rounded-xl p-6 border border-gray-700/50 mb-6">
        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="text-2xl">{info.icon}</span>
              <span className="text-[10px] px-2 py-0.5 rounded" style={{
                background: (DIFF_COLORS[m.difficulty] || '#888') + '33',
                color: DIFF_COLORS[m.difficulty] || '#888',
              }}>
                {DIFF_LABELS[m.difficulty]}
              </span>
              <span className="text-[10px] px-2 py-0.5 rounded bg-gray-700 text-gray-400">{m.estimated_time}</span>
            </div>
            <h2 className="text-xl font-bold text-white">#{m.id} {m.title_ko}</h2>
            <p className="text-sm text-gray-400 mt-1">{m.title}</p>
          </div>
          <button
            onClick={handleLoadExample}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white text-sm rounded-lg font-medium"
          >
            ▶ 캔버스에 로드
          </button>
        </div>
        <p className="text-sm text-gray-300 mt-4 leading-relaxed">{m.description_ko}</p>

        <div className="flex gap-2 mt-3 flex-wrap">
          {m.tags.map(tag => (
            <span key={tag} className="text-[10px] px-2 py-0.5 rounded bg-gray-700/50 text-gray-500">{tag}</span>
          ))}
        </div>
      </div>

      {/* 데이터셋 정보 */}
      {m.dataset_info && (
        <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700/50 mb-6">
          <h3 className="text-sm font-bold text-white mb-3">📊 데이터셋</h3>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="bg-gray-900/50 rounded p-2">
              <span className="text-gray-500">이름:</span>
              <span className="text-white ml-2">{m.dataset_info.name}</span>
            </div>
            <div className="bg-gray-900/50 rounded p-2">
              <span className="text-gray-500">출처:</span>
              <span className="text-white ml-2">{m.dataset_info.source}</span>
            </div>
            <div className="bg-gray-900/50 rounded p-2">
              <span className="text-gray-500">샘플 수:</span>
              <span className="text-blue-400 ml-2 font-mono">{m.dataset_info.samples?.toLocaleString()}</span>
            </div>
            <div className="bg-gray-900/50 rounded p-2">
              <span className="text-gray-500">특성:</span>
              <span className="text-white ml-2">{m.dataset_info.features}</span>
            </div>
            {m.dataset_info.classes && (
              <div className="bg-gray-900/50 rounded p-2">
                <span className="text-gray-500">클래스:</span>
                <span className="text-green-400 ml-2 font-mono">{m.dataset_info.classes}</span>
              </div>
            )}
            {m.dataset_info.target && (
              <div className="bg-gray-900/50 rounded p-2">
                <span className="text-gray-500">타겟:</span>
                <span className="text-white ml-2">{m.dataset_info.target}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* 파이프라인 다이어그램 */}
      <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700/50 mb-6">
        <h3 className="text-sm font-bold text-white mb-3">🔗 파이프라인 구조</h3>
        <PipelineDiagram nodes={example.nodes} edges={example.edges} catalog={catalog} />

        {/* 노드별 상세 */}
        <div className="mt-4 space-y-2">
          {example.nodes.map(node => {
            const catItem = catalog.find(c => c.type === node.type);
            const cat = catItem?.category || 'misc';
            const nodeDesc = NODE_DESCRIPTIONS[node.type];
            const params = Object.entries(node.params).filter(([k, v]) => v !== '' && v !== null && !k.startsWith('_'));

            return (
              <div key={node.id} className="flex gap-3 items-start bg-gray-900/30 rounded p-2">
                <div className="w-2 h-full rounded-full flex-shrink-0 mt-1" style={{ background: CAT_COLORS[cat] || '#6b7280', minHeight: 20 }} />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-bold text-white">{node.type}</span>
                    <span className="text-[9px] text-gray-600">{node.id}</span>
                  </div>
                  {nodeDesc && <p className="text-[10px] text-gray-500">{nodeDesc.desc_ko}</p>}
                  {params.length > 0 && (
                    <div className="flex gap-2 mt-0.5 flex-wrap">
                      {params.map(([k, v]) => (
                        <span key={k} className="text-[9px] text-gray-600">
                          <span className="text-gray-500">{k}=</span>
                          <span className="text-gray-400">{String(v).substring(0, 25)}</span>
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* 연결(엣지) 설명 */}
      <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700/50 mb-6">
        <h3 className="text-sm font-bold text-white mb-3">🔗 노드 연결</h3>
        <div className="space-y-1">
          {example.edges.map((edge, i) => {
            const srcNode = example.nodes.find(n => n.id === edge.source);
            const tgtNode = example.nodes.find(n => n.id === edge.target);
            return (
              <div key={i} className="flex items-center gap-2 text-[10px]">
                <span className="text-blue-400 font-mono">{srcNode?.type || edge.source}</span>
                <span className="text-gray-600">[{edge.sourceHandle || 'output'}]</span>
                <span className="text-gray-500">→</span>
                <span className="text-green-400 font-mono">{tgtNode?.type || edge.target}</span>
                <span className="text-gray-600">[{edge.targetHandle || 'input'}]</span>
              </div>
            );
          })}
        </div>
      </div>

      {/* 학습 목표 */}
      {m.learning_objectives && m.learning_objectives.length > 0 && (
        <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700/50 mb-6">
          <h3 className="text-sm font-bold text-white mb-3">🎯 학습 목표</h3>
          <ul className="space-y-1.5">
            {m.learning_objectives.map((obj, i) => (
              <li key={i} className="flex gap-2 text-xs text-gray-400">
                <span className="text-green-500">✓</span>
                {obj}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* 가이드 스텝 */}
      {m.guide_steps && m.guide_steps.length > 0 && (
        <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700/50">
          <h3 className="text-sm font-bold text-white mb-3">📝 단계별 가이드</h3>
          <div className="space-y-2">
            {m.guide_steps.map(step => (
              <div key={step.step} className="flex gap-3 items-start">
                <div className="w-6 h-6 rounded-full bg-blue-600/30 flex items-center justify-center text-xs text-blue-400 flex-shrink-0">
                  {step.step}
                </div>
                <div>
                  <span className="text-xs font-bold text-white">{step.node}</span>
                  <p className="text-xs text-gray-400">{step.text}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
