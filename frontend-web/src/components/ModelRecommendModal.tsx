import React, { useState } from 'react';
import { useStore } from '../store/useStore';

interface Props {
  onClose: () => void;
  onLoadExample: (id: number) => void;
}

/* ── 데이터 형식 ────────────────────────────────── */
const DATA_TYPES = [
  { id: 'csv', label: 'CSV / 엑셀', icon: '📊', desc: '표 형태 데이터 (수치, 범주형)' },
  { id: 'image', label: '이미지', icon: '🖼️', desc: '사진, 의료영상, 위성 등' },
  { id: 'text', label: '텍스트', icon: '📝', desc: '문장, 문서, 리뷰 등' },
  { id: 'timeseries', label: '시계열', icon: '📈', desc: '센서, 주가, 날씨 등 시간 데이터' },
  { id: 'audio', label: '음성', icon: '🎵', desc: 'MFCC 특성 추출 후 학습' },
  { id: 'other', label: '기타', icon: '📦', desc: '혼합 또는 커스텀 데이터' },
] as const;

/* ── ML 방법 (데이터 유형별 동적) ─────────────── */
interface MethodItem { id: string; label: string; icon: string; desc: string }

function getMethodsForDataType(dataType: string): MethodItem[] {
  switch (dataType) {
    case 'csv':
    case 'audio':
    case 'other':
      return [
        { id: 'classification', icon: '🏷️', label: '분류', desc: '카테고리 예측 (스팸/정상, 품종, 유형)' },
        { id: 'binary', icon: '⚖️', label: '이진 분류', desc: '두 가지 중 하나 예측 (양성/음성, 사기/정상)' },
        { id: 'imbalanced', icon: '📊', label: '불균형 데이터 분류', desc: '희귀 클래스 탐지 (사기, 결함, 희귀질환)' },
        { id: 'regression', icon: '📈', label: '회귀 (수치 예측)', desc: '가격, 온도, 수량 등 연속값 예측' },
        { id: 'anomaly', icon: '🔍', label: '이상치 탐지', desc: 'Autoencoder로 정상/비정상 구분' },
        { id: 'clustering', icon: '🔵', label: '클러스터링 / 시각화', desc: '데이터 군집 탐색, 차원 축소' },
        { id: 'kfold', icon: '🔄', label: '교차 검증 (K-Fold)', desc: '신뢰성 높은 모델 평가' },
        { id: 'ensemble', icon: '🏆', label: '앙상블 / 모델 비교', desc: '여러 모델 병렬 학습 후 비교' },
        { id: 'unknown', icon: '❓', label: '잘 모르겠음', desc: '데이터에 맞는 방법을 추천받겠습니다' },
      ];
    case 'image':
      return [
        { id: 'img_classification', icon: '🏷️', label: '이미지 분류', desc: '사진을 카테고리로 분류 (개/고양이, 질병)' },
        { id: 'img_transfer', icon: '🔄', label: '전이학습 분류', desc: 'ResNet/EfficientNet 등 사전학습 모델 활용' },
        { id: 'img_regression', icon: '📐', label: '이미지 회귀', desc: '나이 예측, 바운딩 박스, 좌표 추정' },
        { id: 'img_segmentation', icon: '✂️', label: '세그멘테이션', desc: '픽셀 단위 분류 (U-Net, 의료영상)' },
        { id: 'img_similarity', icon: '🔗', label: '유사도 비교', desc: '시암 네트워크 — 두 이미지 유사도 판별' },
        { id: 'img_superres', icon: '🔎', label: '초해상도 / 복원', desc: '저해상도 → 고해상도 이미지 변환' },
        { id: 'img_anomaly', icon: '🔍', label: '이상 탐지 (결함)', desc: 'Conv Autoencoder로 결함/이상 검출' },
        { id: 'img_interpret', icon: '🧠', label: '모델 해석 (Grad-CAM)', desc: '학습된 CNN의 판단 근거 시각화' },
        { id: 'img_embed', icon: '🗺️', label: '임베딩 시각화', desc: '이미지 특성 벡터를 2D/3D로 시각화' },
        { id: 'img_mixed', icon: '🔀', label: '혼합 입력 (이미지+수치)', desc: '이미지와 표 데이터를 함께 사용' },
        { id: 'unknown', icon: '❓', label: '잘 모르겠음', desc: '데이터에 맞는 방법을 추천받겠습니다' },
      ];
    case 'text':
      return [
        { id: 'txt_classification', icon: '🏷️', label: '텍스트 분류', desc: '감성 분석, 스팸 탐지, 주제 분류' },
        { id: 'txt_multilabel', icon: '🏷️', label: '다중 레이블 분류', desc: '여러 카테고리 동시 예측 (영화 장르)' },
        { id: 'txt_ner', icon: '📌', label: '시퀀스 라벨링 (NER)', desc: '토큰별 레이블 예측 (개체명 인식)' },
        { id: 'txt_regression', icon: '📈', label: '텍스트 회귀', desc: '감성 강도, 유사도 점수 예측' },
        { id: 'txt_generation', icon: '✏️', label: '텍스트 생성', desc: '다음 단어 예측, 언어 모델' },
        { id: 'txt_similarity', icon: '🔗', label: '문장 유사도', desc: '두 문장의 유사도/관계 판별' },
        { id: 'txt_clustering', icon: '🔵', label: '텍스트 클러스터링', desc: '문서 군집 탐색, 임베딩 시각화' },
        { id: 'unknown', icon: '❓', label: '잘 모르겠음', desc: '데이터에 맞는 방법을 추천받겠습니다' },
      ];
    case 'timeseries':
      return [
        { id: 'ts_classification', icon: '🏷️', label: '시계열 분류', desc: '센서 패턴 분류, 행동 인식' },
        { id: 'ts_forecast', icon: '📈', label: '단일 스텝 예측', desc: '다음 값 1개 예측 (기온, 주가)' },
        { id: 'ts_multistep', icon: '📊', label: '멀티 스텝 예측', desc: '미래 N개 값 예측 (풍속, 판매량)' },
        { id: 'ts_anomaly', icon: '🔍', label: '이상 탐지', desc: '시계열 패턴 이상 감지 (ECG, IoT)' },
        { id: 'ts_hybrid', icon: '🔀', label: 'CNN + RNN 하이브리드', desc: 'Conv1D + LSTM 조합 예측' },
        { id: 'ts_transformer', icon: '⚡', label: 'Transformer 예측', desc: 'Self-Attention 기반 시계열 예측' },
        { id: 'ts_seq2seq', icon: '↔️', label: 'Seq2Seq (인코더-디코더)', desc: '시퀀스 → 시퀀스 변환/예측' },
        { id: 'unknown', icon: '❓', label: '잘 모르겠음', desc: '데이터에 맞는 방법을 추천받겠습니다' },
      ];
    default:
      return [
        { id: 'classification', icon: '🏷️', label: '분류', desc: '카테고리 예측' },
        { id: 'regression', icon: '📈', label: '회귀', desc: '수치 예측' },
        { id: 'unsupervised', icon: '🔵', label: '비지도', desc: '클러스터링, 이상치 탐지' },
        { id: 'unknown', icon: '❓', label: '잘 모르겠음', desc: '추천받겠습니다' },
      ];
  }
}

/* ── 데이터 크기 ─────────────────────────────── */
const DATA_SIZES = [
  { id: 'tiny', label: '100건 미만', desc: '매우 소량 — 전이학습 권장' },
  { id: 'small', label: '100 ~ 1,000건', desc: '소량 — 간단한 모델 권장' },
  { id: 'medium', label: '1,000 ~ 10,000건', desc: '보통 — 대부분 모델 가능' },
  { id: 'large', label: '10,000건 이상', desc: '대량 — 깊은 네트워크 가능' },
  { id: 'unknown', label: '잘 모르겠음', desc: '' },
] as const;

/* ── 추천 로직 ───────────────────────────────── */
interface Recommendation {
  model: string;
  description: string;
  layers: string[];
  tips: string[];
  exampleIds: number[];
}

function getRecommendations(
  dataType: string, method: string, size: string
): Recommendation[] {
  const r: Recommendation[] = [];
  const m = method;
  const isSmall = size === 'tiny' || size === 'small';

  /* ═══════════════════════════════════════════
     CSV / 엑셀 / 음성(MFCC) / 기타 (표 형태)
     ═══════════════════════════════════════════ */
  if (dataType === 'csv' || dataType === 'audio' || dataType === 'other') {

    // ── 분류 (다중 클래스) ──
    if (m === 'classification' || m === 'unknown') {
      r.push({
        model: 'Dense Network (MLP) 분류',
        description: '표 형태 데이터의 기본 분류 모델. 수치/범주형 특성 모두 처리 가능.',
        layers: ['CSVLoader', 'StandardScaler', 'TrainValSplit', 'Dense×2~3', 'Optimizer', 'LossFunction', 'Trainer'],
        tips: [
          '특성 정규화(StandardScaler)를 반드시 적용',
          '출력층: softmax(다중 클래스)',
          isSmall ? '과적합 주의 — Dropout(0.3~0.5) 추가' : '은닉층 유닛: 64 → 32 → 클래스 수',
        ],
        exampleIds: [1, 11, 12, 15, 21],
      });
      r.push({
        model: 'Dense + LabelEncoder (범주형 포함)',
        description: '문자열 레이블이 포함된 데이터. LabelEncoder로 인코딩 후 학습.',
        layers: ['CSVLoader', 'LabelEncoder', 'TrainValSplit', 'Dense×2', 'Trainer'],
        tips: ['target 컬럼이 문자열이면 LabelEncoder 필수', '범주형 특성이 많으면 OneHot도 고려'],
        exampleIds: [2, 9, 10],
      });
      r.push({
        model: 'Dense + PCA (차원 축소)',
        description: 'PCA로 고차원 특성을 줄인 후 분류. 특성 수가 매우 많을 때.',
        layers: ['CSVLoader', 'PCA', 'TrainValSplit', 'Dense×2', 'Trainer'],
        tips: ['PCA 컴포넌트 수: 전체 분산의 95% 유지 권장', '시각화도 가능 (2~3차원)'],
        exampleIds: [18],
      });
    }

    // ── 이진 분류 ──
    if (m === 'binary' || m === 'unknown') {
      r.push({
        model: 'Dense 이진 분류 (sigmoid)',
        description: '두 가지 중 하나를 예측. 양성/음성, 생존/사망 등.',
        layers: ['CSVLoader', 'LabelEncoder', 'StandardScaler', 'TrainValSplit', 'Dense×2', 'Dense(sigmoid)', 'Trainer(BCE)'],
        tips: ['출력층: units=1, activation=sigmoid', 'Loss: BinaryCrossentropy', '임계값 0.5 기준으로 분류'],
        exampleIds: [2, 12, 13],
      });
      r.push({
        model: 'Dense + MinMaxScaler + Dropout 이진 분류',
        description: 'Dropout으로 과적합 방지. 의료/금융 등 정밀도 중요한 이진 분류.',
        layers: ['CSVLoader', 'MinMaxScaler', 'TrainValSplit', 'Dense', 'Dropout', 'Dense(sigmoid)', 'Trainer'],
        tips: ['MinMaxScaler: 0~1 범위 정규화', 'Dropout: 0.3~0.5'],
        exampleIds: [13, 14],
      });
    }

    // ── 불균형 데이터 ──
    if (m === 'imbalanced' || m === 'unknown') {
      r.push({
        model: 'ClassWeight 불균형 처리',
        description: '소수 클래스에 높은 가중치를 부여하여 균형 있는 학습.',
        layers: ['CSVLoader', 'StandardScaler', 'TrainValSplit', 'Dense', 'Dropout', 'Dense', 'Trainer(ClassWeight)'],
        tips: ['ClassWeight로 소수 클래스에 높은 가중치', 'Recall/F1-Score로 평가 (Accuracy 부적합)'],
        exampleIds: [14, 97],
      });
      r.push({
        model: 'FocalLoss 불균형 처리',
        description: 'Focal Loss로 어려운 샘플에 집중하여 학습.',
        layers: ['CSVLoader', 'StandardScaler', 'TrainValSplit', 'Dense×2', 'Trainer(FocalLoss)'],
        tips: ['FocalLoss: 쉬운 샘플의 loss를 줄여 어려운 샘플에 집중', '감마(γ): 2.0이 일반적'],
        exampleIds: [97],
      });
      r.push({
        model: 'Oversampling + Dense',
        description: '소수 클래스를 복제/생성하여 데이터 균형을 맞춘 후 학습.',
        layers: ['CSVLoader', 'StandardScaler', 'Oversampling', 'TrainValSplit', 'Dense×2', 'Trainer'],
        tips: ['SMOTE 등 합성 오버샘플링', '검증 세트에는 오버샘플링 적용 금지'],
        exampleIds: [97],
      });
    }

    // ── 회귀 ──
    if (m === 'regression' || m === 'unknown') {
      r.push({
        model: 'Dense Regression (수치 예측)',
        description: '표 데이터에서 수치를 예측하는 기본 회귀 모델.',
        layers: ['CSVLoader', 'StandardScaler', 'TrainValSplit', 'Dense×3~4', 'Trainer(MSE)'],
        tips: ['출력층: activation=linear, units=1', 'Loss: MSE 또는 MAE', '특성 스케일링이 회귀에서 특히 중요'],
        exampleIds: [26, 33, 36, 39],
      });
      r.push({
        model: 'Dense + MinMaxScaler 회귀',
        description: 'MinMax 정규화 기반 회귀. 출력 범위가 제한적일 때 적합.',
        layers: ['CSVLoader', 'MinMaxScaler', 'TrainValSplit', 'Dense×2~3', 'Trainer'],
        tips: ['특성값 범위가 다양하면 MinMax가 효과적', 'Dropout으로 과적합 방지'],
        exampleIds: [27, 40],
      });
      r.push({
        model: 'Dense + OneHotEncoder 회귀',
        description: '범주형 변수가 포함된 데이터의 회귀. OneHot 인코딩 후 학습.',
        layers: ['CSVLoader', 'OneHotEncoder', 'StandardScaler', 'TrainValSplit', 'Dense×3~4', 'Trainer'],
        tips: ['범주형(지역, 직업 등)은 OneHot 필수', '인코딩 후 StandardScaler 적용'],
        exampleIds: [35, 45],
      });
    }

    // ── 이상치 탐지 ──
    if (m === 'anomaly' || m === 'unknown') {
      r.push({
        model: 'Dense Autoencoder (이상치 탐지)',
        description: '정상 패턴을 학습하고 복원 오차로 이상치를 탐지.',
        layers: ['CSVLoader', 'StandardScaler', 'Dense(enc 64→32→16)', 'Dense(dec 16→32→64)', 'Trainer(MSE)'],
        tips: ['인코더: 차원을 점진적으로 축소', '디코더: 대칭 구조', 'Loss: MSE (입력↔복원 차이)'],
        exampleIds: [47, 48],
      });
    }

    // ── 클러스터링/시각화 ──
    if (m === 'clustering' || m === 'unknown') {
      r.push({
        model: 'PCA + t-SNE 시각화 (클러스터링)',
        description: '고차원 데이터를 2D/3D로 투영하여 군집 구조를 시각화.',
        layers: ['CSVLoader', 'StandardScaler', 'PCA', 'tSNEViewer'],
        tips: ['학습 없이 데이터 구조 탐색 가능', 'K-Means 등 클러스터링과 결합'],
        exampleIds: [46, 50],
      });
      r.push({
        model: 'PCA + UMAP 시각화',
        description: 't-SNE보다 빠르고 전역 구조를 잘 보존하는 시각화.',
        layers: ['CSVLoader', 'StandardScaler', 'PCA', 'UMAPViewer'],
        tips: ['대용량 데이터에 t-SNE보다 빠름', '클러스터 경계가 더 명확'],
        exampleIds: [51],
      });
    }

    // ── K-Fold 교차 검증 ──
    if (m === 'kfold' || m === 'unknown') {
      r.push({
        model: 'K-Fold 교차 검증',
        description: '데이터를 K개로 분할하여 더 신뢰할 수 있는 평가 수행.',
        layers: ['CSVLoader', 'StandardScaler', 'Dense', 'BatchNorm', 'Dropout', 'KFoldTrainer'],
        tips: ['K=5가 일반적', '소량 데이터에서 과적합 방지에 효과적'],
        exampleIds: [98],
      });
    }

    // ── 앙상블 / 모델 비교 ──
    if (m === 'ensemble' || m === 'unknown') {
      r.push({
        model: '앙상블 모델 비교',
        description: '동일 데이터에 여러 모델을 병렬 학습 후 성능 비교.',
        layers: ['데이터 → 3개 모델 병렬 학습 → Metrics 비교'],
        tips: ['동일 데이터/동일 분할로 공정 비교', '최적 모델 선택의 근거 마련'],
        exampleIds: [92],
      });
      r.push({
        model: '하이퍼파라미터 탐색 (Grid Search)',
        description: '학습률, 유닛 수, Dropout 등 최적 조합을 자동 탐색.',
        layers: ['데이터 → StandardScaler → Dense → Trainer(GridSearch)'],
        tips: ['탐색 범위를 좁게 시작', '학습률, 은닉 유닛 수가 가장 영향 큼'],
        exampleIds: [93],
      });
    }
  }

  /* ═══════════════════════════════════════════
     이미지
     ═══════════════════════════════════════════ */
  if (dataType === 'image') {

    // ── 이미지 분류 (직접 CNN) ──
    if (m === 'img_classification' || m === 'unknown') {
      r.push({
        model: 'CNN (Conv2D + Pooling)',
        description: '이미지 분류의 표준 모델. 합성곱으로 특성 추출.',
        layers: ['ImageFolder', 'Conv2D×2~4', 'MaxPooling2D', 'BatchNorm', 'Flatten', 'Dense', 'Trainer'],
        tips: ['필터 수: 32→64→128 순서로 증가', 'BatchNorm + Dropout 조합', isSmall ? 'Conv 2~3층이면 충분' : '깊은 네트워크(4~5 Conv) 가능'],
        exampleIds: [4, 5, 6, 7, 16, 19, 24, 65],
      });
      r.push({
        model: 'CNN + GlobalAveragePooling',
        description: 'Flatten 대신 GlobalAvgPool 사용. 파라미터 수 감소로 과적합 방지.',
        layers: ['ImageFolder', 'Conv2D×2~3', 'GlobalAveragePooling2D', 'Dense', 'Trainer'],
        tips: ['Flatten보다 경량', '공간 정보를 평균으로 요약'],
        exampleIds: [5, 7, 66, 67],
      });
      r.push({
        model: 'CNN + 데이터 증강 비교',
        description: '증강(flip, rotate, crop) 유무에 따른 성능 차이 실험.',
        layers: ['ImageFolder', 'Augmentation', 'Conv2D×2', 'MaxPooling2D', 'GlobalAvgPool', 'Dense', 'Trainer'],
        tips: ['동일 모델에 증강 유/무로 비교', '소량 데이터에서 증강 효과가 큼'],
        exampleIds: [58],
      });
    }

    // ── 전이학습 분류 ──
    if (m === 'img_transfer' || m === 'unknown') {
      r.push({
        model: '전이학습 — MobileNetV2',
        description: '경량 모델. 빠른 학습과 적은 메모리 사용.',
        layers: ['ImageFolder', 'Augmentation', 'PretrainedModel(MobileNetV2)', 'GlobalAvgPool', 'Dense', 'Dropout', 'Dense', 'Trainer'],
        tips: ['소량 데이터에 특히 효과적', 'Pretrained는 frozen으로 시작', '모바일 배포에도 적합'],
        exampleIds: [8, 70],
      });
      r.push({
        model: '전이학습 — ResNet50',
        description: '깊은 Residual 네트워크. 복잡한 이미지 분류에 강력.',
        layers: ['ImageFolder', 'Augmentation', 'PretrainedModel(ResNet50)', 'GlobalAvgPool', 'Dense', 'Trainer'],
        tips: ['Skip Connection으로 깊은 학습 가능', '대용량 데이터에서 높은 성능'],
        exampleIds: [22, 56],
      });
      r.push({
        model: '전이학습 — EfficientNet',
        description: 'EfficientNet 기반. 정확도 대비 연산량이 가장 효율적.',
        layers: ['ImageFolder', 'Augmentation', 'PretrainedModel(EfficientNetB0)', 'GlobalAvgPool', 'Dense', 'Dropout', 'Dense', 'Trainer'],
        tips: ['ResNet보다 적은 파라미터로 높은 성능', '의료/피부 등 세밀한 분류에 적합'],
        exampleIds: [17, 57, 64],
      });
      r.push({
        model: '점진적 전이학습 (Progressive Fine-tuning)',
        description: '단계별로 레이어를 해동하며 세밀하게 학습.',
        layers: ['ImageFolder', 'Augmentation', 'PretrainedModel(frozen→unfreeze)', 'Dense', 'Trainer'],
        tips: ['1단계: 헤드만 학습 → 2단계: 일부 레이어 unfreeze', '학습률을 단계별로 낮춤'],
        exampleIds: [95],
      });
    }

    // ── 이미지 회귀 ──
    if (m === 'img_regression' || m === 'unknown') {
      r.push({
        model: 'CNN Regression (수치 예측)',
        description: '이미지에서 수치를 예측 (나이, 크기 등).',
        layers: ['ImageFolder', 'Conv2D×3', 'MaxPooling2D', 'GlobalAvgPool', 'Dense(linear)', 'Trainer(MSE)'],
        tips: ['출력층: units=1, activation=linear', 'Loss: MSE'],
        exampleIds: [31],
      });
      r.push({
        model: '바운딩 박스 회귀',
        description: '이미지에서 객체 위치(x, y, w, h)를 예측.',
        layers: ['ImageFolder', 'Conv2D×3', 'GlobalAvgPool', 'Dense(4출력, linear)', 'Trainer(MSE)'],
        tips: ['출력층: units=4 (x,y,w,h)', '좌표 정규화(0~1)하면 학습 안정적'],
        exampleIds: [62],
      });
    }

    // ── 세그멘테이션 ──
    if (m === 'img_segmentation' || m === 'unknown') {
      r.push({
        model: 'U-Net (이미지 세그멘테이션)',
        description: '픽셀 단위 분류. 의료 영상, 위성 이미지 등.',
        layers: ['ImageFolder', 'Conv2D(enc)', 'MaxPooling2D', 'Conv2DTranspose(dec)', 'Add(skip)', 'Trainer'],
        tips: ['인코더-디코더 + Skip Connection 구조', '출력 shape = 입력 shape', 'Loss: BinaryCE 또는 Dice'],
        exampleIds: [61],
      });
    }

    // ── 유사도 비교 ──
    if (m === 'img_similarity' || m === 'unknown') {
      r.push({
        model: '시암 네트워크 (Siamese Network)',
        description: '두 이미지의 유사도를 학습. 얼굴 인증, 제품 매칭 등.',
        layers: ['ImageFolder×2', '공유 Conv2D', 'GlobalAvgPool', 'Dense', 'Trainer(ContrastiveLoss)'],
        tips: ['두 입력이 같은 가중치를 공유', 'Contrastive 또는 Triplet Loss'],
        exampleIds: [63],
      });
    }

    // ── 초해상도/복원 ──
    if (m === 'img_superres' || m === 'unknown') {
      r.push({
        model: '초해상도 (SRCNN)',
        description: '저해상도 이미지를 고해상도로 복원하는 CNN.',
        layers: ['NumpyInput', 'Conv2D×3', 'Trainer(MSE)'],
        tips: ['입력: 저해상도, 출력: 고해상도', 'Loss: MSE (픽셀 단위 비교)'],
        exampleIds: [68],
      });
    }

    // ── 이상 탐지 (결함) ──
    if (m === 'img_anomaly' || m === 'unknown') {
      r.push({
        model: 'Conv Autoencoder (이미지 이상 탐지)',
        description: '합성곱 오토인코더로 이미지 결함/이상 검출.',
        layers: ['NumpyInput', 'Conv2D(enc)', 'Conv2DTranspose(dec)', 'Trainer(MSE)'],
        tips: ['인코더: Conv+Pool로 압축', '디코더: ConvTranspose로 복원', '복원 오차로 이상 탐지'],
        exampleIds: [49],
      });
    }

    // ── 모델 해석 (Grad-CAM) ──
    if (m === 'img_interpret' || m === 'unknown') {
      r.push({
        model: 'Grad-CAM 시각화',
        description: '학습된 CNN이 어떤 영역을 보고 판단했는지 히트맵으로 시각화.',
        layers: ['학습된 CNN 모델', 'GradCAMViewer'],
        tips: ['학습 완료 후 적용', '의료/자율주행 등 판단 근거 확인에 필수'],
        exampleIds: [59],
      });
      r.push({
        model: 'Feature Map 시각화',
        description: '각 Conv 레이어의 활성화 맵을 시각화하여 학습 과정 이해.',
        layers: ['학습된 CNN 모델', 'FeatureMapViewer'],
        tips: ['얕은 층: 엣지/텍스처, 깊은 층: 고수준 패턴', '디버깅과 모델 이해에 유용'],
        exampleIds: [60],
      });
    }

    // ── 임베딩 시각화 ──
    if (m === 'img_embed' || m === 'unknown') {
      r.push({
        model: 'Pretrained + UMAP 임베딩 시각화',
        description: '사전학습 모델의 특성 벡터를 2D/3D로 시각화.',
        layers: ['ImageFolder', 'PretrainedModel', 'GlobalAvgPool', 'UMAPViewer'],
        tips: ['학습 없이 이미지 유사도 구조 탐색', '클래스별 분포 확인에 유용'],
        exampleIds: [54, 69],
      });
    }

    // ── 혼합 입력 ──
    if (m === 'img_mixed' || m === 'unknown') {
      r.push({
        model: '혼합 입력 모델 (이미지 + 수치)',
        description: '이미지와 표 데이터를 동시에 입력하는 모델.',
        layers: ['ImageFolder', 'Conv2D', 'GlobalAvgPool', '+ CSVLoader', 'Dense', 'Concat', 'Dense', 'Trainer'],
        tips: ['두 경로를 Concat으로 합침', '각 경로에 독립적인 전처리 필요'],
        exampleIds: [94, 100],
      });
    }
  }

  /* ═══════════════════════════════════════════
     텍스트
     ═══════════════════════════════════════════ */
  if (dataType === 'text') {

    // ── 텍스트 분류 ──
    if (m === 'txt_classification' || m === 'unknown') {
      r.push({
        model: 'Embedding + LSTM 분류',
        description: '문장의 순서를 고려한 텍스트 분류의 기본 모델.',
        layers: ['CSVLoader', 'Embedding', 'LSTM', 'Dropout', 'Dense', 'Trainer'],
        tips: ['Embedding 차원: 64~128', 'LSTM units: 64~128', '긴 텍스트는 max_length로 절단'],
        exampleIds: [9, 10, 71, 80],
      });
      r.push({
        model: 'Embedding + GRU 분류',
        description: 'LSTM보다 빠르고 비슷한 성능. 짧은 텍스트에 효과적.',
        layers: ['CSVLoader', 'Embedding', 'GRU', 'Dense', 'Trainer'],
        tips: ['LSTM 대비 파라미터 적음 → 빠른 학습', '소량 데이터에서 유리할 수 있음'],
        exampleIds: [72, 78],
      });
      r.push({
        model: 'Stacked LSTM (깊은 텍스트 분류)',
        description: '2층 LSTM으로 더 복잡한 텍스트 패턴 학습.',
        layers: ['CSVLoader', 'Embedding', 'LSTM', 'LSTM', 'Dense', 'Trainer'],
        tips: ['첫 LSTM: return_sequences=True', '두 번째 LSTM이 최종 표현 생성'],
        exampleIds: [73, 77],
      });
      r.push({
        model: 'Transformer Block 분류',
        description: 'Self-Attention 기반. 긴 문서에서 전역 의존성 포착.',
        layers: ['CSVLoader', 'Embedding', 'TransformerBlock', 'Dense', 'Trainer'],
        tips: ['Attention으로 중요 토큰에 집중', 'LSTM보다 병렬 처리 가능'],
        exampleIds: [74, 79],
      });
      r.push({
        model: 'Conv1D 텍스트 분류',
        description: 'N-gram 패턴을 합성곱으로 추출. 코드/짧은 텍스트에 효과적.',
        layers: ['CSVLoader', 'Embedding', 'Conv1D', 'GlobalAvgPool1D', 'Dense', 'Trainer'],
        tips: ['커널 크기 = N-gram 크기 (3~5)', 'GlobalAvgPool로 가변 길이 처리'],
        exampleIds: [81],
      });
    }

    // ── 다중 레이블 분류 ──
    if (m === 'txt_multilabel' || m === 'unknown') {
      r.push({
        model: '다중 레이블 분류 (Multi-Label)',
        description: '하나의 입력에 여러 레이블 동시 예측 (영화 장르 등).',
        layers: ['CSVLoader', 'Embedding', 'LSTM', 'Dense(sigmoid)', 'Trainer(BCE)'],
        tips: ['출력층: sigmoid (각 레이블 독립)', 'Loss: BinaryCrossentropy', 'threshold 0.5로 다중 레이블 결정'],
        exampleIds: [25],
      });
    }

    // ── 시퀀스 라벨링 (NER) ──
    if (m === 'txt_ner' || m === 'unknown') {
      r.push({
        model: 'Bidirectional LSTM (양방향)',
        description: '양방향으로 문맥을 파악. 개체명 인식(NER), POS 태깅에 적합.',
        layers: ['CSVLoader', 'Embedding', 'Bidirectional LSTM', 'Dense', 'Trainer'],
        tips: ['양방향 → 앞뒤 문맥 동시 고려', '토큰 단위 분류에 적합', 'return_sequences=True'],
        exampleIds: [76],
      });
    }

    // ── 텍스트 회귀 ──
    if (m === 'txt_regression' || m === 'unknown') {
      r.push({
        model: 'LSTM 텍스트 회귀',
        description: '텍스트에서 수치 예측 (감성 점수, 유사도 등).',
        layers: ['CSVLoader', 'Embedding', 'LSTM', 'Dense(linear)', 'Trainer(MSE)'],
        tips: ['출력: activation=linear', 'Loss: MSE'],
        exampleIds: [75, 82],
      });
    }

    // ── 텍스트 생성 ──
    if (m === 'txt_generation' || m === 'unknown') {
      r.push({
        model: 'Stacked LSTM 언어 모델',
        description: '다음 단어를 예측하는 언어 모델. 텍스트 생성의 기본.',
        layers: ['CSVLoader', 'Embedding', 'LSTM×2', 'Dense(softmax)', 'Trainer(CE)'],
        tips: ['입력: 이전 N 단어, 출력: 다음 단어 확률', 'Temperature로 생성 다양성 조절'],
        exampleIds: [77],
      });
    }

    // ── 문장 유사도 ──
    if (m === 'txt_similarity' || m === 'unknown') {
      r.push({
        model: '문장 유사도 예측 (Dual LSTM)',
        description: '두 문장을 각각 인코딩 후 유사도를 비교/예측.',
        layers: ['CSVLoader', 'Embedding×2', 'LSTM×2', 'Add', 'Dense', 'Trainer'],
        tips: ['두 문장을 동일 LSTM으로 인코딩', 'Add/Concat으로 합친 후 비교'],
        exampleIds: [75],
      });
    }

    // ── 텍스트 클러스터링 ──
    if (m === 'txt_clustering' || m === 'unknown') {
      r.push({
        model: 'Embedding + LSTM → t-SNE 시각화',
        description: '텍스트 임베딩을 2D로 시각화. 문서 유사도/클러스터 탐색.',
        layers: ['CSVLoader', 'Embedding', 'LSTM', 'tSNEViewer'],
        tips: ['학습 없이 텍스트 분포 탐색', '클러스터링과 결합 가능'],
        exampleIds: [52, 55],
      });
    }
  }

  /* ═══════════════════════════════════════════
     시계열
     ═══════════════════════════════════════════ */
  if (dataType === 'timeseries') {

    // ── 시계열 분류 ──
    if (m === 'ts_classification' || m === 'unknown') {
      r.push({
        model: 'LSTM 시계열 분류',
        description: '시간 순서 패턴 분류. 센서, 신호 데이터에 적합.',
        layers: ['CSVLoader', 'StandardScaler', 'TrainValSplit', 'LSTM', 'Dense', 'Trainer'],
        tips: ['입력 shape: (timesteps, features)', 'LSTM units: 32~128'],
        exampleIds: [83, 89, 90],
      });
      r.push({
        model: 'GRU 시계열 분류',
        description: 'LSTM보다 빠른 시계열 분류. 진동, 고장 예측 등.',
        layers: ['CSVLoader', 'StandardScaler', 'TrainValSplit', 'GRU', 'Dense', 'Trainer'],
        tips: ['GRU: LSTM보다 파라미터 적음', '실시간 처리에 유리'],
        exampleIds: [87],
      });
      r.push({
        model: '멀티채널 LSTM (Reshape 포함)',
        description: '다변량 시계열을 Reshape 후 LSTM으로 분류.',
        layers: ['CSVLoader', 'TrainValSplit', 'Reshape', 'LSTM', 'Dense', 'Trainer'],
        tips: ['Reshape으로 (samples, timesteps, channels) 변환', '다변량 센서 데이터에 적합'],
        exampleIds: [90],
      });
    }

    // ── 단일 스텝 예측 ──
    if (m === 'ts_forecast' || m === 'unknown') {
      r.push({
        model: 'LSTM 시계열 예측',
        description: '과거 데이터로 미래 값 1개 예측. 주가, 기온, 전력 등.',
        layers: ['CSVLoader', 'MinMaxScaler', 'TrainValSplit', 'LSTM', 'Dense', 'Trainer'],
        tips: ['MinMaxScaler(0~1)가 시계열에 효과적', '윈도우 크기(lookback): 10~50'],
        exampleIds: [28, 34, 41, 42, 44],
      });
      r.push({
        model: 'Stacked LSTM (깊은 시계열 예측)',
        description: '2층 LSTM으로 복잡한 시간 패턴 학습.',
        layers: ['CSVLoader', 'MinMaxScaler', 'TrainValSplit', 'LSTM×2', 'Dense', 'Trainer'],
        tips: ['첫 LSTM: return_sequences=True', '주가, 풍속 등 복잡한 패턴에 효과적'],
        exampleIds: [29, 37],
      });
      r.push({
        model: 'GRU 시계열 예측',
        description: 'LSTM 대비 빠른 시계열 예측. 전력, 교통량 등.',
        layers: ['CSVLoader', 'StandardScaler', 'TrainValSplit', 'GRU', 'Dense', 'Trainer'],
        tips: ['단일/Stacked GRU 모두 가능', '실시간 예측에 적합'],
        exampleIds: [30, 38, 43],
      });
    }

    // ── 멀티 스텝 예측 ──
    if (m === 'ts_multistep' || m === 'unknown') {
      r.push({
        model: 'Stacked LSTM 멀티스텝 예측',
        description: '미래 N개의 값을 한 번에 예측.',
        layers: ['CSVLoader', 'MinMaxScaler', 'TrainValSplit', 'LSTM×2', 'Dense(N출력)', 'Trainer'],
        tips: ['출력층: units=예측 스텝 수', '단일스텝 반복보다 누적 오차 적음'],
        exampleIds: [84, 37],
      });
      r.push({
        model: 'Seq2Seq 인코더-디코더',
        description: '시퀀스 입력 → 시퀀스 출력. 멀티스텝 예측에 최적.',
        layers: ['CSVLoader', 'MinMaxScaler', 'TrainValSplit', 'LSTM(enc)', 'LSTM(dec)', 'Dense', 'Trainer'],
        tips: ['인코더가 문맥 벡터 생성', '디코더가 여러 스텝을 순차 출력'],
        exampleIds: [91],
      });
    }

    // ── 이상 탐지 ──
    if (m === 'ts_anomaly' || m === 'unknown') {
      r.push({
        model: 'LSTM Autoencoder (시계열 이상 탐지)',
        description: '정상 시계열 패턴을 학습, 복원 오차로 이상 구간 탐지.',
        layers: ['CSVLoader', 'MinMaxScaler', 'TrainValSplit', 'LSTM(enc)', 'LSTM(dec)', 'Dense', 'Trainer(MSE)'],
        tips: ['정상 데이터만으로 학습', '복원 오차 임계값으로 이상 판별', 'ECG, IoT 센서 등에 활용'],
        exampleIds: [53, 86],
      });
    }

    // ── CNN + RNN 하이브리드 ──
    if (m === 'ts_hybrid' || m === 'unknown') {
      r.push({
        model: 'CNN + LSTM 하이브리드',
        description: 'Conv1D로 단기 패턴, LSTM으로 장기 의존성을 동시 포착.',
        layers: ['CSVLoader', 'MinMaxScaler', 'TrainValSplit', 'Conv1D', 'LSTM', 'Dense', 'Trainer'],
        tips: ['Conv1D로 지역 특성 추출 후 LSTM 입력', '복잡한 시계열에 효과적'],
        exampleIds: [85],
      });
    }

    // ── Transformer ──
    if (m === 'ts_transformer' || m === 'unknown') {
      r.push({
        model: 'Transformer 시계열 예측',
        description: 'Self-Attention으로 시계열의 전역 의존성 포착.',
        layers: ['CSVLoader', 'MinMaxScaler', 'TrainValSplit', 'TransformerBlock', 'Dense', 'Trainer'],
        tips: ['Attention이 중요 시점에 집중', 'LSTM보다 긴 시퀀스에 효과적'],
        exampleIds: [88],
      });
    }

    // ── Seq2Seq ──
    if (m === 'ts_seq2seq' || m === 'unknown') {
      r.push({
        model: 'Seq2Seq 인코더-디코더',
        description: '시퀀스 → 시퀀스 변환. 멀티스텝 예측, 변환 등.',
        layers: ['CSVLoader', 'MinMaxScaler', 'TrainValSplit', 'LSTM(enc)', 'LSTM(dec)', 'Dense', 'Trainer'],
        tips: ['인코더가 입력 시퀀스를 요약', '디코더가 출력 시퀀스를 생성'],
        exampleIds: [91],
      });
    }
  }

  /* ═══════════════════════════════════════════
     공통 추천 (하이퍼파라미터 탐색, ONNX)
     ═══════════════════════════════════════════ */
  r.push({
    model: '하이퍼파라미터 탐색 (Grid Search)',
    description: '학습률, 유닛 수, Dropout 등 최적 조합을 자동 탐색.',
    layers: ['데이터 → StandardScaler → Dense → Trainer(GridSearch)'],
    tips: ['탐색 범위를 좁게 시작', '학습률, 은닉 유닛 수가 가장 영향 큼'],
    exampleIds: [93],
  });
  r.push({
    model: 'ONNX Export (모델 배포)',
    description: '학습된 모델을 ONNX 포맷으로 내보내어 C#/모바일에서 추론.',
    layers: ['학습된 모델 → ONNXExport → OnnxRuntime(C#)'],
    tips: ['PyTorch/TF 모두 ONNX 변환 가능', 'C# OnnxRuntime으로 크로스플랫폼 배포'],
    exampleIds: [99],
  });

  // 폴백
  if (r.length <= 2) {
    r.unshift({
      model: 'Dense Network (범용 시작점)',
      description: '어떤 데이터든 시작할 수 있는 기본 모델.',
      layers: ['데이터 로더', 'StandardScaler', 'Dense×2~3', 'Dropout', 'Trainer'],
      tips: ['간단한 모델부터 시작하여 점진적으로 복잡도 증가', '첫 실험은 빠르게 결과 확인'],
      exampleIds: [1, 26],
    });
  }

  return r;
}

/* ── 컴포넌트 ────────────────────────────────── */
export function ModelRecommendModal({ onClose, onLoadExample }: Props) {
  const examples = useStore((s) => s.examples);

  const [step, setStep] = useState(0);
  const [dataType, setDataType] = useState('');
  const [mlMethod, setMlMethod] = useState('');
  const [dataSize, setDataSize] = useState('');

  const mlMethods = getMethodsForDataType(dataType);
  const recommendations = step >= 3 ? getRecommendations(dataType, mlMethod, dataSize) : [];

  const handleBack = () => {
    if (step > 0) setStep(step - 1);
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="bg-gray-900 border border-gray-700 rounded-xl shadow-2xl w-[640px] max-h-[85vh] flex flex-col overflow-hidden">
        {/* 헤더 */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-gray-700 bg-gray-800">
          <div className="flex items-center gap-2">
            <span className="text-lg">🤖</span>
            <h2 className="text-sm font-bold text-white">모델 추천</h2>
            <div className="flex gap-1 ml-3">
              {[0, 1, 2, 3].map((s) => (
                <div
                  key={s}
                  className={`w-2 h-2 rounded-full ${s <= step ? 'bg-blue-500' : 'bg-gray-600'}`}
                />
              ))}
            </div>
          </div>
          <button onClick={onClose} className="text-gray-400 hover:text-white text-lg">✕</button>
        </div>

        {/* 내용 */}
        <div className="flex-1 overflow-y-auto p-5">
          {/* Step 0: 데이터 형식 */}
          {step === 0 && (
            <div>
              <h3 className="text-white font-semibold mb-1">1. 데이터 형식을 선택하세요</h3>
              <p className="text-gray-400 text-xs mb-4">학습에 사용할 데이터의 유형을 선택합니다.</p>
              <div className="grid grid-cols-2 gap-3">
                {DATA_TYPES.map((dt) => (
                  <button
                    key={dt.id}
                    onClick={() => { setDataType(dt.id); setMlMethod(''); setStep(1); }}
                    className={`p-3 rounded-lg border text-left transition-all ${
                      dataType === dt.id
                        ? 'border-blue-500 bg-blue-500/10 ring-1 ring-blue-500'
                        : 'border-gray-700 bg-gray-800 hover:border-gray-500'
                    }`}
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-xl">{dt.icon}</span>
                      <span className="text-white font-medium text-sm">{dt.label}</span>
                    </div>
                    <p className="text-gray-400 text-[10px]">{dt.desc}</p>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Step 1: ML 방법 */}
          {step === 1 && (
            <div>
              <h3 className="text-white font-semibold mb-1">2. ML 방법을 선택하세요</h3>
              <p className="text-gray-400 text-xs mb-4">
                어떤 종류의 문제를 풀고 싶은지 선택합니다.
                <span className="text-gray-500 ml-1">({DATA_TYPES.find(d => d.id === dataType)?.label} 기준)</span>
              </p>
              <div className="grid grid-cols-2 gap-2">
                {mlMethods.map((m) => (
                  <button
                    key={m.id}
                    onClick={() => { setMlMethod(m.id); setStep(2); }}
                    className={`p-2.5 rounded-lg border text-left transition-all ${
                      mlMethod === m.id
                        ? 'border-blue-500 bg-blue-500/10 ring-1 ring-blue-500'
                        : 'border-gray-700 bg-gray-800 hover:border-gray-500'
                    }`}
                  >
                    <div className="flex items-center gap-1.5">
                      <span className="text-sm">{m.icon}</span>
                      <span className="text-white font-medium text-xs">{m.label}</span>
                    </div>
                    <p className="text-gray-400 text-[10px] mt-0.5 ml-5">{m.desc}</p>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Step 2: 데이터 크기 */}
          {step === 2 && (
            <div>
              <h3 className="text-white font-semibold mb-1">3. 데이터 크기를 선택하세요</h3>
              <p className="text-gray-400 text-xs mb-4">데이터 샘플(행) 수에 따라 모델 복잡도를 조절합니다.</p>
              <div className="flex flex-col gap-2">
                {DATA_SIZES.map((ds) => (
                  <button
                    key={ds.id}
                    onClick={() => { setDataSize(ds.id); setStep(3); }}
                    className={`p-3 rounded-lg border text-left transition-all ${
                      dataSize === ds.id
                        ? 'border-blue-500 bg-blue-500/10 ring-1 ring-blue-500'
                        : 'border-gray-700 bg-gray-800 hover:border-gray-500'
                    }`}
                  >
                    <span className="text-white font-medium text-sm">{ds.label}</span>
                    {ds.desc && <span className="text-gray-400 text-[10px] ml-2">{ds.desc}</span>}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Step 3: 추천 결과 */}
          {step >= 3 && (
            <div>
              <div className="flex items-center gap-2 mb-4">
                <span className="text-lg">✨</span>
                <h3 className="text-white font-semibold">추천 결과</h3>
                <span className="text-xs text-gray-500 ml-auto">
                  {DATA_TYPES.find(d => d.id === dataType)?.label} · {mlMethods.find(m => m.id === mlMethod)?.label} · {DATA_SIZES.find(d => d.id === dataSize)?.label}
                </span>
              </div>

              <div className="space-y-4">
                {recommendations.map((rec, ri) => (
                  <div key={ri} className="border border-gray-700 rounded-lg overflow-hidden">
                    {/* 모델 헤더 */}
                    <div className="px-4 py-3 bg-gray-800 border-b border-gray-700">
                      <div className="flex items-center gap-2">
                        {ri === 0 && <span className="bg-yellow-500 text-black text-[9px] font-bold px-2 py-0.5 rounded">추천</span>}
                        <span className="text-white font-semibold text-sm">{rec.model}</span>
                      </div>
                      <p className="text-gray-400 text-xs mt-1">{rec.description}</p>
                    </div>

                    <div className="px-4 py-3 space-y-3">
                      {/* 파이프라인 구성 */}
                      <div>
                        <div className="text-[10px] text-gray-500 font-bold uppercase mb-1">Pipeline</div>
                        <div className="flex flex-wrap gap-1">
                          {rec.layers.map((l, li) => (
                            <React.Fragment key={li}>
                              {li > 0 && <span className="text-gray-600 text-xs">→</span>}
                              <span className="bg-gray-800 text-gray-300 text-[10px] px-2 py-0.5 rounded">{l}</span>
                            </React.Fragment>
                          ))}
                        </div>
                      </div>

                      {/* 팁 */}
                      <div>
                        <div className="text-[10px] text-gray-500 font-bold uppercase mb-1">Tips</div>
                        <ul className="space-y-0.5">
                          {rec.tips.map((tip, ti) => (
                            <li key={ti} className="text-gray-400 text-[11px] flex items-start gap-1">
                              <span className="text-blue-400 mt-0.5">•</span>
                              <span>{tip}</span>
                            </li>
                          ))}
                        </ul>
                      </div>

                      {/* 관련 예제 */}
                      <div>
                        <div className="text-[10px] text-gray-500 font-bold uppercase mb-1">관련 예제</div>
                        <div className="flex flex-wrap gap-1.5">
                          {rec.exampleIds.map((eid) => {
                            const ex = examples.find((e) => e.id === eid);
                            if (!ex) return null;
                            return (
                              <button
                                key={eid}
                                onClick={() => { onLoadExample(eid); onClose(); }}
                                className="bg-gray-800 hover:bg-blue-600/30 border border-gray-700 hover:border-blue-500 text-gray-300 text-[10px] px-2 py-1 rounded transition-colors"
                                title={ex.title}
                              >
                                #{eid} {ex.title_ko || ex.title}
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* 하단 버튼 */}
        <div className="flex items-center justify-between px-5 py-3 border-t border-gray-700 bg-gray-800">
          <button
            onClick={step === 0 ? onClose : handleBack}
            className="px-4 py-1.5 text-xs rounded bg-gray-700 hover:bg-gray-600 text-gray-300"
          >
            {step === 0 ? '닫기' : '← 이전'}
          </button>

          {step >= 3 && (
            <button
              onClick={() => { setStep(0); setDataType(''); setMlMethod(''); setDataSize(''); }}
              className="px-4 py-1.5 text-xs rounded bg-gray-700 hover:bg-gray-600 text-gray-300"
            >
              처음부터 다시
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
