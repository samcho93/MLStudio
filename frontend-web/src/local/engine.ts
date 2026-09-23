// ── 브라우저 파이프라인 실행 엔진 ───────────────────
// 백엔드 pipeline_executor.py + nodes/*.py 의 동작을 브라우저에서 재현한다.
// 학습은 TensorFlow.js 로 수행하며, 백엔드와 동일한 WebSocket 메시지 형식을 emit 한다.

import type { Matrix } from './data';
import {
  isNumericColumn,
  labelEncode,
  minMaxScale,
  oneHot,
  parseCsv,
  shuffledIndices,
  standardScale,
  takeRows,
  toNumericColumn,
} from './data';
import { drawConfusionMatrix } from './viz';

export interface PipelineNode {
  id: string;
  type: string;
  params: Record<string, any>;
}

export interface PipelineEdge {
  source: string;
  target: string;
  sourceHandle?: string;
  targetHandle?: string;
}

export type Emit = (msg: any) => void;

/** 브라우저에서 실행 가능한 노드 타입 */
export const SUPPORTED_NODE_TYPES = new Set([
  'CSVLoader',
  'TrainValSplit',
  'DataInspector',
  'StandardScaler',
  'MinMaxScaler',
  'LabelEncoder',
  'OneHotEncoder',
  'Dense',
  'Dropout',
  'BatchNorm',
  'Flatten',
  'Optimizer',
  'LossFunction',
  'EarlyStopping',
  'LRScheduler',
  'ModelCheckpoint',
  'Trainer',
  'KFoldTrainer',
  'ClassificationMetrics',
  'RegressionMetrics',
  'ConfusionMatrix',
  'ModelSummary',
  'LossCurve',
  'AccuracyCurve',
  'ModelSave',
]);

/** 학습 중단 플래그 (Stop 버튼) */
let abortRequested = false;
export function requestAbort() {
  abortRequested = true;
}

// ── 토폴로지 정렬 (Kahn) ────────────────────────────
function topologicalSort(nodes: PipelineNode[], edges: PipelineEdge[]): string[] {
  const inDegree = new Map<string, number>();
  const adj = new Map<string, string[]>();
  nodes.forEach((n) => inDegree.set(n.id, 0));
  for (const e of edges) {
    if (!inDegree.has(e.source) || !inDegree.has(e.target)) continue;
    adj.set(e.source, [...(adj.get(e.source) || []), e.target]);
    inDegree.set(e.target, (inDegree.get(e.target) || 0) + 1);
  }
  const queue = nodes.filter((n) => inDegree.get(n.id) === 0).map((n) => n.id);
  const order: string[] = [];
  while (queue.length) {
    const id = queue.shift()!;
    order.push(id);
    for (const next of adj.get(id) || []) {
      const deg = (inDegree.get(next) || 0) - 1;
      inDegree.set(next, deg);
      if (deg === 0) queue.push(next);
    }
  }
  if (order.length !== nodes.length) throw new Error('파이프라인에 순환 연결이 있습니다.');
  return order;
}

// ── 입력 수집 (백엔드의 포트 별칭 / labels 전파 규칙과 동일) ──
const PORT_ALIASES: Record<string, string[]> = {
  dataset: ['features', 'output'],
  data: ['features', 'output'],
};

function gatherInputs(
  nodeId: string,
  edges: PipelineEdge[],
  context: Map<string, Record<string, any>>
): Record<string, any> {
  const inputs: Record<string, any> = {};
  for (const edge of edges) {
    if (edge.target !== nodeId) continue;
    const sourceOutput = context.get(edge.source) || {};
    const sourcePort = edge.sourceHandle || 'output';
    const targetPort = edge.targetHandle || 'input';

    let hasPort = sourcePort in sourceOutput;
    let value = sourceOutput[sourcePort];
    if (!hasPort && PORT_ALIASES[sourcePort]) {
      for (const alias of PORT_ALIASES[sourcePort]) {
        if (alias in sourceOutput) {
          value = sourceOutput[alias];
          hasPort = true;
          break;
        }
      }
    }
    if (!hasPort && value === undefined) value = sourceOutput;
    inputs[targetPort] = value;

    if (targetPort === 'input' && 'labels' in sourceOutput && !('_source_labels' in inputs)) {
      inputs._source_labels = sourceOutput.labels;
    }
    if ('_source_labels' in sourceOutput && !('_source_labels' in inputs)) {
      inputs._source_labels = sourceOutput._source_labels;
    }
    if (['train_features', 'val_features', 'test_features'].includes(targetPort)) {
      const labelPort = targetPort.replace('features', 'labels');
      if (!(labelPort in inputs) && labelPort in sourceOutput && sourceOutput[labelPort] != null) {
        inputs[labelPort] = sourceOutput[labelPort];
      }
    }
  }
  return inputs;
}

// ── 값 헬퍼 ─────────────────────────────────────────
function asMatrix(value: any): Matrix | null {
  if (value == null) return null;
  if (Array.isArray(value)) {
    if (value.length === 0) return [];
    if (Array.isArray(value[0])) return value as Matrix;
    return (value as number[]).map((v) => [v]);
  }
  if (typeof value === 'object') {
    return asMatrix(value.features ?? value.output ?? null);
  }
  return null;
}

function asVector(value: any): number[] | null {
  if (value == null) return null;
  if (Array.isArray(value)) {
    if (value.length === 0) return [];
    if (Array.isArray(value[0])) {
      const rows = value as Matrix;
      // (N,1) → (N,), one-hot → argmax
      if (rows[0].length === 1) return rows.map((r) => r[0]);
      return rows.map((r) => r.indexOf(Math.max(...r)));
    }
    return value as number[];
  }
  if (typeof value === 'object') return asVector(value.labels ?? value.output ?? null);
  return null;
}

function pickInput(inputs: Record<string, any>, keys: string[]): any {
  for (const key of keys) {
    const v = inputs[key];
    if (v != null) return v;
  }
  return null;
}

function shapeOf(value: any): number[] {
  const m = asMatrix(value);
  if (!m) return [];
  return m.length ? [m.length, m[0].length] : [0];
}

// ── CSV 로드 ────────────────────────────────────────
const csvCache = new Map<string, string>();

async function loadCsvText(rawPath: string): Promise<string> {
  const fileName = rawPath.split(/[\\/]/).pop() || rawPath;
  if (csvCache.has(fileName)) return csvCache.get(fileName)!;
  const url = `${import.meta.env.BASE_URL}datasets/${fileName}`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(
      `데이터 파일을 찾을 수 없습니다: ${fileName}\n` +
        `브라우저 모드에서는 저장소에 포함된 CSV 데이터셋만 사용할 수 있습니다.`
    );
  }
  const text = await res.text();
  csvCache.set(fileName, text);
  return text;
}

// ── 노드 실행기 ─────────────────────────────────────
type NodeRunner = (
  params: Record<string, any>,
  inputs: Record<string, any>,
  emit: Emit
) => Promise<Record<string, any>>;

const runners: Record<string, NodeRunner> = {
  CSVLoader: async (params, _inputs, _emit) => {
    if (params.tokenize) {
      throw new Error('CSVLoader: 텍스트 토크나이징(NLP)은 브라우저 모드에서 지원되지 않습니다.');
    }
    const text = await loadCsvText(String(params.file_path || ''));
    const table = parseCsv(text, params.separator || ',');
    if (table.columns.length === 0) throw new Error('CSVLoader: 비어 있는 CSV 입니다.');

    const columnValues = table.columns.map((_, c) => table.rows.map((r) => r[c] ?? ''));

    // 타겟 컬럼 결정 (대소문자 무시, 없으면 마지막 컬럼)
    let targetCol: string = params.target_column || '';
    if (targetCol && !table.columns.includes(targetCol)) {
      const found = table.columns.find((c) => c.toLowerCase() === targetCol.toLowerCase());
      targetCol = found || '';
    }
    let targetIdx = targetCol ? table.columns.indexOf(targetCol) : -1;
    if (targetIdx < 0 && table.columns.length >= 2) targetIdx = table.columns.length - 1;

    const featureIdx = table.columns.map((_, i) => i).filter((i) => i !== targetIdx);
    const featureCols = featureIdx.map((i) =>
      isNumericColumn(columnValues[i])
        ? toNumericColumn(columnValues[i])
        : labelEncode(columnValues[i]).codes
    );
    const X: Matrix = table.rows.map((_, r) => featureCols.map((col) => col[r]));

    let y: number[] | null = null;
    let classNames: string[] | null = null;
    if (targetIdx >= 0) {
      const raw = columnValues[targetIdx];
      if (isNumericColumn(raw)) {
        y = toNumericColumn(raw);
      } else {
        const enc = labelEncode(raw);
        y = enc.codes;
        classNames = enc.classes;
      }
    }

    return {
      features: X,
      labels: y,
      dataframe: { columns: table.columns, rowCount: table.rows.length },
      _class_names: classNames,
    };
  },

  TrainValSplit: async (params, inputs) => {
    const X = asMatrix(pickInput(inputs, ['features', 'input', 'output', 'data']));
    if (!X) throw new Error('TrainValSplit: features 입력이 없습니다.');
    const yRaw = pickInput(inputs, ['labels']);
    const y = yRaw == null ? null : asVector(yRaw);

    if (y && y.length !== X.length) {
      throw new Error(
        `TrainValSplit: features(${X.length}개)와 labels(${y.length}개)의 샘플 수가 다릅니다.`
      );
    }

    const valRatio = Number(params.val_ratio ?? 0.2);
    const testRatio = Number(params.test_ratio ?? 0);
    const seed = Number(params.random_seed ?? 42);

    const idx = shuffledIndices(X.length, seed);
    const nTest = Math.round(X.length * testRatio);
    const testIdx = idx.slice(0, nTest);
    const rest = idx.slice(nTest);
    const nVal = Math.round(rest.length * (testRatio < 1 ? valRatio / (1 - testRatio) : valRatio));
    const valIdx = rest.slice(0, nVal);
    const trainIdx = rest.slice(nVal);

    const pick = (indices: number[]) => ({
      X: takeRows(X, indices),
      y: y ? takeRows(y, indices) : null,
    });
    const tr = pick(trainIdx);
    const va = pick(valIdx);
    const te = pick(testIdx);

    return {
      train_features: tr.X,
      train_labels: tr.y,
      val_features: va.X,
      val_labels: va.y,
      test_features: te.X,
      test_labels: te.y,
    };
  },

  DataInspector: async (_params, inputs, emit) => {
    const data = pickInput(inputs, ['input', 'features', 'output']);
    const shape = shapeOf(data);
    emit({ type: 'DATA_INSPECT', info: { shape, dtype: 'float32' } });
    return { output: data };
  },

  StandardScaler: async (_params, inputs) => {
    const X = asMatrix(pickInput(inputs, ['input', 'features', 'output', 'data']));
    if (!X) throw new Error('StandardScaler: input이 없습니다.');
    return { output: standardScale(X) };
  },

  MinMaxScaler: async (_params, inputs) => {
    const X = asMatrix(pickInput(inputs, ['input', 'features', 'output', 'data']));
    if (!X) throw new Error('MinMaxScaler: input이 없습니다.');
    return { output: minMaxScale(X) };
  },

  LabelEncoder: async (_params, inputs) => {
    const value = pickInput(inputs, ['input', 'features', 'output', 'labels']);
    const matrix = asMatrix(value);
    if (matrix && matrix.length && matrix[0].length > 1) return { output: matrix };
    const vec = asVector(value);
    if (!vec) throw new Error('LabelEncoder: input이 없습니다.');
    const enc = labelEncode(vec.map((v) => String(v)));
    return { output: enc.codes };
  },

  OneHotEncoder: async (_params, inputs) => {
    const vec = asVector(pickInput(inputs, ['input', 'features', 'output', 'labels']));
    if (!vec) throw new Error('OneHotEncoder: input이 없습니다.');
    return { output: oneHot(vec) };
  },

  Optimizer: async (params) => ({
    optimizer_config: {
      type: params.type || 'adam',
      learning_rate: Number(params.learning_rate ?? 0.001),
      momentum: Number(params.momentum ?? 0.9),
      weight_decay: Number(params.weight_decay ?? 0),
    },
  }),

  LossFunction: async (params) => ({
    loss_config: { type: params.type || 'sparse_categorical_crossentropy' },
  }),

  EarlyStopping: async (params) => ({
    callback_config: {
      type: 'EarlyStopping',
      patience: Number(params.patience ?? 5),
      monitor: params.monitor || 'val_loss',
      min_delta: Number(params.min_delta ?? 0.001),
    },
  }),

  LRScheduler: async (params) => ({
    scheduler_config: {
      type: params.type || 'ReduceOnPlateau',
      factor: Number(params.factor ?? 0.1),
      patience: Number(params.patience ?? 10),
      step_size: Number(params.step_size ?? 10),
    },
  }),

  ModelCheckpoint: async (params) => ({
    callback_config: {
      type: 'ModelCheckpoint',
      filepath: params.filepath || 'outputs/best_model.keras',
      monitor: params.monitor || 'val_loss',
    },
  }),

  ModelSave: async (params) => ({ model_path: params.filepath || '(브라우저 메모리)' }),

  LossCurve: async (_params, _inputs, emit) => {
    emit({ type: 'INFO', message: 'Loss 곡선은 상단 차트에 실시간으로 표시됩니다.' });
    return { image_b64: null };
  },

  AccuracyCurve: async (_params, _inputs, emit) => {
    emit({ type: 'INFO', message: 'Accuracy 곡선은 상단 차트에 실시간으로 표시됩니다.' });
    return { image_b64: null };
  },

  ModelSummary: async (_params, inputs, emit) => {
    const model = inputs.model;
    if (!model?.layers) return { summary: '' };
    const lines: string[] = [];
    let total = 0;
    for (const layer of model.layers) {
      const count = layer.countParams();
      total += count;
      lines.push(
        `${layer.name.padEnd(24)} ${JSON.stringify(layer.outputShape)}  params=${count}`
      );
    }
    lines.push(`Total params: ${total}`);
    const summary = lines.join('\n');
    emit({ type: 'MODEL_SUMMARY', summary });
    return { summary };
  },
};

// ── 레이어 노드 ─────────────────────────────────────
function layerRunner(build: (params: Record<string, any>) => Record<string, any>): NodeRunner {
  return async (params, inputs) => {
    const prev = Array.isArray(inputs.input) && inputs.input.every((i: any) => i?.type)
      ? inputs.input
      : [];
    const result: Record<string, any> = { output: [...prev, build(params)] };
    if (inputs._source_labels != null) result._source_labels = inputs._source_labels;
    return result;
  };
}

runners.Dense = layerRunner((p) => ({
  type: 'Dense',
  units: Number(p.units ?? 128),
  activation: p.activation || 'relu',
}));
runners.Dropout = layerRunner((p) => ({ type: 'Dropout', rate: Number(p.rate ?? 0.5) }));
runners.BatchNorm = layerRunner(() => ({ type: 'BatchNorm' }));
runners.Flatten = layerRunner(() => ({ type: 'Flatten' }));

// ── Trainer (TensorFlow.js) ─────────────────────────
const REGRESSION_LOSSES = ['mse', 'mae', 'mean_squared_error', 'mean_absolute_error'];

function mapActivation(act: string): string {
  const supported = ['relu', 'sigmoid', 'tanh', 'softmax', 'linear', 'elu', 'selu', 'softplus'];
  if (supported.includes(act)) return act;
  return 'relu'; // swish 등 미지원 활성화는 relu 로 대체
}

async function runTrainer(
  params: Record<string, any>,
  inputs: Record<string, any>,
  emit: Emit
): Promise<Record<string, any>> {
  const tf = await import('@tensorflow/tfjs');

  const epochs = Number(params.epochs ?? 10);
  const batchSize = Number(params.batch_size ?? 32);

  const rawLayers = Array.isArray(inputs.layers) ? inputs.layers : inputs.layers ? [inputs.layers] : [];
  const layers = rawLayers.filter((l: any) => l && typeof l === 'object' && 'type' in l);
  if (layers.length === 0) throw new Error('Trainer: 연결된 레이어가 없습니다.');

  const XtrainM = asMatrix(inputs.train_features);
  if (!XtrainM || XtrainM.length === 0) throw new Error('Trainer: 학습 데이터가 없습니다.');
  let ytrain = asVector(inputs.train_labels ?? inputs._source_labels);
  const XvalM = asMatrix(inputs.val_features);
  let yval = asVector(inputs.val_labels);

  const lossType = inputs.loss_config?.type || params.loss || 'sparse_categorical_crossentropy';
  const optCfg = inputs.optimizer_config || {
    type: params.optimizer || 'adam',
    learning_rate: Number(params.learning_rate ?? 0.001),
  };
  const isRegression = REGRESSION_LOSSES.includes(lossType);
  const isBinary = lossType === 'binary_crossentropy';

  if (!ytrain) throw new Error('Trainer: 라벨(labels)이 없습니다.');

  // 분류 라벨 0-based 보정 (백엔드와 동일)
  let numClasses = 0;
  if (!isRegression && !isBinary) {
    const min = Math.min(...ytrain);
    if (min !== 0) {
      ytrain = ytrain.map((v) => v - min);
      if (yval) yval = yval.map((v) => v - min);
    }
    numClasses = Math.max(...ytrain) + 1;
  }

  // 마지막 Dense 유닛 수를 클래스 수에 맞게 조정
  const built = layers.map((l: any) => ({ ...l }));
  if (numClasses > 0) {
    for (let i = built.length - 1; i >= 0; i--) {
      if (built[i].type === 'Dense') {
        built[i].units = numClasses;
        built[i].activation = 'softmax';
        break;
      }
    }
  } else if (isRegression) {
    for (let i = built.length - 1; i >= 0; i--) {
      if (built[i].type === 'Dense') {
        built[i].units = 1;
        built[i].activation = 'linear';
        break;
      }
    }
  } else if (isBinary) {
    for (let i = built.length - 1; i >= 0; i--) {
      if (built[i].type === 'Dense') {
        built[i].units = 1;
        built[i].activation = 'sigmoid';
        break;
      }
    }
  }

  // 모델 조립
  const model = tf.sequential();
  const inputDim = XtrainM[0].length;
  let first = true;
  for (const cfg of built) {
    const options: any = first ? { inputShape: [inputDim] } : {};
    first = false;
    switch (cfg.type) {
      case 'Dense':
        model.add(
          tf.layers.dense({ units: cfg.units, activation: mapActivation(cfg.activation) as any, ...options })
        );
        break;
      case 'Dropout':
        model.add(tf.layers.dropout({ rate: cfg.rate ?? 0.5, ...options }));
        break;
      case 'BatchNorm':
        model.add(tf.layers.batchNormalization(options));
        break;
      case 'Flatten':
        model.add(tf.layers.flatten(options));
        break;
      default:
        throw new Error(`Trainer: 브라우저 모드에서 지원하지 않는 레이어입니다 — ${cfg.type}`);
    }
  }

  // 옵티마이저 / 손실
  const lr = Number(optCfg.learning_rate ?? 0.001);
  let optimizer: any;
  switch (optCfg.type) {
    case 'sgd':
      optimizer = tf.train.momentum(lr, Number(optCfg.momentum ?? 0.9));
      break;
    case 'rmsprop':
      optimizer = tf.train.rmsprop(lr);
      break;
    default:
      optimizer = tf.train.adam(lr); // adamw 는 adam 으로 대체
  }
  const tfLoss = isRegression
    ? lossType.startsWith('ma')
      ? 'meanAbsoluteError'
      : 'meanSquaredError'
    : isBinary
    ? 'binaryCrossentropy'
    : 'sparseCategoricalCrossentropy';

  model.compile({
    optimizer,
    loss: tfLoss,
    metrics: isRegression ? ['mse'] : ['accuracy'],
  });

  emit({ type: 'DEVICE_INFO', device: tf.getBackend(), gpu_name: `브라우저 (${tf.getBackend()})` });

  // 텐서 변환
  const xs = tf.tensor2d(XtrainM);
  const ys = isRegression || isBinary ? tf.tensor2d(ytrain.map((v) => [v])) : tf.tensor1d(ytrain, 'float32');
  const hasVal = !!(XvalM && XvalM.length && yval && yval.length);
  const xv = hasVal ? tf.tensor2d(XvalM!) : null;
  const yv = hasVal
    ? isRegression || isBinary
      ? tf.tensor2d(yval!.map((v) => [v]))
      : tf.tensor1d(yval!, 'float32')
    : null;

  const history: Record<string, number[]> = { loss: [], val_loss: [], accuracy: [], val_accuracy: [] };
  const patience = Number(inputs.callback_config?.patience ?? 0);
  let best = Infinity;
  let wait = 0;
  const startedAt = Date.now();

  try {
    await model.fit(xs, ys, {
      epochs,
      batchSize,
      shuffle: true,
      validationData: hasVal ? ([xv, yv] as any) : undefined,
      callbacks: {
        onEpochEnd: async (epoch: number, logs: any) => {
          const loss = logs?.loss ?? 0;
          const valLoss = logs?.val_loss ?? 0;
          const acc = logs?.acc ?? logs?.accuracy ?? 0;
          const valAcc = logs?.val_acc ?? logs?.val_accuracy ?? 0;
          history.loss.push(loss);
          history.val_loss.push(valLoss);
          history.accuracy.push(acc);
          history.val_accuracy.push(valAcc);

          emit({
            type: 'EPOCH_UPDATE',
            epoch: epoch + 1,
            total_epochs: epochs,
            loss,
            val_loss: valLoss,
            accuracy: acc,
            val_accuracy: valAcc,
            lr,
            elapsed_sec: Number(((Date.now() - startedAt) / 1000).toFixed(2)),
          });

          // 조기 종료 / 중단 처리
          if (patience > 0 && hasVal) {
            if (valLoss < best - 1e-6) {
              best = valLoss;
              wait = 0;
            } else if (++wait >= patience) {
              (model as any).stopTraining = true;
              emit({ type: 'INFO', message: `Early stopping (patience=${patience})` });
            }
          }
          if (abortRequested) (model as any).stopTraining = true;

          await tf.nextFrame(); // UI 갱신 양보
        },
      },
    });
  } finally {
    xs.dispose();
    ys.dispose();
    xv?.dispose();
    yv?.dispose();
  }

  const last = (arr: number[]) => (arr.length ? arr[arr.length - 1] : 0);
  emit({
    type: 'TRAINING_COMPLETE',
    model_path: '(브라우저 메모리)',
    metrics: {
      loss: last(history.loss),
      accuracy: last(history.accuracy),
      val_loss: last(history.val_loss),
      val_accuracy: last(history.val_accuracy),
    },
  });

  return { model, history, model_path: '(브라우저 메모리)', _is_regression: isRegression };
}

runners.Trainer = runTrainer;
runners.KFoldTrainer = runTrainer;

// ── 평가 노드 ───────────────────────────────────────
async function predictClasses(model: any, X: Matrix): Promise<number[]> {
  const tf = await import('@tensorflow/tfjs');
  const input = tf.tensor2d(X);
  const out = model.predict(input) as any;
  const arr = (await out.array()) as number[][];
  input.dispose();
  out.dispose();
  if (arr[0].length === 1) return arr.map((r) => (r[0] >= 0.5 ? 1 : 0));
  return arr.map((r) => r.indexOf(Math.max(...r)));
}

async function predictValues(model: any, X: Matrix): Promise<number[]> {
  const tf = await import('@tensorflow/tfjs');
  const input = tf.tensor2d(X);
  const out = model.predict(input) as any;
  const arr = (await out.array()) as number[][];
  input.dispose();
  out.dispose();
  return arr.map((r) => r[0]);
}

function testData(inputs: Record<string, any>): { X: Matrix; y: number[] } | null {
  const X = asMatrix(pickInput(inputs, ['test_features', 'features', 'input']));
  const y = asVector(pickInput(inputs, ['test_labels', 'labels']));
  if (!X || !y || X.length === 0 || y.length === 0) return null;
  return { X, y };
}

runners.ClassificationMetrics = async (_params, inputs, emit) => {
  const data = testData(inputs);
  if (!inputs.model || !data) return { metrics: {} };
  const yPred = await predictClasses(inputs.model, data.X);
  const yTrue = data.y;
  const classes = Array.from(new Set([...yTrue, ...yPred])).sort((a, b) => a - b);

  let correct = 0;
  const stats = new Map(classes.map((c) => [c, { tp: 0, fp: 0, fn: 0, support: 0 }]));
  for (let i = 0; i < yTrue.length; i++) {
    const t = yTrue[i];
    const p = yPred[i];
    stats.get(t)!.support++;
    if (t === p) {
      correct++;
      stats.get(t)!.tp++;
    } else {
      stats.get(p)!.fp++;
      stats.get(t)!.fn++;
    }
  }
  let precision = 0;
  let recall = 0;
  let f1 = 0;
  for (const [, s] of stats) {
    const p = s.tp + s.fp ? s.tp / (s.tp + s.fp) : 0;
    const r = s.tp + s.fn ? s.tp / (s.tp + s.fn) : 0;
    const f = p + r ? (2 * p * r) / (p + r) : 0;
    const w = s.support / yTrue.length;
    precision += p * w;
    recall += r * w;
    f1 += f * w;
  }
  const metrics = {
    accuracy: correct / yTrue.length,
    precision,
    recall,
    f1,
  };
  emit({ type: 'METRICS', metrics });
  return { metrics };
};

runners.RegressionMetrics = async (_params, inputs, emit) => {
  const data = testData(inputs);
  if (!inputs.model || !data) return { metrics: {} };
  const yPred = await predictValues(inputs.model, data.X);
  const yTrue = data.y;
  const n = yTrue.length;
  const mae = yTrue.reduce((s, t, i) => s + Math.abs(t - yPred[i]), 0) / n;
  const mse = yTrue.reduce((s, t, i) => s + (t - yPred[i]) ** 2, 0) / n;
  const mean = yTrue.reduce((a, b) => a + b, 0) / n;
  const ssTot = yTrue.reduce((s, t) => s + (t - mean) ** 2, 0);
  const metrics = { mae, mse, rmse: Math.sqrt(mse), r2: ssTot ? 1 - mse * n / ssTot : 0 };
  emit({ type: 'METRICS', metrics });
  return { metrics };
};

runners.ConfusionMatrix = async (_params, inputs, emit) => {
  const data = testData(inputs);
  if (!inputs.model || !data) return { image_b64: null };
  const yPred = await predictClasses(inputs.model, data.X);
  const classes = Array.from(new Set([...data.y, ...yPred])).sort((a, b) => a - b);
  const index = new Map(classes.map((c, i) => [c, i]));
  const matrix = classes.map(() => classes.map(() => 0));
  for (let i = 0; i < data.y.length; i++) {
    matrix[index.get(data.y[i])!][index.get(yPred[i])!]++;
  }
  const imageB64 = drawConfusionMatrix(matrix, classes.map(String));
  emit({ type: 'VISUALIZATION', viz_type: 'confusion_matrix', image_b64: imageB64 });
  return { image_b64: imageB64 };
};

// ── 실행 ────────────────────────────────────────────
export function unsupportedNodeTypes(nodes: PipelineNode[]): string[] {
  return Array.from(new Set(nodes.map((n) => n.type).filter((t) => !SUPPORTED_NODE_TYPES.has(t))));
}

export async function executePipeline(
  nodes: PipelineNode[],
  edges: PipelineEdge[],
  emit: Emit
): Promise<void> {
  abortRequested = false;
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const context = new Map<string, Record<string, any>>();

  const order = topologicalSort(nodes, edges);
  emit({ type: 'PIPELINE_START', execution_order: order, total_nodes: order.length });

  for (let idx = 0; idx < order.length; idx++) {
    if (abortRequested) {
      emit({ type: 'INFO', message: '사용자가 학습을 중단했습니다.' });
      break;
    }
    const node = byId.get(order[idx])!;
    const runner = runners[node.type];
    if (!runner) {
      throw new Error(
        `브라우저 모드에서 지원하지 않는 노드입니다: ${node.type}\n` +
          `학습 서버(백엔드)를 연결하면 모든 노드를 사용할 수 있습니다.`
      );
    }

    emit({ type: 'NODE_START', node_id: node.id, node_type: node.type, index: idx });
    const inputs = gatherInputs(node.id, edges, context);

    let outputs: Record<string, any>;
    try {
      outputs = await runner(node.params || {}, inputs, emit);
    } catch (e: any) {
      throw new Error(`노드 '${node.id}' (${node.type}) 실행 중 오류: ${e?.message || e}`);
    }
    context.set(node.id, outputs);

    emit({
      type: 'NODE_COMPLETE',
      node_id: node.id,
      node_type: node.type,
      index: idx,
      output_summary: summarize(outputs),
    });
    await new Promise((r) => setTimeout(r, 0)); // UI 갱신 양보
  }

  emit({ type: 'PIPELINE_COMPLETE' });
}

// ── 출력 요약 (노드 카드에 표시) ─────────────────────
function summarize(outputs: Record<string, any>): Record<string, any> {
  const summary: Record<string, any> = {};
  for (const [port, value] of Object.entries(outputs)) {
    if (port.startsWith('_') || value == null) continue;
    if (Array.isArray(value)) {
      if (value.length && Array.isArray(value[0])) {
        summary[port] = { type: 'ndarray', shape: [value.length, value[0].length], dtype: 'float32' };
      } else if (value.length && typeof value[0] === 'object' && 'type' in value[0]) {
        summary[port] = {
          type: 'layers',
          count: value.length,
          layers: value.map((l: any) =>
            l.units ? `${l.type}(units=${l.units})` : l.type
          ),
        };
      } else {
        summary[port] = { type: 'ndarray', shape: [value.length], dtype: 'float32' };
      }
    } else if (typeof value === 'string') {
      summary[port] = { type: 'str', value: value.slice(0, 120) };
    } else if (typeof value === 'number' || typeof value === 'boolean') {
      summary[port] = { type: typeof value, value };
    } else if (typeof value === 'object') {
      summary[port] = { type: 'config', keys: Object.keys(value).slice(0, 20) };
    }
  }
  return summary;
}
