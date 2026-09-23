// ── 브라우저 실행용 데이터 유틸 ─────────────────────
// 백엔드(pandas / scikit-learn)가 하던 전처리를 브라우저에서 동일한 규칙으로 수행한다.

export type Matrix = number[][];

export interface CsvTable {
  columns: string[];
  rows: string[][];
}

/** 따옴표 / 줄바꿈을 처리하는 CSV 파서 */
export function parseCsv(text: string, separator = ','): CsvTable {
  const sep = separator || ',';
  const rows: string[][] = [];
  let row: string[] = [];
  let field = '';
  let inQuotes = false;

  const src = text.replace(/^﻿/, '');
  for (let i = 0; i < src.length; i++) {
    const ch = src[i];
    if (inQuotes) {
      if (ch === '"') {
        if (src[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        field += ch;
      }
      continue;
    }
    if (ch === '"') {
      inQuotes = true;
    } else if (ch === sep) {
      row.push(field);
      field = '';
    } else if (ch === '\r') {
      // 무시 (\r\n 처리)
    } else if (ch === '\n') {
      row.push(field);
      rows.push(row);
      row = [];
      field = '';
    } else {
      field += ch;
    }
  }
  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }

  const nonEmpty = rows.filter((r) => r.some((c) => c.trim() !== ''));
  if (nonEmpty.length === 0) return { columns: [], rows: [] };
  const columns = nonEmpty[0].map((c) => c.trim());
  return { columns, rows: nonEmpty.slice(1) };
}

/** 컬럼이 전부 숫자인지 판정 (빈 값은 무시) */
export function isNumericColumn(values: string[]): boolean {
  let seen = 0;
  for (const v of values) {
    const s = v.trim();
    if (s === '' || s.toLowerCase() === 'nan' || s.toLowerCase() === 'na') continue;
    if (!Number.isFinite(Number(s))) return false;
    seen++;
  }
  return seen > 0;
}

/** sklearn LabelEncoder 와 동일: 값을 정렬한 순서대로 0,1,2... 부여 */
export function labelEncode(values: string[]): { codes: number[]; classes: string[] } {
  const classes = Array.from(new Set(values.map((v) => v.trim()))).sort((a, b) => {
    const na = Number(a);
    const nb = Number(b);
    if (Number.isFinite(na) && Number.isFinite(nb)) return na - nb;
    return a < b ? -1 : a > b ? 1 : 0;
  });
  const index = new Map(classes.map((c, i) => [c, i]));
  return { codes: values.map((v) => index.get(v.trim()) ?? 0), classes };
}

/** 숫자 변환 (빈 값/NaN 은 컬럼 평균으로 대체) */
export function toNumericColumn(values: string[]): number[] {
  const nums = values.map((v) => {
    const s = v.trim();
    if (s === '') return NaN;
    const n = Number(s);
    return Number.isFinite(n) ? n : NaN;
  });
  const valid = nums.filter((n) => Number.isFinite(n));
  const mean = valid.length ? valid.reduce((a, b) => a + b, 0) / valid.length : 0;
  return nums.map((n) => (Number.isFinite(n) ? n : mean));
}

export function standardScale(X: Matrix): Matrix {
  if (X.length === 0) return X;
  const cols = X[0].length;
  const mean = new Array(cols).fill(0);
  const std = new Array(cols).fill(0);
  for (const row of X) for (let c = 0; c < cols; c++) mean[c] += row[c];
  for (let c = 0; c < cols; c++) mean[c] /= X.length;
  for (const row of X) for (let c = 0; c < cols; c++) std[c] += (row[c] - mean[c]) ** 2;
  for (let c = 0; c < cols; c++) {
    std[c] = Math.sqrt(std[c] / X.length);
    if (std[c] === 0) std[c] = 1; // sklearn 과 동일하게 상수 컬럼은 0 으로 유지
  }
  return X.map((row) => row.map((v, c) => (v - mean[c]) / std[c]));
}

export function minMaxScale(X: Matrix): Matrix {
  if (X.length === 0) return X;
  const cols = X[0].length;
  const min = new Array(cols).fill(Infinity);
  const max = new Array(cols).fill(-Infinity);
  for (const row of X) {
    for (let c = 0; c < cols; c++) {
      if (row[c] < min[c]) min[c] = row[c];
      if (row[c] > max[c]) max[c] = row[c];
    }
  }
  return X.map((row) =>
    row.map((v, c) => {
      const range = max[c] - min[c];
      return range === 0 ? 0 : (v - min[c]) / range;
    })
  );
}

export function oneHot(y: number[]): Matrix {
  const classes = Array.from(new Set(y)).sort((a, b) => a - b);
  const index = new Map(classes.map((c, i) => [c, i]));
  return y.map((v) => {
    const row = new Array(classes.length).fill(0);
    row[index.get(v) ?? 0] = 1;
    return row;
  });
}

/** 재현 가능한 난수 (mulberry32) */
function makeRng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** sklearn train_test_split 과 같은 역할 (셔플 후 비율 분할) */
export function shuffledIndices(n: number, seed: number): number[] {
  const rng = makeRng(seed);
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  return idx;
}

export function takeRows<T>(arr: T[], indices: number[]): T[] {
  return indices.map((i) => arr[i]);
}
