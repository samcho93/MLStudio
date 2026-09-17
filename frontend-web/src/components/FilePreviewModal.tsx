import React, { useEffect, useState } from 'react';
import { apiFetch, apiUrl } from '../api';

interface PreviewData {
  type: string;
  path: string;
  // CSV
  num_rows?: number;
  num_columns?: number;
  columns?: { name: string; dtype: string }[];
  sample?: Record<string, any>[];
  // Folder
  num_classes?: number;
  total_images?: number;
  root_images?: number;
  root_samples?: string[];
  classes?: { name: string; count: number; samples?: string[] }[];
  // Numpy
  shape?: number[];
  dtype?: string;
  size_mb?: number;
  arrays?: Record<string, { shape: number[]; dtype: string }>;
  // File
  filename?: string;
  extension?: string;
  // Error
  error?: string;
}

interface Props {
  filePath: string;
  onClose: () => void;
}

export function FilePreviewModal({ filePath, onClose }: Props) {
  const [data, setData] = useState<PreviewData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchPreview = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await apiFetch(
          `/api/preview?path=${encodeURIComponent(filePath)}`
        );
        const json = await res.json();
        if (!res.ok) {
          setError(json.error || `HTTP ${res.status}`);
        } else {
          setData(json);
        }
      } catch (e: any) {
        setError(e.message || '서버 연결 실패');
      } finally {
        setLoading(false);
      }
    };
    fetchPreview();
  }, [filePath]);

  // 배경 클릭 시 닫기
  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) onClose();
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={handleBackdropClick}
    >
      <div className="bg-gray-800 border border-gray-600 rounded-xl shadow-2xl w-[90vw] max-w-[900px] max-h-[85vh] flex flex-col overflow-hidden">
        {/* 헤더 */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
          <div className="flex items-center gap-2">
            <span className="text-base">🔍</span>
            <h3 className="text-sm font-bold text-white">파일 미리보기</h3>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white text-lg leading-none px-1"
          >
            ✕
          </button>
        </div>

        {/* 경로 표시 */}
        <div className="px-4 py-2 bg-gray-750 border-b border-gray-700">
          <code className="text-[11px] text-gray-400 break-all">{filePath}</code>
        </div>

        {/* 내용 */}
        <div className="flex-1 overflow-y-auto p-4">
          {loading && (
            <div className="flex items-center justify-center py-8">
              <div className="animate-spin w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full" />
              <span className="ml-3 text-sm text-gray-400">불러오는 중...</span>
            </div>
          )}

          {error && (
            <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
              <p className="text-sm text-red-400">⚠ {error}</p>
            </div>
          )}

          {data && !error && <PreviewContent data={data} />}
        </div>
      </div>
    </div>
  );
}

// ── 미리보기 내용 렌더링 ────────────────────────────
function PreviewContent({ data }: { data: PreviewData }) {
  if (data.type === 'csv') return <CSVPreview data={data} />;
  if (data.type === 'folder') return <FolderPreview data={data} />;
  if (data.type === 'numpy') return <NumpyPreview data={data} />;
  if (data.type === 'numpy_archive') return <NumpyArchivePreview data={data} />;
  return <GenericFilePreview data={data} />;
}

// ── CSV 미리보기 ────────────────────────────────────
function CSVPreview({ data }: { data: PreviewData }) {
  const columns = data.columns || [];
  const sample = data.sample || [];
  const [showFullData, setShowFullData] = useState(false);

  if (showFullData) {
    return <CSVFullDataViewer filePath={data.path} onBack={() => setShowFullData(false)} />;
  }

  return (
    <div className="space-y-3">
      {/* 요약 정보 */}
      <div className="grid grid-cols-3 gap-2">
        <StatCard label="행 수" value={data.num_rows?.toLocaleString() || '0'} />
        <StatCard label="열 수" value={String(data.num_columns || 0)} />
        <StatCard label="파일 형식" value="CSV" />
      </div>

      {/* 컬럼 정보 */}
      <div>
        <h4 className="text-[11px] font-bold text-gray-500 uppercase mb-1.5">
          컬럼 ({columns.length}개)
        </h4>
        <div className="flex flex-wrap gap-1">
          {columns.map((col) => (
            <span
              key={col.name}
              className="text-[10px] px-2 py-0.5 rounded-full bg-blue-500/15 text-blue-300 border border-blue-500/20"
              title={`dtype: ${col.dtype}`}
            >
              {col.name}
              <span className="ml-1 text-blue-500/60">{col.dtype}</span>
            </span>
          ))}
        </div>
      </div>

      {/* 샘플 데이터 테이블 */}
      {sample.length > 0 && (
        <div>
          <h4 className="text-[11px] font-bold text-gray-500 uppercase mb-1.5">
            샘플 데이터 (상위 {sample.length}행)
          </h4>
          <div className="overflow-x-auto rounded-lg border border-gray-700">
            <table className="w-full text-[10px]">
              <thead>
                <tr className="bg-gray-700/50">
                  <th className="px-2 py-1.5 text-left text-gray-400 font-medium">#</th>
                  {columns.map((col) => (
                    <th
                      key={col.name}
                      className="px-2 py-1.5 text-left text-gray-400 font-medium whitespace-nowrap"
                    >
                      {col.name}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sample.map((row, i) => (
                  <tr
                    key={i}
                    className="border-t border-gray-700/50 hover:bg-gray-700/30"
                  >
                    <td className="px-2 py-1 text-gray-500">{i + 1}</td>
                    {columns.map((col) => (
                      <td
                        key={col.name}
                        className="px-2 py-1 text-gray-300 whitespace-nowrap max-w-[120px] truncate"
                        title={String(row[col.name] ?? '')}
                      >
                        {row[col.name] === null || row[col.name] === undefined
                          ? <span className="text-gray-600 italic">null</span>
                          : String(row[col.name])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* 전체 데이터 보기 버튼 */}
      <button
        type="button"
        onClick={() => setShowFullData(true)}
        className="w-full py-2 rounded-lg bg-blue-500/15 border border-blue-500/30 text-blue-300 text-xs font-medium hover:bg-blue-500/25 hover:border-blue-500/50 transition-colors flex items-center justify-center gap-1.5"
      >
        <span>📋</span>
        전체 데이터 보기 ({data.num_rows?.toLocaleString() || 0}행)
      </button>
    </div>
  );
}

// ── CSV 전체 데이터 뷰어 ──────────────────────────────
interface FullDataResponse {
  columns: { name: string; dtype: string }[];
  rows: Record<string, any>[];
  page: number;
  page_size: number;
  total_rows: number;
  total_pages: number;
  start_row: number;
  end_row: number;
}

function CSVFullDataViewer({ filePath, onBack }: { filePath: string; onBack: () => void }) {
  const [data, setData] = useState<FullDataResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const pageSize = 100;

  useEffect(() => {
    const fetchPage = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await apiFetch(
          `/api/preview/csv-full?path=${encodeURIComponent(filePath)}&page=${page}&page_size=${pageSize}`
        );
        const json = await res.json();
        if (!res.ok) {
          setError(json.error || `HTTP ${res.status}`);
        } else {
          setData(json);
        }
      } catch (e: any) {
        setError(e.message || '서버 연결 실패');
      } finally {
        setLoading(false);
      }
    };
    fetchPage();
  }, [filePath, page]);

  return (
    <div className="space-y-3">
      {/* 헤더: 뒤로가기 + 정보 */}
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={onBack}
          className="shrink-0 w-6 h-6 flex items-center justify-center rounded bg-gray-700 border border-gray-600 text-gray-400 hover:text-white hover:border-gray-500 transition-colors text-xs"
          title="미리보기로 돌아가기"
        >
          ←
        </button>
        <h4 className="text-xs font-bold text-white">전체 데이터</h4>
        {data && (
          <span className="text-[10px] text-gray-500 ml-auto">
            {data.total_rows.toLocaleString()}행 x {data.columns.length}열
          </span>
        )}
      </div>

      {loading && (
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full" />
          <span className="ml-2 text-xs text-gray-400">불러오는 중...</span>
        </div>
      )}

      {error && (
        <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
          <p className="text-sm text-red-400">{error}</p>
        </div>
      )}

      {data && !error && (
        <>
          {/* 테이블 */}
          <div className="overflow-x-auto rounded-lg border border-gray-700">
            <table className="w-full text-[10px]">
              <thead className="sticky top-0 z-10">
                <tr className="bg-gray-700">
                  <th className="px-2 py-1.5 text-left text-gray-400 font-medium">#</th>
                  {data.columns.map((col) => (
                    <th
                      key={col.name}
                      className="px-2 py-1.5 text-left text-gray-400 font-medium whitespace-nowrap"
                    >
                      <div>{col.name}</div>
                      <div className="text-[8px] text-gray-500 font-normal">{col.dtype}</div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.rows.map((row, i) => (
                  <tr
                    key={i}
                    className="border-t border-gray-700/50 hover:bg-gray-700/30"
                  >
                    <td className="px-2 py-1 text-gray-500">
                      {data.start_row + i}
                    </td>
                    {data.columns.map((col) => (
                      <td
                        key={col.name}
                        className="px-2 py-1 text-gray-300 whitespace-nowrap max-w-[150px] truncate"
                        title={String(row[col.name] ?? '')}
                      >
                        {row[col.name] === null || row[col.name] === undefined
                          ? <span className="text-gray-600 italic">null</span>
                          : String(row[col.name])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* 페이지네이션 */}
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-gray-500">
              {data.start_row.toLocaleString()} - {data.end_row.toLocaleString()} / {data.total_rows.toLocaleString()}행
            </span>
            <div className="flex items-center gap-1">
              <button
                type="button"
                disabled={page <= 1}
                onClick={() => setPage(1)}
                className="px-1.5 py-0.5 rounded text-[10px] bg-gray-700 border border-gray-600 text-gray-400 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed"
              >
                ««
              </button>
              <button
                type="button"
                disabled={page <= 1}
                onClick={() => setPage((p) => p - 1)}
                className="px-1.5 py-0.5 rounded text-[10px] bg-gray-700 border border-gray-600 text-gray-400 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed"
              >
                «
              </button>
              <span className="text-[10px] text-gray-300 px-2">
                {page} / {data.total_pages}
              </span>
              <button
                type="button"
                disabled={page >= data.total_pages}
                onClick={() => setPage((p) => p + 1)}
                className="px-1.5 py-0.5 rounded text-[10px] bg-gray-700 border border-gray-600 text-gray-400 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed"
              >
                »
              </button>
              <button
                type="button"
                disabled={page >= data.total_pages}
                onClick={() => setPage(data.total_pages)}
                className="px-1.5 py-0.5 rounded text-[10px] bg-gray-700 border border-gray-600 text-gray-400 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed"
              >
                »»
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// ── 이미지 폴더 미리보기 ────────────────────────────
function FolderPreview({ data }: { data: PreviewData }) {
  const classes = data.classes || [];
  const [expandedClass, setExpandedClass] = useState<string | null>(null);

  return (
    <div className="space-y-3">
      {/* 요약 */}
      <div className="grid grid-cols-3 gap-2">
        <StatCard label="클래스 수" value={String(data.num_classes || 0)} />
        <StatCard label="총 이미지" value={data.total_images?.toLocaleString() || '0'} />
        <StatCard label="유형" value="이미지 폴더" />
      </div>

      {data.root_images && data.root_images > 0 && (
        <div>
          <p className="text-[11px] text-yellow-400 mb-2">
            ⚠ 루트 폴더에 {data.root_images}개의 이미지가 직접 있습니다 (서브폴더 분류 권장)
          </p>
          {data.root_samples && data.root_samples.length > 0 && (
            <ImageThumbnailGrid images={data.root_samples} label="루트 이미지" />
          )}
        </div>
      )}

      {/* 클래스별 목록 + 이미지 썸네일 */}
      {classes.length > 0 && (
        <div>
          <h4 className="text-[11px] font-bold text-gray-500 uppercase mb-1.5">
            클래스별 이미지
          </h4>
          <div className="space-y-1.5">
            {classes.map((cls) => {
              const maxCount = Math.max(...classes.map((c) => c.count), 1);
              const pct = (cls.count / maxCount) * 100;
              const isExpanded = expandedClass === cls.name;
              const hasSamples = cls.samples && cls.samples.length > 0;

              return (
                <div key={cls.name}>
                  {/* 클래스 바 (클릭하면 이미지 펼침) */}
                  <button
                    type="button"
                    onClick={() => hasSamples && setExpandedClass(isExpanded ? null : cls.name)}
                    className={`w-full flex items-center gap-2 group ${
                      hasSamples ? 'cursor-pointer' : 'cursor-default'
                    }`}
                  >
                    {/* 펼침 화살표 */}
                    <span className={`text-[9px] text-gray-500 w-3 shrink-0 transition-transform ${
                      isExpanded ? 'rotate-90' : ''
                    } ${hasSamples ? '' : 'invisible'}`}>
                      ▶
                    </span>
                    <span
                      className="text-[10px] text-gray-300 w-24 truncate shrink-0 text-left group-hover:text-white transition-colors"
                      title={cls.name}
                    >
                      {cls.name}
                    </span>
                    <div className="flex-1 bg-gray-700/50 rounded-full h-3 overflow-hidden">
                      <div
                        className="h-full bg-emerald-500/60 rounded-full transition-all"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                    <span className="text-[10px] text-gray-400 w-10 text-right shrink-0">
                      {cls.count}
                    </span>
                  </button>

                  {/* 펼쳐진 이미지 썸네일 그리드 */}
                  {isExpanded && hasSamples && (
                    <div className="ml-5 mt-1.5 mb-1">
                      <ImageThumbnailGrid
                        images={cls.samples!}
                        label={`${cls.name} 샘플`}
                      />
                      {cls.count > cls.samples!.length && (
                        <p className="text-[9px] text-gray-600 mt-1">
                          ... 외 {cls.count - cls.samples!.length}개
                        </p>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* 불균형 경고 */}
          {classes.length >= 2 && (() => {
            const counts = classes.map((c) => c.count);
            const maxC = Math.max(...counts);
            const minC = Math.min(...counts);
            if (maxC > minC * 3) {
              return (
                <p className="text-[10px] text-yellow-400 mt-2">
                  ⚠ 클래스 불균형이 감지되었습니다 (최대 {maxC} vs 최소 {minC}).
                  클래스 가중치 또는 오버샘플링을 고려하세요.
                </p>
              );
            }
            return null;
          })()}
        </div>
      )}

      {classes.length === 0 && !data.root_images && (
        <p className="text-[11px] text-gray-500">
          서브폴더가 없습니다. ImageFolder 노드는 서브폴더명을 클래스 레이블로 사용합니다.
        </p>
      )}
    </div>
  );
}

// ── 이미지 썸네일 그리드 ────────────────────────────
function ImageThumbnailGrid({ images, label }: { images: string[]; label: string }) {
  const [failedSet, setFailedSet] = useState<Set<number>>(new Set());

  return (
    <div>
      <div className="grid grid-cols-3 gap-1.5">
        {images.map((imgPath, idx) => (
          <div
            key={idx}
            className="relative aspect-square rounded-lg overflow-hidden bg-gray-700/50 border border-gray-600/50 group"
          >
            {failedSet.has(idx) ? (
              <div className="w-full h-full flex items-center justify-center">
                <span className="text-[9px] text-gray-500">로드 실패</span>
              </div>
            ) : (
              <img
                src={apiUrl(`/api/preview/image?path=${encodeURIComponent(imgPath)}`)}
                alt={`${label} ${idx + 1}`}
                loading="lazy"
                className="w-full h-full object-cover"
                onError={() => setFailedSet((prev) => new Set(prev).add(idx))}
              />
            )}
            {/* 파일명 오버레이 */}
            <div className="absolute bottom-0 left-0 right-0 bg-black/60 px-1 py-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
              <p className="text-[8px] text-gray-300 truncate">
                {imgPath.split(/[/\\]/).pop()}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Numpy 미리보기 ──────────────────────────────────
function NumpyPreview({ data }: { data: PreviewData }) {
  return (
    <div className="grid grid-cols-3 gap-2">
      <StatCard label="Shape" value={`(${data.shape?.join(', ')})`} />
      <StatCard label="DType" value={data.dtype || '-'} />
      <StatCard label="크기" value={`${data.size_mb} MB`} />
    </div>
  );
}

function NumpyArchivePreview({ data }: { data: PreviewData }) {
  const arrays = data.arrays || {};
  return (
    <div className="space-y-3">
      <StatCard label="파일 형식" value="NumPy Archive (.npz)" />
      <div>
        <h4 className="text-[11px] font-bold text-gray-500 uppercase mb-1.5">
          포함된 배열 ({Object.keys(arrays).length}개)
        </h4>
        <div className="space-y-1">
          {Object.entries(arrays).map(([name, info]) => (
            <div key={name} className="flex items-center justify-between bg-gray-700/30 px-3 py-1.5 rounded">
              <span className="text-[11px] text-blue-300 font-mono">{name}</span>
              <span className="text-[10px] text-gray-400">
                ({info.shape.join(', ')}) {info.dtype}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── 기타 파일 미리보기 ──────────────────────────────
function GenericFilePreview({ data }: { data: PreviewData }) {
  return (
    <div className="grid grid-cols-3 gap-2">
      <StatCard label="파일명" value={data.filename || '-'} />
      <StatCard label="확장자" value={data.extension || '-'} />
      <StatCard label="크기" value={`${data.size_mb} MB`} />
    </div>
  );
}

// ── 통계 카드 ───────────────────────────────────────
function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-gray-700/40 rounded-lg px-3 py-2 text-center">
      <div className="text-[10px] text-gray-500 mb-0.5">{label}</div>
      <div className="text-xs text-white font-semibold truncate" title={value}>
        {value}
      </div>
    </div>
  );
}
