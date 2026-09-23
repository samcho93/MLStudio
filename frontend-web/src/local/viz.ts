// ── 브라우저에서 생성하는 시각화 (백엔드 Matplotlib 대체) ──

/** 혼동 행렬 히트맵을 캔버스로 그려 base64 PNG(접두사 제외)로 반환 */
export function drawConfusionMatrix(matrix: number[][], labels: string[]): string | null {
  const n = matrix.length;
  if (n === 0) return null;

  const cell = n <= 5 ? 64 : n <= 10 ? 44 : 30;
  const margin = { left: 70, top: 46, right: 20, bottom: 56 };
  const width = margin.left + cell * n + margin.right;
  const height = margin.top + cell * n + margin.bottom;

  const canvas = document.createElement('canvas');
  const scale = 2; // 선명하게
  canvas.width = width * scale;
  canvas.height = height * scale;
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  ctx.scale(scale, scale);

  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, width, height);

  const max = Math.max(...matrix.flat(), 1);

  ctx.font = '12px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const value = matrix[r][c];
      const t = value / max;
      // 파랑 계열 그라데이션
      const shade = Math.round(255 - t * 200);
      ctx.fillStyle = `rgb(${shade}, ${Math.round(shade * 0.95 + 10)}, 255)`;
      const x = margin.left + c * cell;
      const y = margin.top + r * cell;
      ctx.fillRect(x, y, cell, cell);
      ctx.strokeStyle = '#e5e7eb';
      ctx.strokeRect(x, y, cell, cell);
      ctx.fillStyle = t > 0.6 ? '#ffffff' : '#111827';
      ctx.fillText(String(value), x + cell / 2, y + cell / 2);
    }
  }

  ctx.fillStyle = '#111827';
  ctx.font = 'bold 14px sans-serif';
  ctx.fillText('Confusion Matrix', width / 2, 18);

  ctx.font = '11px sans-serif';
  for (let i = 0; i < n; i++) {
    ctx.textAlign = 'right';
    ctx.fillText(labels[i], margin.left - 8, margin.top + i * cell + cell / 2);
    ctx.textAlign = 'center';
    ctx.fillText(labels[i], margin.left + i * cell + cell / 2, margin.top + n * cell + 14);
  }

  ctx.font = '12px sans-serif';
  ctx.fillText('Predicted', margin.left + (cell * n) / 2, height - 12);
  ctx.save();
  ctx.translate(16, margin.top + (cell * n) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('True', 0, 0);
  ctx.restore();

  return canvas.toDataURL('image/png').split(',')[1] || null;
}
