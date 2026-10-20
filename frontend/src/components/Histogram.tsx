import { useState } from 'react'
import type { Schemas } from '../api/client'
import { formatCount } from '../format'

type Bin = Schemas['HistogramBin']

const HEIGHT = 180
const PAD = { top: 12, right: 16, bottom: 28, left: 52 }
const BIN = 2

function niceStep(max: number): number {
  const raw = max / 4
  const pow = 10 ** Math.floor(Math.log10(raw || 1))
  return [1, 2, 2.5, 5, 10].map((m) => m * pow).find((s) => s >= raw) ?? pow * 10
}

/** Columns of best-score-per-A-row counts; bins at or above the threshold use the accent. */
export function Histogram({
  bins,
  from,
  threshold,
  width,
}: {
  bins: Bin[]
  from: number
  threshold: number
  width: number
}) {
  const [hover, setHover] = useState<number | null>(null)
  const start = Math.floor(from / BIN) * BIN
  const slots = Math.max(1, Math.ceil((100 - start) / BIN))
  const counts = new Map(bins.map((b) => [b.start, b.count]))
  const max = Math.max(1, ...bins.map((b) => b.count))
  const step = niceStep(max)
  const top = Math.ceil(max / step) * step
  const plotW = Math.max(100, width - PAD.left - PAD.right)
  const plotH = HEIGHT - PAD.top - PAD.bottom
  const slot = plotW / slots
  const barW = Math.min(24, Math.max(2, slot - 2))
  const y = (v: number) => PAD.top + plotH - (v / top) * plotH
  const x = (score: number) => PAD.left + ((score - start) / (100 - start)) * plotW
  const ticks = Array.from({ length: Math.round(top / step) + 1 }, (_, i) => i * step)
  const xTicks = Array.from({ length: 11 }, (_, i) => i * 10).filter((t) => t >= start)
  const hovered = hover === null ? null : { start: start + hover * BIN, count: counts.get(start + hover * BIN) ?? 0 }

  return (
    <div className="histogram" style={{ position: 'relative' }}>
      <svg width={width} height={HEIGHT} role="img" aria-label="Distribution of best match scores">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={PAD.left} x2={PAD.left + plotW} y1={y(t)} y2={y(t)} className="grid" />
            <text x={PAD.left - 8} y={y(t)} dy="0.32em" textAnchor="end" className="axis">
              {formatCount(t)}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={t} x={x(t)} y={HEIGHT - 8} textAnchor="middle" className="axis">
            {t}
          </text>
        ))}
        {Array.from({ length: slots }, (_, i) => {
          const s = start + i * BIN
          const count = counts.get(s) ?? 0
          const h = (count / top) * plotH
          const bx = PAD.left + i * slot + (slot - barW) / 2
          const r = Math.min(4, barW / 2, h)
          const yTop = PAD.top + plotH - h
          // A bin split by an odd threshold counts as below; the stats tiles carry exact numbers.
          const above = s >= threshold
          return (
            <g
              key={s}
              onMouseEnter={() => setHover(i)}
              onMouseLeave={() => setHover(null)}
            >
              <rect x={PAD.left + i * slot} y={PAD.top} width={slot} height={plotH} fill="transparent" />
              {h > 0 && (
                <path
                  className={above ? 'bar bar-above' : 'bar bar-below'}
                  data-active={hover === i}
                  d={`M${bx},${PAD.top + plotH} V${yTop + r} Q${bx},${yTop} ${bx + r},${yTop} H${bx + barW - r} Q${bx + barW},${yTop} ${bx + barW},${yTop + r} V${PAD.top + plotH} Z`}
                />
              )}
            </g>
          )
        })}
        <line x1={PAD.left} x2={PAD.left + plotW} y1={PAD.top + plotH} y2={PAD.top + plotH} className="baseline" />
        <line
          x1={x(threshold)}
          x2={x(threshold)}
          y1={PAD.top - 4}
          y2={PAD.top + plotH}
          className="threshold-line"
        />
      </svg>
      {hovered && (
        <div
          className="tooltip"
          style={{
            left: Math.min(width - 150, Math.max(0, PAD.left + (hover! + 0.5) * slot - 75)),
            top: 0,
          }}
        >
          <strong className="num">{formatCount(hovered.count)}</strong> A rows
          <div className="subtle num">
            best score {hovered.start}–{hovered.start + BIN}
          </div>
        </div>
      )}
      <div className="legend">
        <span>
          <i className="swatch bar-above" /> At or above threshold
        </span>
        <span>
          <i className="swatch bar-below" /> Below threshold
        </span>
        <span>
          <i className="swatch-line" /> Threshold
        </span>
      </div>
    </div>
  )
}
