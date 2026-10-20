import { useEffect, useRef, useState } from 'react'
import {
  ApiError,
  exportUrl,
  type Cardinality,
  type Comparison,
  type ExportKind,
  type JobStatus,
  type JoinMode,
} from '../api/client'
import { useHistogram, useMatches, useStats } from '../api/hooks'
import { Histogram } from '../components/Histogram'
import { ErrorNotice, Field, Segmented, Spinner } from '../components/ui'
import { CARDINALITIES } from '../options'
import { formatCount, formatPercent } from '../format'

const PAGE = 25

function useDebounced<T>(value: T, ms: number): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const t = setTimeout(() => setDebounced(value), ms)
    return () => clearTimeout(t)
  }, [value, ms])
  return debounced
}

function useWidth() {
  const ref = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(0)
  useEffect(() => {
    if (!ref.current) return
    const observer = new ResizeObserver(([entry]) => setWidth(Math.floor(entry.contentRect.width)))
    observer.observe(ref.current)
    return () => observer.disconnect()
  }, [])
  return [ref, width] as const
}

export function ResultsStep({
  job,
  comparisons,
  initialThreshold,
  initialCardinality,
}: {
  job: JobStatus
  comparisons: Comparison[]
  initialThreshold: number
  initialCardinality: Cardinality
}) {
  const summary = job.summary!
  const floor = Math.ceil(summary.keep_from)
  const [threshold, setThreshold] = useState(Math.max(floor, initialThreshold))
  const [cardinality, setCardinality] = useState(initialCardinality)
  const committed = useDebounced(threshold, 250)
  const stats = useStats(job.id, committed, cardinality)
  const histogram = useHistogram(job.id)
  const [chartRef, chartWidth] = useWidth()
  const s = stats.data
  const totalSeconds = Object.values(summary.seconds ?? {}).reduce((a, b) => a + b, 0)

  return (
    <div className="stack">
      <section className="card" aria-label="Result controls">
        <div className="card-body stack">
          <div className="results-controls">
            <Field label={`Threshold · ${threshold}`}>
              <input
                className="range"
                type="range"
                min={floor}
                max={100}
                step={1}
                aria-label="Threshold"
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
              />
            </Field>
            <Field label="Matches per row">
              <Segmented
                label="Matches per row"
                value={cardinality}
                onChange={setCardinality}
                options={CARDINALITIES.map((c) => ({ value: c.value, label: c.label }))}
              />
            </Field>
          </div>
          {floor > 0 && (
            <span className="subtle">
              Pairs below {floor} were discarded during the run; re-run with a lower threshold to
              go lower.
            </span>
          )}
        </div>
      </section>

      <div className="tiles" aria-busy={stats.isFetching}>
        <Tile label="Matches" value={s ? formatCount(s.matches) : '–'} sub={s?.mean_score != null ? `mean score ${s.mean_score}` : ' '} />
        <Tile
          label="Matched in A"
          value={s ? formatCount(s.matched_a) : '–'}
          sub={s ? `${formatPercent(s.matched_a / Math.max(1, summary.rows_a))} of ${formatCount(summary.rows_a)}` : ' '}
        />
        <Tile
          label="Matched in B"
          value={s ? formatCount(s.matched_b) : '–'}
          sub={s ? `${formatPercent(s.matched_b / Math.max(1, summary.rows_b))} of ${formatCount(summary.rows_b)}` : ' '}
        />
        <Tile label="Unmatched in A" value={s ? formatCount(s.unmatched_a) : '–'} sub=" " />
        <Tile label="Unmatched in B" value={s ? formatCount(s.unmatched_b) : '–'} sub=" " />
      </div>
      <ErrorNotice error={stats.error} />

      <section className="card" aria-label="Score distribution">
        <div className="card-head">
          <div className="titles">
            <h3>Best score per A row</h3>
            <span className="subtle num">
              {formatCount(summary.candidate_pairs)} pairs scored in {totalSeconds.toFixed(1)}s
            </span>
          </div>
        </div>
        <div className="card-body" ref={chartRef}>
          {histogram.data && chartWidth > 0 && (
            <Histogram bins={histogram.data} from={floor} threshold={threshold} width={chartWidth} />
          )}
          <ErrorNotice error={histogram.error} />
        </div>
      </section>

      <Downloads jobId={job.id} threshold={committed} cardinality={cardinality} />

      <MatchSample jobId={job.id} threshold={committed} cardinality={cardinality} comparisons={comparisons} />
    </div>
  )
}

function Tile({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div className="tile">
      <span className="label">{label}</span>
      <div className="value">{value}</div>
      <span className="sub">{sub}</span>
    </div>
  )
}

const JOINS: { value: JoinMode; label: string }[] = [
  { value: 'matched', label: 'Matched pairs only' },
  { value: 'left', label: 'Every A row' },
  { value: 'full', label: 'Every A and B row' },
]

function Downloads({
  jobId,
  threshold,
  cardinality,
}: {
  jobId: string
  threshold: number
  cardinality: Cardinality
}) {
  const [format, setFormat] = useState<'csv' | 'xlsx'>('csv')
  const [join, setJoin] = useState<JoinMode>('matched')
  const [pending, setPending] = useState<ExportKind | null>(null)
  const [error, setError] = useState<unknown>(null)

  const download = async (kind: ExportKind) => {
    setPending(kind)
    setError(null)
    try {
      const url = exportUrl(jobId, { kind, threshold, cardinality, format, join: kind === 'merged' ? join : undefined })
      const response = await fetch(url)
      if (!response.ok) {
        const body = await response.json().catch(() => null)
        throw new ApiError(response.status, body?.detail ?? 'Export failed')
      }
      const blob = await response.blob()
      const link = document.createElement('a')
      link.href = URL.createObjectURL(blob)
      link.download = `${kind}.${format}`
      link.click()
      setTimeout(() => URL.revokeObjectURL(link.href), 10_000)
    } catch (e) {
      setError(e)
    } finally {
      setPending(null)
    }
  }

  const button = (kind: ExportKind, label: string, primary = false) => (
    <button
      type="button"
      className={`btn ${primary ? 'btn-primary' : ''}`}
      disabled={pending !== null}
      onClick={() => download(kind)}
    >
      {pending === kind ? <Spinner label="Preparing…" /> : label}
    </button>
  )

  return (
    <section className="card" aria-label="Downloads">
      <div className="card-head">
        <div className="titles">
          <h3>Download</h3>
          <span className="subtle">Uses the threshold and matches-per-row setting above.</span>
        </div>
        <Segmented
          label="Format"
          value={format}
          onChange={setFormat}
          options={[
            { value: 'csv', label: 'CSV' },
            { value: 'xlsx', label: 'Excel' },
          ]}
        />
      </div>
      <div className="card-body stack">
        <div className="download-row">
          <Field label="Merged file includes">
            <select className="select" value={join} onChange={(e) => setJoin(e.target.value as JoinMode)}>
              {JOINS.map((j) => (
                <option key={j.value} value={j.value}>
                  {j.label}
                </option>
              ))}
            </select>
          </Field>
          {button('merged', 'Merged file', true)}
          <span className="divider" aria-hidden />
          {button('unmatched_a', 'Unmatched A rows')}
          {button('unmatched_b', 'Unmatched B rows')}
        </div>
        <ErrorNotice error={error} />
      </div>
    </section>
  )
}

function MatchSample({
  jobId,
  threshold,
  cardinality,
  comparisons,
}: {
  jobId: string
  threshold: number
  cardinality: Cardinality
  comparisons: Comparison[]
}) {
  const [order, setOrder] = useState<'asc' | 'desc'>('asc')
  // Changing what is listed starts again from the first page.
  const filters = `${threshold}|${cardinality}|${order}`
  const [paging, setPaging] = useState({ filters, offset: 0 })
  const offset = paging.filters === filters ? paging.offset : 0
  const setOffset = (value: number) => setPaging({ filters, offset: value })
  const page = useMatches(jobId, threshold, cardinality, order, offset, PAGE)
  const data = page.data

  return (
    <section className="card" aria-label="Matches">
      <div className="card-head">
        <div className="titles">
          <h3>Matches</h3>
          <span className="subtle">
            Check the lowest scores to see whether the threshold lets in false matches.
          </span>
        </div>
        <Segmented
          label="Order"
          value={order}
          onChange={setOrder}
          options={[
            { value: 'asc', label: 'Lowest first' },
            { value: 'desc', label: 'Highest first' },
          ]}
        />
      </div>
      <div className="card-body stack">
        <ErrorNotice error={page.error} />
        {data && data.total === 0 && <div className="empty">No matches at this threshold.</div>}
        {data && data.total > 0 && (
          <>
            <div className="table-wrap" style={{ maxHeight: 560 }}>
              <table className="matches">
                <thead>
                  <tr>
                    <th className="right">Score</th>
                    <th className="right">Rows</th>
                    {comparisons.map((c) => (
                      <th key={c.id} colSpan={2} className="group">
                        {c.column_a === c.column_b ? c.column_a : `${c.column_a} ↔ ${c.column_b}`}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.rows.map((r) => (
                    <tr key={`${r.row_a}-${r.row_b}`}>
                      <td className="right num score">{r.score.toFixed(1)}</td>
                      <td className="right num subtle">
                        {r.row_a + 1} · {r.row_b + 1}
                      </td>
                      {r.values.map((v) => (
                        <PairCells key={v.comparison_id} a={v.a} b={v.b} score={v.score} />
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="spread">
              <span className="subtle num">
                {formatCount(offset + 1)}–{formatCount(Math.min(offset + PAGE, data.total))} of{' '}
                {formatCount(data.total)}
              </span>
              <div className="row">
                {page.isFetching && <Spinner />}
                <button type="button" className="btn btn-sm" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>
                  Previous
                </button>
                <button
                  type="button"
                  className="btn btn-sm"
                  disabled={offset + PAGE >= data.total}
                  onClick={() => setOffset(offset + PAGE)}
                >
                  Next
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </section>
  )
}

function PairCells({ a, b, score }: { a: string | null; b: string | null; score: number | null }) {
  return (
    <>
      <td className="pair-values">
        <div className={a === null ? 'null' : undefined} title={a ?? undefined}>
          {a ?? 'empty'}
        </div>
        <div className={b === null ? 'null' : undefined} title={b ?? undefined}>
          {b ?? 'empty'}
        </div>
      </td>
      <td className="right num pair-score">{score === null ? '–' : Math.round(score)}</td>
    </>
  )
}
