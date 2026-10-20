import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useRef, useState, type DragEvent } from 'react'
import { api, unwrap, type DatasetMeta } from '../api/client'
import { uploadDataset, useDataset, useRows } from '../api/hooks'
import { DataTable } from '../components/DataTable'
import { ErrorNotice, Field, Spinner } from '../components/ui'
import type { Side } from '../draft'
import { formatCount, formatPercent } from '../format'

const ACCEPT = '.csv,.tsv,.txt,.xlsx,.xlsm'
const PAGE = 50

const DELIMITERS = [
  { value: ',', label: 'Comma (,)' },
  { value: ';', label: 'Semicolon (;)' },
  { value: '\t', label: 'Tab' },
  { value: '|', label: 'Pipe (|)' },
]

const ENCODINGS = ['utf-8', 'utf-16', 'cp1252', 'latin-1', 'iso8859-15', 'shift_jis', 'gb18030']

export function FilesStep({
  datasetA,
  datasetB,
  onChange,
}: {
  datasetA: string | null
  datasetB: string | null
  onChange: (side: Side, dataset: DatasetMeta | null) => void
}) {
  return (
    <div className="grid-2">
      <DatasetCard side="a" datasetId={datasetA} onChange={(d) => onChange('a', d)} />
      <DatasetCard side="b" datasetId={datasetB} onChange={(d) => onChange('b', d)} />
    </div>
  )
}

function DatasetCard({
  side,
  datasetId,
  onChange,
}: {
  side: Side
  datasetId: string | null
  onChange: (dataset: DatasetMeta | null) => void
}) {
  const title = side === 'a' ? 'File A' : 'File B'
  const dataset = useDataset(datasetId)
  const queryClient = useQueryClient()
  const [progress, setProgress] = useState<number | null>(null)

  const upload = useMutation({
    mutationFn: (file: File) => uploadDataset(file, setProgress),
    onSuccess: (meta) => {
      queryClient.setQueryData(['dataset', meta.id], meta)
      onChange(meta)
    },
    onSettled: () => setProgress(null),
  })

  const reparse = useMutation({
    mutationFn: (options: { delimiter?: string; encoding?: string; sheet?: string }) =>
      unwrap(
        api.POST('/api/datasets/{dataset_id}/parse', {
          params: { path: { dataset_id: datasetId! } },
          body: { ...dataset.data!.parse, ...options },
        }),
      ),
    onSuccess: (meta) => {
      queryClient.setQueryData(['dataset', meta.id], meta)
      queryClient.removeQueries({ queryKey: ['rows', meta.id] })
      queryClient.removeQueries({ queryKey: ['preview', meta.id] })
      onChange(meta)
    },
  })

  const busy = upload.isPending || reparse.isPending
  const meta = dataset.data

  return (
    <section className="card" aria-label={title}>
      <div className="card-head">
        <div className="titles">
          <h2>{title}</h2>
          {meta && (
            <span className="subtle" title={meta.filename}>
              {meta.filename}
            </span>
          )}
        </div>
        {meta && !busy && (
          <div className="row">
            <ReplaceButton onFile={(f) => upload.mutate(f)} />
            <button type="button" className="btn btn-sm btn-ghost" onClick={() => onChange(null)}>
              Remove
            </button>
          </div>
        )}
      </div>
      <div className="card-body stack">
        {upload.isPending ? (
          <UploadProgress progress={progress} />
        ) : !datasetId ? (
          <DropZone onFile={(f) => upload.mutate(f)} />
        ) : dataset.isPending ? (
          <Spinner label="Loading…" />
        ) : dataset.isError ? (
          <>
            <ErrorNotice error={dataset.error} />
            <DropZone onFile={(f) => upload.mutate(f)} />
          </>
        ) : (
          meta && (
            <DatasetDetails
              meta={meta}
              busy={reparse.isPending}
              onReparse={(o) => reparse.mutate(o)}
            />
          )
        )}
        <ErrorNotice error={upload.error ?? reparse.error} />
      </div>
    </section>
  )
}

function UploadProgress({ progress }: { progress: number | null }) {
  const reading = progress === null || progress >= 1
  return (
    <div className="stack" style={{ padding: '24px 0' }}>
      <div className="spread">
        <Spinner label={reading ? 'Reading and profiling the file…' : 'Uploading…'} />
        {!reading && <span className="num muted">{formatPercent(progress, 0)}</span>}
      </div>
      <div className="progress" aria-hidden>
        <div style={{ width: `${(progress ?? 1) * 100}%` }} />
      </div>
    </div>
  )
}

function DropZone({ onFile }: { onFile: (file: File) => void }) {
  const input = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)

  const drop = (e: DragEvent) => {
    e.preventDefault()
    setOver(false)
    const file = e.dataTransfer.files[0]
    if (file) onFile(file)
  }

  return (
    <div
      className="dropzone"
      data-over={over}
      onDragOver={(e) => {
        e.preventDefault()
        setOver(true)
      }}
      onDragLeave={() => setOver(false)}
      onDrop={drop}
    >
      <svg width="28" height="28" viewBox="0 0 24 24" aria-hidden>
        <path
          d="M12 16V4m0 0l-4 4m4-4l4 4M4 15v3a2 2 0 002 2h12a2 2 0 002-2v-3"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      <p>
        Drop a file here or{' '}
        <button type="button" className="link-btn" onClick={() => input.current?.click()}>
          browse
        </button>
      </p>
      <p className="subtle">CSV, TSV or Excel (.xlsx)</p>
      <input
        ref={input}
        type="file"
        accept={ACCEPT}
        hidden
        data-testid="file-input"
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) onFile(file)
          e.target.value = ''
        }}
      />
    </div>
  )
}

function ReplaceButton({ onFile }: { onFile: (file: File) => void }) {
  const input = useRef<HTMLInputElement>(null)
  return (
    <>
      <button type="button" className="btn btn-sm" onClick={() => input.current?.click()}>
        Replace
      </button>
      <input
        ref={input}
        type="file"
        accept={ACCEPT}
        hidden
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) onFile(file)
          e.target.value = ''
        }}
      />
    </>
  )
}

function DatasetDetails({
  meta,
  busy,
  onReparse,
}: {
  meta: DatasetMeta
  busy: boolean
  onReparse: (options: { delimiter?: string; encoding?: string; sheet?: string }) => void
}) {
  const [offset, setOffset] = useState(0)
  const rows = useRows(meta.id, offset, PAGE)
  const types = new Map(meta.columns.map((c) => [c.name, c.inferred_type]))
  const { parse } = meta
  const delimiters =
    parse.delimiter && !DELIMITERS.some((d) => d.value === parse.delimiter)
      ? [...DELIMITERS, { value: parse.delimiter, label: `"${parse.delimiter}"` }]
      : DELIMITERS
  const encodings =
    parse.encoding && !ENCODINGS.includes(parse.encoding)
      ? [parse.encoding, ...ENCODINGS]
      : ENCODINGS

  return (
    <>
      <div className="row">
        <span className="badge num">{formatCount(meta.rows)} rows</span>
        <span className="badge num">{meta.columns.length} columns</span>
        {busy && <Spinner label="Re-reading…" />}
      </div>
      <div className="fields">
        {parse.format === 'xlsx' ? (
          <Field label="Sheet">
            <select
              className="select"
              value={parse.sheet ?? ''}
              disabled={busy}
              onChange={(e) => onReparse({ sheet: e.target.value })}
            >
              {(parse.sheets ?? []).map((s) => (
                <option key={s}>{s}</option>
              ))}
            </select>
          </Field>
        ) : (
          <>
            <Field label="Delimiter">
              <select
                className="select"
                value={parse.delimiter ?? ','}
                disabled={busy}
                onChange={(e) => onReparse({ delimiter: e.target.value })}
              >
                {delimiters.map((d) => (
                  <option key={d.value} value={d.value}>
                    {d.label}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Encoding">
              <select
                className="select"
                value={parse.encoding ?? 'utf-8'}
                disabled={busy}
                onChange={(e) => onReparse({ encoding: e.target.value })}
              >
                {encodings.map((e) => (
                  <option key={e}>{e}</option>
                ))}
              </select>
            </Field>
          </>
        )}
      </div>
      {rows.data && (
        <>
          <DataTable
            columns={rows.data.columns}
            rows={rows.data.rows}
            header={(c) => (
              <span className="col-head">
                {c}
                <span className="type">{types.get(c)}</span>
              </span>
            )}
          />
          <div className="spread">
            <span className="subtle num">
              Rows {formatCount(offset + 1)}–{formatCount(Math.min(offset + PAGE, rows.data.total))}{' '}
              of {formatCount(rows.data.total)}
            </span>
            <div className="row">
              <button
                type="button"
                className="btn btn-sm"
                disabled={offset === 0}
                onClick={() => setOffset(Math.max(0, offset - PAGE))}
              >
                Previous
              </button>
              <button
                type="button"
                className="btn btn-sm"
                disabled={offset + PAGE >= rows.data.total}
                onClick={() => setOffset(offset + PAGE)}
              >
                Next
              </button>
            </div>
          </div>
        </>
      )}
      <ErrorNotice error={rows.error} />
    </>
  )
}
