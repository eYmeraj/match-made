import { useMutation } from '@tanstack/react-query'
import { useState, type Dispatch } from 'react'
import { api, unwrap, type ColumnType, type Comparison, type DatasetMeta, type Meta } from '../api/client'
import { CleaningOptions } from '../components/CleaningOptions'
import { CleaningPreview } from '../components/CleaningPreview'
import { ErrorNotice, Field, Notice, RemoveButton, Spinner } from '../components/ui'
import { changeType, comparisonId, makeComparison, type Draft, type DraftAction } from '../draft'

const TYPES: { value: ColumnType; label: string }[] = [
  { value: 'text', label: 'Text' },
  { value: 'number', label: 'Number' },
  { value: 'date', label: 'Date' },
  { value: 'categorical', label: 'Category' },
  { value: 'boolean', label: 'Yes / no' },
]

const METRICS: Record<string, string> = {
  token_sort: 'Edit distance, word order ignored',
  ratio: 'Edit distance',
  token_set: 'Shared words, extra words ignored',
  partial: 'Best matching substring',
  wratio: 'Weighted mix of edit distance measures',
  jaro_winkler: 'Jaro-Winkler, for short strings',
}

export function ComparisonsStep({
  draft,
  dispatch,
  meta,
  datasetA,
  datasetB,
}: {
  draft: Draft
  dispatch: Dispatch<DraftAction>
  meta: Meta
  datasetA: DatasetMeta
  datasetB: DatasetMeta
}) {
  const [expanded, setExpanded] = useState<string | null>(null)
  const [confirmSuggest, setConfirmSuggest] = useState(false)

  const suggest = useMutation({
    mutationFn: () =>
      unwrap(api.POST('/api/suggest', { body: { dataset_a: datasetA.id, dataset_b: datasetB.id } })),
    onSuccess: (s) => {
      dispatch({ type: 'replaceSetup', comparisons: s.comparisons, rules: s.blocking.rules ?? [] })
      setExpanded(null)
      setConfirmSuggest(false)
    },
  })

  const runSuggest = () => (draft.comparisons.length > 0 ? setConfirmSuggest(true) : suggest.mutate())

  return (
    <div className="stack">
      <div className="spread">
        <p className="muted" style={{ maxWidth: 680 }}>
          Pick the column pairs that identify the same entity. Each comparison scores 0–100
          after cleaning, and the match score is their weighted average.
        </p>
        <button type="button" className="btn" disabled={suggest.isPending} onClick={runSuggest}>
          {suggest.isPending ? <Spinner label="Suggesting…" /> : 'Auto-suggest'}
        </button>
      </div>
      {confirmSuggest && (
        <Notice tone="warning">
          <div className="spread">
            <span>Replace your comparisons and blocking rules with suggestions?</span>
            <div className="row">
              <button type="button" className="btn btn-sm" onClick={() => setConfirmSuggest(false)}>
                Cancel
              </button>
              <button
                type="button"
                className="btn btn-sm btn-primary"
                disabled={suggest.isPending}
                onClick={() => suggest.mutate()}
              >
                Replace
              </button>
            </div>
          </div>
        </Notice>
      )}
      <ErrorNotice error={suggest.error} />

      {draft.comparisons.map((c) => (
        <ComparisonCard
          key={c.id}
          comparison={c}
          meta={meta}
          datasetA={datasetA}
          datasetB={datasetB}
          expanded={expanded === c.id}
          onToggle={() => setExpanded(expanded === c.id ? null : c.id)}
          onChange={(next) => dispatch({ type: 'updateComparison', id: c.id, comparison: next })}
          onRemove={() => dispatch({ type: 'removeComparison', id: c.id })}
        />
      ))}

      <AddComparison
        meta={meta}
        datasetA={datasetA}
        datasetB={datasetB}
        taken={draft.comparisons.map((c) => c.id)}
        first={draft.comparisons.length === 0}
        onAdd={(c) => {
          dispatch({ type: 'addComparison', comparison: c })
          setExpanded(c.id)
        }}
      />
    </div>
  )
}

function columnOptions(dataset: DatasetMeta) {
  return dataset.columns.map((c) => (
    <option key={c.name} value={c.name}>
      {c.name}
    </option>
  ))
}

function inferredType(dataset: DatasetMeta, column: string): ColumnType {
  return dataset.columns.find((c) => c.name === column)?.inferred_type ?? 'text'
}

function closestColumn(dataset: DatasetMeta, name: string): string {
  const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, '')
  return (
    dataset.columns.find((c) => norm(c.name) === norm(name))?.name ?? dataset.columns[0]?.name ?? ''
  )
}

function AddComparison({
  meta,
  datasetA,
  datasetB,
  taken,
  first,
  onAdd,
}: {
  meta: Meta
  datasetA: DatasetMeta
  datasetB: DatasetMeta
  taken: string[]
  first: boolean
  onAdd: (comparison: Comparison) => void
}) {
  const [columnA, setColumnA] = useState(datasetA.columns[0]?.name ?? '')
  const [columnB, setColumnB] = useState(() => closestColumn(datasetB, columnA))
  const [type, setType] = useState<ColumnType>(() => inferredType(datasetA, columnA))

  const pickA = (name: string) => {
    setColumnA(name)
    setColumnB(closestColumn(datasetB, name))
    setType(inferredType(datasetA, name))
  }

  return (
    <section className="card" aria-label="Add a comparison">
      <div className="card-body stack">
        {first && (
          <p className="muted">
            No comparisons yet. Add one below, or let Auto-suggest propose a starting point.
          </p>
        )}
        <div className="add-row">
          <Field label="Column in file A">
            <select className="select" value={columnA} onChange={(e) => pickA(e.target.value)}>
              {columnOptions(datasetA)}
            </select>
          </Field>
          <Field label="Column in file B">
            <select className="select" value={columnB} onChange={(e) => setColumnB(e.target.value)}>
              {columnOptions(datasetB)}
            </select>
          </Field>
          <Field label="Compare as">
            <select
              className="select"
              value={type}
              onChange={(e) => setType(e.target.value as ColumnType)}
            >
              {TYPES.map((t) => (
                <option key={t.value} value={t.value}>
                  {t.label}
                </option>
              ))}
            </select>
          </Field>
          <button
            type="button"
            className="btn btn-primary"
            disabled={!columnA || !columnB}
            onClick={() => onAdd(makeComparison(meta, type, columnA, columnB, comparisonId(columnA, taken)))}
          >
            Add comparison
          </button>
        </div>
      </div>
    </section>
  )
}

function summary(c: Comparison): string {
  const type = TYPES.find((t) => t.value === c.type)?.label ?? c.type
  switch (c.type) {
    case 'text':
      return `${type} · ${METRICS[c.metric ?? 'token_sort'].toLowerCase()}`
    case 'number':
      return c.tolerance
        ? `${type} · within ${c.tolerance_mode === 'relative' ? `${+(c.tolerance * 100).toFixed(4)}%` : c.tolerance}`
        : `${type} · exact`
    case 'date':
      return c.tolerance_days ? `${type} · within ${c.tolerance_days} days` : `${type} · exact`
    default:
      return `${type} · exact`
  }
}

function ComparisonCard({
  comparison: c,
  meta,
  datasetA,
  datasetB,
  expanded,
  onToggle,
  onChange,
  onRemove,
}: {
  comparison: Comparison
  meta: Meta
  datasetA: DatasetMeta
  datasetB: DatasetMeta
  expanded: boolean
  onToggle: () => void
  onChange: (c: Comparison) => void
  onRemove: () => void
}) {
  const separateB = c.type === 'number' || c.type === 'date'
  const cleaningA = (c.cleaning ?? {}) as Record<string, unknown>
  const cleaningB =
    separateB && c.cleaning_b ? (c.cleaning_b as Record<string, unknown>) : cleaningA

  return (
    <section className="card" aria-label={`Comparison ${c.column_a} and ${c.column_b}`}>
      <div className="card-head comparison-head">
        <button
          type="button"
          className="disclosure"
          aria-expanded={expanded}
          onClick={onToggle}
        >
          <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden>
            <path d="M4 2.5L7.5 6 4 9.5" fill="none" stroke="currentColor" strokeWidth="1.6" />
          </svg>
          <span className="titles">
            <span className="pair">
              <strong>{c.column_a}</strong>
              <span className="subtle">↔</span>
              <strong>{c.column_b}</strong>
            </span>
            <span className="subtle">{summary(c)}</span>
          </span>
        </button>
        <div className="row" style={{ flexWrap: 'nowrap' }}>
          <label className="weight">
            <span className="subtle">Weight</span>
            <input
              className="input num"
              type="number"
              min={0.1}
              max={100}
              step={0.5}
              value={c.weight}
              onChange={(e) => onChange({ ...c, weight: Math.max(0.1, Number(e.target.value) || 1) })}
            />
          </label>
          <RemoveButton label="Remove comparison" onClick={onRemove} />
        </div>
      </div>
      {expanded && (
        <div className="card-body stack">
          <div className="fields">
            <Field label="Column in file A">
              <select
                className="select"
                value={c.column_a}
                onChange={(e) => onChange({ ...c, column_a: e.target.value })}
              >
                {columnOptions(datasetA)}
              </select>
            </Field>
            <Field label="Column in file B">
              <select
                className="select"
                value={c.column_b}
                onChange={(e) => onChange({ ...c, column_b: e.target.value })}
              >
                {columnOptions(datasetB)}
              </select>
            </Field>
            <Field label="Compare as">
              <select
                className="select"
                value={c.type}
                onChange={(e) => onChange(changeType(meta, c, e.target.value as ColumnType))}
              >
                {TYPES.map((t) => (
                  <option key={t.value} value={t.value}>
                    {t.label}
                  </option>
                ))}
              </select>
            </Field>
            <SimilarityFields comparison={c} onChange={onChange} />
          </div>

          <div className="section-title">
            <h3>Cleaning</h3>
          </div>
          <CleaningOptions
            type={c.type}
            idPrefix={`${c.id}-a`}
            value={cleaningA}
            onChange={(v) => onChange({ ...c, cleaning: v } as Comparison)}
          />
          {separateB && (
            <div className="stack">
              <label className="check">
                <input
                  type="checkbox"
                  checked={!!c.cleaning_b}
                  onChange={(e) =>
                    onChange({
                      ...c,
                      cleaning_b: e.target.checked ? structuredClone(cleaningA) : null,
                    } as Comparison)
                  }
                />
                <span>File B formats {c.type === 'date' ? 'dates' : 'numbers'} differently</span>
              </label>
              {c.cleaning_b && (
                <div className="nested">
                  <span className="label">File B</span>
                  <CleaningOptions
                    type={c.type}
                    idPrefix={`${c.id}-b`}
                    value={cleaningB}
                    onChange={(v) => onChange({ ...c, cleaning_b: v } as Comparison)}
                  />
                </div>
              )}
            </div>
          )}

          <div className="grid-2">
            <CleaningPreview
              title="File A"
              datasetId={datasetA.id}
              column={c.column_a}
              type={c.type}
              cleaning={cleaningA}
            />
            <CleaningPreview
              title="File B"
              datasetId={datasetB.id}
              column={c.column_b}
              type={c.type}
              cleaning={cleaningB}
            />
          </div>
        </div>
      )}
    </section>
  )
}

function SimilarityFields({
  comparison: c,
  onChange,
}: {
  comparison: Comparison
  onChange: (c: Comparison) => void
}) {
  switch (c.type) {
    case 'text':
      return (
        <Field label="Similarity">
          <select
            className="select"
            value={c.metric}
            onChange={(e) => onChange({ ...c, metric: e.target.value as typeof c.metric })}
          >
            {Object.entries(METRICS).map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        </Field>
      )
    case 'number': {
      const relative = c.tolerance_mode === 'relative'
      const shown = relative ? +((c.tolerance ?? 0) * 100).toFixed(6) : (c.tolerance ?? 0)
      return (
        <>
          <Field label="Tolerance" hint="Score falls to 0 at this difference">
            <input
              className="input num"
              type="number"
              min={0}
              step="any"
              value={shown}
              onChange={(e) => {
                const v = Math.max(0, Number(e.target.value) || 0)
                onChange({ ...c, tolerance: relative ? v / 100 : v })
              }}
            />
          </Field>
          <Field label="Measured as">
            <select
              className="select"
              value={c.tolerance_mode}
              onChange={(e) => onChange({ ...c, tolerance_mode: e.target.value as 'absolute' | 'relative' })}
            >
              <option value="absolute">Absolute difference</option>
              <option value="relative">Percent of the larger value</option>
            </select>
          </Field>
        </>
      )
    }
    case 'date':
      return (
        <Field label="Tolerance (days)" hint="Score falls to 0 at this difference">
          <input
            className="input num"
            type="number"
            min={0}
            step={1}
            value={c.tolerance_days ?? 0}
            onChange={(e) => onChange({ ...c, tolerance_days: Math.max(0, Math.round(Number(e.target.value) || 0)) })}
          />
        </Field>
      )
    default:
      return null
  }
}
