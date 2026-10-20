import { useMutation } from '@tanstack/react-query'
import { useState, type Dispatch } from 'react'
import { api, unwrap, type ColumnType, type Comparison, type DatasetMeta, type Meta } from '../api/client'
import { CleaningOptions } from '../components/CleaningOptions'
import { CleaningPreview } from '../components/CleaningPreview'
import { NumberField } from '../components/NumberField'
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
          Add a pair for every column in file A that has a counterpart in file B, as many as
          you need. Each pair scores 0–100 after cleaning, and the match score is their
          weighted average.
        </p>
        <button type="button" className="btn" disabled={suggest.isPending} onClick={runSuggest}>
          {suggest.isPending ? <Spinner label="Suggesting…" /> : 'Auto-suggest'}
        </button>
      </div>
      {confirmSuggest && (
        <Notice tone="warning">
          <div className="spread">
            <span>Replace your column pairs and blocking rules with suggestions?</span>
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
        // A fresh form after each add starts from the next column not matched on yet.
        key={draft.comparisons.length}
        meta={meta}
        datasetA={datasetA}
        datasetB={datasetB}
        comparisons={draft.comparisons}
        onAdd={(c) => dispatch({ type: 'addComparison', comparison: c })}
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

/** The column named like `name`, else the first one not matched on yet, else the first. */
function closestColumn(dataset: DatasetMeta, name: string, used: Set<string>): string {
  const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, '')
  const columns = dataset.columns.map((c) => c.name)
  return (
    columns.find((c) => norm(c) === norm(name)) ??
    columns.find((c) => !used.has(c)) ??
    columns[0] ??
    ''
  )
}

function AddComparison({
  meta,
  datasetA,
  datasetB,
  comparisons,
  onAdd,
}: {
  meta: Meta
  datasetA: DatasetMeta
  datasetB: DatasetMeta
  comparisons: Comparison[]
  onAdd: (comparison: Comparison) => void
}) {
  const usedA = new Set(comparisons.map((c) => c.column_a))
  const usedB = new Set(comparisons.map((c) => c.column_b))
  const [columnA, setColumnA] = useState(
    () => datasetA.columns.find((c) => !usedA.has(c.name))?.name ?? datasetA.columns[0]?.name ?? '',
  )
  const [columnB, setColumnB] = useState(() => closestColumn(datasetB, columnA, usedB))
  const [type, setType] = useState<ColumnType>(() => inferredType(datasetA, columnA))
  const duplicate = comparisons.some((c) => c.column_a === columnA && c.column_b === columnB)

  const pickA = (name: string) => {
    setColumnA(name)
    setColumnB(closestColumn(datasetB, name, usedB))
    setType(inferredType(datasetA, name))
  }

  return (
    <section className="card" aria-label="Add a column pair">
      <div className="card-body stack">
        {comparisons.length === 0 && (
          <p className="muted">
            No column pairs yet. Add the first one below, or let Auto-suggest propose some.
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
            disabled={!columnA || !columnB || duplicate}
            onClick={() =>
              onAdd(
                makeComparison(
                  meta,
                  type,
                  columnA,
                  columnB,
                  comparisonId(
                    columnA,
                    comparisons.map((c) => c.id),
                  ),
                ),
              )
            }
          >
            Add column pair
          </button>
        </div>
        {duplicate && <span className="subtle">This pair is already in the list.</span>}
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
    <section className="card" aria-label={`Column pair ${c.column_a} and ${c.column_b}`}>
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
            <NumberField
              className="input num"
              min={0.1}
              max={100}
              step={0.5}
              value={c.weight}
              onChange={(weight) => onChange({ ...c, weight })}
            />
          </label>
          <RemoveButton label={`Remove pair ${c.column_a} and ${c.column_b}`} onClick={onRemove} />
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
            <NumberField
              className="input num"
              min={0}
              step="any"
              value={shown}
              onChange={(v) => onChange({ ...c, tolerance: relative ? v / 100 : v })}
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
          <NumberField
            className="input num"
            min={0}
            integer
            value={c.tolerance_days ?? 0}
            onChange={(days) => onChange({ ...c, tolerance_days: days })}
          />
        </Field>
      )
    default:
      return null
  }
}
