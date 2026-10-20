import type { ColumnType, Schemas } from '../api/client'
import { Field, RemoveButton } from './ui'

type TextCleaning = Schemas['TextCleaning']
type Replacement = Schemas['Replacement']

const TEXT_TOGGLES: { key: keyof TextCleaning; label: string; hint?: string }[] = [
  { key: 'lowercase', label: 'Lowercase' },
  { key: 'strip_accents', label: 'Remove accents', hint: 'é → e' },
  { key: 'remove_punctuation', label: 'Remove punctuation' },
  { key: 'remove_legal_suffixes', label: 'Remove legal suffixes', hint: 'Inc, Ltd, GmbH, LLC…' },
  { key: 'expand_abbreviations', label: 'Expand abbreviations', hint: 'intl → international' },
  { key: 'remove_stopwords', label: 'Remove stopwords', hint: 'the, of, and…' },
  { key: 'remove_digits', label: 'Remove digits' },
]

type Options = Record<string, unknown>

export function CleaningOptions({
  type,
  value,
  onChange,
  idPrefix,
}: {
  type: ColumnType
  value: Options
  onChange: (value: Options) => void
  idPrefix: string
}) {
  const set = (key: string, v: unknown) => onChange({ ...value, [key]: v })

  switch (type) {
    case 'text':
      return (
        <div className="stack">
          <div className="checks">
            {TEXT_TOGGLES.map((t) => (
              <label key={t.key} className="check">
                <input
                  type="checkbox"
                  checked={Boolean(value[t.key])}
                  onChange={(e) => set(t.key, e.target.checked)}
                />
                <span>
                  {t.label}
                  {t.hint && <span className="subtle"> · {t.hint}</span>}
                </span>
              </label>
            ))}
          </div>
          <Replacements
            value={(value.replacements as Replacement[] | undefined) ?? []}
            onChange={(r) => set('replacements', r)}
          />
        </div>
      )
    case 'categorical':
      return (
        <div className="checks">
          <label className="check">
            <input
              type="checkbox"
              checked={Boolean(value.lowercase)}
              onChange={(e) => set('lowercase', e.target.checked)}
            />
            <span>Lowercase</span>
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={Boolean(value.strip_accents)}
              onChange={(e) => set('strip_accents', e.target.checked)}
            />
            <span>Remove accents</span>
          </label>
        </div>
      )
    case 'number':
      return (
        <div className="fields">
          <Field label="Decimal separator">
            <select
              className="select"
              value={String(value.decimal_separator ?? '.')}
              onChange={(e) => set('decimal_separator', e.target.value)}
            >
              <option value=".">Point · 1,234.56</option>
              <option value=",">Comma · 1.234,56</option>
            </select>
          </Field>
          <label className="check" style={{ alignSelf: 'end', paddingBottom: 8 }}>
            <input
              type="checkbox"
              checked={Boolean(value.percent_as_fraction)}
              onChange={(e) => set('percent_as_fraction', e.target.checked)}
            />
            <span>Read 15% as 0.15</span>
          </label>
        </div>
      )
    case 'date':
      return (
        <div className="fields">
          <Field label="Format" hint="Leave empty to detect common formats">
            <input
              className="input"
              id={`${idPrefix}-format`}
              placeholder="e.g. %d/%m/%Y"
              value={String(value.format ?? '')}
              onChange={(e) => set('format', e.target.value || null)}
            />
          </Field>
          <label className="check" style={{ alignSelf: 'start', paddingTop: 27 }}>
            <input
              type="checkbox"
              checked={Boolean(value.day_first)}
              onChange={(e) => set('day_first', e.target.checked)}
            />
            <span>Day before month · 03/04 is 3 April</span>
          </label>
        </div>
      )
    case 'boolean':
      return (
        <div className="fields">
          <Field label="Values meaning true" hint="Comma separated">
            <ListInput value={value.true_values} onChange={(v) => set('true_values', v)} />
          </Field>
          <Field label="Values meaning false" hint="Comma separated">
            <ListInput value={value.false_values} onChange={(v) => set('false_values', v)} />
          </Field>
        </div>
      )
  }
}

function ListInput({ value, onChange }: { value: unknown; onChange: (v: string[]) => void }) {
  const list = Array.isArray(value) ? (value as string[]) : []
  return (
    <input
      className="input"
      defaultValue={list.join(', ')}
      onBlur={(e) =>
        onChange(
          e.target.value
            .split(',')
            .map((s) => s.trim())
            .filter(Boolean),
        )
      }
    />
  )
}

function Replacements({
  value,
  onChange,
}: {
  value: Replacement[]
  onChange: (value: Replacement[]) => void
}) {
  const update = (i: number, patch: Partial<Replacement>) =>
    onChange(value.map((r, j) => (j === i ? { ...r, ...patch } : r)))

  return (
    <div className="stack" style={{ gap: 8 }}>
      <div className="spread">
        <span className="label">Replacements</span>
        <button
          type="button"
          className="btn btn-sm btn-ghost"
          onClick={() => onChange([...value, { find: '', replace: '', regex: false }])}
        >
          + Add replacement
        </button>
      </div>
      {value.length === 0 && (
        <span className="subtle">Applied after the options above, in order.</span>
      )}
      {value.map((r, i) => (
        <div key={i} className="replacement">
          <input
            className="input"
            aria-label="Find"
            placeholder="Find"
            value={r.find}
            aria-invalid={r.find === ''}
            onChange={(e) => update(i, { find: e.target.value })}
          />
          <span className="subtle">→</span>
          <input
            className="input"
            aria-label="Replace with"
            placeholder="Replace with (empty removes)"
            value={r.replace ?? ''}
            onChange={(e) => update(i, { replace: e.target.value })}
          />
          <label className="check">
            <input
              type="checkbox"
              checked={Boolean(r.regex)}
              onChange={(e) => update(i, { regex: e.target.checked })}
            />
            <span>Regex</span>
          </label>
          <RemoveButton
            label="Remove replacement"
            onClick={() => onChange(value.filter((_, j) => j !== i))}
          />
        </div>
      ))}
    </div>
  )
}
