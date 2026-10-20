import { useMutation } from '@tanstack/react-query'
import type { Dispatch } from 'react'
import {
  api,
  unwrap,
  type BlockingRule,
  type Comparison,
  type KeyPart,
  type KeyRule,
  type MatchConfig,
  type Schemas,
  type TfidfRule,
} from '../api/client'
import { ErrorNotice, Field, Notice, RemoveButton, Spinner } from '../components/ui'
import {
  needsParam,
  supportsTfidf,
  toConfig,
  transformsFor,
  type Draft,
  type DraftAction,
} from '../draft'
import { formatCount } from '../format'

type Estimate = Schemas['Estimate']

const TRANSFORMS: Record<KeyPart['transform'], string> = {
  exact: 'Exact value',
  prefix: 'First characters',
  first_token: 'First word',
  sorted_tokens: 'All words, any order',
  phonetic: 'Sounds alike',
  year: 'Year',
  year_month: 'Year and month',
  round: 'Rounded into buckets',
}

export function BlockingStep({
  draft,
  dispatch,
  rowsA,
  rowsB,
}: {
  draft: Draft
  dispatch: Dispatch<DraftAction>
  rowsA: number
  rowsB: number
}) {
  const config = toConfig(draft)
  const estimate = useMutation({
    mutationFn: (body: MatchConfig) => unwrap(api.POST('/api/estimate', { body })),
  })
  const stale = !!estimate.data && JSON.stringify(estimate.variables) !== JSON.stringify(config)
  const result = estimate.data && !stale ? estimate.data : null

  const rules = draft.rules
  const setRules = (next: BlockingRule[]) => dispatch({ type: 'setRules', rules: next })
  const update = (i: number, rule: BlockingRule) => setRules(rules.map((r, j) => (j === i ? rule : r)))
  const textComparisons = draft.comparisons.filter((c) => supportsTfidf(c.type))
  const first = draft.comparisons[0]

  return (
    <div className="stack">
      <p className="muted" style={{ maxWidth: 720 }}>
        Blocking decides which row pairs get scored. Comparing every row of A with every row of B
        means <span className="num">{formatCount(rowsA * rowsB)}</span> pairs. A pair is scored
        when any rule below selects it.
      </p>

      {rules.length === 0 && (
        <Notice tone={rowsA * rowsB > 10_000_000 ? 'warning' : 'info'}>
          No rules: every pair will be compared.
          {rowsA * rowsB > 10_000_000 && ' That is too many for files this size; add a rule.'}
        </Notice>
      )}

      {rules.map((rule, i) => (
        <section key={i} className="card" aria-label={`Blocking rule ${i + 1}`}>
          <div className="card-head">
            <div className="titles">
              <h3>
                Rule {i + 1} ·{' '}
                {rule.kind === 'key' ? 'Same key' : 'Similar text'}
              </h3>
              <span className="subtle">
                {rule.kind === 'key'
                  ? 'Pairs whose values agree on every part.'
                  : 'Each A value is paired with its most similar B values by character n-grams.'}
              </span>
            </div>
            <div className="row" style={{ flexWrap: 'nowrap' }}>
              {result?.rules[i] && <RuleBadge estimate={result.rules[i]} />}
              <RemoveButton label={`Remove rule ${i + 1}`} onClick={() => setRules(rules.filter((_, j) => j !== i))} />
            </div>
          </div>
          <div className="card-body">
            {rule.kind === 'key' ? (
              <KeyRuleEditor rule={rule} comparisons={draft.comparisons} onChange={(r) => update(i, r)} />
            ) : (
              <TfidfRuleEditor rule={rule} comparisons={textComparisons} onChange={(r) => update(i, r)} />
            )}
          </div>
        </section>
      ))}

      <div className="row">
        <button
          type="button"
          className="btn"
          disabled={!first}
          onClick={() =>
            setRules([
              ...rules,
              { kind: 'key', parts: [{ comparison_id: first.id, transform: 'exact', param: null }] },
            ])
          }
        >
          + Same key rule
        </button>
        <button
          type="button"
          className="btn"
          disabled={textComparisons.length === 0}
          title={textComparisons.length === 0 ? 'Needs a text or category comparison' : undefined}
          onClick={() =>
            setRules([
              ...rules,
              { kind: 'tfidf', comparison_id: textComparisons[0].id, top_k: 10, min_similarity: 0.3 },
            ])
          }
        >
          + Similar text rule
        </button>
      </div>

      <section className="card" aria-label="Estimate">
        <div className="card-head">
          <div className="titles">
            <h3>Candidate pairs</h3>
            <span className="subtle">
              Large key blocks (often blank or very common values) are skipped above{' '}
              <span className="num">{formatCount(draft.maxBlockPairs)}</span> pairs.
            </span>
          </div>
          <button
            type="button"
            className="btn"
            disabled={!config || estimate.isPending}
            onClick={() => config && estimate.mutate(config)}
          >
            {estimate.isPending ? <Spinner label="Estimating…" /> : stale ? 'Re-estimate' : 'Estimate'}
          </button>
        </div>
        <div className="card-body stack">
          <Field label="Skip key blocks larger than (pairs)">
            <input
              className="input num"
              style={{ maxWidth: 200 }}
              type="number"
              min={1}
              step={100000}
              value={draft.maxBlockPairs}
              onChange={(e) =>
                dispatch({ type: 'setMaxBlockPairs', value: Math.max(1, Math.round(Number(e.target.value) || 1)) })
              }
            />
          </Field>
          {stale && <span className="subtle">The setup changed since the last estimate.</span>}
          {result && <EstimateSummary estimate={result} hasRules={rules.length > 0} />}
          <ErrorNotice error={estimate.error} />
        </div>
      </section>
    </div>
  )
}

function RuleBadge({ estimate }: { estimate: Estimate['rules'][number] }) {
  return (
    <span className="badge badge-accent num" title={estimate.upper_bound ? 'Upper bound' : undefined}>
      {estimate.upper_bound ? '≤ ' : ''}
      {formatCount(estimate.pairs)} pairs
    </span>
  )
}

function EstimateSummary({ estimate, hasRules }: { estimate: Estimate; hasRules: boolean }) {
  const total = hasRules
    ? estimate.rules.reduce((sum, r) => sum + r.pairs, 0)
    : estimate.all_pairs
  // Rules are OR-ed and may overlap, so a sum over several rules is only an upper bound.
  const bound = hasRules && (estimate.rules.length > 1 || estimate.rules.some((r) => r.upper_bound))
  const skipped = estimate.rules.reduce((sum, r) => sum + r.skipped_blocks, 0)
  const over = total > estimate.max_candidates
  return (
    <div className="stack" style={{ gap: 8 }}>
      <div className="tiles">
        <div className="tile">
          <span className="label">Pairs to score</span>
          <div className="value">
            {bound ? '≤ ' : ''}
            {formatCount(total)}
          </div>
          <span className="sub">
            {estimate.all_pairs > 0
              ? `${((total / estimate.all_pairs) * 100).toPrecision(2)}% of all pairs`
              : 'No rows'}
          </span>
        </div>
        <div className="tile">
          <span className="label">Limit</span>
          <div className="value">{formatCount(estimate.max_candidates)}</div>
          <span className="sub">pairs per run</span>
        </div>
      </div>
      {over && (
        <Notice tone={bound ? 'warning' : 'error'}>
          {bound ? 'This may be' : 'This is'} above the limit. Make the rules more specific
          {estimate.rules.some((r) => r.upper_bound) ? ', or lower Top matches per value' : ''}.
        </Notice>
      )}
      {skipped > 0 && (
        <Notice tone="warning">
          {formatCount(skipped)} oversized key block{skipped === 1 ? '' : 's'} will be skipped (
          {formatCount(estimate.rules.reduce((s, r) => s + r.skipped_pairs, 0))} pairs). Rows that
          only share such a key will not be compared by that rule.
        </Notice>
      )}
    </div>
  )
}

function comparisonOptions(comparisons: Comparison[]) {
  return comparisons.map((c) => (
    <option key={c.id} value={c.id}>
      {c.column_a} ↔ {c.column_b}
    </option>
  ))
}

function KeyRuleEditor({
  rule,
  comparisons,
  onChange,
}: {
  rule: KeyRule
  comparisons: Comparison[]
  onChange: (rule: KeyRule) => void
}) {
  const types = new Map(comparisons.map((c) => [c.id, c.type]))
  const setPart = (i: number, part: KeyPart) =>
    onChange({ ...rule, parts: rule.parts.map((p, j) => (j === i ? part : p)) })

  return (
    <div className="stack" style={{ gap: 10 }}>
      {rule.parts.map((part, i) => {
        const type = types.get(part.comparison_id) ?? 'text'
        return (
          <div key={i} className="key-part">
            <span className="and subtle">{i === 0 ? 'Where' : 'and'}</span>
            <select
              className="select"
              aria-label="Comparison"
              value={part.comparison_id}
              onChange={(e) => {
                const nextType = types.get(e.target.value) ?? 'text'
                const keep = transformsFor(nextType).includes(part.transform)
                setPart(i, {
                  comparison_id: e.target.value,
                  transform: keep ? part.transform : 'exact',
                  param: keep ? part.param : null,
                })
              }}
            >
              {comparisonOptions(comparisons)}
            </select>
            <select
              className="select"
              aria-label="Key"
              value={part.transform}
              onChange={(e) => {
                const transform = e.target.value as KeyPart['transform']
                setPart(i, {
                  ...part,
                  transform,
                  param: needsParam(transform) ? (transform === 'prefix' ? 3 : 1000) : null,
                })
              }}
            >
              {transformsFor(type).map((t) => (
                <option key={t} value={t}>
                  {TRANSFORMS[t]}
                </option>
              ))}
            </select>
            {needsParam(part.transform) ? (
              <input
                className="input num"
                aria-label={part.transform === 'prefix' ? 'Characters' : 'Bucket size'}
                title={part.transform === 'prefix' ? 'Characters' : 'Bucket size'}
                type="number"
                min={part.transform === 'prefix' ? 1 : 0}
                step={part.transform === 'prefix' ? 1 : 'any'}
                value={part.param ?? ''}
                onChange={(e) => {
                  const v = Number(e.target.value)
                  setPart(i, { ...part, param: v > 0 ? v : null })
                }}
              />
            ) : (
              <span />
            )}
            {rule.parts.length > 1 ? (
              <RemoveButton
                label="Remove part"
                onClick={() => onChange({ ...rule, parts: rule.parts.filter((_, j) => j !== i) })}
              />
            ) : (
              <span />
            )}
          </div>
        )
      })}
      <div>
        <button
          type="button"
          className="btn btn-sm btn-ghost"
          onClick={() =>
            onChange({
              ...rule,
              parts: [
                ...rule.parts,
                { comparison_id: comparisons[0].id, transform: 'exact', param: null },
              ],
            })
          }
        >
          + And another part
        </button>
      </div>
    </div>
  )
}

function TfidfRuleEditor({
  rule,
  comparisons,
  onChange,
}: {
  rule: TfidfRule
  comparisons: Comparison[]
  onChange: (rule: TfidfRule) => void
}) {
  return (
    <div className="fields">
      <Field label="Comparison">
        <select
          className="select"
          value={rule.comparison_id}
          onChange={(e) => onChange({ ...rule, comparison_id: e.target.value })}
        >
          {comparisonOptions(comparisons)}
        </select>
      </Field>
      <Field label="Top matches per value" hint="1–100; higher finds more, costs more">
        <input
          className="input num"
          type="number"
          min={1}
          max={100}
          value={rule.top_k}
          onChange={(e) =>
            onChange({ ...rule, top_k: Math.min(100, Math.max(1, Math.round(Number(e.target.value) || 1))) })
          }
        />
      </Field>
      <Field label="Minimum similarity" hint="0–0.99 cosine">
        <input
          className="input num"
          type="number"
          min={0}
          max={0.99}
          step={0.05}
          value={rule.min_similarity}
          onChange={(e) =>
            onChange({ ...rule, min_similarity: Math.min(0.99, Math.max(0, Number(e.target.value) || 0)) })
          }
        />
      </Field>
    </div>
  )
}
