import type {
  BlockingRule,
  Cardinality,
  ColumnType,
  Comparison,
  DatasetMeta,
  KeyPart,
  MatchConfig,
  Meta,
  MissingPolicy,
} from './api/client'

export type Side = 'a' | 'b'

/** The run configuration being edited, before it is submitted as a job. */
export interface Draft {
  datasetA: string | null
  datasetB: string | null
  comparisons: Comparison[]
  rules: BlockingRule[]
  maxBlockPairs: number
  threshold: number
  cardinality: Cardinality
  missing: MissingPolicy
}

export const emptyDraft: Draft = {
  datasetA: null,
  datasetB: null,
  comparisons: [],
  rules: [],
  maxBlockPairs: 1_000_000,
  threshold: 80,
  cardinality: 'one_to_one',
  missing: 'ignore',
}

export type DraftAction =
  | { type: 'setDataset'; side: Side; dataset: DatasetMeta | null }
  | { type: 'addComparison'; comparison: Comparison }
  | { type: 'updateComparison'; id: string; comparison: Comparison }
  | { type: 'removeComparison'; id: string }
  | { type: 'replaceSetup'; comparisons: Comparison[]; rules: BlockingRule[] }
  | { type: 'setRules'; rules: BlockingRule[] }
  | { type: 'setMaxBlockPairs'; value: number }
  | { type: 'setRun'; patch: Partial<Pick<Draft, 'threshold' | 'cardinality' | 'missing'>> }
  | { type: 'load'; config: MatchConfig }
  | { type: 'reset' }

export function draftReducer(draft: Draft, action: DraftAction): Draft {
  switch (action.type) {
    case 'setDataset': {
      const key = action.side === 'a' ? 'datasetA' : 'datasetB'
      const next = { ...draft, [key]: action.dataset?.id ?? null }
      if (!action.dataset) return next
      // A different file keeps the comparisons whose columns it still has.
      const columns = new Set(action.dataset.columns.map((c) => c.name))
      const column = action.side === 'a' ? 'column_a' : 'column_b'
      const kept = draft.comparisons.filter((c) => columns.has(c[column]))
      return { ...next, comparisons: kept, rules: reconcileRules(draft.rules, kept) }
    }
    case 'addComparison':
      return { ...draft, comparisons: [...draft.comparisons, action.comparison] }
    case 'updateComparison': {
      const comparisons = draft.comparisons.map((c) =>
        c.id === action.id ? action.comparison : c,
      )
      return { ...draft, comparisons, rules: reconcileRules(draft.rules, comparisons) }
    }
    case 'removeComparison': {
      const comparisons = draft.comparisons.filter((c) => c.id !== action.id)
      return { ...draft, comparisons, rules: reconcileRules(draft.rules, comparisons) }
    }
    case 'replaceSetup':
      return {
        ...draft,
        comparisons: action.comparisons,
        rules: reconcileRules(action.rules, action.comparisons),
      }
    case 'setRules':
      return { ...draft, rules: reconcileRules(action.rules, draft.comparisons) }
    case 'setMaxBlockPairs':
      return { ...draft, maxBlockPairs: action.value }
    case 'setRun':
      return { ...draft, ...action.patch }
    case 'load':
      return {
        datasetA: action.config.dataset_a,
        datasetB: action.config.dataset_b,
        comparisons: action.config.comparisons,
        rules: action.config.blocking?.rules ?? [],
        maxBlockPairs: action.config.blocking?.max_block_pairs ?? emptyDraft.maxBlockPairs,
        threshold: action.config.threshold,
        cardinality: action.config.cardinality,
        missing: action.config.missing,
      }
    case 'reset':
      return emptyDraft
  }
}

// Mirrors the backend's allowed transforms; the server stays the authority.
const TRANSFORMS: Record<ColumnType, KeyPart['transform'][]> = {
  text: ['exact', 'prefix', 'first_token', 'sorted_tokens', 'phonetic'],
  categorical: ['exact', 'prefix', 'first_token', 'phonetic'],
  number: ['exact', 'round'],
  date: ['exact', 'year', 'year_month'],
  boolean: ['exact'],
}

export function transformsFor(type: ColumnType): KeyPart['transform'][] {
  return TRANSFORMS[type]
}

export function needsParam(transform: KeyPart['transform']): boolean {
  return transform === 'prefix' || transform === 'round'
}

export function supportsTfidf(type: ColumnType): boolean {
  return type === 'text' || type === 'categorical'
}

/**
 * Keep blocking rules consistent with the comparisons they reference: parts pointing at a
 * removed comparison are dropped, and transforms invalid for a changed type fall back to
 * exact. Rules left with nothing to block on are removed.
 */
export function reconcileRules(rules: BlockingRule[], comparisons: Comparison[]): BlockingRule[] {
  const types = new Map(comparisons.map((c) => [c.id, c.type]))
  const result: BlockingRule[] = []
  for (const rule of rules) {
    if (rule.kind === 'tfidf') {
      const type = types.get(rule.comparison_id)
      if (type && supportsTfidf(type)) result.push(rule)
      continue
    }
    const parts = rule.parts.flatMap((part): KeyPart[] => {
      const type = types.get(part.comparison_id)
      if (!type) return []
      if (transformsFor(type).includes(part.transform)) return [part]
      return [{ comparison_id: part.comparison_id, transform: 'exact', param: null }]
    })
    if (parts.length > 0) result.push({ ...rule, parts })
  }
  return result
}

/** A short, unique, URL-safe id for a comparison, derived from its column name. */
export function comparisonId(column: string, taken: Iterable<string>): string {
  const used = new Set(taken)
  const base =
    column
      .toLowerCase()
      .normalize('NFKD')
      .replace(/[^a-z0-9]+/g, '_')
      .replace(/^_+|_+$/g, '')
      .slice(0, 48) || 'field'
  let id = base
  for (let n = 2; used.has(id); n++) id = `${base}_${n}`
  return id
}

export function defaultCleaning(meta: Meta, type: ColumnType): Record<string, unknown> {
  return structuredClone(meta.default_cleaning[type] ?? {})
}

/** A new comparison of the given type with the server's default cleaning for it. */
export function makeComparison(
  meta: Meta,
  type: ColumnType,
  columnA: string,
  columnB: string,
  id: string,
  weight = 1,
): Comparison {
  const base = { id, column_a: columnA, column_b: columnB, weight }
  const cleaning = defaultCleaning(meta, type)
  switch (type) {
    case 'text':
      return { ...base, type, cleaning, metric: 'token_sort' } as Comparison
    case 'number':
      return {
        ...base,
        type,
        cleaning,
        cleaning_b: null,
        tolerance: 0,
        tolerance_mode: 'absolute',
      } as Comparison
    case 'date':
      return { ...base, type, cleaning, cleaning_b: null, tolerance_days: 0 } as Comparison
    case 'boolean':
    case 'categorical':
      return { ...base, type, cleaning } as Comparison
  }
}

export function changeType(meta: Meta, comparison: Comparison, type: ColumnType): Comparison {
  if (comparison.type === type) return comparison
  const { id, column_a, column_b, weight } = comparison
  return makeComparison(meta, type, column_a, column_b, id, weight)
}

/** Drop half-typed replacement rows (no "find" text yet), which the server rejects. */
export function sanitizeCleaning<T extends Record<string, unknown>>(cleaning: T): T {
  const replacements = cleaning.replacements
  if (!Array.isArray(replacements)) return cleaning
  return { ...cleaning, replacements: replacements.filter((r: { find: string }) => r.find !== '') }
}

function sanitizeComparison(comparison: Comparison): Comparison {
  if (comparison.type !== 'text' || !comparison.cleaning) return comparison
  return { ...comparison, cleaning: sanitizeCleaning(comparison.cleaning) }
}

export function toConfig(draft: Draft): MatchConfig | null {
  if (!draft.datasetA || !draft.datasetB || draft.comparisons.length === 0) return null
  return {
    dataset_a: draft.datasetA,
    dataset_b: draft.datasetB,
    comparisons: draft.comparisons.map(sanitizeComparison),
    blocking: { rules: draft.rules, max_block_pairs: draft.maxBlockPairs },
    threshold: draft.threshold,
    cardinality: draft.cardinality,
    missing: draft.missing,
  }
}
