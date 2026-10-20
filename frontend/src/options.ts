import type { Cardinality } from './api/client'

export const CARDINALITIES: { value: Cardinality; label: string; help: string }[] = [
  {
    value: 'one_to_one',
    label: 'One to one',
    help: 'Each row matches at most one row on the other side. Highest scores win.',
  },
  {
    value: 'best_per_a',
    label: 'Best match per A row',
    help: 'Each A row gets its best B match. A B row can match several A rows.',
  },
  {
    value: 'all',
    label: 'All pairs above threshold',
    help: 'Every pair that scores above the threshold. Rows can match many.',
  },
]
