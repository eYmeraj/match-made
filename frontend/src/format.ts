const integer = new Intl.NumberFormat('en-US')

export function formatCount(n: number): string {
  return integer.format(n)
}

export function formatPercent(fraction: number, digits = 1): string {
  return `${(fraction * 100).toFixed(digits)}%`
}
