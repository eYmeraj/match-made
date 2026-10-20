import { useState, type InputHTMLAttributes } from 'react'

type Props = Omit<
  InputHTMLAttributes<HTMLInputElement>,
  'type' | 'value' | 'onChange' | 'min' | 'max'
> & {
  value: number | null
  onChange: (value: number) => void
  min?: number
  max?: number
  integer?: boolean
}

const asText = (value: number | null) => (value === null ? '' : String(value))

/**
 * A number input that keeps what is typed and only reports valid numbers. Bound straight
 * to a number, clearing the field refills it with a default, so deleting "1" and typing
 * "3" would give 13.
 */
export function NumberField({ value, onChange, min, max, integer, onBlur, ...rest }: Props) {
  const [text, setText] = useState(asText(value))
  const [reported, setReported] = useState(value)
  // Follow changes made elsewhere, such as Auto-suggest, without fighting the typing.
  if (value !== reported) {
    setReported(value)
    setText(asText(value))
  }

  const parse = (raw: string): number | null => {
    const n = raw.trim() === '' ? NaN : Number(raw)
    if (!Number.isFinite(n) || (integer && !Number.isInteger(n))) return null
    if ((min !== undefined && n < min) || (max !== undefined && n > max)) return null
    return n
  }

  return (
    <input
      {...rest}
      type="number"
      min={min}
      max={max}
      value={text}
      aria-invalid={parse(text) === null}
      onChange={(e) => {
        setText(e.target.value)
        const n = parse(e.target.value)
        if (n !== null) {
          setReported(n)
          onChange(n)
        }
      }}
      onBlur={(e) => {
        setText(asText(value))
        onBlur?.(e)
      }}
    />
  )
}
