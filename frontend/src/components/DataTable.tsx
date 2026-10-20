export function DataTable({
  columns,
  rows,
  maxHeight = 320,
  header,
}: {
  columns: string[]
  rows: (string | null)[][]
  maxHeight?: number
  header?: (column: string) => React.ReactNode
}) {
  return (
    <div className="table-wrap" style={{ maxHeight }}>
      <table>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c} title={c}>
                {header ? header(c) : c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {row.map((v, j) =>
                v === null ? (
                  <td key={j} className="null">
                    empty
                  </td>
                ) : (
                  <td key={j} title={v}>
                    {v}
                  </td>
                ),
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
