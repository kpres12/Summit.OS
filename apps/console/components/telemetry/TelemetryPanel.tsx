'use client'

interface TelemetryPanelProps {
  telemetry: any[]
}

export function TelemetryPanel({ telemetry }: TelemetryPanelProps) {
  return (
    <div className="h-full overflow-auto p-4">
      <h2 className="text-lg font-semibold mb-3">Telemetry</h2>
      <ul className="space-y-2 text-sm">
        {telemetry?.length ? (
          telemetry.map((t, i) => (
            <li key={i} className="rounded border border-border p-2">
              <pre className="whitespace-pre-wrap break-all text-xs">{JSON.stringify(t, null, 2)}</pre>
            </li>
          ))
        ) : (
          <li className="text-muted-foreground">No telemetry</li>
        )}
      </ul>
    </div>
  )
}
