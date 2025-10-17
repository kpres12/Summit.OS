'use client'

interface MissionPanelProps {
  missions: any[]
}

export function MissionPanel({ missions }: MissionPanelProps) {
  return (
    <div className="h-full overflow-auto p-4">
      <h2 className="text-lg font-semibold mb-3">Missions</h2>
      <ul className="space-y-2 text-sm">
        {missions?.length ? (
          missions.map((m, i) => (
            <li key={i} className="rounded border border-border p-2">
              <pre className="whitespace-pre-wrap break-all text-xs">{JSON.stringify(m, null, 2)}</pre>
            </li>
          ))
        ) : (
          <li className="text-muted-foreground">No missions</li>
        )}
      </ul>
    </div>
  )
}
