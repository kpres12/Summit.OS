'use client'

export function Sidebar() {
  return (
    <aside className="w-64 border-r border-border p-4 hidden md:block">
      <div className="font-semibold mb-2">Sidebar</div>
      <ul className="space-y-1 text-sm text-muted-foreground">
        <li>Layers</li>
        <li>Filters</li>
        <li>Settings</li>
      </ul>
    </aside>
  )
}
