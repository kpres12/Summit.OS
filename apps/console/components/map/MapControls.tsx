'use client'

interface MapControlsProps {
  map: any
}

export function MapControls({ map }: MapControlsProps) {
  // Minimal placeholder controls
  const zoomIn = () => map?.zoomIn && map.zoomIn();
  const zoomOut = () => map?.zoomOut && map.zoomOut();

  return (
    <div className="absolute top-2 left-2 z-10 bg-background/80 backdrop-blur rounded shadow border border-border p-1 flex gap-1">
      <button className="px-2 py-1 border border-border rounded" onClick={zoomIn}>+</button>
      <button className="px-2 py-1 border border-border rounded" onClick={zoomOut}>-</button>
    </div>
  )
}
