'use client'

interface StatusBarProps {
  isConnected: boolean
}

export function StatusBar({ isConnected }: StatusBarProps) {
  return (
    <div className="h-9 px-3 flex items-center border-t border-border text-xs gap-2">
      <span className={`inline-block h-2 w-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
      <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
    </div>
  )
}
