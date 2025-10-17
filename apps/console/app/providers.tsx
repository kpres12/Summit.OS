'use client'

import { createContext, useContext, useEffect, useState } from 'react'

interface SummitContextType {
  ws: WebSocket | null
  isConnected: boolean
  alerts: any[]
  telemetry: any[]
  missions: any[]
}

const SummitContext = createContext<SummitContextType>({
  ws: null,
  isConnected: false,
  alerts: [],
  telemetry: [],
  missions: []
})

export function Providers({ children }: { children: React.ReactNode }) {
  const [ws, setWs] = useState<WebSocket | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [alerts, setAlerts] = useState<any[]>([])
  const [telemetry, setTelemetry] = useState<any[]>([])
  const [missions, setMissions] = useState<any[]>([])

  useEffect(() => {
    const base = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8001'
    const url = base.endsWith('/ws') ? base : `${base}/ws`
    const socket = new WebSocket(url)

    socket.onopen = () => {
      setIsConnected(true)
    }
    socket.onclose = () => {
      setIsConnected(false)
    }
    socket.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data)
        if (msg?.type === 'telemetry') {
          setTelemetry(prev => [...prev.slice(-99), msg.data])
        } else if (msg?.type === 'alert') {
          setAlerts(prev => [msg.data, ...prev.slice(0, 99)])
        } else if (msg?.type === 'mission' || msg?.type === 'mission_event') {
          const data = msg.data
          setMissions(prev => {
            const existing = prev.find((m: any) => m.mission_id === data.mission_id)
            if (existing) {
              return prev.map((m: any) => m.mission_id === data.mission_id ? data : m)
            }
            return [data, ...prev.slice(0, 99)]
          })
        }
      } catch {}
    }

    setWs(socket)
    return () => {
      try { socket.close() } catch {}
    }
  }, [])

  return (
    <SummitContext.Provider value={{
      ws,
      isConnected,
      alerts,
      telemetry,
      missions
    }}>
      {children}
    </SummitContext.Provider>
  )
}

export const useSummit = () => {
  const context = useContext(SummitContext)
  if (!context) {
    throw new Error('useSummit must be used within a SummitProvider')
  }
  return context
}
