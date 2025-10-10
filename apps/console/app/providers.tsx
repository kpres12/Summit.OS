'use client'

import { createContext, useContext, useEffect, useState } from 'react'
import { io, Socket } from 'socket.io-client'

interface SummitContextType {
  socket: Socket | null
  isConnected: boolean
  alerts: any[]
  telemetry: any[]
  missions: any[]
}

const SummitContext = createContext<SummitContextType>({
  socket: null,
  isConnected: false,
  alerts: [],
  telemetry: [],
  missions: []
})

export function Providers({ children }: { children: React.ReactNode }) {
  const [socket, setSocket] = useState<Socket | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [alerts, setAlerts] = useState<any[]>([])
  const [telemetry, setTelemetry] = useState<any[]>([])
  const [missions, setMissions] = useState<any[]>([])

  useEffect(() => {
    // Initialize socket connection
    const newSocket = io('ws://localhost:8001', {
      transports: ['websocket']
    })

    newSocket.on('connect', () => {
      console.log('Connected to Summit.OS')
      setIsConnected(true)
    })

    newSocket.on('disconnect', () => {
      console.log('Disconnected from Summit.OS')
      setIsConnected(false)
    })

    newSocket.on('telemetry', (data) => {
      setTelemetry(prev => [...prev.slice(-99), data])
    })

    newSocket.on('alert', (data) => {
      setAlerts(prev => [data, ...prev.slice(0, 99)])
    })

    newSocket.on('mission', (data) => {
      setMissions(prev => {
        const existing = prev.find(m => m.mission_id === data.mission_id)
        if (existing) {
          return prev.map(m => m.mission_id === data.mission_id ? data : m)
        }
        return [data, ...prev.slice(0, 99)]
      })
    })

    setSocket(newSocket)

    return () => {
      newSocket.close()
    }
  }, [])

  return (
    <SummitContext.Provider value={{
      socket,
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
