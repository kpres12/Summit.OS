'use client'

import { useSummit } from './providers'
import { MapView } from '@/components/map/MapView'
import { AlertPanel } from '@/components/alerts/AlertPanel'
import { TelemetryPanel } from '@/components/telemetry/TelemetryPanel'
import { MissionPanel } from '@/components/missions/MissionPanel'
import { StatusBar } from '@/components/layout/StatusBar'
import { Sidebar } from '@/components/layout/Sidebar'
import { Header } from '@/components/layout/Header'

export default function Home() {
  const { isConnected, alerts, telemetry, missions } = useSummit()

  return (
    <div className="h-screen flex flex-col bg-background">
      <Header />
      
      <div className="flex-1 flex overflow-hidden">
        <Sidebar />
        
        <main className="flex-1 flex flex-col">
          <div className="flex-1 relative">
            <MapView 
              alerts={alerts}
              telemetry={telemetry}
              missions={missions}
            />
          </div>
          
          <div className="flex h-80 border-t border-border">
            <div className="w-1/3 border-r border-border">
              <AlertPanel alerts={alerts} />
            </div>
            <div className="w-1/3 border-r border-border">
              <TelemetryPanel telemetry={telemetry} />
            </div>
            <div className="w-1/3">
              <MissionPanel missions={missions} />
            </div>
          </div>
        </main>
      </div>
      
      <StatusBar isConnected={isConnected} />
    </div>
  )
}
