import type { NextPage } from 'next'
import { useEffect, useState } from 'react'

const API = process.env.NEXT_PUBLIC_API_URL || ''

type Device = { device_id: string; lat?: number; lon?: number; status?: string }
type Alert = { severity?: string; description?: string; lat?: number; lon?: number }

const CopPage: NextPage = () => {
  const [devices, setDevices] = useState<Device[]>([])
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [ts, setTs] = useState<string>('')

  useEffect(() => {
    const fetchWS = async () => {
      try {
        const r = await fetch(`${API}/v1/worldstate`)
        const j = await r.json()
        setDevices(j.devices || [])
        setAlerts(j.alerts || [])
        setTs(j.ts_iso || '')
      } catch {}
    }
    fetchWS()
    const id = setInterval(fetchWS, 2000)
    return () => clearInterval(id)
  }, [])

  return (
    <div style={{ padding: 16 }}>
      <h1>COP</h1>
      <p>ts: {ts}</p>
      <h2>Devices ({devices.length})</h2>
      <ul>
        {devices.map((d: Device, i: number) => (
          <li key={i}>{d.device_id} {d.lat?.toFixed?.(4)},{d.lon?.toFixed?.(4)} {d.status}</li>
        ))}
      </ul>
      <h2>Alerts ({alerts.length})</h2>
      <ul>
        {alerts.map((a: Alert, i: number) => (
          <li key={i}>{a.severity} {a.description} {a.lat?.toFixed?.(4)},{a.lon?.toFixed?.(4)}</li>
        ))}
      </ul>
    </div>
  )
}

export default CopPage
