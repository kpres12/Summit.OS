import type { NextPage } from 'next'
import { useEffect, useState } from 'react'

const API = process.env.NEXT_PUBLIC_API_URL || ''

type Advisory = { risk_level: string; message: string; confidence: number }

const AdvisoriesPage: NextPage = () => {
  const [advisories, setAdvisories] = useState<Advisory[]>([])

  useEffect(() => {
    const load = async () => {
      try {
        const r = await fetch(`${API}/v1/advisories`)
        const j = await r.json()
        setAdvisories(j.advisories || [])
      } catch {}
    }
    load()
    const id = setInterval(load, 5000)
    return () => clearInterval(id)
  }, [])

  return (
    <div style={{ padding: 16 }}>
      <h1>Advisories</h1>
      <ul>
        {advisories.map((a: Advisory, i: number) => (
          <li key={i}>{a.risk_level} {a.message} [{(a.confidence*100).toFixed?.(0)}%]</li>
        ))}
      </ul>
    </div>
  )
}

export default AdvisoriesPage
