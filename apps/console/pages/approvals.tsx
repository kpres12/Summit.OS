import type { NextPage } from 'next'
import { useEffect, useState } from 'react'

const API = process.env.NEXT_PUBLIC_API_URL || ''

type PendingTask = { task_id: string; asset_id: string; action: string; risk_level: string; created_at: string }

const ApprovalsPage: NextPage = () => {
  const [pending, setPending] = useState<PendingTask[]>([])
  const [msg, setMsg] = useState<string>('')

  const reload = async () => {
    try {
      const r = await fetch(`${API}/v1/tasks/pending`)
      const j = await r.json()
      setPending(j.pending_tasks || [])
    } catch {}
  }

  useEffect(() => {
    reload()
  }, [])

  const approve = async (task_id: string) => {
    try {
      const r = await fetch(`${API}/v1/tasks/${task_id}/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ approved_by: 'operator' })
      })
      const j = await r.json()
      setMsg(`Approved ${j.task_id}`)
      reload()
    } catch (e) {
      setMsg(`Error: ${e?.message || 'approve failed'}`)
    }
  }

  return (
    <div style={{ padding: 16 }}>
      <h1>Approvals</h1>
      {msg && <p>{msg}</p>}
      <ul>
        {pending.map((p: PendingTask) => (
          <li key={p.task_id}>
            <code>{p.task_id}</code> {p.asset_id} {p.action} [{p.risk_level}] {p.created_at}
            <button style={{ marginLeft: 8 }} onClick={() => approve(p.task_id)}>Approve</button>
          </li>
        ))}
      </ul>
    </div>
  )
}

export default ApprovalsPage
