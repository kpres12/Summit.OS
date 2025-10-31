import { useEffect, useState } from 'react'

const NAMES = [
  'detection_event',
  'track',
  'mission_intent',
  'task_assignment',
  'vehicle_telemetry',
  'action_ack',
]

export default function ContractsExamples() {
  const [selected, setSelected] = useState<string>(NAMES[0])
  const [data, setData] = useState<Record<string, unknown> | null>(null)
  const [error, setError] = useState<string>('')

  useEffect(() => {
    const fetchData = async () => {
      setError('')
      try {
        const resp = await fetch(`http://localhost:8000/contracts/example/${selected}`)
        if (!resp.ok) throw new Error(`${resp.status}`)
        setData(await resp.json())
      } catch (e) {
        setError(`Failed to load example: ${e}`)
        setData(null)
      }
    }
    fetchData()
  }, [selected])

  return (
    <div className="p-8 max-w-5xl mx-auto">
      <h1 className="text-3xl font-bold mb-6">Contract Examples</h1>
      <div className="flex gap-2 mb-4 flex-wrap">
        {NAMES.map((n) => (
          <button
            key={n}
            onClick={() => setSelected(n)}
            className={`px-3 py-1 rounded border ${selected === n ? 'bg-blue-500 text-white' : 'bg-white'}`}
          >
            {n}
          </button>
        ))}
      </div>
      {error && <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">{error}</div>}
      <pre className="text-xs bg-gray-50 border p-3 rounded overflow-auto min-h-[300px]">
        {data ? JSON.stringify(data, null, 2) : 'No data'}
      </pre>
    </div>
  )
}
