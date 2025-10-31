import { useEffect, useState } from 'react'

interface FeatureFlags {
  features?: {
    ui?: Record<string, boolean>
    packs?: Record<string, boolean>
  }
  domain?: string
  [key: string]: unknown
}

export default function FlagsPage() {
  const [flags, setFlags] = useState<FeatureFlags | null>(null)
  const [domain, setDomain] = useState<string>('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string>('')

  const fetchFlags = async (d?: string) => {
    setLoading(true)
    setError('')
    try {
      const url = d ? `/feature_flags?domain=${d}` : '/feature_flags'
      // In local dev, proxy through API gateway at localhost:8000
      const resp = await fetch(`http://localhost:8000${url}`)
      if (!resp.ok) {
        throw new Error(`${resp.status}: ${await resp.text()}`)
      }
      const data = await resp.json()
      setFlags(data)
    } catch (e) {
      setError(`Failed to fetch flags: ${e}`)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchFlags()
  }, [])

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <div className="flex items-center gap-4 mb-6">
        <h1 className="text-3xl font-bold">Feature Flags</h1>
        <div className="flex gap-2">
          <input
            type="text"
            placeholder="Domain (e.g., wildfire)"
            value={domain}
            onChange={(e) => setDomain(e.target.value)}
            className="border px-3 py-1 rounded"
          />
          <button
            onClick={() => fetchFlags(domain || undefined)}
            className="bg-blue-500 text-white px-4 py-1 rounded hover:bg-blue-600"
          >
            Load
          </button>
        </div>
      </div>

      {loading && <div className="text-gray-600">Loading flags...</div>}
      
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}

      {flags && (
        <div className="space-y-6">
          {flags.domain && (
            <div className="bg-blue-50 border border-blue-200 p-4 rounded">
              <strong>Domain:</strong> {flags.domain}
            </div>
          )}
          
          <div className="bg-gray-50 border p-4 rounded">
            <h2 className="text-xl font-semibold mb-3">UI Features</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {Object.entries(flags.features?.ui || {}).map(([key, enabled]) => (
                <div key={key} className="flex items-center gap-2">
                  <span className={`w-3 h-3 rounded-full ${enabled ? 'bg-green-500' : 'bg-gray-400'}`}></span>
                  <span className="text-sm">{key}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-gray-50 border p-4 rounded">
            <h2 className="text-xl font-semibold mb-3">Domain Packs</h2>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
              {Object.entries(flags.features?.packs || {}).map(([key, enabled]) => (
                <div key={key} className="flex items-center gap-2">
                  <span className={`w-3 h-3 rounded-full ${enabled ? 'bg-green-500' : 'bg-gray-400'}`}></span>
                  <span className="text-sm font-medium">{key}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-gray-50 border p-4 rounded">
            <h2 className="text-xl font-semibold mb-3">Raw JSON</h2>
            <pre className="text-xs bg-white border p-3 rounded overflow-auto">
              {JSON.stringify(flags, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  )
}