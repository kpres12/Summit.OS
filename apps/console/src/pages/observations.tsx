import { useEffect, useState } from 'react';
import Map, { Marker, Popup } from 'react-map-gl';
import 'maplibre-gl/dist/maplibre-gl.css';

interface Observation {
  id: number;
  cls: string;
  lat: number | null;
  lon: number | null;
  confidence: number;
  ts: string;
  source: string | null;
  attributes: Record<string, any> | null;
}

export default function ObservationsPage() {
  const [observations, setObservations] = useState<Observation[]>([]);
  const [selectedObs, setSelectedObs] = useState<Observation | null>(null);
  const [classFilter, setClassFilter] = useState<string>('');
  const [confFilter, setConfFilter] = useState<number>(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchObservations();
    const interval = setInterval(fetchObservations, 5000); // Refresh every 5s
    return () => clearInterval(interval);
  }, [classFilter, confFilter]);

  const fetchObservations = async () => {
    try {
      const params = new URLSearchParams({ limit: '100' });
      if (classFilter) params.append('cls', classFilter);
      
      const response = await fetch(`http://localhost:8000/v1/observations?${params}`);
      const data = await response.json();
      
      // Filter by confidence client-side
      const filtered = (data.observations || []).filter(
        (obs: Observation) => obs.confidence >= confFilter && obs.lat !== null && obs.lon !== null
      );
      
      setObservations(filtered);
      setLoading(false);
    } catch (error) {
      console.error('Failed to fetch observations:', error);
      setLoading(false);
    }
  };

  const getMarkerColor = (obs: Observation) => {
    if (obs.cls.includes('fire') || obs.cls.includes('ignition')) return '#ef4444'; // red
    if (obs.cls.includes('smoke')) return '#f97316'; // orange
    return '#3b82f6'; // blue
  };

  const getMarkerSize = (conf: number) => {
    if (conf >= 0.8) return 12;
    if (conf >= 0.6) return 10;
    return 8;
  };

  return (
    <div className="h-screen flex flex-col">
      {/* Header */}
      <div className="bg-gray-900 border-b border-gray-800 p-4 flex items-center gap-4">
        <h1 className="text-xl font-bold text-white">Observations</h1>
        
        {/* Filters */}
        <div className="flex gap-4">
          <div>
            <label className="text-sm text-gray-400 mr-2">Class:</label>
            <select
              value={classFilter}
              onChange={(e) => setClassFilter(e.target.value)}
              className="bg-gray-800 text-white rounded px-3 py-1 text-sm"
            >
              <option value="">All</option>
              <option value="smoke">Smoke</option>
              <option value="fire.ignition">Fire/Ignition</option>
            </select>
          </div>
          
          <div>
            <label className="text-sm text-gray-400 mr-2">
              Min Confidence: {confFilter.toFixed(1)}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={confFilter}
              onChange={(e) => setConfFilter(parseFloat(e.target.value))}
              className="w-32"
            />
          </div>
        </div>
        
        <div className="ml-auto text-sm text-gray-400">
          {observations.length} observations
        </div>
      </div>

      {/* Map */}
      <div className="flex-1 relative">
        {loading ? (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-950">
            <div className="text-white">Loading observations...</div>
          </div>
        ) : (
          <Map
            initialViewState={{
              latitude: observations[0]?.lat || 37.7749,
              longitude: observations[0]?.lon || -122.4194,
              zoom: 10
            }}
            style={{ width: '100%', height: '100%' }}
            mapStyle="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
          >
            {observations.map((obs) => (
              <Marker
                key={obs.id}
                latitude={obs.lat!}
                longitude={obs.lon!}
                onClick={(e) => {
                  e.originalEvent.stopPropagation();
                  setSelectedObs(obs);
                }}
              >
                <div
                  style={{
                    width: getMarkerSize(obs.confidence),
                    height: getMarkerSize(obs.confidence),
                    backgroundColor: getMarkerColor(obs),
                    borderRadius: '50%',
                    border: '2px solid white',
                    cursor: 'pointer'
                  }}
                />
              </Marker>
            ))}

            {selectedObs && (
              <Popup
                latitude={selectedObs.lat!}
                longitude={selectedObs.lon!}
                onClose={() => setSelectedObs(null)}
                closeButton={true}
                closeOnClick={false}
              >
                <div className="p-2 text-sm">
                  <div className="font-bold mb-1">{selectedObs.cls}</div>
                  <div className="text-gray-600">
                    Confidence: {(selectedObs.confidence * 100).toFixed(0)}%
                  </div>
                  <div className="text-gray-600">
                    Time: {new Date(selectedObs.ts).toLocaleString()}
                  </div>
                  {selectedObs.source && (
                    <div className="text-gray-600">Source: {selectedObs.source}</div>
                  )}
                  <div className="text-xs text-gray-500 mt-1">
                    ({selectedObs.lat!.toFixed(4)}, {selectedObs.lon!.toFixed(4)})
                  </div>
                </div>
              </Popup>
            )}
          </Map>
        )}
      </div>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-gray-900 border border-gray-800 rounded p-3 text-sm">
        <div className="font-bold mb-2 text-white">Legend</div>
        <div className="flex items-center gap-2 mb-1">
          <div className="w-3 h-3 rounded-full bg-red-500" />
          <span className="text-gray-300">Fire/Ignition</span>
        </div>
        <div className="flex items-center gap-2 mb-1">
          <div className="w-3 h-3 rounded-full bg-orange-500" />
          <span className="text-gray-300">Smoke</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-blue-500" />
          <span className="text-gray-300">Other</span>
        </div>
        <div className="text-xs text-gray-500 mt-2">
          Marker size = confidence
        </div>
      </div>
    </div>
  );
}
