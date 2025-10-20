# Summit.OS Tactical Interface

A Soviet-era inspired tactical command interface for autonomous operations coordination built with Next.js 15 and React 19.

## Design Philosophy

**Aesthetic**: Soviet-era tactical terminal meets modern AI command deck
- **Primary Color**: Phosphorescent Soviet-green (#00FF91)
- **Background**: Deep matte black (#0A0A0A)
- **Accent**: Muted amber/orange for alerts (#FF9933)
- **Typography**: IBM Plex Mono (monospaced terminal font)
- **Effects**: CRT scanlines, subtle flicker, phosphor glow, bloom

## Architecture

### Layout Structure
```
┌─────────────────────────────────────────────┐
│           TopOverlay (Header)               │
├──────────┬─────────────────┬────────────────┤
│  Asset   │                 │    Mission     │
│   Log    │  TacticalMap    │     Feed       │
│ (Left)   │   (Center)      │   (Right)      │
│          │                 │                │
├──────────┴─────────────────┴────────────────┤
│          CommandBar (Footer)                │
└─────────────────────────────────────────────┘
```

### Components

#### `/components/tactical/`

1. **TacticalLayout.tsx**
   - Main container orchestrating all subcomponents
   - Fullscreen layout with corner dust/wear textures

2. **TopOverlay.tsx**
   - Mission time (UTC)
   - Grid coordinates
   - Status indicators (Weather, AI, Comms, Encryption)
   - BigMT.ai branding

3. **AssetLog.tsx** (Left Sidebar, 320px)
   - Scrolling list of active nodes
   - Real-time telemetry: battery, temperature, signal strength
   - Status indicators with color coding

4. **TacticalMap.tsx** (Center Panel)
   - Canvas-based wireframe grid with 3D terrain contours
   - Asset markers with glowing icons (UAV, GND, TWR, SEN)
   - Network connection lines
   - Hover interactions for node details
   - Compass orientation

5. **MissionFeed.tsx** (Right Sidebar, 320px)
   - Live event stream
   - Color-coded event types (INFO, ALERT, THREAT, TASK, AI)
   - Timestamp and icon indicators

6. **CommandBar.tsx** (Bottom, 80px)
   - System stats (CPU, Network, Power) with dynamic bars
   - Command prompt input with history navigation (↑/↓)
   - Glowing cursor and text effects

### Styling (`app/globals.css`)

- **CRT Effects**:
  - Scanline overlay (4px repeating gradient)
  - Subtle flicker animation (0.15s intervals)
  - Phosphor glow on text and elements

- **Color System**:
  ```css
  --soviet-green: #00FF91
  --soviet-green-dim: #00CC74
  --soviet-green-dark: #00AA5E
  --alert-amber: #FF9933
  --alert-red: #FF3333
  --background: #0A0A0A
  ```

- **Animations**:
  - `scanline-drift`: Slow vertical movement
  - `crt-flicker`: Subtle opacity fluctuation
  - `phosphor-glow`: Pulsing glow on elements
  - `blink`: Binary on/off for indicators

## Integration

### Real-time Data Hook (`hooks/useRealtimeData.ts`)

```typescript
const {
  assets,      // Array of asset telemetry
  events,      // Array of mission events
  stats,       // System statistics
  connected,   // WebSocket connection status
  sendCommand, // Function to send commands to backend
} = useRealtimeData();
```

### API Endpoints (Expected)

- `GET /api/assets` - Fetch all assets
- `GET /api/events` - Fetch recent events
- `POST /api/command` - Execute command

### WebSocket Messages

```json
{
  "type": "asset_update",
  "data": { "id": "UAV-01", "battery": 87, ... }
}

{
  "type": "new_event",
  "data": { "id": "123", "type": "ALERT", "message": "..." }
}

{
  "type": "stats_update",
  "data": { "cpu": 45, "network": 230, "power": 2.8 }
}
```

## Development

```bash
# Start dev server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

## Environment Variables

Create `.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
```

## Features

✅ Fullscreen 21:9 tactical interface
✅ CRT scanline and flicker effects
✅ Phosphorescent green wireframe graphics
✅ Real-time asset telemetry display
✅ Live mission event feed
✅ Interactive tactical map with node markers
✅ Command prompt with history navigation
✅ System resource monitoring
✅ WebSocket integration for live updates
✅ Matte black industrial aesthetic
✅ Monospaced terminal typography
✅ Glowing vector lines and bloom effects

## Future Enhancements

- [ ] 3D WebGL map rendering
- [ ] Geospatial data integration (Leaflet/Mapbox)
- [ ] Voice command input
- [ ] Multi-mission support
- [ ] Playback/replay mode
- [ ] Export mission logs
- [ ] Thermal/IR layer overlays
- [ ] Predictive path visualization

## Technical Stack

- **Next.js 15** (App Router)
- **React 19**
- **TypeScript**
- **Tailwind CSS v4**
- **Canvas API** (Map rendering)
- **WebSocket** (Real-time data)
- **IBM Plex Mono** (Typography)

---

Built for BigMT.ai autonomous operations platform.
