# Programmatic Data Sources for Heli.OS

Authoritative list of data sources Heli.OS can pull from **right now,
programmatically**, without manual portal acrobatics. Replaces the
earlier list of dead-end sources (paid Sentinel Hub w/o credentials,
xView3 portal-only access, DeepSig portal-only access) that don't
support automation.

Each source below has either:
- ✅ no auth at all (HTTP / S3 anonymous), or
- 🔑 single email registration → bearer token → fully programmatic.

Anything that requires a manual login dance, click-through, or per-file
download UI is out of scope — those don't go in here.

---

## Earth observation imagery

| Source | Auth | Coverage | Wired |
|---|---|---|---|
| **Copernicus Data Space (CDSE)** | 🔑 username/password (free at dataspace.copernicus.eu) | Sentinel-1 SAR / Sentinel-2 / Sentinel-3 / Sentinel-5P | ✅ `copernicus_dataspace_adapter.py` |
| **Sentinel Hub** (paid) | 🔑 OAuth client_credentials (manual setup) | Same Sentinel constellations + Landsat + MODIS | ✅ `sentinel_hub_adapter.py` (works once creds are issued) |
| **AWS Open Data — Sentinel-2 COGs** | ✅ none | s3://sentinel-s2-l2a-cogs/ | available, not yet wired as adapter |
| **AWS Open Data — Landsat** | ✅ none | s3://landsat-pds/ | available |
| **AWS Open Data — Sentinel-1** | ✅ none | s3://sentinel-s1-l1c/ | available |
| **AWS Open Data — GOES-16/17/18** | ✅ none | s3://noaa-goes16/ etc. (geostationary weather, full-disk every 10 min) | available |
| **Microsoft Planetary Computer STAC** | 🔑 free SAS-token issuance via API | Sentinel-1/2/3, Landsat, MODIS, NAIP | available |
| **EuroSAT (Sentinel-2 land cover)** | ✅ none, direct HTTPS | 27k labeled patches × 10 LULC classes | ✅ `datasets/eurosat.py` |
| **NASA EarthData / CMR** | 🔑 free login, bearer token | All NASA EOS missions | available |
| **NASA GIBS WMS/WMTS** | ✅ none | Visualization-grade global mosaics, all wavelengths | partial via existing adapters |

## Weather / atmosphere

| Source | Auth | Coverage | Wired |
|---|---|---|---|
| **NEXRAD Level-II** | ✅ none, S3 anonymous | s3://noaa-nexrad-level2/ — 160+ US WSR-88D radars, ~5 min cadence | ✅ `datasets/nexrad.py` |
| **NOAA NDBC buoys** | ✅ none | Realtime + historical stdmet, every coastal moored buoy | ✅ `train_maritime_sar.py` |
| **Open-Meteo historical archive** | ✅ none (rate-limited free tier) | Global daily from 1940 | ✅ used across many trainers |
| **NOAA Storm Events** | ✅ none, FTP CSV | All US severe weather events 1950-present | ✅ `train_storm_severity.py` |
| **NASA FIRMS — public CSV** | ✅ none, 7-day | Global active fires, last 7 days | ✅ `firms_weather.py` |
| **NASA FIRMS — keyed archive** | 🔑 free MAP_KEY (firms.modaps.eosdis.nasa.gov) | Multi-year VIIRS / MODIS archive | ✅ `train_lightning_ignition.py` |
| **USDM Drought Monitor** | ✅ none, JSON API | Weekly US drought class by state / county | ✅ `train_drought_severity.py` |
| **NASA EONET** | ✅ none, JSON API | Curated current natural events globally | ✅ `train_eonet_classifier.py` |
| **GDACS** | ✅ none, RSS+JSON | Global disaster alerts | wired as data source in download_real_data.py |

## Tracking & telemetry

| Source | Auth | Coverage | Wired |
|---|---|---|---|
| **OpenSky Network** (anonymous) | ✅ none, 100 req/day | Global ADS-B aircraft state vectors | ✅ `opensky_adapter.py` + `train_aircraft_anomaly_lstm.py` |
| **adsb.lol** | ✅ none | Alternative ADS-B feed | available |
| **MarineCadastre.gov AIS** | ✅ none, daily ZIP CSV | All US-waters AIS broadcasts 2017-present | ✅ `datasets/marinecadastre.py` |
| **AISStream.io** | 🔑 free key | Realtime global AIS WebSocket | ✅ `aisstream_adapter.py` |
| **Global Fishing Watch v3** | 🔑 free token | Vessel events (gaps = candidate dark vessels), encounters, port visits, fishing | ✅ `global_fishing_watch_adapter.py` |
| **Space-Track.org** | 🔑 free username/password | USSF orbital catalog, TLEs, decays | ✅ `space_track_adapter.py` |
| **CelesTrak** | ✅ none | Public TLEs, no rate limits | ✅ `celestrak.adapter` |

## Earthquakes / seismic

| Source | Auth | Coverage | Wired |
|---|---|---|---|
| **USGS ComCat** | ✅ none, FDSN Event API | Global earthquake catalog, real-time | ✅ `train_aftershock_lstm.py` (54k events ingested) |
| **USGS Earthquake Hazards** | ✅ none | ShakeMaps, PAGER alerts | available |
| **EMSC** | ✅ none, JSON | European seismic, alternative source | available |

## Conflict / events / OSINT

| Source | Auth | Coverage | Wired |
|---|---|---|---|
| **GDELT 2.0** | ✅ none, 15-min update | Global geocoded events from worldwide media | ✅ `gdelt_adapter.py` |
| **ACLED** | 🔑 free academic / research key | Armed conflict events, ~1M+ records | available |
| **ReliefWeb** | ✅ none, JSON API | Humanitarian crises, disasters, reports | available |
| **UCDP** (Uppsala Conflict Data Program) | ✅ none, JSON API | Battle deaths, conflict episodes | available |

## Air quality / public health

| Source | Auth | Coverage | Wired |
|---|---|---|---|
| **OpenAQ** | ✅ none (free tier) | Global air-quality stations | available |
| **PurpleAir** | 🔑 free key | Hyperlocal PM2.5 sensors | available |
| **WHO emergency events** | ✅ none, JSON | Global health emergencies | available |

## Wildlife / biodiversity (poaching, conservation)

| Source | Auth | Coverage | Wired |
|---|---|---|---|
| **GBIF** | ✅ none, REST | 2B+ wildlife observations | wired as fetch source |
| **iNaturalist** | ✅ none, REST | 100M+ verified observations | wired as fetch source |

## Defense-relevant ML datasets (free + programmatic)

| Dataset | Auth | What it is | Use for |
|---|---|---|---|
| **xBD / xView2** (Maxar) | ✅ direct download | Pre/post disaster building damage, 4-class | Already trained → `damage_classifier` |
| **EuroSAT** | ✅ direct HTTPS | Sentinel-2 LULC, 27k images × 10 classes | Pretrain backbone, sanity test |
| **fMoW (IARPA)** | ✅ AWS public, s3://spacenet-dataset/ | 60+ class overhead, ~1M images | ATR pretrain |
| **SpaceNet 1-8** | ✅ AWS public, s3://spacenet-dataset/ | Building footprints, road extraction | BDA, route planning |
| **BigEarthNet** | ✅ direct HTTPS | Sentinel-2 multilabel land cover, 600k patches | LULC pretrain |
| **DOTA-v2** | ✅ direct download | Aerial 18-class detection, 11k images | ATR overhead |
| **DroneRF / drone acoustic** | ✅ Mendeley / IEEE DataPort | Drone radar / audio signatures | Counter-UAS |
| **MIT-LL DroneAcoustic** | ✅ public download | Drone audio, multiple types | Counter-UAS detection |
| **VisDrone** | ✅ direct download | UAV-perspective detection, 10k images | UAS surveillance |
| **NEU-AAR** | ✅ direct download | Aerial action recognition | Pattern of life |

## Datasets behind credentialed portals (NOT in this list)

The following are real data and the trainers we've shipped will use them
once the data lands on disk, but **they cannot be auto-downloaded** —
the user has to register manually and place files in `packages/training/data/`:

- xView3 SAR vessel detection (https://iuu.xview.us — manual portal only)
- xView (1) ATR (http://xviewdataset.org — registration + per-file download)
- DeepSig RadioML 2016/2018 (https://deepsig.ai/datasets — registration + manual download)

These trainers (`train_xview3_vessel_detector.py`,
`train_radioml_classifier.py`, etc.) have synthetic-fallback paths that
prove the pipeline works; they retrain on the real data after the user
drops the manifest+chips into the data dirs.

---

## Quick start: enable a new programmatic source

1. **Copernicus Data Space** (free Sentinel-1/2/3 alternative):
   ```
   # Register at https://dataspace.copernicus.eu (1-screen form)
   echo "CDSE_USERNAME=you@example.com" >> .env
   echo "CDSE_PASSWORD=..."             >> .env
   ```

2. **Global Fishing Watch** (vessel + dark-vessel events):
   ```
   # Register at https://globalfishingwatch.org/our-apis/
   echo "GFW_API_TOKEN=..." >> .env
   ```

3. **Space-Track.org** (orbital catalog, decays, TLEs):
   ```
   # Register at https://www.space-track.org
   echo "SPACETRACK_USER=you@example.com" >> .env
   echo "SPACETRACK_PASS=..."             >> .env
   ```

4. **GDELT 2.0** (no setup):
   ```
   # Just enable the adapter — it pulls from data.gdeltproject.org/gdeltv2/
   ```

5. **NEXRAD radar** (no setup):
   ```
   # Use packages/training/datasets/nexrad.py directly — anonymous S3
   ```

6. **MarineCadastre AIS** (no setup):
   ```
   # Use packages/training/datasets/marinecadastre.py directly — anonymous HTTPS
   ```

7. **EuroSAT** (no setup):
   ```
   from packages.training.datasets.eurosat import ensure_downloaded, load_eurosat_rgb
   ensure_downloaded()
   ```
