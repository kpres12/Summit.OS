"""
Heli.OS — Weather REST Adapter
==================================

Polls weather APIs and emits WEATHER_STATION entities at configured geographic
locations. Supports National Weather Service (NWS) and OpenWeatherMap.

Dependencies
------------
    pip install aiohttp
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

try:
    import aiohttp
except ImportError:
    raise ImportError(
        "aiohttp is required for WeatherAdapter. Install with: pip install aiohttp>=3.9.0"
    )

from .base import AdapterConfig, BaseAdapter

logger = logging.getLogger("heli.adapters.weather")

# NWS base URL
_NWS_BASE = "https://api.weather.gov"
# OWM base URL
_OWM_BASE = "https://api.openweathermap.org/data/2.5"


def _fire_weather_index(
    temp_c: Optional[float],
    humidity_pct: Optional[float],
    wind_speed_mps: Optional[float],
) -> Optional[float]:
    """
    Simple Fire Weather Index approximation.
    Higher = greater fire risk. Returns 0-100 scaled value.
    """
    if temp_c is None or humidity_pct is None or wind_speed_mps is None:
        return None
    # Normalise components: high temp, low humidity, high wind = high risk
    temp_factor = max(0, min(1, (temp_c - 10) / 40))
    humidity_factor = max(0, min(1, 1 - humidity_pct / 100))
    wind_factor = max(0, min(1, wind_speed_mps / 20))
    fwi = (temp_factor * 0.4 + humidity_factor * 0.4 + wind_factor * 0.2) * 100
    return round(fwi, 1)


class WeatherAdapter(BaseAdapter):
    """
    Polls NWS or OpenWeatherMap and emits WEATHER_STATION observations.

    Config extras
    -------------
    source                : "nws" | "openweathermap"
    api_key               : str     (required for openweathermap)
    locations             : list of {"name": str, "lat": float, "lon": float}
    poll_interval_seconds : float   (default 300.0)
    """

    adapter_type = "weather"

    def __init__(self, config: AdapterConfig, mqtt_client=None) -> None:
        super().__init__(config, mqtt_client)
        ex = config.extra

        self._source: str = ex.get("source", "nws")
        self._api_key: str = ex.get("api_key", "")
        self._locations: list[dict] = ex.get("locations", [])
        self._poll_interval: float = float(
            ex.get("poll_interval_seconds", config.poll_interval_seconds or 300.0)
        )

        if not self._locations:
            raise ValueError("At least one location must be configured")

        if self._source == "openweathermap" and not self._api_key:
            raise ValueError("api_key is required for OpenWeatherMap source")

        self._session: Optional[aiohttp.ClientSession] = None
        # NWS: cache gridpoint URLs per location to avoid repeated /points calls
        self._nws_observation_urls: dict[str, str] = {}

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession(
            headers={"User-Agent": "HeliOS-WeatherAdapter/1.0 (heli-os)"}
        )
        self._log.info(
            "Weather adapter ready: source=%s, locations=%d",
            self._source,
            len(self._locations),
        )

    async def disconnect(self) -> None:
        try:
            if self._session is not None:
                await self._session.close()
                self._session = None
        except Exception:
            pass

    async def stream_observations(self) -> AsyncIterator[dict]:
        while not self._stop_event.is_set():
            for loc in self._locations:
                try:
                    obs = await self._fetch_location(loc)
                    if obs is not None:
                        yield obs
                except Exception as exc:
                    self._log.warning(
                        "Weather fetch error [%s]: %s", loc.get("name"), exc
                    )
            await self._interruptible_sleep(self._poll_interval)

    async def _fetch_location(self, loc: dict) -> Optional[dict]:
        if self._source == "nws":
            return await self._fetch_nws(loc)
        elif self._source == "openweathermap":
            return await self._fetch_owm(loc)
        else:
            raise ValueError(f"Unknown weather source: {self._source!r}")

    # -------------------------------------------------------------------------
    # National Weather Service
    # -------------------------------------------------------------------------

    async def _fetch_nws(self, loc: dict) -> Optional[dict]:
        lat = float(loc["lat"])
        lon = float(loc["lon"])
        name = loc.get("name", f"{lat},{lon}")
        loc_key = f"{lat:.4f},{lon:.4f}"

        # Resolve observation URL (cached)
        if loc_key not in self._nws_observation_urls:
            obs_url = await self._nws_resolve_observation_url(lat, lon)
            if obs_url is None:
                return None
            self._nws_observation_urls[loc_key] = obs_url

        obs_url = self._nws_observation_urls[loc_key]
        async with self._session.get(
            obs_url, timeout=aiohttp.ClientTimeout(total=15)
        ) as resp:
            if resp.status == 404:
                # Invalidate cache and return
                self._nws_observation_urls.pop(loc_key, None)
                return None
            resp.raise_for_status()
            data = await resp.json()

        props = data.get("properties", {})
        return self._nws_props_to_obs(name, lat, lon, props)

    async def _nws_resolve_observation_url(
        self, lat: float, lon: float
    ) -> Optional[str]:
        """Use /points to find the nearest station and return its observation URL."""
        points_url = f"{_NWS_BASE}/points/{lat:.4f},{lon:.4f}"
        try:
            async with self._session.get(
                points_url, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except Exception as exc:
            self._log.warning("NWS /points failed for %.4f,%.4f: %s", lat, lon, exc)
            return None

        # Get observation stations list
        obs_stations_url = data.get("properties", {}).get("observationStations")
        if not obs_stations_url:
            return None

        try:
            async with self._session.get(
                obs_stations_url, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                resp.raise_for_status()
                stations_data = await resp.json()
        except Exception as exc:
            self._log.warning("NWS stations fetch failed: %s", exc)
            return None

        features = stations_data.get("features", [])
        if not features:
            return None

        station_id = features[0].get("properties", {}).get("stationIdentifier")
        if not station_id:
            return None

        return f"{_NWS_BASE}/stations/{station_id}/observations/latest"

    def _nws_props_to_obs(self, name: str, lat: float, lon: float, props: dict) -> dict:
        def nws_val(field: str) -> Optional[float]:
            obj = props.get(field, {})
            if isinstance(obj, dict):
                v = obj.get("value")
                return float(v) if v is not None else None
            return None

        temp_c = nws_val("temperature")
        humidity = nws_val("relativeHumidity")
        wind_speed_mps = nws_val("windSpeed")
        if wind_speed_mps is not None:
            wind_speed_mps = wind_speed_mps / 3.6  # km/h → m/s

        wind_dir = nws_val("windDirection")
        visibility_m = nws_val("visibility")
        cloud_layers = props.get("cloudLayers", [])
        cloud_cover_pct: Optional[float] = None
        if cloud_layers:
            # Use the lowest cloud layer's amount as a rough proxy
            amount_map = {"CLR": 0, "FEW": 15, "SCT": 40, "BKN": 70, "OVC": 100}
            top_amount = cloud_layers[-1].get("amount", "")
            cloud_cover_pct = float(amount_map.get(top_amount, 50))

        precip = nws_val("precipitationLastHour")

        text_description = props.get("textDescription", "")
        conditions = _classify_conditions(text_description)

        fwi = _fire_weather_index(temp_c, humidity, wind_speed_mps)

        return self._build_obs(
            name=name,
            lat=lat,
            lon=lon,
            temp_c=temp_c,
            humidity_pct=humidity,
            wind_speed_mps=wind_speed_mps,
            wind_direction_deg=wind_dir,
            visibility_m=visibility_m,
            cloud_cover_pct=cloud_cover_pct,
            precipitation_mm_hr=precip,
            conditions=conditions,
            fire_weather_index=fwi,
            raw_description=text_description,
        )

    # -------------------------------------------------------------------------
    # OpenWeatherMap
    # -------------------------------------------------------------------------

    async def _fetch_owm(self, loc: dict) -> Optional[dict]:
        lat = float(loc["lat"])
        lon = float(loc["lon"])
        name = loc.get("name", f"{lat},{lon}")
        url = (
            f"{_OWM_BASE}/weather"
            f"?lat={lat}&lon={lon}"
            f"&appid={self._api_key}&units=metric"
        )
        async with self._session.get(
            url, timeout=aiohttp.ClientTimeout(total=15)
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        return self._owm_data_to_obs(name, lat, lon, data)

    def _owm_data_to_obs(self, name: str, lat: float, lon: float, data: dict) -> dict:
        main = data.get("main", {})
        wind = data.get("wind", {})
        clouds = data.get("clouds", {})
        weather = data.get("weather", [{}])[0]
        rain = data.get("rain", {})
        visibility_m = data.get("visibility")

        temp_c = main.get("temp")
        humidity = main.get("humidity")
        wind_speed_mps = wind.get("speed")
        wind_dir = wind.get("deg")
        cloud_cover_pct = float(clouds.get("all", 0))
        precip_mm_hr = rain.get("1h")
        description = weather.get("description", "")
        conditions = _classify_conditions(description)

        fwi = _fire_weather_index(temp_c, humidity, wind_speed_mps)

        return self._build_obs(
            name=name,
            lat=lat,
            lon=lon,
            temp_c=temp_c,
            humidity_pct=float(humidity) if humidity is not None else None,
            wind_speed_mps=wind_speed_mps,
            wind_direction_deg=wind_dir,
            visibility_m=float(visibility_m) if visibility_m else None,
            cloud_cover_pct=cloud_cover_pct,
            precipitation_mm_hr=precip_mm_hr,
            conditions=conditions,
            fire_weather_index=fwi,
            raw_description=description,
        )

    # -------------------------------------------------------------------------
    # Common observation builder
    # -------------------------------------------------------------------------

    def _build_obs(
        self,
        name: str,
        lat: float,
        lon: float,
        temp_c: Optional[float],
        humidity_pct: Optional[float],
        wind_speed_mps: Optional[float],
        wind_direction_deg: Optional[float],
        visibility_m: Optional[float],
        cloud_cover_pct: Optional[float],
        precipitation_mm_hr: Optional[float],
        conditions: str,
        fire_weather_index: Optional[float],
        raw_description: str,
    ) -> dict:
        now = datetime.now(timezone.utc)
        entity_id = f"weather-{self.config.adapter_id}-{name.lower().replace(' ', '-')}"
        return {
            "source_id": f"{entity_id}:{now.timestamp():.3f}",
            "adapter_id": self.config.adapter_id,
            "adapter_type": self.adapter_type,
            "entity_id": entity_id,
            "callsign": name,
            "position": {"lat": lat, "lon": lon, "alt_m": None},
            "velocity": None,
            "entity_type": "WEATHER_STATION",
            "classification": None,
            "metadata": {
                "temperature_c": temp_c,
                "humidity_pct": humidity_pct,
                "wind_speed_mps": wind_speed_mps,
                "wind_direction_deg": wind_direction_deg,
                "visibility_m": visibility_m,
                "cloud_cover_pct": cloud_cover_pct,
                "precipitation_mm_hr": precipitation_mm_hr,
                "conditions": conditions,
                "fire_weather_index": fire_weather_index,
                "raw_description": raw_description,
                "source": self._source,
            },
            "ts_iso": now.isoformat(),
        }


def _classify_conditions(description: str) -> str:
    """Classify a free-text weather description into a simple condition string."""
    d = description.lower()
    if any(w in d for w in ("snow", "sleet", "blizzard", "flurr")):
        return "snow"
    if any(w in d for w in ("fog", "mist", "haze")):
        return "fog"
    if any(w in d for w in ("rain", "drizzle", "shower", "storm", "thunder")):
        return "rain"
    if any(w in d for w in ("cloud", "overcast", "partly")):
        return "cloudy"
    if any(w in d for w in ("clear", "sunny", "fair", "fine")):
        return "clear"
    return "unknown"
