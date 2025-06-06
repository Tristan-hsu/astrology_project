from typing import List, Dict

import pytz
import requests
from dateutil import parser as dtparser
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
import swisseph as swe
import os

import datetime as dt

def _init_sweph() -> None:
    """Initialise Swiss Ephemeris with ephemeris path if provided via env."""
    ephe_path = os.getenv("SWEPH_PATH")
    if ephe_path and os.path.isdir(ephe_path):
        swe.set_ephe_path(ephe_path)

_init_sweph()

# ---------------------------------------------------------------------------
# Helper utilities 
# ---------------------------------------------------------------------------

def _get_coordinates(place: str) -> tuple[float, float]:
    """Geocode *place* to (lat, lon) using OSM Nominatim."""
    geocoder = Nominatim(user_agent="astro_gradio")
    location = geocoder.geocode(place)
    if not location:
        raise ValueError(f"Could not geocode location: {place}")
    return location.latitude, location.longitude


def _get_timezone(lat: float, lon: float, when_utc: dt.datetime) -> pytz.tzinfo.BaseTzInfo:
    """Return the timezone for the given coordinates, preferring Google API."""
    api_key = os.getenv("GOOGLE_TZ_API_KEY")
    if api_key:
        url = (
            "https://maps.googleapis.com/maps/api/timezone/json?location="
            f"{lat},{lon}&timestamp={int(when_utc.timestamp())}&key={api_key}"
        )
        resp = requests.get(url, timeout=5)
        data = resp.json()
        if data.get("status") == "OK":
            return pytz.timezone(data["timeZoneId"])
        # silently fall through to offline fallback

    tf = TimezoneFinder()
    tzid = tf.timezone_at(lat=lat, lng=lon) or "UTC"
    return pytz.timezone(tzid)


def _to_julian_day(dt_utc: dt.datetime) -> float:
    """Convert *aware* UTC datetime to Julian Day."""
    return swe.julday(
        dt_utc.year,
        dt_utc.month,
        dt_utc.day,
        dt_utc.hour + dt_utc.minute / 60 + dt_utc.second / 3600,
    )


def deg_to_sign(degree: float) -> tuple[int, float]:
    sign_index = int(degree // 30)
    sign_degree = degree % 30
    return sign_index, sign_degree


def _planet_positions(jd_utc: float) -> List[Dict[str, float | str]]:
    """Calculate ecliptic longitude (J2000) and zodiac sign for 10 bodies."""
    planets = {
        "Sun": swe.SUN,
        "Moon": swe.MOON,
        "Mercury": swe.MERCURY,
        "Venus": swe.VENUS,
        "Mars": swe.MARS,
        "Jupiter": swe.JUPITER,
        "Saturn": swe.SATURN,
        "Uranus": swe.URANUS,
        "Neptune": swe.NEPTUNE,
        "Pluto": swe.PLUTO,
    }
    signs = [
        "Aries",
        "Taurus",
        "Gemini",
        "Cancer",
        "Leo",
        "Virgo",
        "Libra",
        "Scorpio",
        "Sagittarius",
        "Capricorn",
        "Aquarius",
        "Pisces",
    ]

    results: List[Dict[str, str | float]] = []
    for name, pid in planets.items():
        position, _ = swe.calc_ut(jd_utc, pid)
        degree = position[0]
        sign_index, sign_degree = deg_to_sign(degree)
        speed = position[3] 
        results.append({
            "planet": name,
            "degree": round(degree, 4),
            "sign": signs[sign_index],
            "sign_degree": round(sign_degree, 2),
            'speed': speed,
            'retrograde': speed < 0
        })
    return results


# ---------------------------------------------------------------------------
# Core calculation function 
# ---------------------------------------------------------------------------

def astro_data(datetime_str: str, location: str) -> Dict:
    """Return planetary positions for *datetime_str* (ISO‑8601) at *location*."""
    try:
        naive_local_dt = dtparser.isoparse(datetime_str)
    except Exception as exc:
        raise ValueError(f"Invalid datetime format: {exc}") from exc

    # 1. Geocode → lat/lon
    lat, lon = _get_coordinates(location)

    # 2. Determine local timezone (online Google API preferred)
    tz = _get_timezone(lat, lon, dt.datetime.utcnow().replace(tzinfo=pytz.utc))

    # 3. Convert local → UTC (assume naive means local)
    if naive_local_dt.tzinfo is None:
        local_dt = tz.localize(naive_local_dt)
    else:
        local_dt = naive_local_dt.astimezone(tz)
    dt_utc = local_dt.astimezone(pytz.utc)

    # 4. Planetary positions
    jd = _to_julian_day(dt_utc)
    positions = _planet_positions(jd)

    return {
        "datetime_utc": dt_utc.isoformat(),
        "location": {
            "place": location,
            "latitude": lat,
            "longitude": lon,
            "timezone": tz.zone,
        },
        "bodies": positions,
    }
