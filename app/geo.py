"""
EcoAlert — GPS & Geocoding utilities
=====================================
- extract_gps_from_exif(image_bytes)  →  (lat, lng) | (None, None)
- reverse_geocode(lat, lng)           →  address string | None
- get_image_location(image_bytes, fallback_lat, fallback_lng)
  Tries EXIF first, falls back to the provided browser coordinates.

Uses:
  - Pillow (already installed: pip install Pillow)
  - httpx (async HTTP client)  ← install if missing
  - OpenStreetMap Nominatim   ← free, no API key needed
"""

from __future__ import annotations

import io
import struct
import logging
from typing import Optional, Tuple

log = logging.getLogger(__name__)

# ── EXIF GPS extraction ───────────────────────────────────────────────────────

def _rational_to_float(value) -> float:
    """Convert a Pillow IFDRational or (num, den) tuple to a Python float."""
    try:
        # PIL IFDRational
        return float(value)
    except TypeError:
        # (numerator, denominator) tuple
        num, den = value
        return num / den if den != 0 else 0.0


def _dms_to_decimal(dms, ref: str) -> float:
    """
    Convert DMS (degrees, minutes, seconds) + reference to decimal degrees.
    dms = [degrees, minutes, seconds]  (each may be IFDRational or tuple)
    ref = 'N' | 'S' | 'E' | 'W'
    """
    degrees = _rational_to_float(dms[0])
    minutes = _rational_to_float(dms[1])
    seconds = _rational_to_float(dms[2])
    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    if ref in ('S', 'W'):
        decimal = -decimal
    return decimal


def extract_gps_from_exif(image_bytes: bytes) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse a JPEG/PNG/WEBP image byte string and return (latitude, longitude)
    from EXIF GPS data.  Returns (None, None) if no GPS info is found.

    Works with images taken on:
    - Android phones (standard JPEG EXIF)
    - iPhones (HEIC converted to JPEG, standard EXIF)
    - Most modern digital cameras
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS, GPSTAGS
    except ImportError:
        log.warning("Pillow not installed — EXIF GPS extraction disabled.")
        return None, None

    try:
        img = Image.open(io.BytesIO(image_bytes))
        exif_data = img._getexif()  # type: ignore[attr-defined]
        if not exif_data:
            return None, None

        # Map tag IDs to names
        labeled = {TAGS.get(k, k): v for k, v in exif_data.items()}
        gps_info_raw = labeled.get("GPSInfo")
        if not gps_info_raw:
            return None, None

        # Map GPS tag IDs to names
        gps = {GPSTAGS.get(k, k): v for k, v in gps_info_raw.items()}

        lat_dms = gps.get("GPSLatitude")
        lat_ref = gps.get("GPSLatitudeRef", "N")
        lng_dms = gps.get("GPSLongitude")
        lng_ref = gps.get("GPSLongitudeRef", "E")

        if not (lat_dms and lng_dms):
            return None, None

        lat = _dms_to_decimal(lat_dms, lat_ref)
        lng = _dms_to_decimal(lng_dms, lng_ref)

        # Sanity check
        if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
            return None, None
        if lat == 0.0 and lng == 0.0:
            return None, None

        log.info("EXIF GPS extracted: lat=%.6f, lng=%.6f", lat, lng)
        return round(lat, 7), round(lng, 7)

    except Exception as exc:
        log.debug("EXIF GPS extraction failed: %s", exc)
        return None, None


# ── Reverse geocoding via OpenStreetMap Nominatim ─────────────────────────────

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
NOMINATIM_HEADERS = {
    "User-Agent": "EcoAlert/1.0 (trash-detection community app)",
    "Accept-Language": "ro,en",
}


async def reverse_geocode(lat: float, lng: float) -> Optional[str]:
    """
    Convert (lat, lng) to a human-readable address using OpenStreetMap Nominatim.
    Returns a compact address string like:
        "Strada Republicii 15, Alba Iulia, Alba, România"
    or None on failure.

    Nominatim usage policy: max 1 req/sec.  For production use, consider
    caching or a self-hosted instance.
    """
    try:
        import httpx
    except ImportError:
        log.warning("httpx not installed — reverse geocoding disabled. Run: pip install httpx")
        return None

    try:
        params = {
            "lat": lat,
            "lon": lng,
            "format": "jsonv2",
            "zoom": 18,          # street-level detail
            "addressdetails": 1,
        }
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(NOMINATIM_URL, params=params, headers=NOMINATIM_HEADERS)
            r.raise_for_status()
            data = r.json()

        # Build a compact address from components
        addr = data.get("address", {})
        parts = []

        # Street + number
        road = addr.get("road") or addr.get("pedestrian") or addr.get("path") or addr.get("footway")
        number = addr.get("house_number")
        if road:
            parts.append(f"{road} {number}".strip() if number else road)

        # City / town / suburb
        city = (addr.get("city") or addr.get("town") or
                addr.get("municipality") or addr.get("village") or
                addr.get("county"))
        if city:
            parts.append(city)

        # Country
        country = addr.get("country")
        if country and country.lower() not in ("românia", "romania"):
            parts.append(country)
        elif country:
            parts.append("România")

        result = ", ".join(parts) if parts else data.get("display_name", "")
        log.info("Reverse geocode (%.4f, %.4f) → %s", lat, lng, result)
        return result or None

    except Exception as exc:
        log.debug("Reverse geocode failed: %s", exc)
        return None


# ── Main helper used by the detect endpoint ──────────────────────────────────

async def get_image_location(
    image_bytes: bytes,
    fallback_lat: Optional[float] = None,
    fallback_lng: Optional[float] = None,
) -> dict:
    """
    Attempt to determine the location where the photo was taken.

    Priority:
      1. GPS from EXIF metadata (most accurate — the actual location)
      2. Browser GPS provided by the user (fallback)
      3. Nothing → lat/lng/address all None

    Returns a dict:
      {
        "latitude": float | None,
        "longitude": float | None,
        "address": str | None,
        "gps_source": "exif" | "browser" | None
      }
    """
    # 1. Try EXIF
    exif_lat, exif_lng = extract_gps_from_exif(image_bytes)

    if exif_lat is not None and exif_lng is not None:
        address = await reverse_geocode(exif_lat, exif_lng)
        return {
            "latitude": exif_lat,
            "longitude": exif_lng,
            "address": address,
            "gps_source": "exif",
        }

    # 2. Browser GPS fallback
    if fallback_lat is not None and fallback_lng is not None:
        address = await reverse_geocode(fallback_lat, fallback_lng)
        return {
            "latitude": fallback_lat,
            "longitude": fallback_lng,
            "address": address,
            "gps_source": "browser",
        }

    # 3. No GPS available
    return {
        "latitude": None,
        "longitude": None,
        "address": None,
        "gps_source": None,
    }
