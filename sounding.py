"""
sounding.py
-----------
Helpers for the Cape Canaveral Lightning Probability Tool.

Provides three things the app needs:

1. fetch_sounding(year, month, day, hour)
       Pull the XMR (72210, Cape Canaveral) sounding from the University of
       Wyoming archive via Siphon, returning a tidy DataFrame.

2. parse_wmo_temp(text)
       Decode an uploaded raw WMO TEMP message (TTAA mandatory + TTBB
       significant levels) into the same DataFrame format.

3. compute_indices(df)
       From a sounding DataFrame, compute exactly the model features:
           - Thompson Index (K-Index - Lifted Index)
           - wind_average  (TRUE vector-mean U over 1000-700 mb)   <-- see NOTE
           - RH            (700-500 mb mean relative humidity, %)
           - PWAT          (precipitable water, mm)

Returned DataFrame schema (one row per level, sorted by descending pressure):
    pressure   (hPa)
    height     (m)
    temperature(degC)
    dewpoint   (degC)
    direction  (deg)
    speed      (kts)
    u          (kts)   eastward component
    v          (kts)   northward component

NOTE on wind_average
--------------------
The original app computed:
    wind_average = mean_speed * cos(deg2rad(270 - mean_direction))
i.e. it averaged speed and direction separately, then took a U-like
component. At your request this module instead returns the TRUE
vector-mean U-wind over the layer (mean of the per-level u components).
These can differ noticeably when wind veers/backs with height.

>>> IMPORTANT <<<  Your models were trained on whatever definition built
the training CSVs. If those were built with the old speed/direction
formula, feeding true vector-mean U here is an inconsistency. Verify
against New_Lightning_Probability_Tool.ipynb, or retrain. To revert,
flip USE_VECTOR_MEAN_U to False below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import metpy.calc as mpcalc
from metpy.units import units

# Toggle the wind definition in one place.
USE_VECTOR_MEAN_U = True

CAPE_CANAVERAL_STNM = "74794"  # XMR


# ---------------------------------------------------------------------------
# 1. Fetch sounding from University of Wyoming (via Siphon)
# ---------------------------------------------------------------------------

from datetime import datetime

CAPE_CANAVERAL_STNM = "74794"  # XMR


def fetch_sounding(year: int, month: int, day: int, preferred_hour: int):
    """
    Fetch the most recent available XMR sounding.

    Tries:
        preferred hour
        15Z
        12Z
        10Z
        00Z

    Returns:
        df, hour_used
    """

    candidate_hours = []

    if preferred_hour == 10:
           fallback_hours = [10, 12, 0]
    elif preferred_hour == 15:
           fallback_hours = [15, 12, 10, 0]
    else:
           fallback_hours = [preferred_hour, 15, 12, 10, 0]
           
           for h in fallback_hours:
              if h not in candidate_hours:
                     candidate_hours.append(h)

    errors = []

    for hour in candidate_hours:
        when = datetime(
            int(year),
            int(month),
            int(day),
            int(hour)
        )

        try:
            df = _fetch_wyoming_sounding(when)
            return df, hour

        except Exception as exc:
            errors.append(f"{hour:02d}Z: {exc}")

    raise RuntimeError(
        f"No XMR sounding available for {year}-{month:02d}-{day:02d}\n\n"
        + "\n".join(errors)
    )


def _fetch_wyoming_sounding(when: datetime) -> pd.DataFrame:
    """
    Retrieve a single sounding from the Wyoming archive.
    """

    try:
        from siphon.simplewebservice.wyoming import WyomingUpperAir

    except ImportError as exc:
        raise RuntimeError(
            "The 'siphon' package is not installed. "
            "Add 'siphon' to requirements.txt."
        ) from exc

    try:
        raw = WyomingUpperAir.request_data(
            when,
            CAPE_CANAVERAL_STNM
        )

    except Exception as exc:
        raise RuntimeError(
            f"No data available ({exc})"
        ) from exc

    df = pd.DataFrame(
        {
            "pressure": raw["pressure"].astype(float),
            "height": raw["height"].astype(float),
            "temperature": raw["temperature"].astype(float),
            "dewpoint": raw["dewpoint"].astype(float),
            "direction": raw["direction"].astype(float),
            "speed": raw["speed"].astype(float),
            "u": raw["u_wind"].astype(float),
            "v": raw["v_wind"].astype(float),
        }
    )

    return _clean(df)

# ---------------------------------------------------------------------------
# 2. Decode a raw WMO TEMP message (TTAA + TTBB)
# ---------------------------------------------------------------------------
#
# This is a pragmatic decoder for the FM 35 TEMP alphanumeric code, covering
# the parts needed for these indices:
#   TTAA - mandatory pressure levels (1000,925,850,700,500,400,300,...) with
#          temperature, dewpoint depression, and wind.
#   TTBB - significant levels w.r.t. temperature/humidity (P, T, Td-dep),
#          which sharpen the PWAT and RH integrals.
# TTCC/TTDD (above 100 hPa) are not needed for lightning indices and are
# ignored if present.
#
# The TEMP code is terse and has many regional quirks; this handles the
# common WMO encoding. Always sanity-check decoded output against the source.
# ---------------------------------------------------------------------------

_MANDATORY_PP = {  # TTAA level indicator -> pressure (hPa)
    "99": None,    # surface (pressure encoded in the group)
    "00": 1000.0,
    "92": 925.0,
    "85": 850.0,
    "70": 700.0,
    "50": 500.0,
    "40": 400.0,
    "30": 300.0,
    "25": 250.0,
    "20": 200.0,
    "15": 150.0,
    "10": 100.0,
}


def _decode_ttaa_temp_dewpoint(grp: str):
    """Decode a 5-char TTaTaTaDaDa temperature/dewpoint-depression group."""
    if grp is None or len(grp) < 5 or not grp[:3].isdigit():
        return np.nan, np.nan
    t_tenths = int(grp[:3])
    # Tens digit of tenths odd => negative temperature.
    temp = t_tenths / 10.0
    if int(grp[2]) % 2 == 1:
        temp = -temp
    dd = grp[3:5]
    if dd.isdigit():
        dd = int(dd)
        # 00-50 => tenths of a degree; 56-99 => (value-50) whole degrees.
        dep = dd / 10.0 if dd <= 50 else (dd - 50)
    else:
        dep = np.nan
    dew = temp - dep if not np.isnan(dep) else np.nan
    return temp, dew


def _decode_wind(grp: str):
    """Decode a 5-char dddff wind group (dir in deg, speed in kt).
    Direction is rounded to nearest 5 deg; the units digit of direction
    carries the hundreds of the speed per WMO convention."""
    if grp is None or len(grp) < 5 or not grp.isdigit():
        return np.nan, np.nan
    ddd = int(grp[:3])
    ff = int(grp[3:5])
    # If direction not a multiple of 5, the remainder (1 or 6/.. ) adds 100/500
    # to the speed. Common convention: units digit 1 -> +100, etc. We handle
    # the standard "direction rounded down to 5, remainder*100 added to speed".
    rem = ddd % 5
    direction = ddd - rem
    speed = ff + rem * 100
    if direction == 0 and speed == 0:
        return np.nan, np.nan
    return float(direction), float(speed)


def parse_wmo_temp(text: str) -> pd.DataFrame:
    """
    Decode a raw WMO TEMP message into the standard sounding DataFrame.

    Accepts the full bulletin text (may include TTAA and TTBB sections,
    line breaks, and the leading TTAA/TTBB identifier groups). Groups may
    be separated by spaces and/or newlines.
    """
    if not text or not text.strip():
        raise RuntimeError("Empty TEMP message.")

    tokens = text.replace("\n", " ").split()

    # Split into sections by identifier.
    sections: dict[str, list[str]] = {}
    current = None
    for tok in tokens:
        if tok in ("TTAA", "TTBB", "TTCC", "TTDD"):
            current = tok
            sections[current] = []
        elif current:
            sections[current].append(tok)

    if "TTAA" not in sections and "TTBB" not in sections:
        raise RuntimeError(
            "No TTAA or TTBB section found. Paste the full WMO TEMP bulletin "
            "(it should contain a 'TTAA' and ideally a 'TTBB' line)."
        )

    rows = []

    # ---- TTAA: mandatory levels ----
    if "TTAA" in sections:
        g = sections["TTAA"]
        # Skip the first two header groups (YYGGId, IIiii station) when present.
        # We scan for level groups beginning with a known PP indicator.
        i = 0
        # Drop leading header groups heuristically (day/hour, station id).
        while i < len(g) and not g[i][:2] in _MANDATORY_PP:
            i += 1
        while i < len(g):
            grp = g[i]
            pp = grp[:2]
            if pp not in _MANDATORY_PP:
                i += 1
                continue
            if pp == "99":  # surface
                # 99PPP : surface pressure (PPP, add 1000 if < 100)
                ppp = grp[2:5]
                pres = np.nan
                if ppp.isdigit():
                    pres = float(ppp)
                    pres += 1000.0 if pres < 100 else 0.0
            else:
                pres = _MANDATORY_PP[pp]
            temp = dew = direction = speed = np.nan
            if i + 1 < len(g):
                temp, dew = _decode_ttaa_temp_dewpoint(g[i + 1])
            if i + 2 < len(g):
                direction, speed = _decode_wind(g[i + 2])
            if pres is not None and not np.isnan(pres):
                rows.append(
                    dict(pressure=pres, height=np.nan, temperature=temp,
                         dewpoint=dew, direction=direction, speed=speed)
                )
            i += 3

    # ---- TTBB: significant levels (P + T/Td only, no wind) ----
    if "TTBB" in sections:
        g = sections["TTBB"]
        i = 0
        # significant-level groups come in pairs: (nnPPP, TTTDD)
        # nn is an ordinal indicator (00,11,22,...), PPP the pressure.
        while i + 1 < len(g):
            idx_grp = g[i]
            td_grp = g[i + 1]
            if len(idx_grp) == 5 and idx_grp[2:].isdigit():
                ppp = idx_grp[2:]
                pres = float(ppp)
                # significant-level pressure: values < ~100 imply +1000 hPa
                if pres < 100:
                    pres += 1000.0
                temp, dew = _decode_ttaa_temp_dewpoint(td_grp)
                if not np.isnan(temp):
                    rows.append(
                        dict(pressure=pres, height=np.nan, temperature=temp,
                             dewpoint=dew, direction=np.nan, speed=np.nan)
                    )
            i += 2

    if not rows:
        raise RuntimeError(
            "Could not decode any levels from the TEMP message. "
            "Double-check it is a standard WMO TTAA/TTBB bulletin."
        )

    df = pd.DataFrame(rows)
    # Derive u/v from direction/speed where available.
    df = _add_uv(df)
    return _clean(df)


# ---------------------------------------------------------------------------
# Shared cleanup helpers
# ---------------------------------------------------------------------------
def _add_uv(df: pd.DataFrame) -> pd.DataFrame:
    """Add u/v (kts) columns from direction/speed (meteorological convention)."""
    rad = np.deg2rad(df["direction"].to_numpy(dtype=float))
    spd = df["speed"].to_numpy(dtype=float)
    # Meteorological: wind FROM direction. u = -spd*sin(dir), v = -spd*cos(dir)
    df["u"] = -spd * np.sin(rad)
    df["v"] = -spd * np.cos(rad)
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by descending pressure, drop dup/garbage rows, ensure columns."""
    for col in ["pressure", "height", "temperature", "dewpoint",
                "direction", "speed", "u", "v"]:
        if col not in df.columns:
            df[col] = np.nan
    df = (
        df.dropna(subset=["pressure"])
          .drop_duplicates(subset=["pressure"])
          .sort_values("pressure", ascending=False)
          .reset_index(drop=True)
    )
    return df


# ---------------------------------------------------------------------------
# 3. Compute model features
# ---------------------------------------------------------------------------
def _layer_mean(df, col, p_top, p_bot):
    """Simple mean of `col` for levels within [p_top, p_bot] hPa (inclusive)."""
    m = (df["pressure"] <= p_bot) & (df["pressure"] >= p_top)
    vals = df.loc[m, col].dropna()
    return float(vals.mean()) if len(vals) else np.nan


def compute_indices(df: pd.DataFrame) -> dict:
    """
    Compute the four model features from a sounding DataFrame.

    Returns a dict:
        thompson_index, wind_average, rh, pwat_mm
    plus the intermediate k_index and lifted_index for display/debug.
    Missing pieces come back as NaN rather than raising, so the UI can
    decide what to do.
    """
    out = {
        "k_index": np.nan, "lifted_index": np.nan, "thompson_index": np.nan,
        "wind_average": np.nan, "rh": np.nan, "pwat_mm": np.nan,
    }
    if df is None or df.empty:
        return out

    p = df["pressure"].to_numpy() * units.hPa
    T = df["temperature"].to_numpy() * units.degC
    Td = df["dewpoint"].to_numpy() * units.degC

    # --- K-Index ---
    try:
        out["k_index"] = float(mpcalc.k_index(p, T, Td).m)
    except Exception:
        pass

    # --- Lifted Index (lift a surface parcel) ---
    try:
        prof = mpcalc.parcel_profile(p, T[0], Td[0]).to("degC")
        li = mpcalc.lifted_index(p, T, prof)
        out["lifted_index"] = float(np.atleast_1d(li.m)[0])
    except Exception:
        pass

    if not np.isnan(out["k_index"]) and not np.isnan(out["lifted_index"]):
        out["thompson_index"] = out["k_index"] - out["lifted_index"]

    # --- 700-500 mb mean RH ---
    try:
        rh_profile = mpcalc.relative_humidity_from_dewpoint(T, Td).to("percent").m
        tmp = df.copy()
        tmp["_rh"] = rh_profile
        out["rh"] = _layer_mean(tmp, "_rh", p_top=500, p_bot=700)
    except Exception:
        pass

    # --- 1000-700 mb wind ---
    if USE_VECTOR_MEAN_U:
        # TRUE vector-mean U (eastward component), averaged over the layer.
        out["wind_average"] = _layer_mean(df, "u", p_top=700, p_bot=1000)
    else:
        # Legacy: mean speed & mean direction recombined.
        mean_spd = _layer_mean(df, "speed", p_top=700, p_bot=1000)
        mean_dir = _layer_mean(df, "direction", p_top=700, p_bot=1000)
        out["wind_average"] = mean_spd * np.cos(np.deg2rad(270 - mean_dir))

    # --- PWAT (mm) ---
    try:
        pw = mpcalc.precipitable_water(p, Td)
        out["pwat_mm"] = float(pw.to("mm").m)
    except Exception:
        pass

    return out
