import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

from prediction import predict_15Z, predict_10Z_updated
from sounding import fetch_sounding, parse_wmo_temp, compute_indices


def _indices_from_df(df, label):
    """Compute indices from a sounding DataFrame and show a summary table."""
    ix = compute_indices(df)
    st.subheader(f"Computed {label} indices")
    st.dataframe(
        pd.DataFrame(
            {
                "K-Index": [ix["k_index"]],
                "Lifted Index": [ix["lifted_index"]],
                "Thompson Index": [ix["thompson_index"]],
                "Wind avg (U, kt)": [ix["wind_average"]],
                "700-500 RH (%)": [ix["rh"]],
                "PWAT (mm)": [ix["pwat_mm"]],
            }
        ).round(2),
        hide_index=True,
    )
    return ix


def _run_10Z(thompson, wind_avg, rh):
    feats = {
        "Thompson_Index": thompson,
        "1000-700mb_Average_U-Wind_Component": wind_avg,
        "700-500mb_Average_RH": rh,
    }
    result = predict_10Z_updated(feats)
    st.header("10Z Output (Version 2.0)")
    st.header(f"{int(result[0])}%")


def _run_15Z(thompson, wind_avg, pwat_mm, rh):
    feats = {
        "Thompson_Index": thompson,
        "1000-700mb_Average_U-Wind_Component": wind_avg,
        "PWAT": pwat_mm,
        "700-500mb_Average_RH": rh,
    }
    result = predict_15Z(feats)
    st.header("15Z Output")
    st.header(f"{int(result[0])}%")


def main():
    st.title("Cape Canaveral Lightning Probability Tool")

    mode = st.radio(
        "How would you like to provide the sounding?",
        ["Manual entry", "Auto-fetch", "Upload WMO TEMP file"],
    )

    # =====================================================================
    # MODE 1: MANUAL ENTRY  (original behaviour preserved)
    # =====================================================================
    if mode == "Manual entry":
        st.header("Sounding Parameters from 10Z")
        col1, col2 = st.columns(2)
        with col1:
            ti = st.number_input("Thompson Index (KI - LI)", -30.0, 60.0,
                                 step=0.1, format="%.1f")
            rh = st.number_input("700-500mb Average RH", 0, 100, step=1)
        with col2:
            wdir = st.number_input("1000-700mb Average Wind Direction",
                                   0, 360, step=1)
            wspd = st.number_input("1000-700mb Average Wind Speed in kts",
                                   0.0, 100.0, step=0.1, format="%.1f")
        if st.button("Probability of Lightning 10Z"):
            wind_avg = wspd * np.cos(np.deg2rad(270 - wdir))
            _run_10Z(ti, wind_avg, rh)

        st.header("Sounding Parameters from 15Z")
        col3, col4 = st.columns(2)
        with col3:
            ti15 = st.number_input("15Z Thompson Index (KI - LI)", -30.0, 60.0,
                                   step=0.1, format="%.1f")
            rh15 = st.number_input("15Z 700-500mb Average RH", 0, 100, step=1)
            pwat = st.number_input("PWAT (inches)", 0.00, 5.00,
                                   step=0.01, format="%.2f")
        with col4:
            wdir15 = st.number_input("15Z 1000-700mb Average Wind Direction",
                                     0, 360, step=1)
            wspd15 = st.number_input("15Z 1000-700mb Average Wind Speed in kts",
                                     0.0, 100.0, step=0.1, format="%.1f")
        if st.button("Probability of Lightning 15Z"):
            wind_avg15 = wspd15 * np.cos(np.deg2rad(270 - wdir15))
            _run_15Z(ti15, wind_avg15, pwat * 25.4, rh15)
        return

    # =====================================================================
# MODE 2: AUTO-FETCH XMR SOUNDING
# =====================================================================
if mode.startswith("Auto-fetch"):
    st.header("Auto-fetch Cape Canaveral XMR sounding")

    current_hour = dt.datetime.utcnow().hour

    if 9 <= current_hour <= 12:
        run = "10Z"
        obs_hour = 10

    elif 14 <= current_hour <= 16:
        run = "15Z"
        obs_hour = 15

    else:
        st.warning(
            "Outside operational model windows. "
            "10Z runs from 09-12Z and 15Z runs from 14-16Z."
        )
        return

    st.info(
        f"Current UTC hour: {current_hour:02d}Z | "
        f"Selected model: {run} | "
        f"Preferred sounding: {obs_hour:02d}Z"
    )

    the_date = st.date_input("Date (UTC)", dt.date.today())

    st.caption(
        f"This will try to fetch the {obs_hour:02d}Z XMR sounding first. "
        "If unavailable, it will fall back to nearby available times."
    )

    if st.button("Fetch and compute"):
        try:
            with st.spinner("Fetching XMR sounding..."):
                df, hour_used = fetch_sounding(
                    the_date.year,
                    the_date.month,
                    the_date.day,
                    obs_hour
                )

            st.success(
                f"Retrieved {hour_used:02d}Z sounding with {len(df)} levels."
            )

            with st.expander("View raw levels"):
                st.dataframe(df, hide_index=True)

            ix = _indices_from_df(
                df,
                f"{run} using {hour_used:02d}Z sounding"
            )

            st.session_state["fetched_ix"] = (run, ix)

        except RuntimeError as exc:
            st.error(str(exc))

    if "fetched_ix" in st.session_state:
        run, ix = st.session_state["fetched_ix"]

        if st.button(f"Probability of Lightning {run}"):
            if run == "10Z":
                _run_10Z(
                    ix["thompson_index"],
                    ix["wind_average"],
                    ix["rh"]
                )
            else:
                _run_15Z(
                    ix["thompson_index"],
                    ix["wind_average"],
                    ix["pwat_mm"],
                    ix["rh"]
                )

    return

    # =====================================================================
    # MODE 3: UPLOAD WMO TEMP FILE
    # =====================================================================
    if mode.startswith("Upload"):
        st.header("Upload a raw WMO TEMP sounding")
        st.caption(
            "Paste or upload the bulletin text (TTAA mandatory levels, and "
            "TTBB significant levels if available — TTBB improves PWAT/RH)."
        )
        run = st.selectbox("Model run", ["10Z", "15Z"], key="upload_run")

        uploaded = st.file_uploader("TEMP file (.txt)", type=["txt", "dat"])
        pasted = st.text_area("...or paste the TEMP message here", height=160)

        if st.button("Decode and compute"):
            text = ""
            if uploaded is not None:
                text = uploaded.read().decode("utf-8", errors="replace")
            elif pasted.strip():
                text = pasted
            if not text.strip():
                st.warning("Provide a file or paste a TEMP message first.")
            else:
                try:
                    df = parse_wmo_temp(text)
                    st.success(f"Decoded {len(df)} levels.")
                    with st.expander("View decoded levels"):
                        st.dataframe(df, hide_index=True)
                    ix = _indices_from_df(df, run)
                    st.session_state["uploaded_ix"] = (run, ix)
                except RuntimeError as exc:
                    st.error(str(exc))

        if "uploaded_ix" in st.session_state:
            run, ix = st.session_state["uploaded_ix"]
            if st.button(f"Probability of Lightning {run}", key="upload_predict"):
                if run == "10Z":
                    _run_10Z(ix["thompson_index"], ix["wind_average"], ix["rh"])
                else:
                    _run_15Z(ix["thompson_index"], ix["wind_average"],
                             ix["pwat_mm"], ix["rh"])
        return


if __name__ == "__main__":
    main()
