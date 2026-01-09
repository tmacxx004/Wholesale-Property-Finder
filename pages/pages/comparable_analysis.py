import streamlit as st
import pandas as pd

from hennepin_mpls_deal_crawler import (
    get_parcel_by_pid,
    get_comps_for_pid,
    hennepin_pins_pid_url,
)

st.set_page_config(page_title="Comparable Analysis", layout="wide")

# Prefer query param, fallback to session state
pid = (st.query_params.get("pid") or "").strip()
if not pid:
    pid = str(st.session_state.get("selected_pid", "")).strip()

st.title("Comparable Analysis")
st.caption("Zillow-style snapshot + nearby comparable parcels using Hennepin ArcGIS parcel data.")

if not pid:
    st.warning("No PID provided. Go back to Results and click 📊 on a property.")
    if st.button("⬅️ Back to Results"):
        st.switch_page("app.py")
    st.stop()

# Action bar
colA, colB, colC = st.columns([2, 2, 6])
with colA:
    st.link_button("🏛️ County Property Page", hennepin_pins_pid_url(pid))
with colB:
    if st.button("⬅️ Back to Results"):
        st.switch_page("app.py")

with st.spinner("Loading subject property…"):
    subj = get_parcel_by_pid(pid)

if subj is None:
    st.error("Could not load the subject property from the parcel service.")
    st.stop()

addr = subj.get("situs_address", "")
city = subj.get("situs_city", "")
zipc = subj.get("situs_zip", "")
owner = subj.get("owner_name", "")
ptype = subj.get("property_type", "")
mv = subj.get("market_value_total", None)
sale_date = subj.get("sale_date", "")
sale_price = subj.get("sale_price", None)
homestead = subj.get("homestead_code", "")
years_delq = subj.get("years_delinquent", None)
forfeit = subj.get("forfeited", False)
absentee = subj.get("absentee", False)

st.subheader(f"{addr}, {city} {zipc}")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Market Value", f"${mv:,.0f}" if isinstance(mv, (int, float)) else str(mv))
k2.metric("Last Sale Price", f"${sale_price:,.0f}" if isinstance(sale_price, (int, float)) else str(sale_price))
k3.metric("Last Sale Date", sale_date or "—")
k4.metric("Property Type", ptype or "—")
k5.metric("Owner", owner or "—")

flags = []
if forfeit:
    flags.append("Tax-forfeit indicator")
if years_delq:
    flags.append(f"Delinquent ~{years_delq} yrs")
if absentee:
    flags.append("Absentee owner")
if homestead and str(homestead).upper() != "H":
    flags.append(f"Non-homestead ({homestead})")

if flags:
    st.info(" • ".join(flags))

st.divider()

st.subheader("Comparable Sales / Nearby Parcels (Zillow-style comps)")

c1, c2, c3, c4 = st.columns([2, 2, 2, 4])
with c1:
    radius_m = st.slider("Radius (meters)", 200, 2000, 800, 100)
with c2:
    max_comps = st.slider("Max comps", 5, 50, 15, 1)
with c3:
    value_band = st.slider("Value band (+/- %)", 10, 80, 30, 5)

with st.spinner("Finding comps…"):
    comps_df = get_comps_for_pid(
        pid=pid,
        radius_m=radius_m,
        max_comps=max_comps,
        value_band_pct=value_band,
    )

if comps_df is None or comps_df.empty:
    st.warning("No comps found with current filters. Try increasing radius or value band.")
    st.stop()

vals = comps_df.get("market_value_total")
if vals is not None:
    vals = pd.to_numeric(vals, errors="coerce").dropna()
    if len(vals) >= 3:
        st.success(f"Directional ARV hint (median market value of comps): **${vals.median():,.0f}**")

st.dataframe(comps_df, use_container_width=True)

st.divider()
st.subheader("Next steps (wholesale workflow)")
st.markdown(
    """
- Verify motivation stack: delinquency years, absentee, forfeiture indicators, and any city violations (if stacked).
- Confirm condition: photos, drive-by, neighborhood comps.
- Set offer range: **ARV × (0.70–0.80) − Repairs − Assignment fee** (adjust to your buyer pool).
"""
)
