import os
import tempfile
import pandas as pd
import streamlit as st

from hennepin_mpls_deal_crawler import run

st.set_page_config(page_title="Wholesale Property Finder (Hennepin + MPLS)", layout="wide")
st.title("Wholesale Property Finder — Hennepin County MN (+ optional Minneapolis stacking)")

st.caption(
    "Build a prioritized call list from Hennepin parcel distress signals (delinquency/forfeit) "
    "and optionally stack Minneapolis Tableau datasets (Vacant/Condemned + Violations)."
)

# City/suburb filter
default_city_list = [
    "Minneapolis", "Bloomington", "Brooklyn Park", "Brooklyn Center", "Richfield",
    "Edina", "St. Louis Park", "Plymouth", "Golden Valley", "Eden Prairie",
    "Minnetonka", "Hopkins", "Crystal", "New Hope", "Robbinsdale", "Maple Grove",
]

cities = st.multiselect(
    "Filter to specific cities/suburbs (Hennepin MUNIC_NM field)",
    options=default_city_list,
    default=["Minneapolis"],
)

enable_mpls = st.checkbox("Stack Minneapolis Tableau data (Vacant/Condemned + Violations)", value=True)

top_n = st.slider("Top N leads to export", min_value=100, max_value=5000, value=1000, step=100)

# Helpful controls
st.markdown("### Run")
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    run_btn = st.button("Run crawler", type="primary")

with col2:
    show_debug = st.checkbox("Show debug logs", value=False)

with col3:
    st.info(
        "If Minneapolis Tableau extraction fails (403/format change), the app will still return "
        "a Hennepin distress list and show a warning."
    )

if run_btn:
    with st.spinner("Running… this can take a few minutes depending on filters."):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.close()

        logs = []
        def log(msg: str):
            logs.append(msg)
            if show_debug:
                st.write(msg)

        try:
            run(
                cities=cities if cities else None,
                enable_mpls=enable_mpls,
                top_n=top_n,
                out_csv=tmp.name,
                logger=log,
            )
            df = pd.read_csv(tmp.name)
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

    if df.empty:
        st.warning("No results returned. Try removing city filters or disabling Minneapolis stacking.")
    else:
        st.success(f"Done. Rows: {len(df):,}")
        st.dataframe(df, use_container_width=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv_bytes, "ranked_leads.csv", "text/csv")
