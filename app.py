import os
import tempfile
import pandas as pd
import streamlit as st

from hennepin_mpls_deal_crawler import run, hennepin_pins_pid_url

st.set_page_config(page_title="Wholesale Property Finder", layout="wide")
st.title("Wholesale Property Finder — Hennepin County MN (+ optional Minneapolis stacking)")

st.caption(
    "Generate a prioritized lead list from Hennepin distress signals and optionally stack Minneapolis datasets. "
    "Each result includes actions: 🏛️ county property page + 📊 comparable analysis."
)

default_city_list = [
    "Minneapolis", "Bloomington", "Brooklyn Park", "Brooklyn Center", "Richfield",
    "Edina", "St. Louis Park", "Plymouth", "Golden Valley", "Eden Prairie",
    "Minnetonka", "Hopkins", "Crystal", "New Hope", "Robbinsdale", "Maple Grove",
]

with st.sidebar:
    st.header("Filters")
    cities = st.multiselect(
        "Cities/Suburbs (Hennepin MUNIC_NM)",
        options=default_city_list,
        default=["Minneapolis"],
    )

    enable_mpls = st.checkbox("Stack Minneapolis Tableau (Vacant/Condemned + Violations)", value=True)
    top_n = st.slider("Top N leads", 100, 5000, 1000, 100)

    st.divider()
    st.header("Display")
    page_size = st.slider("Rows per page", 10, 200, 50, 10)
    show_debug = st.checkbox("Show debug logs", value=False)

run_btn = st.button("Run crawler", type="primary")

if "results_df" not in st.session_state:
    st.session_state.results_df = None

if run_btn:
    logs = []
    def log(msg: str):
        logs.append(msg)
        if show_debug:
            st.write(msg)

    with st.spinner("Running…"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.close()
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

    st.session_state.results_df = df

df = st.session_state.results_df

if df is None:
    st.info("Click **Run crawler** to generate results.")
    st.stop()

if df.empty:
    st.warning("No results returned. Try removing city filters or disabling Minneapolis stacking.")
    st.stop()

# Download
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv_bytes, "ranked_leads.csv", "text/csv")

st.subheader("Results")

# Simple pagination
total = len(df)
total_pages = max(1, (total + page_size - 1) // page_size)
page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
start = (page - 1) * page_size
end = min(total, start + page_size)
slice_df = df.iloc[start:end].copy()

st.caption(f"Showing rows {start+1:,}–{end:,} of {total:,}")

# Render as a Zillow-style list (fast + works great with icons)
for _, row in slice_df.iterrows():
    pid = str(row.get("pid", "")).strip()
    addr = str(row.get("situs_address", "")).strip()
    city = str(row.get("situs_city", "")).strip()
    zipc = str(row.get("situs_zip", "")).strip()
    owner = str(row.get("owner_name", "")).strip()
    score = row.get("score", "")
    notes = str(row.get("score_notes", ""))

    county_url = hennepin_pins_pid_url(pid)

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([6, 2, 1, 1])

        with c1:
            st.markdown(f"### {addr}, {city} {zipc}")
            st.write(f"**PID:** {pid}  •  **Owner:** {owner}")
            st.write(f"**Score:** {score}  •  {notes}")

        with c2:
            st.write(f"**Market Value:** {row.get('market_value_total', '')}")
            st.write(f"**Last Sale:** {row.get('sale_date_raw', '')}")
            st.write(f"**Sale Price:** {row.get('sale_price', '')}")

        with c3:
            # 🏛️ county property page link
            st.link_button("🏛️", county_url, help="Open Hennepin County property search (PID)")

        with c4:
            # 📊 comparable analysis inside Streamlit
            # st.page_link works reliably in multipage Streamlit apps
            st.page_link(
                "pages/1_Comparable_Analysis.py",
                label="📊",
                help="Open comparable analysis",
                query_params={"pid": pid}
            )
