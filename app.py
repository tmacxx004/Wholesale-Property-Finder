import os
import tempfile
import urllib.parse
import pandas as pd
import streamlit as st

from hennepin_mpls_deal_crawler import (
    run,
    hennepin_pins_pid_url,
    get_parcel_by_pid,
    get_comps_for_pid,
    format_pid,
    digits_only,
)

st.set_page_config(page_title="Wholesale Property Finder", layout="wide")

# -------------------------
# Router via query params
# -------------------------
view = (st.query_params.get("view") or "results").strip().lower()
pid_qp = (st.query_params.get("pid") or "").strip()

# -------------------------
# Sidebar (search inputs only)
# -------------------------
default_city_list = [
    "Minneapolis", "Bloomington", "Brooklyn Park", "Brooklyn Center", "Richfield",
    "Edina", "St. Louis Park", "Plymouth", "Golden Valley", "Eden Prairie",
    "Minnetonka", "Hopkins", "Crystal", "New Hope", "Robbinsdale", "Maple Grove",
]

with st.sidebar:
    st.header("Search Filters")

    cities = st.multiselect(
        "Cities/Suburbs (Hennepin MUNIC_NM)",
        options=default_city_list,
        default=["Minneapolis"],
    )

    st.subheader("Minneapolis stacking (optional)")
    enable_vbr = st.checkbox("Vacant/Condemned (VBR)", value=True)
    enable_viol = st.checkbox("Violations", value=True)

    top_n = st.slider("Top N leads", 100, 5000, 1000, 100)

    st.divider()
    st.header("Display")
    page_size = st.slider("Rows per page", 10, 200, 50, 10)
    show_debug = st.checkbox("Show debug logs", value=False)

# Persist results + UI filters across reruns
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "property_type_filter" not in st.session_state:
    st.session_state.property_type_filter = []


def zillow_search_url(address: str, city: str, state: str, zipc: str) -> str:
    """Link-out only (no scraping). Opens Zillow search results for the address."""
    q = f"{address}, {city}, {state} {zipc}".strip().strip(",")
    return f"https://www.zillow.com/homes/{urllib.parse.quote_plus(q)}_rb/"


def _enforce_pid_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Force pid_raw digits-only and pid formatted ##-###-##-###-####."""
    if df is None or df.empty:
        return df
    if "pid_raw" in df.columns:
        df["pid_raw"] = df["pid_raw"].astype(str).map(digits_only)
        df["pid"] = df["pid_raw"].map(format_pid)
    return df


def _property_type_options_from_results(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty or "property_type" not in df.columns:
        return []
    opts = sorted({str(x).strip() for x in df["property_type"].tolist() if str(x).strip()})
    return opts


def _apply_property_type_filter(df: pd.DataFrame, selected: list[str]) -> pd.DataFrame:
    """Apply property type filter (case-insensitive exact match)."""
    if df is None or df.empty:
        return df
    if not selected:
        return df
    if "property_type" not in df.columns:
        return df

    allowed = {x.strip().upper() for x in selected if x.strip()}
    d2 = df.copy()
    d2["_pt"] = d2["property_type"].astype(str).str.strip().str.upper()
    d2 = d2[d2["_pt"].isin(allowed)].drop(columns=["_pt"])
    return d2


# -------------------------
# Results view
# -------------------------
def render_results():
    st.title("Wholesale Property Finder — Hennepin County MN (+ optional Minneapolis stacking)")
    st.caption(
        "Generate a prioritized lead list from Hennepin distress signals and optionally stack Minneapolis datasets. "
        "Actions: 🏛️ county page • 🔎 Zillow link • 📊 comps"
    )

    run_btn = st.button("Run crawler", type="primary")

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
                # NOTE: property_types filter is intentionally post-search now
                run(
                    cities=cities if cities else None,
                    enable_vbr=enable_vbr,
                    enable_viol=enable_viol,
                    property_types=None,   # <-- IMPORTANT: do not filter in crawler
                    top_n=top_n,
                    out_csv=tmp.name,
                    logger=log,
                )
                df = pd.read_csv(
                    tmp.name,
                    dtype={"pid": "string", "pid_raw": "string"},
                    keep_default_na=False,
                )
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

        df = _enforce_pid_columns(df)
        st.session_state.results_df = df

        # Reset property type filter when you re-run a new search
        st.session_state.property_type_filter = []

    df = st.session_state.results_df
    if df is None:
        st.info("Click **Run crawler** to generate results.")
        return

    if df.empty:
        st.warning("No results returned. Try changing city or Minneapolis stacking options.")
        return

    # -------------------------
    # Static Property Type filter ABOVE results (post-search only)
    # -------------------------
    st.markdown("## Property Type Filter")

    pt_options = _property_type_options_from_results(df)
    if not pt_options:
        st.caption("No property type field available in results.")
        df_ui = df
    else:
        # Keep any existing selection that still exists in options
        st.session_state.property_type_filter = [
            x for x in st.session_state.property_type_filter if x in pt_options
        ]

        st.session_state.property_type_filter = st.multiselect(
            "Select property types (only values returned in this search)",
            options=pt_options,
            default=st.session_state.property_type_filter,
            help="This filter is applied AFTER the crawler runs. Clear selection to show all types.",
        )

        df_ui = _apply_property_type_filter(df, st.session_state.property_type_filter)

    st.divider()

    if df_ui.empty:
        st.warning("No results after applying the property type filter.")
        return

    # Download filtered view
    csv_bytes = df_ui.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV (filtered)", csv_bytes, "ranked_leads_filtered.csv", "text/csv")

    st.subheader("Results")

    # Pagination
    total = len(df_ui)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start = (page - 1) * page_size
    end = min(total, start + page_size)
    slice_df = df_ui.iloc[start:end].copy()

    st.caption(f"Showing rows {start+1:,}–{end:,} of {total:,}")

    for idx, row in slice_df.iterrows():
        pid_raw = digits_only(str(row.get("pid_raw", "")).strip())
        pid = format_pid(pid_raw)

        addr = str(row.get("situs_address", "")).strip()
        city = str(row.get("situs_city", "")).strip()
        zipc = str(row.get("situs_zip", "")).strip()
        owner = str(row.get("owner_name", "")).strip()
        score = row.get("score", "")
        notes = str(row.get("score_notes", ""))
        ptype = str(row.get("property_type", "")).strip()

        county_url = hennepin_pins_pid_url(pid_raw)
        z_url = zillow_search_url(addr, city, "MN", zipc)

        with st.container(border=True):
            c1, c2, c3, c4, c5 = st.columns([6, 2, 1, 1, 1])

            with c1:
                st.markdown(f"### {addr}, {city} {zipc}")
                st.write(f"**PID:** {pid}  •  **Type:** {ptype or '—'}  •  **Owner:** {owner}")
                st.write(f"**Score:** {score}  •  {notes}")

            with c2:
                st.write(f"**Market Value:** {row.get('market_value_total', '')}")
                st.write(f"**Last Sale:** {row.get('sale_date_raw', '')}")
                st.write(f"**Sale Price:** {row.get('sale_price', '')}")

            with c3:
                st.link_button("🏛️", county_url, help="Open Hennepin County property search (PID)")

            with c4:
                st.link_button("🔎", z_url, help="Open Zillow (search) for this address")

            with c5:
                if st.button("📊", key=f"comp_{pid_raw}_{idx}", help="Open comparable analysis"):
                    st.query_params["view"] = "comps"
                    st.query_params["pid"] = pid_raw
                    st.rerun()


# -------------------------
# Comparable analysis view
# -------------------------
def render_comps(pid_any: str):
    pid_raw = digits_only(pid_any)
    pid_fmt = format_pid(pid_raw)

    st.title("Comparable Analysis (County + Zillow link)")
    st.caption("County parcel snapshot + nearby county comps. Zillow opens via link-out (no scraping).")

    colA, colB, colC = st.columns([2, 2, 6])
    with colA:
        st.link_button("🏛️ County Property Page", hennepin_pins_pid_url(pid_raw))
    with colB:
        if st.button("⬅️ Back to Results"):
            st.query_params.clear()
            st.rerun()

    if not pid_raw:
        st.warning("No PID provided. Go back to Results and click 📊 on a property.")
        return

    with st.spinner("Loading county parcel…"):
        subj = get_parcel_by_pid(pid_raw)

    if subj is None:
        st.error("Could not load the subject property from the parcel service.")
        return

    addr = subj.get("situs_address", "")
    city = subj.get("situs_city", "")
    zipc = subj.get("situs_zip", "")
    owner = subj.get("owner_name", "")
    ptype = subj.get("property_type", "")
    mv = subj.get("market_value_total", None)
    sale_date = subj.get("sale_date", "")
    sale_price = subj.get("sale_price", None)

    z_url = zillow_search_url(addr, city, "MN", zipc)

    st.subheader(f"{addr}, {city} {zipc}")
    st.write(f"**PID:** {pid_fmt}  •  **Type:** {ptype or '—'}  •  **Owner:** {owner or '—'}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("County Market Value", f"${mv:,.0f}" if isinstance(mv, (int, float)) else str(mv))
    k2.metric("Last Sale Price", f"${sale_price:,.0f}" if isinstance(sale_price, (int, float)) else str(sale_price))
    k3.metric("Last Sale Date", sale_date or "—")
    k4.metric("Property Type", ptype or "—")

    st.link_button("Open on Zillow (search)", z_url)

    st.divider()
    st.subheader("Nearby County Comps (cross-check)")

    c1, c2, c3, _ = st.columns([2, 2, 2, 4])
    with c1:
        radius_m = st.slider("Radius (meters)", 200, 2000, 800, 100)
    with c2:
        max_comps = st.slider("Max comps", 5, 50, 15, 1)
    with c3:
        value_band = st.slider("County value band (+/- %)", 10, 80, 30, 5)

    with st.spinner("Finding county comps…"):
        comps_df = get_comps_for_pid(
            pid_any=pid_raw,
            radius_m=radius_m,
            max_comps=max_comps,
            value_band_pct=value_band,
        )

    if comps_df is None or comps_df.empty:
        st.warning("No county comps found with current filters.")
        return

    vals = pd.to_numeric(comps_df.get("market_value_total"), errors="coerce").dropna()
    if len(vals) >= 3:
        st.success(f"Directional ARV hint (median county market value of comps): **${vals.median():,.0f}**")

    st.dataframe(comps_df, use_container_width=True)

    st.divider()
    st.subheader("Next steps (wholesale workflow)")
    st.markdown(
        """
- Verify motivation stack: delinquency years, absentee, forfeiture indicators, and (optional) city signals.
- Confirm condition: drive-by + your buyer pool notes (Zillow link is for viewing only).
- Offer math: **ARV × (0.70–0.80) − Repairs − Assignment fee** (tune to your buyer pool).
"""
    )


# -------------------------
# Route
# -------------------------
if view == "comps":
    render_comps(pid_qp)
else:
    render_results()
