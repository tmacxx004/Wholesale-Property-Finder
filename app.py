import os
import tempfile
import urllib.parse
from datetime import datetime

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

# -------------------------
# Session state
# -------------------------
if "results_df" not in st.session_state:
    st.session_state.results_df = None

# Property type filters (post-search)
if "property_type_filter_results" not in st.session_state:
    st.session_state.property_type_filter_results = []
if "property_type_filter_saved" not in st.session_state:
    st.session_state.property_type_filter_saved = []

# Saved list + interaction cache
if "saved_pids" not in st.session_state:
    st.session_state.saved_pids = set()  # set of pid_raw strings (digits-only)

if "interaction_cache" not in st.session_state:
    # pid_raw -> {"zillow_clicked_at": "...", "comps_opened_at": "...", "saved_at": "...", "unsaved_at": "..."}
    st.session_state.interaction_cache = {}


# -------------------------
# Helpers
# -------------------------
def now_ts() -> str:
    # ISO timestamp (seconds) for easy sorting/display
    return datetime.now().isoformat(timespec="seconds")


def log_interaction(pid_raw: str, field: str):
    pid_raw = digits_only(pid_raw)
    if not pid_raw:
        return
    cache = st.session_state.interaction_cache
    if pid_raw not in cache:
        cache[pid_raw] = {}
    cache[pid_raw][field] = now_ts()


def get_interaction(pid_raw: str, field: str) -> str:
    pid_raw = digits_only(pid_raw)
    return (st.session_state.interaction_cache.get(pid_raw, {}) or {}).get(field, "")


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


def _property_type_options(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty or "property_type" not in df.columns:
        return []
    return sorted({str(x).strip() for x in df["property_type"].tolist() if str(x).strip()})


def _apply_property_type_filter(df: pd.DataFrame, selected: list[str]) -> pd.DataFrame:
    """Case-insensitive exact match filter on property_type."""
    if df is None or df.empty:
        return df
    if not selected or "property_type" not in df.columns:
        return df
    allowed = {x.strip().upper() for x in selected if x.strip()}
    d2 = df.copy()
    d2["_pt"] = d2["property_type"].astype(str).str.strip().str.upper()
    d2 = d2[d2["_pt"].isin(allowed)].drop(columns=["_pt"])
    return d2


def _header_nav():
    """Top navigation buttons."""
    left, mid, right = st.columns([2, 2, 6])
    with left:
        if st.button("📋 Results", use_container_width=True):
            st.query_params.clear()
            st.query_params["view"] = "results"
            st.rerun()
    with mid:
        if st.button("⭐ Saved", use_container_width=True):
            st.query_params.clear()
            st.query_params["view"] = "saved"
            st.rerun()
    with right:
        st.caption("Tip: ⭐ saves a property to your list. Zillow clicks route through the app to log timestamps.")


def _save_toggle(pid_raw: str) -> bool:
    """Toggle saved status. Returns new saved status."""
    pid_raw = digits_only(pid_raw)
    if not pid_raw:
        return False
    if pid_raw in st.session_state.saved_pids:
        st.session_state.saved_pids.remove(pid_raw)
        log_interaction(pid_raw, "unsaved_at")
        return False
    else:
        st.session_state.saved_pids.add(pid_raw)
        log_interaction(pid_raw, "saved_at")
        return True


def _render_property_card(row: pd.Series, idx_key: str, allow_save: bool = True):
    """Reusable property card with actions + click tracking + save feature."""
    pid_raw = digits_only(str(row.get("pid_raw", "")).strip())
    pid_fmt = format_pid(pid_raw)

    addr = str(row.get("situs_address", "")).strip()
    city = str(row.get("situs_city", "")).strip()
    zipc = str(row.get("situs_zip", "")).strip()
    owner = str(row.get("owner_name", "")).strip()
    score = row.get("score", "")
    notes = str(row.get("score_notes", ""))
    ptype = str(row.get("property_type", "")).strip()

    county_url = hennepin_pins_pid_url(pid_raw)

    # Cached interaction timestamps
    z_clicked = get_interaction(pid_raw, "zillow_clicked_at")
    c_opened = get_interaction(pid_raw, "comps_opened_at")
    saved_at = get_interaction(pid_raw, "saved_at")

    is_saved = pid_raw in st.session_state.saved_pids

    with st.container(border=True):
        c1, c2, c3, c4, c5, c6 = st.columns([6, 2, 1, 1, 1, 1])

        with c1:
            st.markdown(f"### {addr}, {city} {zipc}")
            st.write(f"**PID:** {pid_fmt}  •  **Type:** {ptype or '—'}  •  **Owner:** {owner}")
            st.write(f"**Score:** {score}  •  {notes}")

            meta = []
            if z_clicked:
                meta.append(f"🔎 Zillow clicked: {z_clicked}")
            if c_opened:
                meta.append(f"📊 Comps opened: {c_opened}")
            if saved_at:
                meta.append(f"⭐ Saved: {saved_at}")
            if meta:
                st.caption(" | ".join(meta))

        with c2:
            st.write(f"**Market Value:** {row.get('market_value_total', '')}")
            st.write(f"**Last Sale:** {row.get('sale_date_raw', '')}")
            st.write(f"**Sale Price:** {row.get('sale_price', '')}")

        with c3:
            st.link_button("🏛️", county_url, help="Open Hennepin County property search (PID)")

        with c4:
            # Zillow click is routed through internal view to log timestamp
            if st.button("🔎", key=f"z_{idx_key}", help="Open Zillow (via app) and log click timestamp"):
                st.query_params.clear()
                st.query_params["view"] = "out"
                st.query_params["pid"] = pid_raw
                st.query_params["dest"] = "zillow"
                st.rerun()

        with c5:
            if st.button("📊", key=f"c_{idx_key}", help="Open comparable analysis and log open timestamp"):
                log_interaction(pid_raw, "comps_opened_at")
                st.query_params.clear()
                st.query_params["view"] = "comps"
                st.query_params["pid"] = pid_raw
                st.rerun()

        with c6:
            if allow_save:
                label = "★" if is_saved else "☆"
                help_txt = "Unsave" if is_saved else "Save to list"
                if st.button(label, key=f"s_{idx_key}", help=help_txt):
                    _save_toggle(pid_raw)
                    st.rerun()


# -------------------------
# Views
# -------------------------
def render_results():
    st.title("Wholesale Property Finder — Hennepin County MN")
    st.caption("Actions: 🏛️ county page • 🔎 Zillow link (tracked) • 📊 comps (tracked) • ☆/★ save")

    _header_nav()

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
                run(
                    cities=cities if cities else None,
                    enable_vbr=enable_vbr,
                    enable_viol=enable_viol,
                    property_types=None,  # filter happens after search now
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

        # Reset results property type filter when you run a new search
        st.session_state.property_type_filter_results = []

    df = st.session_state.results_df
    if df is None:
        st.info("Click **Run crawler** to generate results.")
        return

    if df.empty:
        st.warning("No results returned. Try changing city or Minneapolis stacking options.")
        return

    # Static property type filter above results (limited to returned values)
    st.markdown("## Property Type Filter")
    pt_options = _property_type_options(df)

    if pt_options:
        # keep selections that still exist
        st.session_state.property_type_filter_results = [
            x for x in st.session_state.property_type_filter_results if x in pt_options
        ]
        st.session_state.property_type_filter_results = st.multiselect(
            "Filter results by property type (values from this search only)",
            options=pt_options,
            default=st.session_state.property_type_filter_results,
        )
        df_ui = _apply_property_type_filter(df, st.session_state.property_type_filter_results)
    else:
        st.caption("No property_type field available in these results.")
        df_ui = df

    st.divider()

    if df_ui.empty:
        st.warning("No results after applying the property type filter.")
        return

    # Download filtered view
    st.download_button(
        "Download CSV (filtered)",
        df_ui.to_csv(index=False).encode("utf-8"),
        "ranked_leads_filtered.csv",
        "text/csv",
    )

    st.subheader("Results")

    total = len(df_ui)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start = (page - 1) * page_size
    end = min(total, start + page_size)
    slice_df = df_ui.iloc[start:end].copy()

    st.caption(f"Showing rows {start+1:,}–{end:,} of {total:,}")

    for i, row in slice_df.iterrows():
        _render_property_card(row, idx_key=f"r_{i}", allow_save=True)


def render_saved():
    st.title("⭐ Saved Properties")
    st.caption("Only properties you’ve starred appear here.")

    _header_nav()

    saved = sorted(st.session_state.saved_pids)
    if not saved:
        st.info("Your saved list is empty. Go to Results and click ☆ to save properties.")
        return

    df = st.session_state.results_df
    if df is None or df.empty:
        st.warning("No results are loaded yet. Run the crawler first so the app can display saved details.")
        st.caption("Your saved PIDs are still stored for this session.")
        # Show raw saved list
        st.write(saved)
        return

    df = _enforce_pid_columns(df)
    df_saved = df[df["pid_raw"].isin(saved)].copy()

    if df_saved.empty:
        st.warning("None of your saved PIDs are in the currently loaded results.")
        st.caption("Run a new search that includes those properties, or keep saving new ones from Results.")
        st.write(saved)
        return

    # Property type filter above saved list (limited to values in saved results)
    st.markdown("## Property Type Filter (Saved)")
    pt_options = _property_type_options(df_saved)

    if pt_options:
        st.session_state.property_type_filter_saved = [
            x for x in st.session_state.property_type_filter_saved if x in pt_options
        ]
        st.session_state.property_type_filter_saved = st.multiselect(
            "Filter saved by property type (values from saved list only)",
            options=pt_options,
            default=st.session_state.property_type_filter_saved,
        )
        df_ui = _apply_property_type_filter(df_saved, st.session_state.property_type_filter_saved)
    else:
        df_ui = df_saved

    st.divider()

    st.download_button(
        "Download CSV (saved, filtered)",
        df_ui.to_csv(index=False).encode("utf-8"),
        "saved_properties_filtered.csv",
        "text/csv",
    )

    st.subheader(f"Saved ({len(df_ui):,})")

    # No pagination needed usually, but keep it consistent
    total = len(df_ui)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = st.number_input("Page (Saved)", min_value=1, max_value=total_pages, value=1, step=1)
    start = (page - 1) * page_size
    end = min(total, start + page_size)
    slice_df = df_ui.iloc[start:end].copy()

    st.caption(f"Showing rows {start+1:,}–{end:,} of {total:,}")

    for i, row in slice_df.iterrows():
        _render_property_card(row, idx_key=f"s_{i}", allow_save=True)


def render_out(pid_any: str):
    """
    Internal "click-out" view.
    We route Zillow clicks here so we can log the click timestamp, then present the external link.
    """
    pid_raw = digits_only(pid_any)
    dest = (st.query_params.get("dest") or "zillow").strip().lower()

    st.title("Leaving the app")
    _header_nav()

    if not pid_raw:
        st.error("Missing PID.")
        return

    # Load parcel to build a clean address-based Zillow search link
    with st.spinner("Preparing link…"):
        subj = get_parcel_by_pid(pid_raw)

    if subj is None:
        st.error("Could not load parcel details to build the external link.")
        return

    addr = subj.get("situs_address", "")
    city = subj.get("situs_city", "")
    zipc = subj.get("situs_zip", "")

    if dest == "zillow":
        log_interaction(pid_raw, "zillow_clicked_at")
        url = zillow_search_url(addr, city, "MN", zipc)
        st.success(f"Zillow click logged for PID {format_pid(pid_raw)} at {get_interaction(pid_raw, 'zillow_clicked_at')}.")
        st.link_button("Open Zillow (search)", url)
    else:
        st.warning("Unknown destination.")
        st.write(dest)

    st.divider()
    if st.button("⬅️ Back", type="primary"):
        st.query_params.clear()
        st.query_params["view"] = "results"
        st.rerun()


def render_comps(pid_any: str):
    pid_raw = digits_only(pid_any)
    pid_fmt = format_pid(pid_raw)

    st.title("Comparable Analysis (County + Zillow link)")
    st.caption("County parcel snapshot + nearby county comps. Zillow is link-out only (no scraping).")

    _header_nav()

    colA, colB, colC = st.columns([2, 2, 6])
    with colA:
        st.link_button("🏛️ County Property Page", hennepin_pins_pid_url(pid_raw))
    with colB:
        if st.button("⬅️ Back to Results"):
            st.query_params.clear()
            st.query_params["view"] = "results"
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

    # Show click tracking
    st.caption(
        " | ".join(
            [x for x in [
                f"🔎 Zillow clicked: {get_interaction(pid_raw, 'zillow_clicked_at')}" if get_interaction(pid_raw, "zillow_clicked_at") else "",
                f"📊 Comps opened: {get_interaction(pid_raw, 'comps_opened_at')}" if get_interaction(pid_raw, "comps_opened_at") else "",
                f"⭐ Saved: {get_interaction(pid_raw, 'saved_at')}" if get_interaction(pid_raw, "saved_at") else "",
            ] if x]
        )
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("County Market Value", f"${mv:,.0f}" if isinstance(mv, (int, float)) else str(mv))
    k2.metric("Last Sale Price", f"${sale_price:,.0f}" if isinstance(sale_price, (int, float)) else str(sale_price))
    k3.metric("Last Sale Date", sale_date or "—")
    k4.metric("Property Type", ptype or "—")

    # Zillow click-out via tracked out-view
    colZ1, colZ2 = st.columns([2, 8])
    with colZ1:
        if st.button("🔎 Track + Zillow"):
            st.query_params.clear()
            st.query_params["view"] = "out"
            st.query_params["pid"] = pid_raw
            st.query_params["dest"] = "zillow"
            st.rerun()
    with colZ2:
        st.link_button("Zillow (direct link-out, not tracked)", z_url)

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
if view == "saved":
    render_saved()
elif view == "comps":
    render_comps(pid_qp)
elif view == "out":
    render_out(pid_qp)
else:
    render_results()
