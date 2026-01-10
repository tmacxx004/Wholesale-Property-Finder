import os
import tempfile
import urllib.parse
from datetime import datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from hennepin_mpls_deal_crawler import (
    run,
    hennepin_pins_pid_url,
    get_parcel_by_pid,
    get_comps_for_pid,
    format_pid,
    digits_only,
)

from signals_startribune import (
    NoticeConfig,
    fetch_startribune_foreclosure_notices,
    fetch_startribune_probate_notices,
    merge_notices_into_leads,
    apply_notice_scoring,
)

st.set_page_config(page_title="Wholesale Property Finder", layout="wide")

# -------------------------
# Router via query params
# -------------------------
view = (st.query_params.get("view") or "results").strip().lower()
pid_qp = (st.query_params.get("pid") or "").strip()

# -------------------------
# Sidebar inputs
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

    st.subheader("Extra distress signals (statewide notices)")
    enable_mtg_notice = st.checkbox("Mortgage foreclosure notices (Star Tribune)", value=False)
    enable_probate_notice = st.checkbox("Probate notices (Star Tribune)", value=False)

    notice_limit = st.slider("Notices per source (limit)", 24, 240, 120, 24)
    notice_age = st.slider("Max notice age (days)", 7, 365, 120, 7)

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

if "property_type_filter_results" not in st.session_state:
    st.session_state.property_type_filter_results = []

if "probate_crosscheck_filter_results" not in st.session_state:
    st.session_state.probate_crosscheck_filter_results = []

if "saved_pids" not in st.session_state:
    st.session_state.saved_pids = set()

if "interaction_cache" not in st.session_state:
    st.session_state.interaction_cache = {}

if "js_open_url" not in st.session_state:
    st.session_state.js_open_url = ""
if "js_open_nonce" not in st.session_state:
    st.session_state.js_open_nonce = 0


# -------------------------
# Helpers
# -------------------------
def now_ts() -> str:
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
    q = f"{address}, {city}, {state} {zipc}".strip().strip(",")
    return f"https://www.zillow.com/homes/{urllib.parse.quote_plus(q)}_rb/"


def _enforce_pid_columns(df: pd.DataFrame) -> pd.DataFrame:
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
    if df is None or df.empty:
        return df
    if not selected or "property_type" not in df.columns:
        return df
    allowed = {x.strip().upper() for x in selected if x.strip()}
    d2 = df.copy()
    d2["_pt"] = d2["property_type"].astype(str).str.strip().str.upper()
    d2 = d2[d2["_pt"].isin(allowed)].drop(columns=["_pt"])
    return d2


def _probate_crosscheck_options(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty or "probate_mail_crosscheck_status" not in df.columns:
        return []
    return sorted({str(x).strip() for x in df["probate_mail_crosscheck_status"].tolist() if str(x).strip()})


def _apply_probate_crosscheck_filter(df: pd.DataFrame, selected: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if not selected or "probate_mail_crosscheck_status" not in df.columns:
        return df
    allowed = {x.strip().upper() for x in selected if x.strip()}
    d2 = df.copy()
    d2["_pcs"] = d2["probate_mail_crosscheck_status"].astype(str).str.strip().str.upper()
    d2 = d2[d2["_pcs"].isin(allowed)].drop(columns=["_pcs"])
    return d2


def _header_nav():
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
        st.caption("Sources show as badges. Zillow opens in a new tab without leaving Results; clicks are timestamped.")


def _save_toggle(pid_raw: str) -> bool:
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


def _queue_js_open(url: str):
    st.session_state.js_open_url = url
    st.session_state.js_open_nonce += 1


def _flush_js_open():
    url = st.session_state.get("js_open_url", "")
    nonce = st.session_state.get("js_open_nonce", 0)
    if not url:
        return
    components.html(
        f"""
        <script>
          (function() {{
            const url = {url!r};
            window.open(url, "_blank", "noopener,noreferrer");
          }})();
        </script>
        <div style="display:none">nonce:{nonce}</div>
        """,
        height=0,
    )
    st.session_state.js_open_url = ""


def _compute_source_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    d = df.copy()

    def sources_for_row(r) -> list[str]:
        sources = ["Hennepin County"]

        # Minneapolis signals from your crawler (if present)
        if bool(r.get("mpls_vacant_condemned", False)):
            sources.append("MPLS VBR")
        try:
            if int(r.get("mpls_open_violation_count", 0) or 0) > 0:
                sources.append("MPLS Violations")
        except Exception:
            pass

        # Notice sources
        if bool(r.get("has_mortgage_foreclosure_notice", False)):
            sources.append("Foreclosure Notice")
        if bool(r.get("has_probate_notice", False)):
            sources.append("Probate Notice")

        return sources

    d["sources_found"] = d.apply(sources_for_row, axis=1)
    d["sources_found_text"] = d["sources_found"].map(lambda xs: " | ".join(xs) if isinstance(xs, list) else str(xs))
    return d


def _render_source_badges(sources: list[str]):
    if not sources:
        return
    badge_map = {
        "Hennepin County": "🏛️ Hennepin",
        "MPLS VBR": "🏚️ VBR",
        "MPLS Violations": "🧾 Violations",
        "Foreclosure Notice": "🚨 Foreclosure",
        "Probate Notice": "⚖️ Probate",
    }
    tags = [badge_map.get(s, s) for s in sources]
    st.caption("  •  ".join(tags))


def _filter_by_selected_extra_distress_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    If the user selects ANY 'Extra distress signals' option(s),
    then only keep properties that match at least one selected signal.
      - Mortgage foreclosure notices -> has_mortgage_foreclosure_notice
      - Probate notices -> has_probate_notice
    """
    if df is None or df.empty:
        return df

    masks = []

    if enable_mtg_notice:
        m = df.get("has_mortgage_foreclosure_notice", pd.Series([False] * len(df)))
        masks.append(m.fillna(False).astype(bool))

    if enable_probate_notice:
        m = df.get("has_probate_notice", pd.Series([False] * len(df)))
        masks.append(m.fillna(False).astype(bool))

    # none selected => no restriction
    if not masks:
        return df

    mask = masks[0]
    for m in masks[1:]:
        mask = mask | m

    return df.loc[mask].copy()


def _render_property_card(row: pd.Series, idx_key: str, allow_save: bool = True):
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

    z_clicked = get_interaction(pid_raw, "zillow_clicked_at")
    c_opened = get_interaction(pid_raw, "comps_opened_at")
    saved_at = get_interaction(pid_raw, "saved_at")
    is_saved = pid_raw in st.session_state.saved_pids

    z_url = zillow_search_url(addr, city, "MN", zipc)

    sources = row.get("sources_found", [])
    if isinstance(sources, str):
        sources = [s.strip() for s in sources.split("|") if s.strip()]

    fc_url = str(row.get("foreclosure_notice_url", "")).strip()
    pb_url = str(row.get("probate_notice_url", "")).strip()

    pcs = str(row.get("probate_mail_crosscheck_status", "")).strip()

    with st.container(border=True):
        c1, c2, c3, c4, c5, c6 = st.columns([6, 2, 1, 1, 1, 1])

        with c1:
            st.markdown(f"### {addr}, {city} {zipc}")
            st.write(f"**PID:** {pid_fmt}  •  **Type:** {ptype or '—'}  •  **Owner:** {owner}")
            _render_source_badges(sources)

            if bool(row.get("has_probate_notice", False)):
                st.caption(f"🔍 Probate cross-check (mailing city vs notice address): **{pcs or 'No data'}**")

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

            link_row = []
            if fc_url:
                link_row.append(("Foreclosure source", fc_url))
            if pb_url:
                link_row.append(("Probate source", pb_url))
            if link_row:
                st.caption(" • ".join([f"[{t}]({u})" for t, u in link_row]))

        with c2:
            st.write(f"**Market Value:** {row.get('market_value_total', '')}")
            st.write(f"**Last Sale:** {row.get('sale_date_raw', '')}")
            st.write(f"**Sale Price:** {row.get('sale_price', '')}")

        with c3:
            st.link_button("🏛️", county_url, help="Open Hennepin County property search (PID)")

        with c4:
            if st.button("🔎", key=f"z_{idx_key}", help="Open Zillow (new tab) and log timestamp"):
                log_interaction(pid_raw, "zillow_clicked_at")
                _queue_js_open(z_url)
                st.rerun()

        with c5:
            if st.button("📊", key=f"c_{idx_key}", help="Open comparable analysis (tracked)"):
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


def _apply_notice_sources(df: pd.DataFrame):
    if df is None or df.empty:
        return df

    cfg = NoticeConfig(limit=notice_limit, max_age_days=notice_age, sleep_s=0.2)

    def log(msg: str):
        if show_debug:
            st.write(msg)

    forecl_df = pd.DataFrame()
    probate_df = pd.DataFrame()

    if enable_mtg_notice:
        with st.spinner("Pulling mortgage foreclosure notices (detail pages)…"):
            # ✅ IMPORTANT: pass cities + county filter so we focus on Hennepin-relevant notices
            forecl_df = fetch_startribune_foreclosure_notices(
                cfg,
                cities=cities,
                county_filter="HENNEPIN",
                logger=log,
            )

    if enable_probate_notice:
        with st.spinner("Pulling probate notices (detail pages)…"):
            probate_df = fetch_startribune_probate_notices(cfg, logger=log)

    if (forecl_df is None or forecl_df.empty) and (probate_df is None or probate_df.empty):
        return df

    merged = merge_notices_into_leads(df, forecl_df, probate_df)
    merged = apply_notice_scoring(merged)
    return merged


# -------------------------
# Views
# -------------------------
def render_results():
    st.title("Wholesale Property Finder — Hennepin County MN")
    st.caption("If you select any extra distress signal, results only show properties matching those signals.")

    _header_nav()
    _flush_js_open()

    run_btn = st.button("Run crawler", type="primary")

    if run_btn:
        def log(msg: str):
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
                    property_types=None,  # post-search filter handles property type
                    top_n=top_n,
                    out_csv=tmp.name,
                    logger=log,
                )
                df = pd.read_csv(tmp.name, dtype={"pid": "string", "pid_raw": "string"}, keep_default_na=False)
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

        df = _enforce_pid_columns(df)
        df = _apply_notice_sources(df)
        df = _compute_source_indicators(df)

        # ✅ restrict results if extra distress signals selected
        df = _filter_by_selected_extra_distress_signals(df)

        st.session_state.results_df = df
        st.session_state.property_type_filter_results = []
        st.session_state.probate_crosscheck_filter_results = []

    df = st.session_state.results_df
    if df is None:
        st.info("Click **Run crawler** to generate results.")
        return
    if df.empty:
        st.warning("No results returned with the current selections.")
        st.caption("Tip: If foreclosure/probate signals are selected, results only include matching properties.")
        return

    # -------------------------
    # Post-search filters (static header)
    # -------------------------
    st.markdown("## Post-search Filters")

    pt_options = _property_type_options(df)
    if pt_options:
        st.session_state.property_type_filter_results = [
            x for x in st.session_state.property_type_filter_results if x in pt_options
        ]
        st.session_state.property_type_filter_results = st.multiselect(
            "Property type (values from this search only)",
            options=pt_options,
            default=st.session_state.property_type_filter_results,
        )

    pcs_options = _probate_crosscheck_options(df)
    if pcs_options:
        st.session_state.probate_crosscheck_filter_results = [
            x for x in st.session_state.probate_crosscheck_filter_results if x in pcs_options
        ]
        st.session_state.probate_crosscheck_filter_results = st.multiselect(
            "Cross-check decedent vs owner mailing address (status from this search only)",
            options=pcs_options,
            default=st.session_state.probate_crosscheck_filter_results,
            help="Match means owner mailing city appears in the probate notice representative address (best available cross-check).",
        )

    df_ui = df
    df_ui = _apply_property_type_filter(df_ui, st.session_state.property_type_filter_results)
    df_ui = _apply_probate_crosscheck_filter(df_ui, st.session_state.probate_crosscheck_filter_results)

    st.divider()

    if df_ui.empty:
        st.warning("No results after applying post-search filters.")
        return

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
    _header_nav()
    _flush_js_open()

    saved = sorted(st.session_state.saved_pids)
    if not saved:
        st.info("Your saved list is empty. Go to Results and click ☆ to save properties.")
        return

    df = st.session_state.results_df
    if df is None or df.empty:
        st.warning("No results are loaded yet. Run the crawler first so the app can display saved details.")
        st.write(saved)
        return

    df = _enforce_pid_columns(df)
    df_saved = df[df["pid_raw"].isin(saved)].copy()

    if df_saved.empty:
        st.warning("None of your saved PIDs are in the currently loaded results.")
        st.write(saved)
        return

    st.download_button(
        "Download CSV (saved)",
        df_saved.to_csv(index=False).encode("utf-8"),
        "saved_properties.csv",
        "text/csv",
    )

    st.subheader(f"Saved ({len(df_saved):,})")

    total = len(df_saved)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = st.number_input("Page (Saved)", min_value=1, max_value=total_pages, value=1, step=1)
    start = (page - 1) * page_size
    end = min(total, start + page_size)
    slice_df = df_saved.iloc[start:end].copy()

    st.caption(f"Showing rows {start+1:,}–{end:,} of {total:,}")

    for i, row in slice_df.iterrows():
        _render_property_card(row, idx_key=f"sv_{i}", allow_save=True)


def render_comps(pid_any: str):
    pid_raw = digits_only(pid_any)
    pid_fmt = format_pid(pid_raw)

    st.title("Comparable Analysis (County + Zillow link)")
    _header_nav()

    colA, colB, _ = st.columns([2, 2, 8])
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
    z_url = zillow_search_url(addr, city, "MN", zipc)

    st.subheader(f"{addr}, {city} {zipc}")
    st.write(f"**PID:** {pid_fmt}")

    if st.button("🔎 Open Zillow (tracked, new tab)"):
        log_interaction(pid_raw, "zillow_clicked_at")
        _queue_js_open(z_url)
        st.rerun()

    st.divider()
    st.subheader("Nearby County Comps")

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

    st.dataframe(comps_df, use_container_width=True)


# -------------------------
# Route
# -------------------------
if view == "saved":
    render_saved()
elif view == "comps":
    render_comps(pid_qp)
else:
    render_results()
