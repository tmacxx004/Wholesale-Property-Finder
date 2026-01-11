import os
import re
import tempfile
import urllib.parse
import hashlib
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

view = (st.query_params.get("view") or "results").strip().lower()
pid_qp = (st.query_params.get("pid") or "").strip()

default_city_list = [
    "Minneapolis", "Bloomington", "Brooklyn Park", "Brooklyn Center", "Richfield",
    "Edina", "St. Louis Park", "Plymouth", "Golden Valley", "Eden Prairie",
    "Minnetonka", "Hopkins", "Crystal", "New Hope", "Robbinsdale", "Maple Grove",
]

# -------------------------
# Sidebar
# -------------------------
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

    st.subheader("Extra distress signals (Star Tribune)")
    enable_mtg_notice = st.checkbox("Mortgage foreclosure notices", value=False)
    enable_probate_notice = st.checkbox("Probate notices", value=False)

    notice_limit = st.slider("Notices per source (limit)", 24, 240, 120, 24)
    notice_age = st.slider("Max notice age (days)", 7, 365, 120, 7)

    st.subheader("Fuzzy Address Match")
    fuzzy_cutoff = st.slider(
        "Fuzzy match threshold (0-100)",
        70, 99, 92, 1,
        help="Used when PID isn't available or exact address match fails. Higher = stricter.",
    )

    top_n = st.slider("Top N Hennepin leads", 100, 5000, 1000, 100)

    st.divider()
    st.header("Display")
    page_size = st.slider("Rows per page", 10, 200, 50, 10)
    show_debug = st.checkbox("Show debug logs", value=False)

# -------------------------
# Session State
# -------------------------
if "results_df" not in st.session_state:
    st.session_state.results_df = None

if "property_type_filter_results" not in st.session_state:
    st.session_state.property_type_filter_results = []

if "probate_crosscheck_filter_results" not in st.session_state:
    st.session_state.probate_crosscheck_filter_results = []

if "saved_pids" not in st.session_state:
    st.session_state.saved_pids = set()

# interaction_cache[entity_key] = {"zillow_clicked_at": "...", ...}
if "interaction_cache" not in st.session_state:
    st.session_state.interaction_cache = {}

# JS open queue
if "js_open_url" not in st.session_state:
    st.session_state.js_open_url = ""
if "js_open_nonce" not in st.session_state:
    st.session_state.js_open_nonce = 0


# -------------------------
# Helpers
# -------------------------
def now_ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def stable_key_for_row(row: pd.Series) -> str:
    """
    Prefer PID-based key. If PID is absent (notice-only rows),
    use a stable hash of (source_url + address).
    """
    pid_raw = digits_only(str(row.get("pid_raw", "")).strip())
    if pid_raw:
        return f"pid:{pid_raw}"

    addr = str(row.get("situs_address", "")).strip().upper()
    fc = str(row.get("foreclosure_notice_url", "")).strip()
    pb = str(row.get("probate_notice_url", "")).strip()
    src = fc or pb or str(row.get("source_url", "")).strip()
    raw = (src + "|" + addr)[:500]
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"notice:{h}"


def log_interaction(entity_key: str, field: str):
    if not entity_key:
        return
    cache = st.session_state.interaction_cache
    if entity_key not in cache:
        cache[entity_key] = {}
    cache[entity_key][field] = now_ts()


def get_interaction(entity_key: str, field: str) -> str:
    return (st.session_state.interaction_cache.get(entity_key, {}) or {}).get(field, "")


def zillow_search_url(address: str, city: str, state: str, zipc: str) -> str:
    q = f"{address}, {city}, {state} {zipc}".strip().strip(",")
    return f"https://www.zillow.com/homes/{urllib.parse.quote_plus(q)}_rb/"


def _queue_js_open(url: str):
    if not url:
        return
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


def open_url_icon_button(
    *,
    icon: str,
    help_text: str,
    url: str,
    entity_key: str,
    audit_field: str,
    key: str,
):
    """
    Clickable icon that:
      - logs audit timestamp
      - opens url in new tab
      - keeps user on current page
    """
    if st.button(icon, key=key, help=help_text):
        log_interaction(entity_key, audit_field)
        _queue_js_open(url)
        st.rerun()


def money_fmt(x) -> str:
    try:
        v = int(pd.to_numeric(x, errors="coerce") or 0)
        if v <= 0:
            return "—"
        return f"${v:,}"
    except Exception:
        return "—"


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
        st.caption("📰 Source icon opens Star Tribune notice (foreclosure preferred, else probate).")


def _save_toggle(pid_raw: str) -> bool:
    pid_raw = digits_only(pid_raw)
    if not pid_raw:
        return False
    if pid_raw in st.session_state.saved_pids:
        st.session_state.saved_pids.remove(pid_raw)
        log_interaction(f"pid:{pid_raw}", "unsaved_at")
        return False
    else:
        st.session_state.saved_pids.add(pid_raw)
        log_interaction(f"pid:{pid_raw}", "saved_at")
        return True


def _compute_source_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    d = df.copy()

    def sources_for_row(r) -> list[str]:
        sources = []
        if bool(r.get("is_notice_only", False)):
            sources.append("StarTribune")
        else:
            sources.append("Hennepin County")

        if bool(r.get("mpls_vacant_condemned", False)):
            sources.append("MPLS VBR")
        try:
            if int(r.get("mpls_open_violation_count", 0) or 0) > 0:
                sources.append("MPLS Violations")
        except Exception:
            pass

        if bool(r.get("has_mortgage_foreclosure_notice", False)):
            sources.append("Foreclosure Notice")

        if str(r.get("foreclosure_match_method", "")).strip().lower() == "fuzzy_address":
            sources.append("Fuzzy Addr Match")

        if bool(r.get("has_probate_notice", False)):
            sources.append("Probate Notice")

        return sources

    d["sources_found"] = d.apply(sources_for_row, axis=1)
    return d


def _render_source_badges(sources: list[str]):
    if not sources:
        return
    badge_map = {
        "Hennepin County": "🏛️ Hennepin",
        "StarTribune": "📰 StarTribune",
        "MPLS VBR": "🏚️ VBR",
        "MPLS Violations": "🧾 Violations",
        "Foreclosure Notice": "🚨 Foreclosure",
        "Probate Notice": "⚖️ Probate",
        "Fuzzy Addr Match": "🧩 Fuzzy",
    }
    tags = [badge_map.get(s, s) for s in sources]
    st.caption("  •  ".join(tags))


def _filter_by_selected_extra_distress_signals_OR(df: pd.DataFrame) -> pd.DataFrame:
    """
    OR logic:
      - if foreclosure selected -> keep foreclosure rows
      - if probate selected -> keep probate rows
      - if both selected -> keep (foreclosure OR probate)
      - if neither selected -> no restriction
    """
    if df is None or df.empty:
        return df

    masks = []
    if enable_mtg_notice:
        masks.append(df.get("has_mortgage_foreclosure_notice", pd.Series([False] * len(df))).fillna(False).astype(bool))
    if enable_probate_notice:
        masks.append(df.get("has_probate_notice", pd.Series([False] * len(df))).fillna(False).astype(bool))

    if not masks:
        return df

    m = masks[0]
    for mm in masks[1:]:
        m = m | mm
    return df.loc[m].copy()


def _parse_city_zip_from_address(address_text: str):
    t = (address_text or "").strip()
    m = re.search(r",\s*([A-Za-z .'-]+)\s*,\s*MN\s+(\d{5})", t, re.I)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = re.search(r"\b([A-Za-z .'-]+)\s+MN\s+(\d{5})\b", t, re.I)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", ""


def _make_notice_only_rows(forecl_df: pd.DataFrame, probate_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if forecl_df is not None and not forecl_df.empty:
        for _, r in forecl_df.iterrows():
            addr = str(r.get("address_text", "")).strip()
            city, zipc = _parse_city_zip_from_address(addr)
            rows.append(
                {
                    "pid_raw": "",
                    "pid": "",
                    "situs_address": addr,
                    "situs_city": city,
                    "situs_zip": zipc,
                    "owner_name": "",
                    "property_type": "",
                    "market_value_total": "",
                    "sale_date_raw": "",
                    "sale_price": "",
                    "score": 35,
                    "score_notes": "Mortgage foreclosure notice (city match)",
                    "is_notice_only": True,
                    "has_mortgage_foreclosure_notice": True,
                    "has_probate_notice": False,
                    "foreclosure_sale_date": str(r.get("event_date", "") or ""),
                    "foreclosure_notice_url": str(r.get("source_url", "") or ""),
                    "foreclosure_notice_details": str(r.get("signal_details", "") or ""),
                    "foreclosure_match_method": "",
                    "foreclosure_fuzzy_score": 0,
                    # mortgage amount (if your signals_startribune provides it; otherwise stays 0)
                    "foreclosure_mortgage_amount": int(pd.to_numeric(r.get("mortgage_amount", 0), errors="coerce") or 0),
                    "probate_notice_url": "",
                    "probate_mail_crosscheck_status": "No data",
                }
            )

    if probate_df is not None and not probate_df.empty:
        for _, r in probate_df.iterrows():
            dec = str(r.get("decedent_name", "")).strip()
            rep_addr = str(r.get("rep_address_text", "")).strip()
            city, zipc = _parse_city_zip_from_address(rep_addr)
            rows.append(
                {
                    "pid_raw": "",
                    "pid": "",
                    "situs_address": rep_addr or "(Probate notice – representative address not found)",
                    "situs_city": city,
                    "situs_zip": zipc,
                    "owner_name": dec,
                    "property_type": "Probate Notice",
                    "market_value_total": "",
                    "sale_date_raw": "",
                    "sale_price": "",
                    "score": 25,
                    "score_notes": "Probate notice (city match)",
                    "is_notice_only": True,
                    "has_mortgage_foreclosure_notice": False,
                    "has_probate_notice": True,
                    "probate_notice_url": str(r.get("source_url", "") or ""),
                    "probate_notice_details": str(r.get("signal_details", "") or ""),
                    "foreclosure_notice_url": "",
                    "probate_mail_crosscheck_status": "No data",
                    "foreclosure_mortgage_amount": 0,
                }
            )

    return pd.DataFrame(rows)


def _apply_notice_sources_and_union(base_df: pd.DataFrame):
    cfg = NoticeConfig(limit=notice_limit, max_age_days=notice_age, sleep_s=0.2)

    def log(msg: str):
        if show_debug:
            st.write(msg)

    forecl_df = pd.DataFrame()
    probate_df = pd.DataFrame()

    if enable_mtg_notice:
        with st.spinner("Pulling mortgage foreclosure notices (city match)…"):
            forecl_df = fetch_startribune_foreclosure_notices(cfg, cities=cities, logger=log)

    if enable_probate_notice:
        with st.spinner("Pulling probate notices (city match)…"):
            probate_df = fetch_startribune_probate_notices(cfg, cities=cities, logger=log)

    enriched = base_df
    if base_df is not None and not base_df.empty and ((not forecl_df.empty) or (not probate_df.empty)):
        # Backwards-compatible call
        try:
            enriched = merge_notices_into_leads(
                base_df,
                forecl_df,
                probate_df,
                fuzzy_score_cutoff=int(fuzzy_cutoff),
            )
        except TypeError:
            enriched = merge_notices_into_leads(base_df, forecl_df, probate_df)

        enriched = apply_notice_scoring(enriched)

    notice_only = _make_notice_only_rows(forecl_df, probate_df)

    if enriched is None or enriched.empty:
        combined = notice_only
    elif notice_only is None or notice_only.empty:
        combined = enriched
    else:
        combined = pd.concat([enriched, notice_only], ignore_index=True)

    # ensure columns exist
    if "foreclosure_mortgage_amount" not in combined.columns:
        combined["foreclosure_mortgage_amount"] = 0

    return combined


def _render_property_card(row: pd.Series, idx_key: str, allow_save: bool = True):
    pid_raw = digits_only(str(row.get("pid_raw", "")).strip())
    pid_fmt = format_pid(pid_raw) if pid_raw else "—"
    is_notice_only = bool(row.get("is_notice_only", False))

    addr = str(row.get("situs_address", "")).strip()
    city = str(row.get("situs_city", "")).strip()
    zipc = str(row.get("situs_zip", "")).strip()
    owner = str(row.get("owner_name", "")).strip()
    score = row.get("score", "")
    notes = str(row.get("score_notes", ""))
    ptype = str(row.get("property_type", "")).strip()

    sources = row.get("sources_found", [])
    if isinstance(sources, str):
        sources = [s.strip() for s in sources.split("|") if s.strip()]

    entity_key = stable_key_for_row(row)

    county_url = hennepin_pins_pid_url(pid_raw) if pid_raw else ""
    z_url = zillow_search_url(addr, city, "MN", zipc) if addr and city else ""

    # Star Tribune notice URLs
    fc_url = str(row.get("foreclosure_notice_url", "")).strip()
    pb_url = str(row.get("probate_notice_url", "")).strip()

    # Combined source target: foreclosure preferred, else probate
    st_url = fc_url or pb_url
    st_tooltip = "Open Star Tribune notice (audited)"
    if fc_url and pb_url:
        st_tooltip = "Open Star Tribune notice (audited) — foreclosure preferred; probate also available"
    elif fc_url:
        st_tooltip = "Open Star Tribune foreclosure notice (audited)"
    elif pb_url:
        st_tooltip = "Open Star Tribune probate notice (audited)"

    # Mortgage amount (from foreclosure signal merge)
    mort_amt = row.get("foreclosure_mortgage_amount", 0)

    # audits
    z_clicked = get_interaction(entity_key, "zillow_clicked_at")
    c_clicked = get_interaction(entity_key, "county_clicked_at")
    comps_opened = get_interaction(entity_key, "comps_opened_at")
    saved_at = get_interaction(entity_key, "saved_at")
    st_clicked = get_interaction(entity_key, "startribune_source_clicked_at")

    is_saved = (pid_raw in st.session_state.saved_pids) if pid_raw else False

    with st.container(border=True):
        c1, c2, c3, c4, c5, c6, c7 = st.columns([6, 2, 1, 1, 1, 1, 1])

        with c1:
            st.markdown(f"### {addr or '—'}")
            st.write(f"**City:** {city or '—'}  •  **ZIP:** {zipc or '—'}")
            st.write(f"**PID:** {pid_fmt}  •  **Type:** {ptype or '—'}  •  **Owner/Decedent:** {owner or '—'}")
            _render_source_badges(sources)
            st.write(f"**Score:** {score}  •  {notes}")

            # Optional: show raw links for visibility (non-audited)
            if fc_url or pb_url:
                links = []
                if fc_url:
                    links.append(f"[Foreclosure Notice]({fc_url})")
                if pb_url:
                    links.append(f"[Probate Notice]({pb_url})")
                st.caption(" • ".join(links))

            meta = []
            if c_clicked:
                meta.append(f"🏛️ County: {c_clicked}")
            if z_clicked:
                meta.append(f"🔎 Zillow: {z_clicked}")
            if st_clicked:
                meta.append(f"📰 Source: {st_clicked}")
            if comps_opened:
                meta.append(f"📊 Comps: {comps_opened}")
            if saved_at:
                meta.append(f"⭐ Saved: {saved_at}")
            if meta:
                st.caption(" | ".join(meta))

        with c2:
            st.write(f"**Market Value:** {row.get('market_value_total', '')}")
            st.write(f"**Last Sale:** {row.get('sale_date_raw', '')}")
            st.write(f"**Sale Price:** {row.get('sale_price', '')}")
            st.write(f"**Mortgage Amount:** {money_fmt(mort_amt)}")

        with c3:
            if county_url:
                open_url_icon_button(
                    icon="🏛️",
                    help_text="Open Hennepin County property page (audited)",
                    url=county_url,
                    entity_key=entity_key,
                    audit_field="county_clicked_at",
                    key=f"county_{idx_key}",
                )
            else:
                st.caption("—")

        with c4:
            if z_url:
                open_url_icon_button(
                    icon="🔎",
                    help_text="Open Zillow (audited)",
                    url=z_url,
                    entity_key=entity_key,
                    audit_field="zillow_clicked_at",
                    key=f"z_{idx_key}",
                )
            else:
                st.caption("—")

        with c5:
            if st_url:
                open_url_icon_button(
                    icon="📰",
                    help_text=st_tooltip,
                    url=st_url,
                    entity_key=entity_key,
                    audit_field="startribune_source_clicked_at",
                    key=f"st_{idx_key}",
                )
            else:
                st.caption("—")

        with c6:
            if (not is_notice_only) and pid_raw:
                if st.button("📊", key=f"c_{idx_key}", help="Open comparable analysis (audited)"):
                    log_interaction(entity_key, "comps_opened_at")
                    st.query_params.clear()
                    st.query_params["view"] = "comps"
                    st.query_params["pid"] = pid_raw
                    st.rerun()
            else:
                st.caption("—")

        with c7:
            if allow_save and pid_raw:
                label = "★" if is_saved else "☆"
                help_txt = "Unsave" if is_saved else "Save to list"
                if st.button(label, key=f"s_{idx_key}", help=help_txt):
                    _save_toggle(pid_raw)
                    st.rerun()
            else:
                st.caption("—")


def render_results():
    st.title("Wholesale Property Finder — Hennepin + Star Tribune Signals")
    _header_nav()
    _flush_js_open()

    run_btn = st.button("Run crawler", type="primary")

    if run_btn:
        def log(msg: str):
            if show_debug:
                st.write(msg)

        with st.spinner("Running Hennepin crawl…"):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            tmp.close()
            try:
                run(
                    cities=cities if cities else None,
                    enable_vbr=enable_vbr,
                    enable_viol=enable_viol,
                    property_types=None,
                    top_n=top_n,
                    out_csv=tmp.name,
                    logger=log,
                )
                base_df = pd.read_csv(tmp.name, dtype={"pid": "string", "pid_raw": "string"}, keep_default_na=False)
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

        base_df = _enforce_pid_columns(base_df)

        combined = _apply_notice_sources_and_union(base_df)
        combined = _compute_source_indicators(combined)
        combined = _filter_by_selected_extra_distress_signals_OR(combined)

        st.session_state.results_df = combined
        st.session_state.property_type_filter_results = []
        st.session_state.probate_crosscheck_filter_results = []

    df = st.session_state.results_df
    if df is None:
        st.info("Click **Run crawler** to generate results.")
        return
    if df.empty:
        st.warning("No results returned with the current selections.")
        return

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
            "Probate cross-check status (values from this search only)",
            options=pcs_options,
            default=st.session_state.probate_crosscheck_filter_results,
        )

    df_ui = df
    df_ui = _apply_property_type_filter(df_ui, st.session_state.property_type_filter_results)
    df_ui = _apply_probate_crosscheck_filter(df_ui, st.session_state.probate_crosscheck_filter_results)

    st.divider()

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
        st.warning("No results are loaded yet. Run the crawler first.")
        st.write(saved)
        return

    df_saved = df[df.get("pid_raw", "").astype(str).map(digits_only).isin(saved)].copy()
    if df_saved.empty:
        st.warning("None of your saved PIDs are in the currently loaded results.")
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
    _flush_js_open()

    if not pid_raw:
        st.warning("No PID provided. Go back to Results and click 📊 on a county property.")
        return

    entity_key = f"pid:{pid_raw}"

    colA, colB, colC = st.columns([2, 2, 8])
    with colA:
        open_url_icon_button(
            icon="🏛️",
            help_text="Open Hennepin County property page (audited)",
            url=hennepin_pins_pid_url(pid_raw),
            entity_key=entity_key,
            audit_field="county_clicked_at",
            key=f"comps_county_{pid_raw}",
        )
    with colB:
        if st.button("⬅️ Back to Results"):
            st.query_params.clear()
            st.query_params["view"] = "results"
            st.rerun()
    with colC:
        st.caption(f"PID: {pid_fmt}")

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

    open_url_icon_button(
        icon="🔎 Open Zillow",
        help_text="Open Zillow (audited)",
        url=z_url,
        entity_key=entity_key,
        audit_field="zillow_clicked_at",
        key=f"comps_z_{pid_raw}",
    )

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
# Router
# -------------------------
if view == "saved":
    render_saved()
elif view == "comps":
    render_comps(pid_qp)
else:
    render_results()
