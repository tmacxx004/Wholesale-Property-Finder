import os
import tempfile
import pandas as pd
import streamlit as st

from hennepin_mpls_deal_crawler import (
    run,
    hennepin_pins_pid_url,
    get_parcel_by_pid,
    get_comps_for_pid,
)

# NorthstarMLS RESO Web API client (you added this file)
from northstar_mls_client import NorthstarResoClient

st.set_page_config(page_title="Wholesale Property Finder", layout="wide")

# -------------------------
# Router via query params
# -------------------------
view = (st.query_params.get("view") or "results").strip().lower()
pid_qp = (st.query_params.get("pid") or "").strip()

# -------------------------
# Shared sidebar
# -------------------------
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

# Persist results
if "results_df" not in st.session_state:
    st.session_state.results_df = None


# -------------------------
# Results view
# -------------------------
def render_results():
    st.title("Wholesale Property Finder — Hennepin County MN (+ optional Minneapolis stacking)")
    st.caption(
        "Generate a prioritized lead list from Hennepin distress signals and optionally stack Minneapolis datasets. "
        "Each result includes actions: 🏛️ county property page + 📊 comparable analysis (with NorthstarMLS details if available)."
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
                run(
                    cities=cities if cities else None,
                    enable_mpls=enable_mpls,
                    top_n=top_n,
                    out_csv=tmp.name,
                    logger=log,
                )
                # ✅ Keep PID text formatting
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

        st.session_state.results_df = df

    df = st.session_state.results_df
    if df is None:
        st.info("Click **Run crawler** to generate results.")
        return

    if df.empty:
        st.warning("No results returned. Try removing city filters or disabling Minneapolis stacking.")
        return

    # Download
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv_bytes, "ranked_leads.csv", "text/csv")

    st.subheader("Results")

    # Pagination
    total = len(df)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start = (page - 1) * page_size
    end = min(total, start + page_size)
    slice_df = df.iloc[start:end].copy()

    st.caption(f"Showing rows {start+1:,}–{end:,} of {total:,}")

    for idx, row in slice_df.iterrows():
        pid = str(row.get("pid", "")).strip()          # formatted ##-###-##-###-####
        pid_raw = str(row.get("pid_raw", "")).strip()  # digits-only
        addr = str(row.get("situs_address", "")).strip()
        city = str(row.get("situs_city", "")).strip()
        zipc = str(row.get("situs_zip", "")).strip()
        owner = str(row.get("owner_name", "")).strip()
        score = row.get("score", "")
        notes = str(row.get("score_notes", ""))

        county_url = hennepin_pins_pid_url(pid_raw or pid)

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
                st.link_button("🏛️", county_url, help="Open Hennepin County property search (PID)")

            with c4:
                if st.button("📊", key=f"comp_{pid}_{idx}", help="Open comparable analysis"):
                    st.query_params["view"] = "comps"
                    # Pass raw digits for best reliability with ArcGIS/MLS
                    st.query_params["pid"] = pid_raw or pid
                    st.rerun()


# -------------------------
# Comparable analysis view (County + NorthstarMLS)
# -------------------------
def render_comps(pid_any: str):
    st.title("Comparable Analysis (County + NorthstarMLS)")
    st.caption(
        "County parcel snapshot + NorthstarMLS listing details and photos (requires valid Northstar credentials/permissions)."
    )

    colA, colB, colC = st.columns([2, 2, 6])
    with colA:
        st.link_button("🏛️ County Property Page", hennepin_pins_pid_url(pid_any))
    with colB:
        if st.button("⬅️ Back to Results"):
            st.query_params.clear()
            st.rerun()

    if not pid_any:
        st.warning("No PID provided. Go back to Results and click 📊 on a property.")
        return

    # -------------------------
    # County parcel (subject)
    # -------------------------
    with st.spinner("Loading county parcel…"):
        subj = get_parcel_by_pid(pid_any)

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
    homestead = subj.get("homestead_code", "")
    years_delq = subj.get("years_delinquent", None)
    forfeit = subj.get("forfeited", False)
    absentee = subj.get("absentee", False)

    st.subheader(f"{addr}, {city} {zipc}")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("County Market Value", f"${mv:,.0f}" if isinstance(mv, (int, float)) else str(mv))
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

    # -------------------------
    # NorthstarMLS section
    # -------------------------
    st.subheader("NorthstarMLS Listing Details (if matched)")
    st.caption("This section requires NORTHSTAR_CLIENT_ID and NORTHSTAR_CLIENT_SECRET set in your environment/secrets.")

    mls_error = None
    mls_property = None
    mls_media = []

    with st.spinner("Searching NorthstarMLS by address…"):
        try:
            mls = NorthstarResoClient()

            # Search by UnparsedAddress contains + City + PostalCode
            matches = mls.search_property_by_address(
                unparsed_address=addr,
                city=city,
                postal_code=zipc,
                top=5,
            )

            if matches:
                # Basic pick: first match
                candidate = matches[0]
                listing_key = str(candidate.get("ListingKey") or "").strip()

                # Fetch full record (try expand Media)
                if listing_key:
                    mls_property = mls.get_property_by_listing_key(listing_key, expand_media=True)

                    # Pull Media
                    if isinstance(mls_property, dict) and isinstance(mls_property.get("Media"), list):
                        mls_media = mls_property.get("Media") or []
                    else:
                        mls_media = mls.get_media_for_listing_key(listing_key, top=200)
                else:
                    # If ListingKey missing for some reason, still show candidate
                    mls_property = candidate

        except Exception as e:
            mls_error = str(e)

    if mls_error:
        st.warning(
            "NorthstarMLS data is not available right now. This typically means credentials/permissions are missing "
            "or the API access is not enabled for your account."
        )
        st.code(mls_error)

    if not mls_property:
        st.info("No NorthstarMLS listing matched this address (or it’s not available via your permissions).")
    else:
        # Photos
        photo_urls = []
        for m in (mls_media or []):
            u = m.get("MediaURL") or m.get("MediaUrl") or m.get("MediaUri")
            if u:
                photo_urls.append(u)

        if photo_urls:
            st.write("### Photos")
            # Render up to 30 photos
            st.image(photo_urls[:30], use_container_width=True)
        else:
            st.info("No MediaURL values were returned for this listing (or media is not exposed to your feed).")

        # Curated quick facts (Zillow-ish)
        def g(*keys):
            for k in keys:
                if k in mls_property and mls_property[k] not in (None, ""):
                    return mls_property[k]
            return None

        st.write("### Key MLS Fields (curated)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("List Price", g("ListPrice", "OriginalListPrice") or "—")
        c2.metric("Beds", g("BedroomsTotal", "BedsTotal") or "—")
        c3.metric("Baths", g("BathroomsTotalInteger", "BathroomsTotal") or "—")
        c4.metric("Sq Ft", g("LivingArea", "LivingAreaTotal") or "—")

        st.write(
            f"**Status:** {g('StandardStatus','MlsStatus') or '—'}  •  "
            f"**DOM:** {g('DaysOnMarket','DaysOnMarketCumulative') or '—'}  •  "
            f"**Year Built:** {g('YearBuilt') or '—'}"
        )

        # Full raw payload (as you requested: include ALL MLS info they provide)
        st.write("### All MLS Fields (raw)")
        st.json(mls_property)

        if mls_media:
            st.write("### All MLS Media Records (raw)")
            st.json(mls_media)

    st.divider()

    # -------------------------
    # County comps (fallback / cross-check)
    # -------------------------
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
            pid_any=pid_any,
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
- Validate motivation: delinquency years, absentee, forfeiture indicators, and any city violations (if stacked).
- Validate condition: MLS photos, drive-by, and neighborhood comps.
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
