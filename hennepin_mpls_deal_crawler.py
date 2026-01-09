import re
import csv
import json
import time
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import requests
import pandas as pd
from dateutil import parser as dateparser
from rapidfuzz import fuzz


# =========================
# Config
# =========================

USER_AGENT = "HennepinMplsDealCrawler/1.0 (+public data; polite)"
PAGE_SIZE = 2000  # ArcGIS layers often cap around this per page

# ---- Hennepin (ArcGIS REST) ----
# This is the same dataset family described by Hennepin GIS Hub "County Parcels" and metadata containing
# FORFEIT_LAND_IND and EARLIEST_DELQ_YR definitions.  :contentReference[oaicite:5]{index=5}
HENNEPIN_LAYER_QUERY_URL = (
    "https://gis.hennepin.us/arcgis/rest/services/HennepinData/LAND_PROPERTY/MapServer/1/query"
)

HENNEPIN_FIELDS = [
    "PID",
    "OWNER_NM",
    "HOUSE_NO",
    "FRAC_HOUSE_NO",
    "STREET_NM",
    "MUNIC_NM",
    "ZIP_CD",
    "MAILING_MUNIC_NM",
    "SALE_DATE",
    "SALE_PRICE",
    "MKT_VAL_TOT",
    "PR_TYP_NM1",
    "HMSTD_CD1",
    "EARLIEST_DELQ_YR",
    "FORFEIT_LAND_IND",
]

# ---- Minneapolis Tableau dashboards (public pages) ----
# Dashboard landing pages:
# Vacant/Condemned: :contentReference[oaicite:6]{index=6}
# Regulatory Violations: :contentReference[oaicite:7]{index=7}
# Direct Tableau views discovered:
MPLS_VBR_VIEW_URL = "https://tableau.minneapolismn.gov/views/MinneapolisVacantCondemnedPropertyInventory/DailyVBRcondemnedpropertylist"
MPLS_VIOL_VIEW_URL = "https://tableau.minneapolismn.gov/views/OpenDataRegulatoryServices-Violations/ViolationDetails"


# =========================
# Helpers
# =========================

def polite_sleep(sec: float = 0.25):
    time.sleep(sec)

def now_year() -> int:
    return datetime.now().year

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def normalize_addr(a: str) -> str:
    a = normalize_ws(a).upper()
    # very basic normalization (enough for 80/20 matching)
    a = a.replace(".", "").replace(",", "").replace("#", " ")
    a = a.replace(" STREET", " ST").replace(" AVENUE", " AVE").replace(" ROAD", " RD")
    a = a.replace(" DRIVE", " DR").replace(" LANE", " LN").replace(" BOULEVARD", " BLVD")
    a = re.sub(r"\s+", " ", a).strip()
    return a

def parse_2digit_year(x: Any) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if not s.isdigit():
        return None
    yy = int(s)
    cy = now_year() % 100
    return (2000 + yy) if yy <= cy else (1900 + yy)

def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(float(x))
    except Exception:
        return None

def build_situs_address(attrs: Dict[str, Any]) -> str:
    house_no = str(attrs.get("HOUSE_NO") or "").strip()
    frac = normalize_ws(str(attrs.get("FRAC_HOUSE_NO") or "")).strip()
    street = normalize_ws(str(attrs.get("STREET_NM") or "")).strip()
    parts = [p for p in [house_no, frac, street] if p]
    return normalize_ws(" ".join(parts))

def is_absentee(attrs: Dict[str, Any]) -> bool:
    situs = normalize_ws(str(attrs.get("MUNIC_NM") or "")).upper()
    mail = normalize_ws(str(attrs.get("MAILING_MUNIC_NM") or "")).upper()
    if not situs or not mail:
        return False
    return situs != mail

def parse_date_maybe(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return dateparser.parse(s, fuzzy=True).date().isoformat()
    except Exception:
        return None


# =========================
# Data models
# =========================

@dataclass
class ParcelLead:
    pid: str
    owner_name: str
    situs_address: str
    situs_city: str
    situs_zip: str

    mailing_city: str
    property_type: str
    homestead_code: str

    sale_date_raw: str
    sale_price: Optional[int]
    market_value_total: Optional[int]

    earliest_delq_year: Optional[int]
    years_delinquent: Optional[int]
    forfeited: bool
    absentee: bool

    # Minneapolis stack fields
    mpls_vacant_condemned: bool
    mpls_condemned_date: Optional[str]
    mpls_vbr_type: Optional[str]

    mpls_violation_count: int
    mpls_open_violation_count: int

    score: float
    score_notes: str


# =========================
# ArcGIS Client (Hennepin)
# =========================

class ArcGISClient:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": USER_AGENT})

    def query(self, where: str, out_fields: List[str], offset: int, count: int) -> Dict[str, Any]:
        params = {
            "where": where,
            "outFields": ",".join(out_fields),
            "f": "json",
            "resultOffset": offset,
            "resultRecordCount": count,
            "returnGeometry": "false",
        }
        r = self.s.get(HENNEPIN_LAYER_QUERY_URL, params=params, timeout=60)
        r.raise_for_status()
        return r.json()

def iter_hennepin_features(where: str, out_fields: List[str]) -> List[Dict[str, Any]]:
    client = ArcGISClient()
    feats_all = []
    offset = 0
    while True:
        data = client.query(where=where, out_fields=out_fields, offset=offset, count=PAGE_SIZE)
        feats = data.get("features") or []
        feats_all.extend(feats)
        if len(feats) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        polite_sleep(0.2)
    return feats_all


# =========================
# Tableau extraction (Minneapolis)
# =========================

class TableauExtractor:
    """
    Tries two strategies:
      1) Simple CSV export endpoints (fast if enabled)
      2) BootstrapSession method (works on many Tableau servers)

    If strategy 1 fails, strategy 2 usually works.
    """
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": USER_AGENT})

    def try_simple_csv(self, view_url: str) -> Optional[pd.DataFrame]:
        """
        Many Tableau servers allow: /views/WB/Sheet.csv?:showVizHome=no
        """
        csv_url = view_url + ".csv"
        params = {":showVizHome": "no"}
        r = self.s.get(csv_url, params=params, timeout=60)
        if r.status_code != 200 or "text/csv" not in (r.headers.get("Content-Type") or ""):
            return None
        from io import StringIO
        return pd.read_csv(StringIO(r.text))

    def bootstrap_df(self, view_url: str) -> pd.DataFrame:
        """
        Bootstrap session technique:
        - GET the viz page to capture a hidden bootstrap session endpoint
        - POST/GET to bootstrapSession to retrieve data segments
        - Parse the 'presModel' chunks to extract tables

        NOTE: Tableau internal formats vary; we implement a practical extraction that works often,
        but may require small tweaks if Minneapolis changes workbook structure.
        """
        # Step 1: request the viz page (no viz home)
        r = self.s.get(view_url, params={":showVizHome": "no"}, timeout=60)
        r.raise_for_status()
        html = r.text

        # Find bootstrapSession endpoint pieces (common patterns)
        # Example pattern in HTML: bootstrapSession/sessions/{id}
        m = re.search(r'"sessionid":"([^"]+)"', html, re.IGNORECASE)
        if not m:
            # Alternate pattern: /bootstrapSession/sessions/
            m2 = re.search(r'(bootstrapSession/sessions/)', html)
            if not m2:
                raise RuntimeError("Could not locate Tableau session info in HTML. Workbook may require auth or JS-only.")

        # Tableau often provides a 'sheetId' etc; easier approach:
        # Use the documented-ish endpoint:
        #   {view_url}/bootstrapSession/sessions/  (POST)
        # with form data: sheet_id, showParams...
        # We'll do a minimal POST known to work on many servers.
        bootstrap_url = view_url + "/bootstrapSession/sessions/"
        payload = {
            "worksheetPortSize": '{"w":1200,"h":800}',
            "dashboardPortSize": '{"w":1200,"h":800}',
            "clientDimension": '{"w":1200,"h":800}',
            ":showVizHome": "no",
        }
        br = self.s.post(bootstrap_url, data=payload, timeout=60)
        br.raise_for_status()
        txt = br.text

        # The response contains a big JSON-ish blob with a "presModel" section.
        # We'll extract any CSV-like data tables embedded in the response.
        # Practical strategy: locate "dataSegments" and decode tables.
        # This is not a full Tableau parser; it’s an 80/20 extractor.

        # Find the presModel JSON chunk
        idx = txt.find('"presModel"')
        if idx == -1:
            raise RuntimeError("Bootstrap response missing presModel; Tableau format changed or blocked.")

        # Heuristic: find the largest JSON object in the response
        # (Tableau responses sometimes have prefix; we find first '{' after presModel)
        start = txt.rfind("{", 0, idx)
        end = txt.rfind("}")
        blob = txt[start:end+1]

        data = json.loads(blob)

        # Extract tables: navigate down to "presModel" -> "dataDictionary" / "dataSegments"
        pres = data.get("presModel", {})
        segments = pres.get("dataSegments") or {}
        if not segments:
            raise RuntimeError("No dataSegments found in Tableau presModel (may require selecting a worksheet/table).")

        # Each segment is a columnar table; we’ll attempt to stitch the first large segment.
        # This is intentionally simple; for production, you’d pick the segment by worksheet name.
        best_key = None
        best_size = 0
        for k, seg in segments.items():
            # seg often contains "dataColumns"
            cols = seg.get("dataColumns") if isinstance(seg, dict) else None
            if cols and len(cols) > best_size:
                best_key = k
                best_size = len(cols)

        if not best_key:
            raise RuntimeError("Could not identify a usable data segment to parse.")

        seg = segments[best_key]
        cols = seg["dataColumns"]

        # Tableau dataColumns: each col has "dataValues"
        table = {}
        row_count = None
        for i, col in enumerate(cols):
            values = col.get("dataValues") or []
            if row_count is None:
                row_count = len(values)
            name = col.get("fieldCaption") or col.get("caption") or f"col_{i}"
            table[name] = values

        df = pd.DataFrame(table)
        return df

    def extract(self, view_url: str) -> pd.DataFrame:
        df = self.try_simple_csv(view_url)
        if df is not None and not df.empty:
            return df
        # fallback
        return self.bootstrap_df(view_url)


# =========================
# Matching + stacking
# =========================

def match_by_address(target_addr: str, target_city: str, df: pd.DataFrame,
                     addr_col_candidates: List[str], city_col_candidates: List[str],
                     min_score: int = 92) -> Optional[int]:
    """
    Returns row index match (best match) or None.
    """
    t_addr = normalize_addr(target_addr)
    t_city = normalize_ws(target_city).upper()

    addr_col = next((c for c in addr_col_candidates if c in df.columns), None)
    city_col = next((c for c in city_col_candidates if c in df.columns), None)
    if not addr_col:
        return None

    best = (None, 0)
    for idx, row in df.iterrows():
        r_addr = normalize_addr(str(row.get(addr_col) or ""))
        if not r_addr:
            continue

        # quick city filter if available
        if city_col:
            r_city = normalize_ws(str(row.get(city_col) or "")).upper()
            if r_city and t_city and r_city != t_city:
                continue

        sc = fuzz.token_sort_ratio(t_addr, r_addr)
        if sc > best[1]:
            best = (idx, sc)

    return best[0] if best[0] is not None and best[1] >= min_score else None


def score_lead(base: Dict[str, Any], mpls_vacant: bool, condemned_date: Optional[str],
               violation_count: int, open_violation_count: int) -> Tuple[float, List[str]]:
    score = 0.0
    notes = []

    forfeited = (str(base.get("FORFEIT_LAND_IND") or "").upper() == "Y")
    if forfeited:
        score += 60
        notes.append("Forfeited land indicator")

    delq_year = parse_2digit_year(base.get("EARLIEST_DELQ_YR"))
    if delq_year:
        yrs = max(0, now_year() - delq_year)
        if yrs >= 3:
            score += 50
            notes.append(f"Delinquent ~{yrs} yrs (>=3)")
        elif yrs == 2:
            score += 35
            notes.append("Delinquent ~2 yrs")
        elif yrs == 1:
            score += 20
            notes.append("Delinquent ~1 yr")

    if is_absentee(base):
        score += 20
        notes.append("Absentee (mail city != situs city)")

    hmstd = str(base.get("HMSTD_CD1") or "").strip().upper()
    if hmstd and hmstd != "H":
        score += 10
        notes.append(f"Non-homestead code ({hmstd})")

    # Minneapolis stacking boosts
    if mpls_vacant:
        score += 35
        notes.append("Mpls vacant/condemned (VBR dashboard match)")
    if condemned_date:
        score += 10
        notes.append(f"Has condemned date ({condemned_date})")

    if open_violation_count >= 5:
        score += 25
        notes.append(f"Open violations >=5 ({open_violation_count})")
    elif open_violation_count >= 1:
        score += 15
        notes.append(f"Open violations ({open_violation_count})")

    if violation_count >= 10:
        score += 10
        notes.append(f"Total violations >=10 ({violation_count})")

    return score, notes


# =========================
# Build pipeline
# =========================

def build_where_clause(cities: Optional[List[str]]) -> str:
    base = "(EARLIEST_DELQ_YR IS NOT NULL) OR (FORFEIT_LAND_IND = 'Y')"
    if not cities:
        return base
    # ArcGIS SQL: MUNIC_NM IN (...)
    quoted = ",".join([f"'{c.upper()}'" for c in cities])
    return f"({base}) AND (UPPER(MUNIC_NM) IN ({quoted}))"

def fetch_mpls_tables(enable_mpls: bool) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if not enable_mpls:
        return None, None
    t = TableauExtractor()
    # These workbook views come from the Tableau URLs discovered for Minneapolis. :contentReference[oaicite:8]{index=8}
    vbr = t.extract(MPLS_VBR_VIEW_URL)
    polite_sleep(0.5)
    viol = t.extract(MPLS_VIOL_VIEW_URL)
    return vbr, viol

def infer_vbr_columns(df: pd.DataFrame) -> Dict[str, str]:
    # Best-effort guessing; if Minneapolis changes labels, adjust here.
    candidates = {
        "address": ["Address", "ADDRESS", "Property Address", "PROPERTY ADDRESS"],
        "condemned_date": ["Condemned Date", "CONDEMNED DATE", "CON1", "CONB", "Condemned"],
        "vbr_type": ["Type", "VBR Type", "VBR_STATUS", "Status", "VBR Status"],
        "city": ["City", "CITY", "Municipality", "MUNIC_NM"],
    }
    resolved = {}
    for key, opts in candidates.items():
        for c in df.columns:
            if c in opts or c.upper() in [o.upper() for o in opts]:
                resolved[key] = c
                break
    return resolved

def infer_violation_columns(df: pd.DataFrame) -> Dict[str, str]:
    candidates = {
        "address": ["Address", "ADDRESS", "Property Address", "PROPERTY ADDRESS"],
        "resolved": ["Violation Resolved?", "Resolved", "VIOLATION RESOLVED?", "RESOLVED"],
        "city": ["City", "CITY", "Municipality", "MUNIC_NM"],
    }
    resolved = {}
    for key, opts in candidates.items():
        for c in df.columns:
            if c in opts or c.upper() in [o.upper() for o in opts]:
                resolved[key] = c
                break
    return resolved

def stack_mpls(pid_row: Dict[str, Any],
               vbr_df: Optional[pd.DataFrame],
               viol_df: Optional[pd.DataFrame]) -> Tuple[bool, Optional[str], Optional[str], int, int]:
    """
    Returns:
      (vacant_flag, condemned_date, vbr_type, violation_count, open_violation_count)
    """
    situs_addr = build_situs_address(pid_row)
    city = normalize_ws(str(pid_row.get("MUNIC_NM") or ""))

    mpls_vacant = False
    condemned_date = None
    vbr_type = None
    violation_count = 0
    open_violation_count = 0

    # VBR match
    if vbr_df is not None and not vbr_df.empty:
        cols = infer_vbr_columns(vbr_df)
        idx = match_by_address(
            target_addr=situs_addr,
            target_city=city,
            df=vbr_df,
            addr_col_candidates=[cols.get("address","Address"), "Address", "ADDRESS", "Property Address"],
            city_col_candidates=[cols.get("city","City"), "City", "CITY"],
            min_score=92
        )
        if idx is not None:
            mpls_vacant = True
            if "condemned_date" in cols:
                condemned_date = parse_date_maybe(vbr_df.loc[idx, cols["condemned_date"]])
            if "vbr_type" in cols:
                vbr_type = normalize_ws(str(vbr_df.loc[idx, cols["vbr_type"]] or ""))

    # Violations aggregation by address match
    if viol_df is not None and not viol_df.empty:
        cols = infer_violation_columns(viol_df)
        addr_col = cols.get("address") or ("Address" if "Address" in viol_df.columns else None)
        resolved_col = cols.get("resolved")
        city_col = cols.get("city")

        if addr_col:
            t_addr = normalize_addr(situs_addr)
            t_city = normalize_ws(city).upper()
            for _, row in viol_df.iterrows():
                r_addr = normalize_addr(str(row.get(addr_col) or ""))
                if not r_addr:
                    continue
                if city_col:
                    r_city = normalize_ws(str(row.get(city_col) or "")).upper()
                    if r_city and t_city and r_city != t_city:
                        continue
                if fuzz.token_sort_ratio(t_addr, r_addr) >= 92:
                    violation_count += 1
                    if resolved_col:
                        resolved_val = str(row.get(resolved_col) or "").strip().upper()
                        if resolved_val in ("NO", "N", "FALSE", "0", ""):
                            open_violation_count += 1
                    else:
                        # If no resolved column, treat as "open-ish" bucket
                        open_violation_count += 1

    return mpls_vacant, condemned_date, vbr_type, violation_count, open_violation_count

def run(cities: Optional[List[str]], enable_mpls: bool, top_n: int, out_csv: str):
    where = build_where_clause(cities)
    feats = iter_hennepin_features(where=where, out_fields=HENNEPIN_FIELDS)

    vbr_df, viol_df = fetch_mpls_tables(enable_mpls=enable_mpls)

    leads: List[ParcelLead] = []
    for f in feats:
        attrs = f.get("attributes") or {}
        pid = str(attrs.get("PID") or "").strip()
        if not pid:
            continue

        situs_addr = build_situs_address(attrs)
        situs_city = normalize_ws(str(attrs.get("MUNIC_NM") or ""))
        situs_zip = normalize_ws(str(attrs.get("ZIP_CD") or ""))
        owner = normalize_ws(str(attrs.get("OWNER_NM") or ""))

        delq_year = parse_2digit_year(attrs.get("EARLIEST_DELQ_YR"))
        yrs_delq = (now_year() - delq_year) if delq_year else None
        forfeited = (str(attrs.get("FORFEIT_LAND_IND") or "").upper() == "Y")
        absentee = is_absentee(attrs)

        mpls_vacant, condemned_date, vbr_type, vio_cnt, open_vio_cnt = stack_mpls(attrs, vbr_df, viol_df) if enable_mpls else (False, None, None, 0, 0)

        score, notes = score_lead(attrs, mpls_vacant, condemned_date, vio_cnt, open_vio_cnt)

        leads.append(ParcelLead(
            pid=pid,
            owner_name=owner,
            situs_address=situs_addr,
            situs_city=situs_city,
            situs_zip=situs_zip,
            mailing_city=normalize_ws(str(attrs.get("MAILING_MUNIC_NM") or "")),
            property_type=normalize_ws(str(attrs.get("PR_TYP_NM1") or "")),
            homestead_code=normalize_ws(str(attrs.get("HMSTD_CD1") or "")),
            sale_date_raw=normalize_ws(str(attrs.get("SALE_DATE") or "")),
            sale_price=safe_int(attrs.get("SALE_PRICE")),
            market_value_total=safe_int(attrs.get("MKT_VAL_TOT")),
            earliest_delq_year=delq_year,
            years_delinquent=yrs_delq,
            forfeited=forfeited,
            absentee=absentee,
            mpls_vacant_condemned=mpls_vacant,
            mpls_condemned_date=condemned_date,
            mpls_vbr_type=vbr_type,
            mpls_violation_count=vio_cnt,
            mpls_open_violation_count=open_vio_cnt,
            score=score,
            score_notes="; ".join(notes),
        ))

    leads.sort(key=lambda x: x.score, reverse=True)
    rows = [asdict(l) for l in leads[:top_n]]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["pid"])
        w.writeheader()
        w.writerows(rows)

    print(f"Exported {len(rows)} leads -> {out_csv}")


# =========================
# CLI
# =========================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", help="City/suburb filter (match Hennepin MUNIC_NM). Example: Minneapolis Bloomington")
    ap.add_argument("--enable-mpls", action="store_true", help="Stack Minneapolis VBR + Violations (Tableau)")
    ap.add_argument("--top", type=int, default=1000, help="Top N to export")
    ap.add_argument("--out", default="ranked_leads_hennepin_mpls.csv", help="Output CSV path")
    args = ap.parse_args()

    run(
        cities=args.cities,
        enable_mpls=args.enable_mpls,
        top_n=args.top,
        out_csv=args.out
    )
