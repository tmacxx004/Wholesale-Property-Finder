import re
import csv
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Callable
from urllib.parse import urlparse, urljoin

import requests
import pandas as pd
from rapidfuzz import fuzz
from dateutil import parser as dateparser

# =========================
# Global Config
# =========================

USER_AGENT = "HennepinMplsDealCrawler/1.1 (+public data; polite)"
PAGE_SIZE = 2000

# Hennepin ArcGIS REST query endpoint
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

# Tableau view URLs (friendly flags)
MPLS_VBR_VIEW_URL = (
    "https://tableau.minneapolismn.gov/views/"
    "MinneapolisVacantCondemnedPropertyInventory/DailyVBRcondemnedpropertylist"
    "?:showVizHome=no&:embed=y&:toolbar=n"
)
MPLS_VIOL_VIEW_URL = (
    "https://tableau.minneapolismn.gov/views/"
    "OpenDataRegulatoryServices-Violations/ViolationDetails"
    "?:showVizHome=no&:embed=y&:toolbar=n"
)


# =========================
# Helpers
# =========================

Logger = Optional[Callable[[str], None]]

def log_msg(logger: Logger, msg: str):
    if logger:
        logger(msg)

def polite_sleep(sec: float = 0.25):
    time.sleep(sec)

def now_year() -> int:
    return datetime.now().year

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def normalize_addr(a: str) -> str:
    a = normalize_ws(a).upper()
    a = a.replace(".", "").replace(",", "").replace("#", " ")
    a = a.replace(" STREET", " ST").replace(" AVENUE", " AVE").replace(" ROAD", " RD")
    a = a.replace(" DRIVE", " DR").replace(" LANE", " LN").replace(" BOULEVARD", " BLVD")
    a = a.replace(" PLACE", " PL").replace(" COURT", " CT")
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
# Data model
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
# Hennepin ArcGIS Client
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

def iter_hennepin_features(where: str, out_fields: List[str], logger: Logger = None) -> List[Dict[str, Any]]:
    client = ArcGISClient()
    feats_all = []
    offset = 0
    while True:
        data = client.query(where=where, out_fields=out_fields, offset=offset, count=PAGE_SIZE)
        feats = data.get("features") or []
        feats_all.extend(feats)
        log_msg(logger, f"Hennepin: fetched {len(feats)} features (offset={offset}). Total={len(feats_all)}")
        if len(feats) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        polite_sleep(0.2)
    return feats_all


# =========================
# Tableau Extraction (Fixed)
# =========================

class TableauExtractor:
    """
    Robust extractor:
      - Try .csv export first
      - Fallback to vizql bootstrapSession:
          1) GET view HTML
          2) Extract correct /vizql/w/.../v/.../bootstrapSession/sessions/ path
          3) POST to that endpoint
          4) Parse JSON payload for dataSegments
    """
    def __init__(self, logger: Logger = None):
        self.logger = logger
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": USER_AGENT})

    def try_simple_csv(self, view_url: str) -> Optional[pd.DataFrame]:
        # Convert https://.../views/WB/Sheet?:... into https://.../views/WB/Sheet.csv?:showVizHome=no
        base = view_url.split("?")[0]
        csv_url = base + ".csv"
        r = self.s.get(csv_url, params={":showVizHome": "no"}, timeout=60, allow_redirects=True)
        ct = (r.headers.get("Content-Type") or "").lower()

        if r.status_code == 200 and ("text/csv" in ct or r.text[:200].count(",") > 5):
            from io import StringIO
            df = pd.read_csv(StringIO(r.text))
            log_msg(self.logger, f"Tableau CSV export succeeded: {base} (rows={len(df)})")
            return df

        log_msg(self.logger, f"Tableau CSV export not available (status={r.status_code}, ct={ct})")
        return None

    def bootstrap_df(self, view_url: str) -> pd.DataFrame:
        r = self.s.get(view_url, timeout=60, allow_redirects=True)
        r.raise_for_status()
        html = r.text

        # Find vizql bootstrap path
        m = re.search(r'(/vizql/w/[^/]+/v/[^/]+/bootstrapSession/sessions/)', html)
        if not m:
            m = re.search(r'(/vizql/w/[^"]+?/bootstrapSession/sessions/)', html)
        if not m:
            raise RuntimeError("Could not find Tableau vizql bootstrapSession endpoint in HTML.")

        vizql_path = m.group(1)
        parsed = urlparse(view_url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        bootstrap_url = urljoin(base, vizql_path)

        log_msg(self.logger, f"Tableau bootstrap endpoint: {bootstrap_url}")

        payload = {
            "worksheetPortSize": '{"w":1200,"h":800}',
            "dashboardPortSize": '{"w":1200,"h":800}',
            "clientDimension": '{"w":1200,"h":800}',
            ":showVizHome": "no",
            ":embed": "y",
        }

        headers = {
            "Referer": view_url,
            "Accept": "application/json,text/plain,*/*",
        }

        br = self.s.post(bootstrap_url, data=payload, headers=headers, timeout=60, allow_redirects=True)
        log_msg(self.logger, f"Tableau bootstrap status={br.status_code} url={br.url}")
        br.raise_for_status()

        txt = br.text

        # Bootstrap response is often JSON-ish; extract largest JSON object
        start = txt.find("{")
        end = txt.rfind("}")
        if start == -1 or end == -1:
            raise RuntimeError("Unexpected Tableau bootstrap response format (no JSON object found).")

        data = json.loads(txt[start:end+1])

        pres = data.get("presModel", {})
        segments = pres.get("dataSegments") or {}
        if not segments:
            raise RuntimeError("No dataSegments found in Tableau presModel.")

        # pick the segment with most columns
        best_key = None
        best_cols = 0
        for k, seg in segments.items():
            cols = seg.get("dataColumns") if isinstance(seg, dict) else None
            if cols and len(cols) > best_cols:
                best_key = k
                best_cols = len(cols)

        seg = segments[best_key]
        cols = seg["dataColumns"]

        table = {}
        for i, col in enumerate(cols):
            values = col.get("dataValues") or []
            name = col.get("fieldCaption") or col.get("caption") or f"col_{i}"
            table[name] = values

        df = pd.DataFrame(table)
        log_msg(self.logger, f"Tableau bootstrap parse OK (rows={len(df)}, cols={len(df.columns)})")
        return df

    def extract(self, view_url: str) -> pd.DataFrame:
        df = self.try_simple_csv(view_url)
        if df is not None and not df.empty:
            return df
        return self.bootstrap_df(view_url)


def fetch_mpls_tables(enable_mpls: bool, logger: Logger = None) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if not enable_mpls:
        return None, None

    t = TableauExtractor(logger=logger)

    vbr = None
    viol = None

    try:
        log_msg(logger, "Fetching Minneapolis VBR (Vacant/Condemned)…")
        vbr = t.extract(MPLS_VBR_VIEW_URL)
    except Exception as e:
        log_msg(logger, f"[WARN] VBR Tableau extract failed: {e}")

    polite_sleep(0.5)

    try:
        log_msg(logger, "Fetching Minneapolis Violations…")
        viol = t.extract(MPLS_VIOL_VIEW_URL)
    except Exception as e:
        log_msg(logger, f"[WARN] Violations Tableau extract failed: {e}")

    return vbr, viol


# =========================
# Matching + Stacking
# =========================

def match_by_address(
    target_addr: str,
    target_city: str,
    df: pd.DataFrame,
    addr_col_candidates: List[str],
    city_col_candidates: List[str],
    min_score: int = 92
) -> Optional[int]:
    t_addr = normalize_addr(target_addr)
    t_city = normalize_ws(target_city).upper()

    addr_col = next((c for c in addr_col_candidates if c and c in df.columns), None)
    city_col = next((c for c in city_col_candidates if c and c in df.columns), None)
    if not addr_col:
        return None

    best_idx = None
    best_score = 0

    for idx, row in df.iterrows():
        r_addr = normalize_addr(str(row.get(addr_col) or ""))
        if not r_addr:
            continue

        if city_col:
            r_city = normalize_ws(str(row.get(city_col) or "")).upper()
            if r_city and t_city and r_city != t_city:
                continue

        sc = fuzz.token_sort_ratio(t_addr, r_addr)
        if sc > best_score:
            best_score = sc
            best_idx = idx

    if best_idx is not None and best_score >= min_score:
        return best_idx
    return None


def infer_vbr_columns(df: pd.DataFrame) -> Dict[str, str]:
    candidates = {
        "address": ["Address", "ADDRESS", "Property Address", "PROPERTY ADDRESS"],
        "condemned_date": ["Condemned Date", "CONDEMNED DATE", "Condemned", "CONDEMNED"],
        "vbr_type": ["Type", "VBR Type", "Status", "VBR Status", "VBR_STATUS"],
        "city": ["City", "CITY", "Municipality", "MUNIC_NM"],
    }
    resolved = {}
    upper_map = {c.upper(): c for c in df.columns}
    for key, opts in candidates.items():
        for o in opts:
            if o.upper() in upper_map:
                resolved[key] = upper_map[o.upper()]
                break
    return resolved


def infer_violation_columns(df: pd.DataFrame) -> Dict[str, str]:
    candidates = {
        "address": ["Address", "ADDRESS", "Property Address", "PROPERTY ADDRESS"],
        "resolved": ["Violation Resolved?", "Resolved", "VIOLATION RESOLVED?", "RESOLVED"],
        "city": ["City", "CITY", "Municipality", "MUNIC_NM"],
    }
    resolved = {}
    upper_map = {c.upper(): c for c in df.columns}
    for key, opts in candidates.items():
        for o in opts:
            if o.upper() in upper_map:
                resolved[key] = upper_map[o.upper()]
                break
    return resolved


def stack_mpls(
    attrs: Dict[str, Any],
    vbr_df: Optional[pd.DataFrame],
    viol_df: Optional[pd.DataFrame],
) -> Tuple[bool, Optional[str], Optional[str], int, int]:
    situs_addr = build_situs_address(attrs)
    city = normalize_ws(str(attrs.get("MUNIC_NM") or ""))

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
            addr_col_candidates=[cols.get("address"), "Address", "PROPERTY ADDRESS", "ADDRESS"],
            city_col_candidates=[cols.get("city"), "City", "MUNIC_NM"],
            min_score=92,
        )
        if idx is not None:
            mpls_vacant = True
            if cols.get("condemned_date"):
                condemned_date = parse_date_maybe(vbr_df.loc[idx, cols["condemned_date"]])
            if cols.get("vbr_type"):
                vbr_type = normalize_ws(str(vbr_df.loc[idx, cols["vbr_type"]] or ""))

    # Violations aggregation
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
                        open_violation_count += 1

    return mpls_vacant, condemned_date, vbr_type, violation_count, open_violation_count


# =========================
# Scoring
# =========================

def score_lead(
    base: Dict[str, Any],
    mpls_vacant: bool,
    condemned_date: Optional[str],
    violation_count: int,
    open_violation_count: int
) -> Tuple[float, List[str]]:
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
        notes.append("Mpls vacant/condemned match")
    if condemned_date:
        score += 10
        notes.append(f"Condemned date ({condemned_date})")

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
# Filters + Runner
# =========================

def build_where_clause(cities: Optional[List[str]]) -> str:
    base = "(EARLIEST_DELQ_YR IS NOT NULL) OR (FORFEIT_LAND_IND = 'Y')"
    if not cities:
        return base
    quoted = ",".join([f"'{c.upper()}'" for c in cities])
    return f"({base}) AND (UPPER(MUNIC_NM) IN ({quoted}))"


def run(
    cities: Optional[List[str]],
    enable_mpls: bool,
    top_n: int,
    out_csv: str,
    logger: Logger = None,
):
    where = build_where_clause(cities)
    log_msg(logger, f"WHERE clause: {where}")

    feats = iter_hennepin_features(where=where, out_fields=HENNEPIN_FIELDS, logger=logger)

    vbr_df, viol_df = fetch_mpls_tables(enable_mpls=enable_mpls, logger=logger)

    if enable_mpls:
        if vbr_df is None:
            log_msg(logger, "[WARN] Minneapolis VBR data unavailable. Continuing without VBR stacking.")
        if viol_df is None:
            log_msg(logger, "[WARN] Minneapolis Violations data unavailable. Continuing without violations stacking.")

    leads: List[ParcelLead] = []

    for i, f in enumerate(feats):
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

        if enable_mpls and (vbr_df is not None or viol_df is not None):
            mpls_vacant, condemned_date, vbr_type, vio_cnt, open_vio_cnt = stack_mpls(attrs, vbr_df, viol_df)
        else:
            mpls_vacant, condemned_date, vbr_type, vio_cnt, open_vio_cnt = (False, None, None, 0, 0)

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

        if i % 2000 == 0 and i > 0:
            log_msg(logger, f"Processed {i} parcels…")
        polite_sleep(0.001)

    leads.sort(key=lambda x: x.score, reverse=True)
    rows = [asdict(l) for l in leads[:top_n]]

    if not rows:
        # still write empty with headers for consistency
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["pid"])
        log_msg(logger, "No rows returned. Wrote empty CSV.")
        return

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    log_msg(logger, f"Exported {len(rows)} rows -> {out_csv}")
