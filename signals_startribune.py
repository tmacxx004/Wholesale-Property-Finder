import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup


DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (WholesalePropertyFinder/1.0; +https://streamlit.io)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _now_utc():
    return datetime.utcnow()


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _digits_only(s: str) -> str:
    return re.sub(r"\D+", "", s or "")


def _norm_addr(s: str) -> str:
    s = _clean(s).upper()
    s = s.replace(" AVENUE", " AVE").replace(" STREET", " ST").replace(" ROAD", " RD")
    s = s.replace(" DRIVE", " DR").replace(" BOULEVARD", " BLVD").replace(" LANE", " LN")
    s = s.replace(" COURT", " CT").replace(" CIRCLE", " CIR").replace(" PARKWAY", " PKWY")
    s = re.sub(r"[#,]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _try_parse_sale_date(text: str) -> Optional[str]:
    # "DATE AND TIME OF SALE: February 24, 2026, 10:00 AM"
    m = re.search(r"DATE\s+AND\s+TIME\s+OF\s+SALE:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})", text, re.I)
    if not m:
        return None
    raw = m.group(1)
    try:
        dt = datetime.strptime(raw, "%B %d, %Y")
        return dt.date().isoformat()
    except Exception:
        return None


def _extract_county(text: str) -> Optional[str]:
    m = re.search(r"COUNTY\s+IN\s+WHICH\s+PROPERTY\s+IS\s+LOCATED:\s*([A-Za-z ]+)", text, re.I)
    return _clean(m.group(1)) if m else None


def _extract_address(text: str) -> Optional[str]:
    m = re.search(r"ADDRESS\s+OF\s+PROPERTY:\s*(.+?)\s+COUNTY\s+IN\s+WHICH", text, re.I)
    if not m:
        m = re.search(r"MORTGAGED\s+PROPERTY\s+ADDRESS:\s*(.+?)\s+TAX\s+PARCEL", text, re.I)
    return _clean(m.group(1)) if m else None


def _extract_tax_parcel(text: str) -> Optional[str]:
    m = re.search(r"TAX\s+PARCEL\s+(?:NO\.?|NO|I\.D\.)\s*[:#]?\s*([0-9A-Za-z\-]+)", text, re.I)
    return _clean(m.group(1)) if m else None


def _extract_probate_file(text: str) -> Optional[str]:
    m = re.search(r"Court\s+File\s+No\.?\s*[:#]?\s*([0-9A-Za-z\-]+)", text, re.I)
    return _clean(m.group(1)) if m else None


def _extract_decedent(text: str) -> Optional[str]:
    m = re.search(r"Estate\s+of\s+(.+?),\s*Decedent", text, re.I)
    return _clean(m.group(1)) if m else None


def _extract_probate_rep_address(text: str) -> str:
    """
    Best-effort: many probate notices include representative mailing address lines.
    We try to capture something like:
      "Address: 123 Main Street, Minneapolis, MN 55401"
    If not found, returns "".
    """
    # Common patterns
    patterns = [
        r"\bAddress:\s*(.+?)(?:\bName\b|\bTelephone\b|\bAttorney\b|\bDated\b|$)",
        r"\bMailing\s+Address:\s*(.+?)(?:\bName\b|\bTelephone\b|\bAttorney\b|\bDated\b|$)",
        r"\bAddress\s+of\s+Personal\s+Representative:\s*(.+?)(?:\bName\b|\bTelephone\b|\bAttorney\b|\bDated\b|$)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            return _clean(m.group(1))
    return ""


@dataclass
class NoticeConfig:
    limit: int = 240
    max_age_days: int = 90
    sleep_s: float = 0.3


def fetch_startribune_foreclosure_notices(
    cfg: NoticeConfig,
    logger: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    """
    StarTribune Foreclosures category. Returns signals that can match by PID guess or address.
    """
    url = f"https://classifieds.startribune.com/mn/foreclosures/search?limit={cfg.limit}"
    if logger:
        logger(f"[StarTribune] Fetch foreclosures: {url}")

    r = requests.get(url, headers=DEFAULT_HEADERS, timeout=30)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")
    page_text = _clean(soup.get_text(" "))

    blocks = re.split(r"\bNOTICE OF MORTGAGE FORECLOSURE SALE\b", page_text, flags=re.I)

    out = []
    cutoff = _now_utc().date() - timedelta(days=cfg.max_age_days)

    for b in blocks[1:]:
        text = _clean("NOTICE OF MORTGAGE FORECLOSURE SALE " + b)

        county = _extract_county(text) or ""
        addr = _extract_address(text) or ""
        tax_parcel = _extract_tax_parcel(text) or ""
        sale_date = _try_parse_sale_date(text) or ""

        if not addr:
            continue

        if sale_date:
            try:
                sd = datetime.fromisoformat(sale_date).date()
                if sd < cutoff:
                    continue
            except Exception:
                pass

        pid_raw_guess = _digits_only(tax_parcel)
        norm_key = _norm_addr(addr)

        out.append(
            {
                "signal_name": "mortgage_foreclosure_notice",
                "source_url": url,
                "event_date": sale_date,
                "county": county,
                "address_text": addr,
                "tax_parcel_text": tax_parcel,
                "pid_raw_guess": pid_raw_guess,
                "norm_key": norm_key,
                "signal_strength": 35,
                "signal_details": f"Foreclosure notice; sale_date={sale_date or 'unknown'}; county={county or 'unknown'}",
            }
        )

    df = pd.DataFrame(out)
    time.sleep(cfg.sleep_s)
    return df


def fetch_startribune_probate_notices(
    cfg: NoticeConfig,
    logger: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    """
    StarTribune Probates category. Returns probate notices.
    """
    url = f"https://classifieds.startribune.com/mn/sub-probates/search?limit={cfg.limit}"
    if logger:
        logger(f"[StarTribune] Fetch probates: {url}")

    r = requests.get(url, headers=DEFAULT_HEADERS, timeout=30)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")
    page_text = _clean(soup.get_text(" "))

    candidates = re.split(r"\bEstate\s+of\b", page_text, flags=re.I)

    out = []
    for c in candidates[1:]:
        text = _clean("Estate of " + c)

        court_file = _extract_probate_file(text) or ""
        decedent = _extract_decedent(text) or ""
        rep_addr = _extract_probate_rep_address(text) or ""

        if not decedent:
            continue

        out.append(
            {
                "signal_name": "probate_notice",
                "source_url": url,
                "event_date": "",
                "court_file_no": court_file,
                "decedent_name": decedent,
                "rep_address_text": rep_addr,
                "norm_decedent": _clean(decedent).upper(),
                "signal_strength": 25,
                "signal_details": f"Probate notice; court_file={court_file or 'unknown'}; decedent={decedent}",
            }
        )

    df = pd.DataFrame(out)
    time.sleep(cfg.sleep_s)
    return df


def merge_notices_into_leads(
    leads_df: pd.DataFrame,
    forecl_df: pd.DataFrame,
    probate_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merges foreclosure/probate signals into leads.

    Foreclosure match:
      1) PID (if parcel number yields digits)
      2) else normalized address match

    Probate match (two-step):
      1) owner_name contains decedent last name (heuristic)
      2) store representative address from notice (if found)
      3) compute cross-check status based on owner mailing city vs rep address text
    """
    df = leads_df.copy()

    # Ensure pid_raw
    if "pid_raw" not in df.columns:
        if "pid" in df.columns:
            df["pid_raw"] = df["pid"].astype(str).map(_digits_only)
        else:
            df["pid_raw"] = ""

    df["pid_raw"] = df["pid_raw"].astype(str).map(_digits_only)

    # Address key for foreclosure fallback
    if "situs_address" in df.columns:
        df["_norm_addr_key"] = df["situs_address"].astype(str).map(_norm_addr)
    else:
        df["_norm_addr_key"] = ""

    # -----------------
    # Foreclosure fields
    # -----------------
    df["has_mortgage_foreclosure_notice"] = False
    df["foreclosure_sale_date"] = ""
    df["foreclosure_notice_url"] = ""
    df["foreclosure_notice_details"] = ""

    if forecl_df is not None and not forecl_df.empty:
        f = forecl_df.copy()
        f["pid_raw_guess"] = f.get("pid_raw_guess", "").astype(str).map(_digits_only)
        f["_norm_addr_key"] = f.get("norm_key", "").astype(str)

        # PID match
        pid_hits = f[f["pid_raw_guess"].astype(bool)].drop_duplicates(subset=["pid_raw_guess"])
        pid_map = pid_hits.set_index("pid_raw_guess").to_dict(orient="index") if not pid_hits.empty else {}

        hit_mask = df["pid_raw"].isin(pid_map.keys())
        df.loc[hit_mask, "has_mortgage_foreclosure_notice"] = True
        df.loc[hit_mask, "foreclosure_sale_date"] = df.loc[hit_mask, "pid_raw"].map(lambda x: pid_map.get(x, {}).get("event_date", ""))
        df.loc[hit_mask, "foreclosure_notice_url"] = df.loc[hit_mask, "pid_raw"].map(lambda x: pid_map.get(x, {}).get("source_url", ""))
        df.loc[hit_mask, "foreclosure_notice_details"] = df.loc[hit_mask, "pid_raw"].map(lambda x: pid_map.get(x, {}).get("signal_details", ""))

        # Address fallback
        addr_hits = f[f["_norm_addr_key"].astype(bool)].drop_duplicates(subset=["_norm_addr_key"])
        addr_map = addr_hits.set_index("_norm_addr_key").to_dict(orient="index") if not addr_hits.empty else {}

        hit_mask2 = df["_norm_addr_key"].isin(addr_map.keys()) & (~df["has_mortgage_foreclosure_notice"])
        df.loc[hit_mask2, "has_mortgage_foreclosure_notice"] = True
        df.loc[hit_mask2, "foreclosure_sale_date"] = df.loc[hit_mask2, "_norm_addr_key"].map(lambda k: addr_map.get(k, {}).get("event_date", ""))
        df.loc[hit_mask2, "foreclosure_notice_url"] = df.loc[hit_mask2, "_norm_addr_key"].map(lambda k: addr_map.get(k, {}).get("source_url", ""))
        df.loc[hit_mask2, "foreclosure_notice_details"] = df.loc[hit_mask2, "_norm_addr_key"].map(lambda k: addr_map.get(k, {}).get("signal_details", ""))

    # -----------------
    # Probate fields
    # -----------------
    df["has_probate_notice"] = False
    df["probate_court_file_no"] = ""
    df["probate_decedent_name"] = ""
    df["probate_rep_address_text"] = ""
    df["probate_notice_url"] = ""
    df["probate_notice_details"] = ""

    # Cross-check status: Match / Mismatch / No data
    df["probate_mail_crosscheck_status"] = "No data"

    if probate_df is not None and not probate_df.empty and "owner_name" in df.columns:
        p = probate_df.copy()
        p["norm_decedent"] = p.get("norm_decedent", "").astype(str)
        p["decedent_last"] = p["norm_decedent"].map(lambda x: x.split()[-1] if x else "")
        p = p[p["decedent_last"].astype(bool)]

        # last_name -> best record
        last_map = p.drop_duplicates(subset=["decedent_last"]).set_index("decedent_last").to_dict(orient="index")

        def _find_probate(owner: str):
            owner = _clean(owner).upper()
            for last, rec in last_map.items():
                if last and last in owner:
                    return rec
            return None

        owner_up = df["owner_name"].astype(str).str.upper()
        recs = owner_up.map(_find_probate)

        df["has_probate_notice"] = recs.notna()

        df.loc[df["has_probate_notice"], "probate_court_file_no"] = recs[df["has_probate_notice"]].map(lambda r: (r or {}).get("court_file_no", ""))
        df.loc[df["has_probate_notice"], "probate_decedent_name"] = recs[df["has_probate_notice"]].map(lambda r: (r or {}).get("decedent_name", ""))
        df.loc[df["has_probate_notice"], "probate_rep_address_text"] = recs[df["has_probate_notice"]].map(lambda r: (r or {}).get("rep_address_text", ""))
        df.loc[df["has_probate_notice"], "probate_notice_url"] = recs[df["has_probate_notice"]].map(lambda r: (r or {}).get("source_url", ""))
        df.loc[df["has_probate_notice"], "probate_notice_details"] = recs[df["has_probate_notice"]].map(lambda r: (r or {}).get("signal_details", ""))

        # Cross-check using MAILING city from county vs rep address text (best available)
        mail_city = df.get("mailing_city", pd.Series([""] * len(df))).astype(str).str.upper()
        rep_addr = df["probate_rep_address_text"].astype(str).str.upper()

        status = []
        for has_pb, mc, ra in zip(df["has_probate_notice"].tolist(), mail_city.tolist(), rep_addr.tolist()):
            mc = _clean(mc).upper()
            ra = _clean(ra).upper()
            if not has_pb:
                status.append("No data")
                continue
            if not mc or not ra:
                status.append("No data")
                continue
            status.append("Match" if mc in ra else "Mismatch")

        df["probate_mail_crosscheck_status"] = status

    df.drop(columns=["_norm_addr_key"], errors="ignore", inplace=True)
    return df


def apply_notice_scoring(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds score boosts + notes based on notices.
    """
    d = df.copy()
    if "score" not in d.columns:
        d["score"] = 0
    if "score_notes" not in d.columns:
        d["score_notes"] = ""

    def add_note(cur: str, note: str) -> str:
        cur = (cur or "").strip()
        if not cur:
            return note
        if note in cur:
            return cur
        return cur + " | " + note

    # Foreclosure notice boost
    if "has_mortgage_foreclosure_notice" in d.columns:
        mask = d["has_mortgage_foreclosure_notice"].fillna(False)
        d.loc[mask, "score"] = pd.to_numeric(d.loc[mask, "score"], errors="coerce").fillna(0) + 35
        d.loc[mask, "score_notes"] = d.loc[mask, "score_notes"].map(lambda s: add_note(s, "Mortgage foreclosure notice"))

    # Probate notice boost
    if "has_probate_notice" in d.columns:
        mask = d["has_probate_notice"].fillna(False)
        d.loc[mask, "score"] = pd.to_numeric(d.loc[mask, "score"], errors="coerce").fillna(0) + 25
        d.loc[mask, "score_notes"] = d.loc[mask, "score_notes"].map(lambda s: add_note(s, "Probate notice (heuristic match)"))

    return d
