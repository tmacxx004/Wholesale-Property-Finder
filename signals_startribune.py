import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional, Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup


DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (WholesalePropertyFinder/1.0; +https://streamlit.io)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

BASE = "https://classifieds.startribune.com"


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


def _abs_url(href: str) -> str:
    href = (href or "").strip()
    if not href:
        return ""
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("/"):
        return BASE + href
    return BASE + "/" + href


def _extract_links_by_prefix(html: str, must_start_with: str) -> list[str]:
    """
    More robust than 'must_contain'. We collect any link starting with a path prefix.
    Example: /mn/foreclosures/...
    """
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(must_start_with):
            links.append(_abs_url(href))

    # de-dupe preserving order
    seen = set()
    out = []
    for u in links:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _city_in_text(city: str, text: str) -> bool:
    city = _clean(city).upper()
    text = _clean(text).upper()
    if not city or not text:
        return False
    return re.search(rf"\b{re.escape(city)}\b", text) is not None


def _try_parse_sale_date(text: str) -> str:
    m = re.search(r"DATE\s+AND\s+TIME\s+OF\s+SALE:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})", text, re.I)
    if not m:
        return ""
    raw = m.group(1)
    try:
        dt = datetime.strptime(raw, "%B %d, %Y")
        return dt.date().isoformat()
    except Exception:
        return ""


def _extract_county(text: str) -> str:
    m = re.search(r"COUNTY\s+IN\s+WHICH\s+PROPERTY\s+IS\s+LOCATED:\s*([A-Za-z ]+)", text, re.I)
    return _clean(m.group(1)) if m else ""


def _extract_address(text: str) -> str:
    m = re.search(r"ADDRESS\s+OF\s+PROPERTY:\s*(.+?)\s+COUNTY\s+IN\s+WHICH", text, re.I)
    if not m:
        m = re.search(r"MORTGAGED\s+PROPERTY\s+ADDRESS:\s*(.+?)\s+TAX\s+PARCEL", text, re.I)
    return _clean(m.group(1)) if m else ""


def _extract_tax_parcel(text: str) -> str:
    m = re.search(r"TAX\s+PARCEL\s+(?:NO\.?|NO|I\.D\.)\s*[:#]?\s*([0-9A-Za-z\-]+)", text, re.I)
    return _clean(m.group(1)) if m else ""


def _extract_probate_file(text: str) -> str:
    m = re.search(r"Court\s+File\s+No\.?\s*[:#]?\s*([0-9A-Za-z\-]+)", text, re.I)
    return _clean(m.group(1)) if m else ""


def _extract_decedent(text: str) -> str:
    m = re.search(r"Estate\s+of\s+(.+?),\s*Decedent", text, re.I)
    return _clean(m.group(1)) if m else ""


def _extract_probate_rep_address(text: str) -> str:
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
    limit: int = 120
    max_age_days: int = 120
    sleep_s: float = 0.2


def fetch_startribune_foreclosure_notices(
    cfg: NoticeConfig,
    *,
    cities: Optional[Iterable[str]] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    """
    Foreclosures:
      - Scrape listing page
      - Extract ad links under /mn/foreclosures/
      - Visit each ad detail page and parse notice body
      - Keep only notices matching selected cities (by address text)
    """
    list_url = f"{BASE}/mn/foreclosures/search?limit=240"
    if logger:
        logger(f"[StarTribune] Foreclosures listing: {list_url}")

    r = requests.get(list_url, headers=DEFAULT_HEADERS, timeout=30)
    r.raise_for_status()

    ad_links = _extract_links_by_prefix(r.text, must_start_with="/mn/foreclosures/")
    # remove listing/search links
    ad_links = [u for u in ad_links if "/search" not in u and "/create" not in u]
    if logger:
        logger(f"[StarTribune] Foreclosures links found: {len(ad_links)}")

    ad_links = ad_links[: max(1, cfg.limit)]
    cutoff = _now_utc().date() - timedelta(days=cfg.max_age_days)

    cities_list = [c for c in (cities or []) if _clean(c)]
    out = []

    for i, url in enumerate(ad_links, start=1):
        try:
            rr = requests.get(url, headers=DEFAULT_HEADERS, timeout=30)
            rr.raise_for_status()
            soup = BeautifulSoup(rr.text, "lxml")
            text = _clean(soup.get_text(" "))

            county = _extract_county(text)
            addr = _extract_address(text)
            tax_parcel = _extract_tax_parcel(text)
            sale_date = _try_parse_sale_date(text)

            if not addr:
                continue

            if sale_date:
                try:
                    sd = datetime.fromisoformat(sale_date).date()
                    if sd < cutoff:
                        continue
                except Exception:
                    pass

            # City match required (your request: show based on city match)
            if cities_list and not any(_city_in_text(c, addr) for c in cities_list):
                continue

            out.append(
                {
                    "signal_name": "mortgage_foreclosure_notice",
                    "source_url": url,
                    "event_date": sale_date,
                    "county": county,
                    "address_text": addr,
                    "tax_parcel_text": tax_parcel,
                    "pid_raw_guess": _digits_only(tax_parcel),
                    "norm_key": _norm_addr(addr),
                    "signal_strength": 35,
                    "signal_details": f"Foreclosure notice; sale_date={sale_date or 'unknown'}; county={county or 'unknown'}",
                }
            )

            time.sleep(cfg.sleep_s)

        except Exception as e:
            if logger:
                logger(f"[StarTribune] Foreclosure ad failed ({i}/{len(ad_links)}): {url} :: {type(e).__name__}")
            continue

    if logger:
        logger(f"[StarTribune] Foreclosures parsed signals: {len(out)}")
    return pd.DataFrame(out)


def fetch_startribune_probate_notices(
    cfg: NoticeConfig,
    *,
    cities: Optional[Iterable[str]] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    """
    Probates:
      - Scrape listing page
      - Extract ad links under /mn/sub-probates/
      - Visit each ad detail page and parse notice body
      - Keep only notices matching selected cities (by representative address text)
    """
    list_url = f"{BASE}/mn/sub-probates/search?limit=240"
    if logger:
        logger(f"[StarTribune] Probates listing: {list_url}")

    r = requests.get(list_url, headers=DEFAULT_HEADERS, timeout=30)
    r.raise_for_status()

    ad_links = _extract_links_by_prefix(r.text, must_start_with="/mn/sub-probates/")
    ad_links = [u for u in ad_links if "/search" not in u and "/create" not in u]
    if logger:
        logger(f"[StarTribune] Probate links found: {len(ad_links)}")

    ad_links = ad_links[: max(1, cfg.limit)]
    cities_list = [c for c in (cities or []) if _clean(c)]

    out = []
    for i, url in enumerate(ad_links, start=1):
        try:
            rr = requests.get(url, headers=DEFAULT_HEADERS, timeout=30)
            rr.raise_for_status()
            soup = BeautifulSoup(rr.text, "lxml")
            text = _clean(soup.get_text(" "))

            decedent = _extract_decedent(text)
            if not decedent:
                continue

            court_file = _extract_probate_file(text)
            rep_addr = _extract_probate_rep_address(text)

            # City match required (use rep address, since probate notice often lacks property address)
            if cities_list:
                hay = rep_addr or text
                if not any(_city_in_text(c, hay) for c in cities_list):
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

            time.sleep(cfg.sleep_s)

        except Exception as e:
            if logger:
                logger(f"[StarTribune] Probate ad failed ({i}/{len(ad_links)}): {url} :: {type(e).__name__}")
            continue

    if logger:
        logger(f"[StarTribune] Probates parsed signals: {len(out)}")
    return pd.DataFrame(out)


def merge_notices_into_leads(
    leads_df: pd.DataFrame,
    forecl_df: pd.DataFrame,
    probate_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge signals into county leads when possible.
    IMPORTANT: This is only for "enrichment". The app will also show notice-only rows.
    """
    df = leads_df.copy()

    if "pid_raw" not in df.columns:
        df["pid_raw"] = df.get("pid", "").astype(str).map(_digits_only)
    df["pid_raw"] = df["pid_raw"].astype(str).map(_digits_only)

    df["_norm_addr_key"] = df.get("situs_address", "").astype(str).map(_norm_addr)

    # Foreclosure enrichment
    df["has_mortgage_foreclosure_notice"] = False
    df["foreclosure_sale_date"] = ""
    df["foreclosure_notice_url"] = ""
    df["foreclosure_notice_details"] = ""

    if forecl_df is not None and not forecl_df.empty:
        f = forecl_df.copy()
        f["pid_raw_guess"] = f.get("pid_raw_guess", "").astype(str).map(_digits_only)
        f["_norm_addr_key"] = f.get("norm_key", "").astype(str)

        # PID match (rare)
        pid_hits = f[f["pid_raw_guess"].astype(bool)].drop_duplicates(subset=["pid_raw_guess"])
        pid_map = pid_hits.set_index("pid_raw_guess").to_dict(orient="index") if not pid_hits.empty else {}

        hit_mask = df["pid_raw"].isin(pid_map.keys())
        df.loc[hit_mask, "has_mortgage_foreclosure_notice"] = True
        df.loc[hit_mask, "foreclosure_sale_date"] = df.loc[hit_mask, "pid_raw"].map(lambda x: pid_map.get(x, {}).get("event_date", ""))
        df.loc[hit_mask, "foreclosure_notice_url"] = df.loc[hit_mask, "pid_raw"].map(lambda x: pid_map.get(x, {}).get("source_url", ""))
        df.loc[hit_mask, "foreclosure_notice_details"] = df.loc[hit_mask, "pid_raw"].map(lambda x: pid_map.get(x, {}).get("signal_details", ""))

        # Address match (primary)
        addr_hits = f[f["_norm_addr_key"].astype(bool)].drop_duplicates(subset=["_norm_addr_key"])
        addr_map = addr_hits.set_index("_norm_addr_key").to_dict(orient="index") if not addr_hits.empty else {}

        hit_mask2 = df["_norm_addr_key"].isin(addr_map.keys()) & (~df["has_mortgage_foreclosure_notice"])
        df.loc[hit_mask2, "has_mortgage_foreclosure_notice"] = True
        df.loc[hit_mask2, "foreclosure_sale_date"] = df.loc[hit_mask2, "_norm_addr_key"].map(lambda k: addr_map.get(k, {}).get("event_date", ""))
        df.loc[hit_mask2, "foreclosure_notice_url"] = df.loc[hit_mask2, "_norm_addr_key"].map(lambda k: addr_map.get(k, {}).get("source_url", ""))
        df.loc[hit_mask2, "foreclosure_notice_details"] = df.loc[hit_mask2, "_norm_addr_key"].map(lambda k: addr_map.get(k, {}).get("signal_details", ""))

    # Probate enrichment (heuristic by owner last name)
    df["has_probate_notice"] = False
    df["probate_court_file_no"] = ""
    df["probate_decedent_name"] = ""
    df["probate_rep_address_text"] = ""
    df["probate_notice_url"] = ""
    df["probate_notice_details"] = ""
    df["probate_mail_crosscheck_status"] = "No data"

    if probate_df is not None and not probate_df.empty and "owner_name" in df.columns:
        p = probate_df.copy()
        p["norm_decedent"] = p.get("norm_decedent", "").astype(str)
        p["decedent_last"] = p["norm_decedent"].map(lambda x: x.split()[-1] if x else "")
        p = p[p["decedent_last"].astype(bool)]

        last_map = p.drop_duplicates(subset=["decedent_last"]).set_index("decedent_last").to_dict(orient="index")

        def _find_probate(owner: str):
            owner = _clean(owner).upper()
            for last, rec in last_map.items():
                if last and last in owner:
                    return rec
            return None

        recs = df["owner_name"].astype(str).str.upper().map(_find_probate)
        df["has_probate_notice"] = recs.notna()

        df.loc[df["has_probate_notice"], "probate_court_file_no"] = recs[df["has_probate_notice"]].map(lambda r: (r or {}).get("court_file_no", ""))
        df.loc[df["has_probate_notice"], "probate_decedent_name"] = recs[df["has_probate_notice"]].map(lambda r: (r or {}).get("decedent_name", ""))
        df.loc[df["has_probate_notice"], "probate_rep_address_text"] = recs[df["has_probate_notice"]].map(lambda r: (r or {}).get("rep_address_text", ""))
        df.loc[df["has_probate_notice"], "probate_notice_url"] = recs[df["has_probate_notice"]].map(lambda r: (r or {}).get("source_url", ""))
        df.loc[df["has_probate_notice"], "probate_notice_details"] = recs[df["has_probate_notice"]].map(lambda r: (r or {}).get("signal_details", ""))

        mail_city = df.get("mailing_city", pd.Series([""] * len(df))).astype(str).str.upper()
        rep_addr = df["probate_rep_address_text"].astype(str).str.upper()

        statuses = []
        for has_pb, mc, ra in zip(df["has_probate_notice"].tolist(), mail_city.tolist(), rep_addr.tolist()):
            mc = _clean(mc).upper()
            ra = _clean(ra).upper()
            if not has_pb or not mc or not ra:
                statuses.append("No data")
            else:
                statuses.append("Match" if re.search(rf"\b{re.escape(mc)}\b", ra) else "Mismatch")
        df["probate_mail_crosscheck_status"] = statuses

    df.drop(columns=["_norm_addr_key"], errors="ignore", inplace=True)
    return df


def apply_notice_scoring(df: pd.DataFrame) -> pd.DataFrame:
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

    if "has_mortgage_foreclosure_notice" in d.columns:
        mask = d["has_mortgage_foreclosure_notice"].fillna(False)
        d.loc[mask, "score"] = pd.to_numeric(d.loc[mask, "score"], errors="coerce").fillna(0) + 35
        d.loc[mask, "score_notes"] = d.loc[mask, "score_notes"].map(lambda s: add_note(s, "Mortgage foreclosure notice"))

    if "has_probate_notice" in d.columns:
        mask = d["has_probate_notice"].fillna(False)
        d.loc[mask, "score"] = pd.to_numeric(d.loc[mask, "score"], errors="coerce").fillna(0) + 25
        d.loc[mask, "score_notes"] = d.loc[mask, "score_notes"].map(lambda s: add_note(s, "Probate notice (heuristic match)"))

    return d
