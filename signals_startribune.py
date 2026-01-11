import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional, Iterable, Dict, Any, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Optional dependency: rapidfuzz (fast)
try:
    from rapidfuzz import fuzz, process  # type: ignore
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False
    import difflib


DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (WholesalePropertyFinder/1.0)",
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
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(must_start_with):
            links.append(_abs_url(href))

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
        return datetime.strptime(raw, "%B %d, %Y").date().isoformat()
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


# ✅ UPDATED: Mortgage amount extraction using "DATE OF NOTICE"
def _extract_mortgage_amount(text: str) -> int:
    """
    Extracts mortgage amount using the phrase 'DATE OF NOTICE'
    Handles formats like:
      DATE OF NOTICE: $123,456.78
      DATE OF NOTICE $123,456
      DATE OF NOTICE ..... $123,456
    Returns integer dollars or 0 if not found.
    """
    patterns = [
        r"DATE\s+OF\s+NOTICE\s*[:\-]?\s*\$?\s*([0-9][0-9,]*\.?\d{0,2})",
        r"DATE\s+OF\s+NOTICE.*?\$([0-9][0-9,]*\.?\d{0,2})",
    ]

    for pat in patterns:
        m = re.search(pat, text, re.I | re.S)
        if m:
            raw = m.group(1).replace(",", "").strip()
            try:
                val = float(raw)
                return int(val) if val > 0 else 0
            except Exception:
                continue

    return 0


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

    list_url = f"{BASE}/mn/foreclosures/search?limit=240"
    r = requests.get(list_url, headers=DEFAULT_HEADERS, timeout=30)
    r.raise_for_status()

    ad_links = _extract_links_by_prefix(r.text, must_start_with="/mn/foreclosures/")
    ad_links = [u for u in ad_links if "/search" not in u][: cfg.limit]

    cutoff = _now_utc().date() - timedelta(days=cfg.max_age_days)
    cities_list = [c for c in (cities or []) if _clean(c)]

    out = []

    for url in ad_links:
        try:
            rr = requests.get(url, headers=DEFAULT_HEADERS, timeout=30)
            rr.raise_for_status()
            soup = BeautifulSoup(rr.text, "lxml")
            text = _clean(soup.get_text(" "))

            addr = _extract_address(text)
            if not addr:
                continue

            sale_date = _try_parse_sale_date(text)
            if sale_date:
                try:
                    if datetime.fromisoformat(sale_date).date() < cutoff:
                        continue
                except Exception:
                    pass

            if cities_list and not any(_city_in_text(c, addr) for c in cities_list):
                continue

            mortgage_amount = _extract_mortgage_amount(text)

            out.append(
                {
                    "signal_name": "mortgage_foreclosure_notice",
                    "source_url": url,
                    "event_date": sale_date,
                    "county": _extract_county(text),
                    "address_text": addr,
                    "tax_parcel_text": _extract_tax_parcel(text),
                    "pid_raw_guess": _digits_only(_extract_tax_parcel(text)),
                    "norm_key": _norm_addr(addr),
                    "mortgage_amount": mortgage_amount,
                    "signal_strength": 35,
                    "signal_details": (
                        f"Foreclosure notice; mortgage_amount=${mortgage_amount:,}"
                        if mortgage_amount
                        else "Foreclosure notice"
                    ),
                }
            )

            time.sleep(cfg.sleep_s)

        except Exception as e:
            if logger:
                logger(f"[StarTribune] Foreclosure parse failed: {url} :: {type(e).__name__}")
            continue

    return pd.DataFrame(out)
