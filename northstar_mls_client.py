import os
import time
import requests
from typing import Any, Dict, List, Optional

TOKEN_URL = "https://reso-auth.northstarmls.com/oauth/token"
BASE_URL = "https://reso.northstarmls.com/reso-web-api"


def _clean(s: str) -> str:
    return (s or "").strip()


class NorthstarResoClient:
    """
    NorthstarMLS RESO Web API client.

    - Auth: Auth0 M2M token (client_credentials) with Basic Auth (client_id:client_secret)
    - Base URL: https://reso.northstarmls.com/reso-web-api
    Docs: https://northstarmls.com/northstarmls-reso-web-api-documentation/ :contentReference[oaicite:4]{index=4}
    """

    def __init__(self):
        self.client_id = _clean(os.getenv("NORTHSTAR_CLIENT_ID", ""))
        self.client_secret = _clean(os.getenv("NORTHSTAR_CLIENT_SECRET", ""))
        self.scope = _clean(os.getenv("NORTHSTAR_SCOPE", "read:reso-resources read:reso-metadata"))

        if not self.client_id or not self.client_secret:
            raise RuntimeError("Missing NORTHSTAR_CLIENT_ID / NORTHSTAR_CLIENT_SECRET environment variables.")

        self._token: Optional[str] = None
        self._token_exp: float = 0.0

        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "WholesalePropertyFinder/1.0 (Northstar RESO Web API)"})

    def _get_token(self) -> str:
        now = time.time()
        if self._token and now < (self._token_exp - 30):
            return self._token

        # Per Northstar docs: client_credentials + scope; Basic Auth header (client_id as username)
        # :contentReference[oaicite:5]{index=5}
        data = {
            "grant_type": "client_credentials",
            "scope": self.scope,
        }

        r = self.s.post(
            TOKEN_URL,
            data=data,
            auth=(self.client_id, self.client_secret),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=60,
        )
        r.raise_for_status()
        payload = r.json()

        token = payload.get("access_token")
        expires_in = int(payload.get("expires_in", 86400))
        if not token:
            raise RuntimeError(f"Token response missing access_token: {payload}")

        self._token = token
        self._token_exp = now + expires_in
        return token

    def _auth_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._get_token()}"}

    def odata_get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{BASE_URL.rstrip('/')}/{path.lstrip('/')}"
        r = self.s.get(url, headers=self._auth_headers(), params=params or {}, timeout=60)
        r.raise_for_status()
        return r.json()

    # ---------- MLS helpers ----------

    def search_property_by_address(
        self,
        unparsed_address: str,
        city: str = "",
        postal_code: str = "",
        top: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search the Property resource by UnparsedAddress + optional City/PostalCode.
        Field names are RESO-standard; confirm exact availability via $metadata if needed.
        """
        addr = (unparsed_address or "").lower().replace("'", "''").strip()
        city = (city or "").replace("'", "''").strip()
        postal_code = (postal_code or "").replace("'", "''").strip()

        # Conservative filter to avoid overly strict matching
        filters = [f"contains(tolower(UnparsedAddress),'{addr}')"] if addr else []
        if city:
            filters.append(f"City eq '{city}'")
        if postal_code:
            filters.append(f"PostalCode eq '{postal_code}'")

        filt = " and ".join(filters) if filters else "true"

        return (self.odata_get("Property", params={"$filter": filt, "$top": str(top)}).get("value") or [])

    def get_property_by_listing_key(self, listing_key: str, expand_media: bool = True) -> Optional[Dict[str, Any]]:
        """
        Northstar docs show Property('7145271') with ListingKey present. :contentReference[oaicite:6]{index=6}
        We optionally try $expand=Media (if the navigation property exists).
        """
        listing_key = _clean(listing_key)
        if not listing_key:
            return None

        params = {}
        if expand_media:
            params["$expand"] = "Media"  # if supported in this dataset

        try:
            return self.odata_get(f"Property('{listing_key}')", params=params)
        except requests.HTTPError:
            # fallback without expand
            return self.odata_get(f"Property('{listing_key}')")

    def get_media_for_listing_key(self, listing_key: str, top: int = 200) -> List[Dict[str, Any]]:
        """
        Media retrieval. Northstar docs show Media('...') returning MediaURL. :contentReference[oaicite:7]{index=7}

        Linking field varies by MLS implementation.
        We try common filters in sequence and return the first non-empty set.
        """
        listing_key = _clean(listing_key)
        if not listing_key:
            return []

        candidates = [
            ("ListingKey", f"ListingKey eq '{listing_key}'"),
            ("ResourceRecordKey", f"ResourceRecordKey eq '{listing_key}'"),
            ("ResourceRecordID", f"ResourceRecordID eq '{listing_key}'"),
        ]

        for _, filt in candidates:
            try:
                data = self.odata_get(
                    "Media",
                    params={"$filter": filt, "$top": str(top), "$orderby": "Order asc"},
                )
                vals = data.get("value") or []
                if vals:
                    return vals
            except requests.HTTPError:
                continue

        return []
