"""Minimal Dryad API v2 client: list a dataset's files, then download them.

Listing is public. Since 2026, Dryad requires a bearer token for every file download,
which broke the download scripts shipped with most BrainGate papers. Get credentials by
creating an API account linked to an ORCID-backed Dryad account (see
https://github.com/datadryad/dryad-app/blob/main/documentation/apis/api_accounts.md),
then set either

* ``DRYAD_CLIENT_ID`` and ``DRYAD_CLIENT_SECRET`` (tokens are fetched and refreshed
  automatically; they last about 10 hours), or
* ``DRYAD_TOKEN`` (a token you fetched yourself).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable

BASE = "https://datadryad.org"
API = f"{BASE}/api/v2"
USER_AGENT = "ibci-datasets (https://github.com/cabibbz/ibci-datasets)"
CHUNK = 1 << 20


class DryadError(RuntimeError):
    pass


class DryadAuthError(DryadError):
    pass


@dataclasses.dataclass(frozen=True)
class DryadFile:
    path: str
    size: int
    digest: str | None
    digest_type: str | None
    download_url: str

    @property
    def safe_name(self) -> str:
        """The file's base name, refusing anything that could escape the target folder."""
        name = PurePosixPath(self.path).name
        if name in ("", ".", "..") or "/" in name or "\\" in name:
            raise DryadError(f"refusing unsafe file name from Dryad: {self.path!r}")
        return name


Opener = Callable[[urllib.request.Request], "object"]


class DryadClient:
    def __init__(
        self,
        token: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        opener: Opener | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._token = token
        self._token_expiry = float("inf") if token else 0.0
        self._client_id = client_id
        self._client_secret = client_secret
        self._open = opener or (lambda req: urllib.request.urlopen(req, timeout=timeout))

    @classmethod
    def from_env(cls, **kwargs) -> "DryadClient":
        return cls(
            token=os.environ.get("DRYAD_TOKEN") or None,
            client_id=os.environ.get("DRYAD_CLIENT_ID") or None,
            client_secret=os.environ.get("DRYAD_CLIENT_SECRET") or None,
            **kwargs,
        )

    # -- metadata (public) ---------------------------------------------------------

    def _get_json(self, url: str) -> dict:
        req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
        with self._open(req) as resp:
            return json.load(resp)

    def dataset(self, doi: str) -> dict:
        return self._get_json(f"{API}/datasets/{urllib.parse.quote(doi, safe='')}")

    def files(self, doi: str) -> list[DryadFile]:
        """Files in the latest public version of a dataset."""
        version_href = self.dataset(doi)["_links"]["stash:version"]["href"]
        out: list[DryadFile] = []
        url: str | None = f"{BASE}{version_href}/files"
        while url:
            page = self._get_json(url)
            for f in page.get("_embedded", {}).get("stash:files", []):
                out.append(
                    DryadFile(
                        path=f["path"],
                        size=int(f["size"]),
                        digest=f.get("digest"),
                        digest_type=f.get("digestType"),
                        download_url=BASE + f["_links"]["stash:download"]["href"],
                    )
                )
            nxt = page.get("_links", {}).get("next", {}).get("href")
            url = BASE + nxt if nxt else None
        return out

    # -- auth ------------------------------------------------------------------------

    def token(self) -> str:
        if self._token and time.time() < self._token_expiry - 300:
            return self._token
        if not (self._client_id and self._client_secret):
            if self._token:
                return self._token
            raise DryadAuthError(
                "Dryad file downloads need credentials. Set DRYAD_CLIENT_ID and "
                "DRYAD_CLIENT_SECRET (or DRYAD_TOKEN); see the ibci_datasets.dryad docstring."
            )
        body = urllib.parse.urlencode(
            {
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "grant_type": "client_credentials",
            }
        ).encode()
        req = urllib.request.Request(
            f"{BASE}/oauth/token",
            data=body,
            headers={
                "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
                "User-Agent": USER_AGENT,
            },
            method="POST",
        )
        with self._open(req) as resp:
            payload = json.load(resp)
        self._token = payload["access_token"]
        self._token_expiry = time.time() + float(payload.get("expires_in", 36000))
        return self._token

    # -- download ----------------------------------------------------------------------

    def download(self, f: DryadFile, dest_dir: str | os.PathLike, verify: bool = True) -> Path:
        """Download one file, resuming a partial download and checking its SHA-256."""
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / f.safe_name
        if target.exists() and target.stat().st_size == f.size:
            if not verify or _matches(target, f):
                return target
            target.unlink()
        part = target.with_name(target.name + ".part")
        have = part.stat().st_size if part.exists() else 0
        if have > f.size:
            part.unlink()
            have = 0
        if have < f.size:
            headers = {"Authorization": f"Bearer {self.token()}", "User-Agent": USER_AGENT}
            if have:
                headers["Range"] = f"bytes={have}-"
            req = urllib.request.Request(f.download_url, headers=headers)
            try:
                resp = self._open(req)
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    raise DryadAuthError(f"Dryad refused the download ({e.code}); check your credentials") from e
                raise
            with resp:
                status = getattr(resp, "status", 200)
                mode = "ab" if have and status == 206 else "wb"
                with open(part, mode) as out:
                    while chunk := resp.read(CHUNK):
                        out.write(chunk)
        if part.stat().st_size != f.size:
            raise DryadError(f"{f.path}: got {part.stat().st_size} bytes, expected {f.size}")
        if verify and not _matches(part, f):
            part.unlink()
            raise DryadError(f"{f.path}: checksum mismatch; the partial file was deleted, retry")
        part.replace(target)
        return target


def _matches(path: Path, f: DryadFile) -> bool:
    if not f.digest or (f.digest_type or "").lower() not in ("sha-256", "sha256"):
        return True  # nothing to check against
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest() == f.digest


def safe_extract(zip_path: str | os.PathLike, dest_dir: str | os.PathLike,
                 suffixes: Iterable[str] | None = None) -> list[Path]:
    """Extract a zip without letting any member escape ``dest_dir``.

    ``suffixes`` limits extraction to matching file types (e.g. ``[".hdf5"]``), which
    also keeps pickles and model files out of the working tree.
    """
    dest = Path(dest_dir).resolve()
    keep = tuple(s.lower() for s in suffixes) if suffixes else None
    written: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if keep and not info.filename.lower().endswith(keep):
                continue
            out = (dest / info.filename).resolve()
            if dest != out and dest not in out.parents:
                raise DryadError(f"refusing zip member outside the target folder: {info.filename!r}")
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(out, "wb") as dst:
                while chunk := src.read(CHUNK):
                    dst.write(chunk)
            written.append(out)
    return written
