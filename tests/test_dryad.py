import hashlib
import io
import json
import urllib.error
import zipfile

import pytest

from ibci_datasets.dryad import DryadAuthError, DryadClient, DryadError, DryadFile, safe_extract

PAYLOAD = b"x" * 3000
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


class FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, status: int = 200):
        super().__init__(body)
        self.status = status


class FakeDryad:
    """Answers the handful of Dryad endpoints the client uses."""

    def __init__(self, require_token="tok-1"):
        self.require_token = require_token
        self.requests = []

    def __call__(self, req):
        self.requests.append(req)
        url = req.full_url
        if url.endswith("/oauth/token"):
            return FakeResponse(json.dumps({"access_token": "tok-1", "expires_in": 36000}).encode())
        if "/api/v2/datasets/" in url:
            return FakeResponse(json.dumps({"_links": {"stash:version": {"href": "/api/v2/versions/9"}}}).encode())
        if url.endswith("/versions/9/files"):
            return FakeResponse(json.dumps({"_embedded": {"stash:files": [{
                "path": "data.zip", "size": len(PAYLOAD), "digest": DIGEST, "digestType": "sha-256",
                "_links": {"stash:download": {"href": "/api/v2/files/1/download"}},
            }]}}).encode())
        if url.endswith("/files/1/download"):
            if req.get_header("Authorization") != f"Bearer {self.require_token}":
                raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, None)
            rng = req.get_header("Range")
            if rng:
                start = int(rng.split("=")[1].rstrip("-"))
                return FakeResponse(PAYLOAD[start:], status=206)
            return FakeResponse(PAYLOAD)
        raise AssertionError(f"unexpected URL {url}")


def test_lists_files_without_credentials():
    client = DryadClient(opener=FakeDryad())
    (f,) = client.files("doi:10.5061/dryad.test")
    assert f.path == "data.zip" and f.size == 3000
    assert f.download_url == "https://datadryad.org/api/v2/files/1/download"


def test_download_without_credentials_explains_how(tmp_path):
    client = DryadClient(opener=FakeDryad())
    (f,) = client.files("doi:x")
    with pytest.raises(DryadAuthError, match="DRYAD_CLIENT_ID"):
        client.download(f, tmp_path)


def test_download_fetches_token_and_verifies(tmp_path):
    fake = FakeDryad()
    client = DryadClient(client_id="id", client_secret="secret", opener=fake)
    (f,) = client.files("doi:x")
    path = client.download(f, tmp_path)
    assert path.read_bytes() == PAYLOAD
    assert sum(r.full_url.endswith("/oauth/token") for r in fake.requests) == 1
    # a second call is a no-op: file already complete and verified
    client.download(f, tmp_path)
    assert sum(r.full_url.endswith("/download") for r in fake.requests) == 1


def test_download_resumes_partial_file(tmp_path):
    fake = FakeDryad()
    client = DryadClient(token="tok-1", opener=fake)
    (f,) = client.files("doi:x")
    (tmp_path / "data.zip.part").write_bytes(PAYLOAD[:1000])
    path = client.download(f, tmp_path)
    assert path.read_bytes() == PAYLOAD
    assert fake.requests[-1].get_header("Range") == "bytes=1000-"


def test_bad_checksum_is_rejected(tmp_path):
    client = DryadClient(token="tok-1", opener=FakeDryad())
    (f,) = client.files("doi:x")
    bad = DryadFile(f.path, f.size, "0" * 64, "sha-256", f.download_url)
    with pytest.raises(DryadError, match="checksum"):
        client.download(bad, tmp_path)
    assert not (tmp_path / "data.zip.part").exists()


def test_rejected_token_raises_auth_error(tmp_path):
    client = DryadClient(token="wrong", opener=FakeDryad())
    (f,) = client.files("doi:x")
    with pytest.raises(DryadAuthError):
        client.download(f, tmp_path)


@pytest.mark.parametrize("name", ["../evil.zip", "a/../../evil", ".."])
def test_unsafe_names_are_refused(name):
    f = DryadFile(name, 1, None, None, "https://example.invalid")
    if name.endswith("evil.zip") or name.endswith("evil"):
        assert f.safe_name in ("evil.zip", "evil")  # only the base name is ever used
    else:
        with pytest.raises(DryadError):
            f.safe_name


def test_safe_extract_blocks_traversal_and_filters(tmp_path):
    zpath = tmp_path / "a.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("hdf5_data_final/t15.2023.08.11/data_train.hdf5", b"h5")
        z.writestr("hdf5_data_final/notes.pkl", b"pickle")
    out = safe_extract(zpath, tmp_path / "out", [".hdf5"])
    assert [p.name for p in out] == ["data_train.hdf5"]

    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as z:
        z.writestr("../escape.txt", b"no")
    with pytest.raises(DryadError, match="outside"):
        safe_extract(evil, tmp_path / "out2")
