import json

from dinogenept.provenance import atomic_write_json, digest_file, sanitized_config


def test_atomic_json_and_digest(tmp_path, monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "never-serialize-this")
    path = tmp_path / "manifest.json"
    payload = sanitized_config(
        model="doubao-embedding-vision",
        base_url="https://ark.cn-beijing.volces.com/api/plan/v3",
        dimensions=1024,
        credential_variable="ARK_API_KEY",
    )
    atomic_write_json(path, payload)
    rendered = path.read_text()
    assert "never-serialize-this" not in rendered
    assert json.loads(rendered)["credential_present"] is True
    assert len(digest_file(path)) == 64

