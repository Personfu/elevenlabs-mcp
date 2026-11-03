"""Regression coverage for CVE-2026-48710 using in-memory ASGI requests."""
import pytest
from starlette.requests import Request


def request_with_host(host):
    return Request({
        "type": "http", "method": "GET", "scheme": "http",
        "path": "/private/tools", "raw_path": b"/private/tools",
        "root_path": "", "query_string": b"", "server": ("127.0.0.1", 8000),
        "headers": [(b"host", host)],
    })


@pytest.mark.parametrize("host", [
    b"example.com/public?ignored=", b"example.com?ignored=",
    b"example.com#ignored", b"user@example.com", b"example.com/../public",
    b"example.com with-space",
])
def test_malformed_host_cannot_hide_the_protected_path(host):
    request = request_with_host(host)
    assert request.url.path == "/private/tools"
    assert request.url.hostname == "127.0.0.1"


def test_valid_host_is_preserved():
    request = request_with_host(b"example.com:8443")
    assert request.url.path == "/private/tools"
    assert request.url.hostname == "example.com"
    assert request.url.port == 8443
