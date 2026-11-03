"""Validate SDK compatibility without invoking provider tools or paid APIs."""
import asyncio
import importlib
import socket
import sys
from unittest.mock import MagicMock


def test_all_tools_register_without_provider_calls(monkeypatch):
    # Windows creates a loopback socket pair when constructing the event loop.
    loop = asyncio.new_event_loop()
    def reject_network(*_args, **_kwargs):
        raise AssertionError("Tool registration must not open a network connection")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "fixture-key-never-valid")
    monkeypatch.setenv("ELEVENLABS_MCP_OUTPUT_MODE", "files")
    provider = MagicMock()
    monkeypatch.setattr("elevenlabs.client.ElevenLabs", provider)
    sys.modules.pop("elevenlabs_mcp.server", None)
    try:
        server = importlib.import_module("elevenlabs_mcp.server")
        tools = loop.run_until_complete(server.mcp.list_tools())
        names = [tool.name for tool in tools]
        assert "text_to_speech" in names
        assert len(names) == len(set(names))
        assert all(tool.inputSchema.get("type") == "object" for tool in tools)
        assert provider.call_count == 1
        assert provider.return_value.mock_calls == []
    finally:
        loop.close()
        if "elevenlabs_mcp.server" in sys.modules:
            sys.modules.pop("elevenlabs_mcp.server").custom_client.close()
