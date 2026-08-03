from __future__ import annotations

import asyncio
import base64
import importlib
import json
import sys
import types
import unittest
from pathlib import Path


class _Logger:
    def debug(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass


ROOT = Path(__file__).resolve().parents[1]
astrbot_module = types.ModuleType("astrbot")
astrbot_api_module = types.ModuleType("astrbot.api")
astrbot_api_module.logger = _Logger()
sys.modules.setdefault("astrbot", astrbot_module)
sys.modules.setdefault("astrbot.api", astrbot_api_module)

plugin_module = types.ModuleType("astrbot_plugin_image_generation")
plugin_module.__path__ = [str(ROOT)]
adapter_module = types.ModuleType("astrbot_plugin_image_generation.adapter")
adapter_module.__path__ = [str(ROOT / "adapter")]
core_module = types.ModuleType("astrbot_plugin_image_generation.core")
core_module.__path__ = [str(ROOT / "core")]
core_adapters_module = types.ModuleType("astrbot_plugin_image_generation.core.adapters")
core_adapters_module.__path__ = [str(ROOT / "core" / "adapters")]
core_shared_module = types.ModuleType("astrbot_plugin_image_generation.core.shared")
core_shared_module.__path__ = [str(ROOT / "core" / "shared")]
sys.modules.setdefault("astrbot_plugin_image_generation", plugin_module)
sys.modules.setdefault("astrbot_plugin_image_generation.adapter", adapter_module)
sys.modules.setdefault("astrbot_plugin_image_generation.core", core_module)
sys.modules.setdefault(
    "astrbot_plugin_image_generation.core.adapters", core_adapters_module
)
sys.modules.setdefault(
    "astrbot_plugin_image_generation.core.shared", core_shared_module
)

OpenAIAdapter = importlib.import_module(
    "astrbot_plugin_image_generation.adapter.openai_adapter"
).OpenAIAdapter
types_module = importlib.import_module(
    "astrbot_plugin_image_generation.core.shared.types"
)
AdapterConfig = types_module.AdapterConfig
AdapterType = types_module.AdapterType
GenerationRequest = types_module.GenerationRequest


class _StreamContent:
    def __init__(self, chunks: list[bytes]):
        self.chunks = chunks

    def iter_any(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self.chunks:
            yield chunk


class _Response:
    status = 200
    headers = {"Content-Type": "application/json"}

    def __init__(self, *, data: bytes = b"image", chunks: list[bytes] | None = None):
        self.data = data
        self.content = _StreamContent(chunks or [])

    async def read(self):
        return self.data

    async def json(self):
        return {"data": [{"b64_json": base64.b64encode(self.data).decode()}]}


class _Context:
    def __init__(
        self,
        response: _Response | None = None,
        error: Exception | None = None,
    ):
        self.response = response
        self.error = error

    async def __aenter__(self):
        if self.error:
            raise self.error
        return self.response

    async def __aexit__(self, *_args):
        return None


class _Session:
    closed = False

    def __init__(self, *, post_response: _Response | None = None, get_error=None):
        self.post_response = post_response or _Response()
        self.get_error = get_error
        self.post_kwargs = None
        self.get_timeout = None

    def post(self, *_args, **kwargs):
        self.post_kwargs = kwargs
        return _Context(self.post_response)

    def get(self, *_args, **kwargs):
        self.get_timeout = kwargs["timeout"]
        return _Context(_Response(), self.get_error)


def _adapter(session: _Session, *, retries: int = 3) -> OpenAIAdapter:
    config = AdapterConfig(
        type=AdapterType.OPENAI,
        name="test",
        base_url="https://example.test",
        api_keys=["key"],
        model="gpt-image-2",
        timeout=330,
        max_retry_attempts=retries,
        extra={"model_family": "gpt-image"},
    )
    adapter = OpenAIAdapter(config)
    adapter._session = session
    return adapter


class OpenAIAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_gpt_image_generation_requests_streaming(self):
        session = _Session()
        images, error = await _adapter(session)._generate_once(
            GenerationRequest(prompt="test")
        )

        self.assertEqual(images, [b"image"])
        self.assertIsNone(error)
        self.assertIs(session.post_kwargs["json"]["stream"], True)

    async def test_sse_completed_event_survives_tcp_chunk_boundaries(self):
        encoded = base64.b64encode(b"image").decode()
        event = json.dumps({"type": "image_generation.completed", "b64_json": encoded})
        wire = f"event: image_generation.completed\ndata: {event}\n\ndata: [DONE]\n\n"
        chunks = [wire[:17].encode(), wire[17:49].encode(), wire[49:].encode()]
        response = _Response(chunks=chunks)

        data = await _adapter(_Session())._read_stream_response(response, "task")

        self.assertEqual(data["data"][0]["b64_json"], encoded)

    async def test_url_download_uses_provider_timeout(self):
        session = _Session()
        images, error = await _adapter(session)._extract_images(
            {"data": [{"url": "https://example.test/image.png"}]}, "task"
        )

        self.assertEqual(images, [b"image"])
        self.assertIsNone(error)
        self.assertEqual(session.get_timeout.total, 330)

    async def test_empty_timeout_exception_preserves_its_type(self):
        session = _Session()
        session.post = lambda *_args, **_kwargs: _Context(error=asyncio.TimeoutError())

        images, error = await _adapter(session)._generate_once(
            GenerationRequest(prompt="test")
        )

        self.assertIsNone(images)
        self.assertEqual(error, "TimeoutError: TimeoutError()")

    async def test_download_failure_does_not_resubmit_generation(self):
        session = _Session(get_error=asyncio.TimeoutError())
        adapter = _adapter(session, retries=3)
        calls = 0

        async def generated_then_download_failed(_request):
            nonlocal calls
            calls += 1
            return await adapter._extract_images(
                {"data": [{"url": "https://example.test/image.png"}]}, "task"
            )

        adapter._generate_once = generated_then_download_failed
        result = await adapter.generate(GenerationRequest(prompt="test"))

        self.assertEqual(calls, 1)
        self.assertIsNone(result.images)
        self.assertIn("TimeoutError", result.error)


if __name__ == "__main__":
    unittest.main()
