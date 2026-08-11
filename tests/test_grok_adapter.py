from __future__ import annotations

import base64
import importlib
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

    def info(self, *_args, **_kwargs):
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

GrokAdapter = importlib.import_module(
    "astrbot_plugin_image_generation.adapter.grok_adapter"
).GrokAdapter
types_module = importlib.import_module(
    "astrbot_plugin_image_generation.core.shared.types"
)
AdapterConfig = types_module.AdapterConfig
AdapterType = types_module.AdapterType
GenerationRequest = types_module.GenerationRequest
ImageData = types_module.ImageData


def _adapter() -> GrokAdapter:
    return GrokAdapter(
        AdapterConfig(
            type=AdapterType.GROK,
            name="xAI",
            model="grok-imagine-image",
            api_keys=["test-key"],
        )
    )


def _png_image(
    *,
    source_url: str | None = None,
    payload: bytes = b"\x89PNG\r\n\x1a\nfake",
) -> ImageData:
    return ImageData(data=payload, mime_type="image/png", source_url=source_url)


class GrokAdapterPayloadTests(unittest.TestCase):
    def test_text_to_image_payload_omits_image_fields(self):
        adapter = _adapter()
        request = GenerationRequest(
            prompt="a cat",
            aspect_ratio="16:9",
            resolution="2K",
        )

        payload = adapter._build_payload(request)

        self.assertEqual(payload["model"], "grok-imagine-image")
        self.assertEqual(payload["prompt"], "a cat")
        self.assertEqual(payload["n"], 1)
        self.assertEqual(payload["response_format"], "b64_json")
        self.assertEqual(payload["aspect_ratio"], "16:9")
        self.assertEqual(payload["resolution"], "2k")
        self.assertNotIn("image", payload)
        self.assertNotIn("images", payload)

    def test_single_edit_uses_image_object(self):
        adapter = _adapter()
        image = _png_image()
        request = GenerationRequest(prompt="make blue", images=[image])

        payload = adapter._build_payload(request)

        self.assertIn("image", payload)
        self.assertNotIn("images", payload)
        self.assertEqual(payload["image"]["type"], "image_url")
        self.assertTrue(payload["image"]["url"].startswith("data:image/png;base64,"))
        encoded = payload["image"]["url"].split(",", 1)[1]
        self.assertEqual(base64.b64decode(encoded), image.data)

    def test_single_edit_prefers_http_source_url(self):
        adapter = _adapter()
        request = GenerationRequest(
            prompt="sketch",
            images=[
                _png_image(source_url="https://example.com/ref.png"),
            ],
        )

        payload = adapter._build_payload(request)

        self.assertEqual(
            payload["image"],
            {
                "url": "https://example.com/ref.png",
                "type": "image_url",
            },
        )

    def test_multi_edit_uses_images_array_and_truncates_to_three(self):
        adapter = _adapter()
        images = [
            _png_image(
                source_url=f"https://example.com/{index}.png", payload=bytes([index])
            )
            for index in range(4)
        ]
        request = GenerationRequest(prompt="blend", images=images)

        payload = adapter._build_payload(request)

        self.assertNotIn("image", payload)
        self.assertEqual(len(payload["images"]), 3)
        self.assertEqual(
            payload["images"][0],
            {"url": "https://example.com/0.png", "type": "image_url"},
        )
        self.assertEqual(
            payload["images"][2],
            {"url": "https://example.com/2.png", "type": "image_url"},
        )

    def test_endpoint_selection(self):
        adapter = _adapter()
        self.assertEqual(
            adapter._endpoint_url("/images/generations"),
            "https://api.x.ai/v1/images/generations",
        )
        adapter.base_url = "https://api.futureppo.top"
        self.assertEqual(
            adapter._endpoint_url("/images/edits"),
            "https://api.futureppo.top/v1/images/edits",
        )


if __name__ == "__main__":
    unittest.main()
