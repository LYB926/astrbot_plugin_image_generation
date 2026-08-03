from __future__ import annotations

import base64
import time
from typing import Any

import aiohttp

from astrbot.api import logger

from ..core.adapters.base import BaseImageAdapter
from ..core.shared.constants import UNSPECIFIED_OPTION
from ..core.shared.logging import safe_log_error_body
from ..core.shared.types import GenerationRequest, ImageCapability


class OpenAIAdapter(BaseImageAdapter):
    """OpenAI image generation adapter for DALL-E and GPT Image models."""

    def get_capabilities(self) -> ImageCapability:
        """Return adapter capabilities."""
        return self._get_configured_capabilities()

    def _is_gpt_image_model(self) -> bool:
        """Return whether the active model is a GPT Image model."""
        model_family = self.config.extra.get("model_family", "auto")
        if model_family == "gpt-image":
            return True
        if model_family == "dall-e":
            return False
        # auto: infer the family from the model name.
        return self.model is not None and "gpt-image" in self.model

    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        """Execute one image generation request."""
        start_time = time.time()
        prefix = self._get_log_prefix(request.task_id)

        is_gpt = self._is_gpt_image_model()
        use_edit = bool(request.images) and is_gpt
        if request.images and not is_gpt:
            logger.warning(
                f"{prefix} 提供了参考图但当前模型不支持图生图，仅 GPT Image 系列支持图生图，参考图将被忽略"
            )
        session = self._get_session()
        base = self.base_url.rstrip("/") if self.base_url else "https://api.openai.com"
        headers = {"Authorization": f"Bearer {self._get_current_api_key()}"}

        if use_edit:
            url = f"{base}/v1/images/edits"
            form = aiohttp.FormData()
            form.add_field("model", self.model or "gpt-image-1")
            form.add_field("prompt", request.prompt)
            form.add_field("n", "1")
            if size := self._map_aspect_ratio_to_size(
                request.aspect_ratio, gpt_model=True
            ):
                form.add_field("size", size)
            for img in request.images:
                form.add_field(
                    "image[]",
                    img.data,
                    content_type=img.mime_type,
                    filename="image",
                )
            kwargs: dict = {"data": form}
        else:
            url = f"{base}/v1/images/generations"
            headers["Content-Type"] = "application/json"
            payload = self._build_payload(request)
            kwargs = {"json": payload}
            self._log_request_overview(request, url, payload=payload)
            self._log_debug_json("请求", payload, request.task_id)
        if use_edit:
            self._log_request_overview(
                request,
                url,
                form_fields=["model", "prompt", "n", "size", "image[]"],
            )

        try:
            async with session.post(
                url,
                headers=headers,
                proxy=self.proxy,
                timeout=self._get_timeout(),
                **kwargs,
            ) as resp:
                duration = time.time() - start_time
                self._log_response_status(request, resp.status, duration)
                if resp.status != 200:
                    error_text = await resp.text()
                    self._log_debug_json_text("响应", error_text, request.task_id)
                    self._log_api_error(request, resp.status, duration, error_text)
                    return None, self._format_api_error_message(
                        resp.status,
                        error_text,
                    )
                data = await self._read_response_json(resp, request.task_id)
                return await self._extract_images(data)
        except Exception as e:
            duration = time.time() - start_time
            self._log_request_exception(request, duration, e)
            return None, safe_log_error_body(e)

    def _build_payload(self, request: GenerationRequest) -> dict:
        """Build the request payload."""
        gpt = self._is_gpt_image_model()
        payload: dict[str, Any] = {
            "model": self.model or "dall-e-3",
            "prompt": request.prompt,
            "n": 1,
        }

        if size := self._map_aspect_ratio_to_size(request.aspect_ratio, gpt_model=gpt):
            payload["size"] = size
        # OpenAI models do not support the plugin resolution setting; quality is separate.
        if not gpt:
            # GPT Image models always return b64_json and do not support response_format.
            payload["response_format"] = "b64_json"

        return payload

    def _map_aspect_ratio_to_size(
        self, aspect_ratio: str | None, gpt_model: bool
    ) -> str | None:
        """Map an aspect ratio to an OpenAI-supported size parameter."""
        if not aspect_ratio or aspect_ratio == UNSPECIFIED_OPTION:
            return None

        if gpt_model:
            # GPT Image models support only square, landscape, and portrait sizes.
            # Map unsupported ratios to the closest supported size.
            mapping = {
                "1:1": "1024x1024",
                "3:2": "1536x1024",
                "16:9": "1536x1024",
                "4:3": "1536x1024",
                "5:4": "1536x1024",
                "21:9": "1536x1024",
                "2:3": "1024x1536",
                "3:4": "1024x1536",
                "9:16": "1024x1536",
                "4:5": "1024x1536",
            }
        else:
            # DALL-E 3 supports only square, landscape, and portrait sizes.
            # Map unsupported ratios to the closest supported size.
            mapping = {
                "1:1": "1024x1024",
                "3:2": "1792x1024",
                "16:9": "1792x1024",
                "4:3": "1792x1024",
                "5:4": "1792x1024",
                "21:9": "1792x1024",
                "2:3": "1024x1792",
                "3:4": "1024x1792",
                "9:16": "1024x1792",
                "4:5": "1024x1792",
            }
        return mapping.get(aspect_ratio)

    async def _extract_images(
        self, response: dict
    ) -> tuple[list[bytes] | None, str | None]:
        """Extract image bytes from the response payload."""
        if "data" not in response:
            return None, "响应中未找到 data 字段"

        images = []
        for item in response["data"]:
            if "b64_json" in item:
                images.append(base64.b64decode(item["b64_json"]))
            elif "url" in item:
                # Download URL results even though b64_json is requested.
                async with self._get_session().get(
                    item["url"], proxy=self.proxy, timeout=self._get_download_timeout()
                ) as resp:
                    if resp.status == 200:
                        images.append(await resp.read())

        if not images:
            return None, "未找到有效的图片数据"

        return images, None
