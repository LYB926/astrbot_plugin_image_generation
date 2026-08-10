from __future__ import annotations

import base64
import time
from typing import Any

import aiohttp

from ..core.adapters.base import BaseImageAdapter
from ..core.shared.constants import UNSPECIFIED_OPTION
from ..core.shared.logging import safe_log_error_body
from ..core.shared.types import GenerationRequest, ImageCapability


class GrokAdapter(BaseImageAdapter):
    """Grok (xAI) image generation adapter."""

    def get_capabilities(self) -> ImageCapability:
        """Return adapter capabilities."""
        return self._get_configured_capabilities()

    # generate() is provided by the base class via the template method pattern.

    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        """Execute one image generation request."""
        start_time = time.time()

        session = self._get_session()

        if request.images:
            end_point = "/images/edits"
        else:
            end_point = "/images/generations"

        if not self.base_url:
            url = f"https://api.x.ai/v1{end_point}"
        else:
            # main.py strips /v1, so add it here consistently.
            url = f"{self.base_url.rstrip('/')}/v1{end_point}"

        headers = {"Authorization": f"Bearer {self._get_current_api_key()}"}
        if request.images:
            form = aiohttp.FormData()
            form.add_field("model", self.model or "grok-imagine-image")
            form.add_field("prompt", request.prompt)
            form.add_field("response_format", "b64_json")
            if ratio := self._get_aspect_ratio(request):
                form.add_field("aspect_ratio", ratio)
            if resolution := self._get_resolution(request):
                form.add_field("resolution", resolution)
            for index, image in enumerate(request.images, start=1):
                form.add_field(
                    "image",
                    image.data,
                    content_type=image.mime_type,
                    filename=self._image_filename(image.mime_type, index),
                )
            kwargs: dict[str, Any] = {"data": form}
            self._log_request_overview(
                request,
                url,
                form_fields=[
                    "model",
                    "prompt",
                    "response_format",
                    "aspect_ratio",
                    "resolution",
                    "image",
                ],
            )
        else:
            payload = self._build_payload(request)
            headers["Content-Type"] = "application/json"
            kwargs = {"json": payload}
            self._log_request_overview(request, url, payload=payload)
            self._log_debug_json("请求", payload, request.task_id)

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

        accept_ratio = [
            "auto",
            "1:1",
            "16:9",
            "9:16",
            "4:3",
            "3:4",
            "3:2",
            "2:3",
            "1:2",
            "2:1",
            "19.5:9",
            "9:19.5",
            "20:9",
            "9:20",
        ]
        accept_resolution = ["1k", "2k"]

        ratio = self._get_aspect_ratio(request, accept_ratio)
        resolution = self._get_resolution(request, accept_resolution)

        images_ref = []
        for image in request.images:
            b64_data = base64.b64encode(image.data).decode("utf-8")
            images_ref.append(
                {
                    "type": "image_url",
                    "url": f"data:{image.mime_type};base64,{b64_data}",
                }
            )

        payload: dict[str, Any] = {
            "model": self.model or "grok-imagine-image",
            "prompt": request.prompt,
            "response_format": "b64_json",
        }
        if ratio:
            payload["aspect_ratio"] = ratio
        if resolution:
            payload["resolution"] = resolution

        if len(images_ref) > 0:
            payload.update({"images": images_ref})

        return payload

    def _get_aspect_ratio(
        self,
        request: GenerationRequest,
        accepted: list[str] | None = None,
    ) -> str | None:
        """Validate and return the requested Grok aspect ratio."""
        accepted = accepted or [
            "auto",
            "1:1",
            "16:9",
            "9:16",
            "4:3",
            "3:4",
            "3:2",
            "2:3",
            "1:2",
            "2:1",
            "19.5:9",
            "9:19.5",
            "20:9",
            "9:20",
        ]
        return request.aspect_ratio if request.aspect_ratio in accepted else None

    def _get_resolution(
        self,
        request: GenerationRequest,
        accepted: list[str] | None = None,
    ) -> str | None:
        """Validate and return the requested Grok resolution."""
        accepted = accepted or ["1k", "2k"]
        value = (request.resolution or "").lower()
        return value if value != UNSPECIFIED_OPTION and value in accepted else None

    def _image_filename(self, mime_type: str, index: int) -> str:
        """Return a filename whose extension matches the uploaded image bytes."""
        extension = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/gif": ".gif",
            "image/heic": ".heic",
            "image/heif": ".heif",
        }.get((mime_type or "").lower(), ".png")
        return f"reference_{index}{extension}"

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
