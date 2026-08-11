from __future__ import annotations

import base64
import time
from typing import Any

from astrbot.api import logger

from ..core.adapters.base import BaseImageAdapter
from ..core.shared.constants import UNSPECIFIED_OPTION
from ..core.shared.logging import safe_log_error_body
from ..core.shared.types import GenerationRequest, ImageCapability, ImageData

# xAI official multi-image editing supports up to 3 source images.
MAX_EDIT_IMAGES = 3
ACCEPTED_ASPECT_RATIOS = (
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
)
ACCEPTED_RESOLUTIONS = ("1k", "2k")


class GrokAdapter(BaseImageAdapter):
    """Grok (xAI) image generation adapter.

    Follows the official xAI Images REST API:
    - text-to-image: POST /v1/images/generations (application/json)
    - image editing: POST /v1/images/edits (application/json)
    - single source image uses ``image``
    - multiple source images use ``images`` (max 3), mutually exclusive with ``image``
    """

    def get_capabilities(self) -> ImageCapability:
        """Return adapter capabilities."""
        return self._get_configured_capabilities()

    # generate() is provided by the base class via the template method pattern.

    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        """Execute one image generation or edit request."""
        start_time = time.time()
        payload = self._build_payload(request)
        session = self._get_session()
        url = self._endpoint_url(
            "/images/edits" if request.images else "/images/generations"
        )
        headers = {
            "Authorization": f"Bearer {self._get_current_api_key()}",
            "Content-Type": "application/json",
        }
        self._log_request_overview(request, url, payload=payload)
        self._log_debug_json("请求", payload, request.task_id)

        try:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                proxy=self.proxy,
                timeout=self._get_timeout(),
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

    def _endpoint_url(self, end_point: str) -> str:
        """Build an xAI v1 image endpoint URL."""
        if not self.base_url:
            return f"https://api.x.ai/v1{end_point}"
        # Config loading may strip a trailing /v1, so re-add it consistently.
        return f"{self.base_url.rstrip('/')}/v1{end_point}"

    def _build_payload(self, request: GenerationRequest) -> dict[str, Any]:
        """Build the official xAI images generations/edits JSON payload."""
        payload: dict[str, Any] = {
            "model": self.model or "grok-imagine-image",
            "prompt": request.prompt,
            "n": 1,
            "response_format": "b64_json",
        }

        if ratio := self._get_aspect_ratio(request):
            payload["aspect_ratio"] = ratio
        if resolution := self._get_resolution(request):
            payload["resolution"] = resolution

        image_refs = self._build_image_refs(request.images, request.task_id)
        if not image_refs:
            return payload

        # Official API: single image uses `image`; multi-image uses `images`.
        if len(image_refs) == 1:
            payload["image"] = image_refs[0]
        else:
            payload["images"] = image_refs
        return payload

    def _build_image_refs(
        self,
        images: list[ImageData],
        task_id: str | None,
    ) -> list[dict[str, str]]:
        """Convert reference images to official xAI image objects."""
        if not images:
            return []

        selected = images
        if len(images) > MAX_EDIT_IMAGES:
            prefix = self._get_log_prefix(task_id)
            logger.warning(
                f"{prefix} xAI 图生图最多支持 {MAX_EDIT_IMAGES} 张参考图，"
                f"已截断多余参考图: 原始={len(images)}"
            )
            selected = images[:MAX_EDIT_IMAGES]

        refs: list[dict[str, str]] = []
        for image in selected:
            refs.append(
                {
                    "url": self._image_url_value(image),
                    "type": "image_url",
                }
            )
        return refs

    def _image_url_value(self, image: ImageData) -> str:
        """Return a public URL or base64 data URL for one reference image."""
        source = (image.source_url or "").strip()
        if source.startswith(("http://", "https://", "data:")):
            return source

        mime_type = (image.mime_type or "image/png").strip() or "image/png"
        b64_data = base64.b64encode(image.data).decode("utf-8")
        return f"data:{mime_type};base64,{b64_data}"

    def _get_aspect_ratio(self, request: GenerationRequest) -> str | None:
        """Validate and return the requested Grok aspect ratio."""
        value = (request.aspect_ratio or "").strip()
        if not value or value == UNSPECIFIED_OPTION:
            return None
        return value if value in ACCEPTED_ASPECT_RATIOS else None

    def _get_resolution(self, request: GenerationRequest) -> str | None:
        """Validate and return the requested Grok resolution."""
        value = (request.resolution or "").strip().lower()
        if not value or value == UNSPECIFIED_OPTION:
            return None
        return value if value in ACCEPTED_RESOLUTIONS else None

    async def _extract_images(
        self, response: dict
    ) -> tuple[list[bytes] | None, str | None]:
        """Extract image bytes from the response payload."""
        if "data" not in response:
            return None, "响应中未找到 data 字段"

        images = []
        for item in response["data"]:
            if not isinstance(item, dict):
                continue
            if "b64_json" in item and item["b64_json"]:
                images.append(base64.b64decode(item["b64_json"]))
            elif "url" in item and item["url"]:
                # Download URL results even though b64_json is requested.
                async with self._get_session().get(
                    item["url"], proxy=self.proxy, timeout=self._get_download_timeout()
                ) as resp:
                    if resp.status == 200:
                        images.append(await resp.read())

        if not images:
            return None, "未找到有效的图片数据"

        return images, None
